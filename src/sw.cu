#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)             \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";          \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

// ============================================================
// Config / helper structs
// ============================================================

struct SWBatchConfig {
    int window_len = 0;        // 0 => auto
    int stride = 0;            // 0 => auto
    int top_k = 8;             // refine top-K windows
    int batch_windows = 1024;  // windows per GPU launch batch
    int threads_per_block = 256;
    int refine_extra = 0;      // extra halo beyond overlap
};

struct WindowSpec {
    int start;
    int len;
};

struct WindowHit {
    int score;
    int start;
    int len;
    int end_col;  // column inside the coarse window
};

// ============================================================
// Device helpers
// ============================================================

__device__ __forceinline__ int sw_max4(int a, int b, int c, int d) {
    return max(max(a, b), max(c, d));
}

__device__ __forceinline__ int diag_len_dev(int k, int q_len, int d_len) {
    return min(min(k, q_len), min(d_len, q_len + d_len - k));
}

__device__ __forceinline__ int diag_start_row_dev(int k, int d_len) {
    return (k <= d_len) ? 1 : (k - d_len + 1);
}

// ============================================================
// Host helpers
// ============================================================

static std::vector<WindowSpec> build_overlapping_windows(
    int target_len,
    int window_len,
    int stride)
{
    std::vector<WindowSpec> out;
    if (target_len <= 0) return out;

    for (int start = 0;; start += stride) {
        int len = std::min(window_len, target_len - start);
        out.push_back({start, len});
        if (start + len >= target_len) break;
    }

    int tail_start = std::max(0, target_len - window_len);
    if (out.empty() || out.back().start != tail_start) {
        out.push_back({tail_start, target_len - tail_start});
    }

    std::sort(out.begin(), out.end(), [](const WindowSpec& a, const WindowSpec& b) {
        return a.start < b.start;
    });

    out.erase(std::unique(out.begin(), out.end(),
                          [](const WindowSpec& a, const WindowSpec& b) {
                              return a.start == b.start;
                          }),
              out.end());

    return out;
}

static std::vector<std::pair<int, int>> merge_intervals(
    std::vector<std::pair<int, int>> intervals)
{
    if (intervals.empty()) return {};

    std::sort(intervals.begin(), intervals.end());

    std::vector<std::pair<int, int>> merged;
    merged.push_back(intervals[0]);

    for (size_t i = 1; i < intervals.size(); ++i) {
        if (intervals[i].first <= merged.back().second) {
            merged.back().second = std::max(merged.back().second, intervals[i].second);
        } else {
            merged.push_back(intervals[i]);
        }
    }

    return merged;
}

// ============================================================
// Exact single-window kernel (your original style, renamed path)
// This is used during refinement for exact traceback.
// ============================================================

__global__ void sw_wavefront_kernel_exact(
    const char* d_q,
    const char* d_d,
    int* d_matrix,
    int q_len,
    int d_len,
    int match_score,
    int mismatch_score,
    int gap_score,
    int k)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int elements_in_diag =
        min(min(k, q_len), min(d_len, q_len + d_len - k));

    if (tid < elements_in_diag) {
        int start_row = (k <= d_len) ? 1 : (k - d_len + 1);
        int row = start_row + tid;
        int col = k - row + 1;
        int width = d_len + 1;

        char q_char = d_q[row - 1];
        char d_char = d_d[col - 1];

        int diag_score = d_matrix[(row - 1) * width + (col - 1)];
        if (q_char == d_char || q_char == 'N' || d_char == 'N') {
            diag_score += match_score;
        } else {
            diag_score += mismatch_score;
        }

        int up_score = d_matrix[(row - 1) * width + col] + gap_score;
        int left_score = d_matrix[row * width + (col - 1)] + gap_score;

        int max_val = max(0, max(diag_score, max(up_score, left_score)));
        d_matrix[row * width + col] = max_val;
    }
}

// ============================================================
// Exact single-window Smith-Waterman with traceback
// Keeps your original behavior, but renamed so the batched path
// can call it for final refinement.
// ============================================================

void run_cuda_smith_waterman_single(
    const std::string& q,
    const std::string& d,
    int& out_score,
    int& out_start,
    int& out_stop,
    std::string& out_aligned_q,
    std::string& out_aligned_d,
    float& out_time_ms)
{
    const int q_len = static_cast<int>(q.length());
    const int d_len = static_cast<int>(d.length());

    if (q_len == 0 || d_len == 0) {
        out_score = 0;
        out_start = 0;
        out_stop = -1;
        out_aligned_q.clear();
        out_aligned_d.clear();
        out_time_ms = 0.0f;
        return;
    }

    const int rows = q_len + 1;
    const int cols = d_len + 1;
    const size_t matrix_bytes =
        static_cast<size_t>(rows) * cols * sizeof(int);

    char* d_q = nullptr;
    char* d_d = nullptr;
    int* d_matrix = nullptr;

    CUDA_CHECK(cudaMalloc(&d_q, q_len * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_d, d_len * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_matrix, matrix_bytes));

    CUDA_CHECK(cudaMemcpy(d_q, q.data(), q_len * sizeof(char),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d, d.data(), d_len * sizeof(char),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_matrix, 0, matrix_bytes));

    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    CUDA_CHECK(cudaEventRecord(start_event));

    const int match = 2;
    const int mismatch = -1;
    const int gap = -1;
    const int total_diagonals = q_len + d_len - 1;

    for (int k = 1; k <= total_diagonals; ++k) {
        int elements =
            std::min(std::min(k, q_len),
                     std::min(d_len, q_len + d_len - k));
        int threadsPerBlock = 256;
        int blocksPerGrid = (elements + threadsPerBlock - 1) / threadsPerBlock;

        sw_wavefront_kernel_exact<<<blocksPerGrid, threadsPerBlock>>>(
            d_q, d_d, d_matrix, q_len, d_len, match, mismatch, gap, k);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&out_time_ms, start_event, stop_event));

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    std::vector<int> h_matrix(static_cast<size_t>(rows) * cols);
    CUDA_CHECK(cudaMemcpy(h_matrix.data(), d_matrix, matrix_bytes,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_d));
    CUDA_CHECK(cudaFree(d_matrix));

    int max_score = 0;
    int max_row = 0;
    int max_col = 0;

    for (int r = 1; r <= q_len; ++r) {
        for (int c = 1; c <= d_len; ++c) {
            if (h_matrix[r * cols + c] > max_score) {
                max_score = h_matrix[r * cols + c];
                max_row = r;
                max_col = c;
            }
        }
    }

    out_score = max_score;

    if (max_score == 0) {
        out_start = 0;
        out_stop = -1;
        out_aligned_q.clear();
        out_aligned_d.clear();
        return;
    }

    out_stop = max_col - 1;

    std::string aligned_q;
    std::string aligned_d;

    int r = max_row;
    int c = max_col;

    while (r > 0 && c > 0 && h_matrix[r * cols + c] > 0) {
        int current_score = h_matrix[r * cols + c];
        int diag_score = h_matrix[(r - 1) * cols + (c - 1)];
        int up_score = h_matrix[(r - 1) * cols + c];
        int left_score = h_matrix[r * cols + (c - 1)];

        char q_char = q[r - 1];
        char d_char = d[c - 1];

        int match_val =
            (q_char == d_char || q_char == 'N' || d_char == 'N')
                ? match
                : mismatch;

        if (current_score == diag_score + match_val) {
            aligned_q.push_back(q_char);
            aligned_d.push_back(d_char);
            --r;
            --c;
        } else if (current_score == left_score + gap) {
            aligned_q.push_back('-');
            aligned_d.push_back(d_char);
            --c;
        } else if (current_score == up_score + gap) {
            aligned_q.push_back(q_char);
            aligned_d.push_back('-');
            --r;
        } else {
            break;
        }
    }

    std::reverse(aligned_q.begin(), aligned_q.end());
    std::reverse(aligned_d.begin(), aligned_d.end());

    out_start = c;
    out_aligned_q = aligned_q;
    out_aligned_d = aligned_d;
}

// ============================================================
// Batched coarse score-only kernel
// One block = one window
// Uses rolling diagonals only
// ============================================================

__global__ void sw_batch_score_kernel(
    const char* __restrict__ d_q,
    const char* __restrict__ d_windows,
    const int* __restrict__ d_win_lens,
    int q_len,
    int window_stride,
    int batch_size,
    int match_score,
    int mismatch_score,
    int gap_score,
    int* __restrict__ d_prev2,
    int* __restrict__ d_prev1,
    int* __restrict__ d_curr,
    int* __restrict__ d_best_scores,
    int* __restrict__ d_best_endcols)
{
    int wid = blockIdx.x;
    if (wid >= batch_size) return;

    int tid = threadIdx.x;
    int rows = q_len + 1;

    const char* win = d_windows + static_cast<size_t>(wid) * window_stride;
    int d_len = d_win_lens[wid];

    int* prev2 = d_prev2 + static_cast<size_t>(wid) * rows;
    int* prev1 = d_prev1 + static_cast<size_t>(wid) * rows;
    int* curr  = d_curr  + static_cast<size_t>(wid) * rows;

    int local_best_score = 0;
    int local_best_endcol = 0;

    for (int k = 1; k <= q_len + d_len - 1; ++k) {
        int start_row = diag_start_row_dev(k, d_len);
        int elems = diag_len_dev(k, q_len, d_len);

        for (int idx = tid; idx < elems; idx += blockDim.x) {
            int row = start_row + idx;
            int col = k - row + 1;

            char q_char = d_q[row - 1];
            char d_char = win[col - 1];

            int sub = (q_char == d_char || q_char == 'N' || d_char == 'N')
                          ? match_score
                          : mismatch_score;

            int up   = (row > 1)             ? prev1[row - 1] : 0;
            int left = (col > 1)             ? prev1[row]     : 0;
            int diag = (row > 1 && col > 1)  ? prev2[row - 1] : 0;

            int score = sw_max4(0, diag + sub, up + gap_score, left + gap_score);
            curr[row] = score;

            if (score > local_best_score) {
                local_best_score = score;
                local_best_endcol = col;
            }
        }

        __syncthreads();

        int* tmp = prev2;
        prev2 = prev1;
        prev1 = curr;
        curr = tmp;

        __syncthreads();
    }

    extern __shared__ int s_reduce[];
    int* s_scores  = s_reduce;
    int* s_endcols = s_reduce + blockDim.x;

    s_scores[tid] = local_best_score;
    s_endcols[tid] = local_best_endcol;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_scores[tid + stride] > s_scores[tid]) {
                s_scores[tid] = s_scores[tid + stride];
                s_endcols[tid] = s_endcols[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_best_scores[wid] = s_scores[0];
        d_best_endcols[wid] = s_endcols[0];
    }
}

// ============================================================
// Batched multi-window runner
// Stage 1: coarse GPU score-only scan over many windows
// Stage 2: exact refinement on top-K expanded merged regions
// ============================================================

void run_cuda_smith_waterman_batched(
    const std::string& q,
    const std::string& target,
    int& out_score,
    int& out_start,
    int& out_stop,
    std::string& out_aligned_q,
    std::string& out_aligned_d,
    float& out_time_ms,
    SWBatchConfig cfg = {})
{
    const int q_len = static_cast<int>(q.size());
    const int d_len = static_cast<int>(target.size());

    out_score = 0;
    out_start = 0;
    out_stop = -1;
    out_aligned_q.clear();
    out_aligned_d.clear();
    out_time_ms = 0.0f;

    if (q_len == 0 || d_len == 0) {
        return;
    }

    // Accuracy-oriented defaults
    if (cfg.window_len == 0) cfg.window_len = std::max(8 * q_len, 1024);
    if (cfg.stride == 0)     cfg.stride     = std::max(2 * q_len, 256);
    if (cfg.stride >= cfg.window_len) {
        cfg.stride = std::max(1, cfg.window_len / 4);
    }

    const int overlap = cfg.window_len - cfg.stride;
    const int halo = overlap + cfg.refine_extra;

    std::vector<WindowSpec> windows =
        build_overlapping_windows(d_len, cfg.window_len, cfg.stride);

    if (windows.empty()) {
        return;
    }

    char* d_q = nullptr;
    char* d_windows = nullptr;
    int* d_win_lens = nullptr;
    int* d_prev2 = nullptr;
    int* d_prev1 = nullptr;
    int* d_curr = nullptr;
    int* d_best_scores = nullptr;
    int* d_best_endcols = nullptr;

    const int rows = q_len + 1;
    const int max_batch = cfg.batch_windows;

    CUDA_CHECK(cudaMalloc(&d_q, q_len * sizeof(char)));
    CUDA_CHECK(cudaMemcpy(d_q, q.data(), q_len * sizeof(char),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_windows,
                          static_cast<size_t>(max_batch) * cfg.window_len * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_win_lens, max_batch * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_prev2,
                          static_cast<size_t>(max_batch) * rows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prev1,
                          static_cast<size_t>(max_batch) * rows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_curr,
                          static_cast<size_t>(max_batch) * rows * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_best_scores, max_batch * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_endcols, max_batch * sizeof(int)));

    std::vector<WindowHit> all_hits;
    all_hits.reserve(windows.size());

    float coarse_ms = 0.0f;

    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    CUDA_CHECK(cudaEventRecord(start_event));

    for (size_t base = 0; base < windows.size(); base += max_batch) {
        int chunk = static_cast<int>(
            std::min<size_t>(max_batch, windows.size() - base));

        std::vector<char> h_windows(static_cast<size_t>(chunk) * cfg.window_len, 'X');
        std::vector<int> h_win_lens(chunk, 0);

        for (int i = 0; i < chunk; ++i) {
            const auto& w = windows[base + i];
            h_win_lens[i] = w.len;
            std::memcpy(h_windows.data() + static_cast<size_t>(i) * cfg.window_len,
                        target.data() + w.start,
                        static_cast<size_t>(w.len) * sizeof(char));
        }

        CUDA_CHECK(cudaMemcpy(
            d_windows,
            h_windows.data(),
            static_cast<size_t>(chunk) * cfg.window_len * sizeof(char),
            cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(
            d_win_lens,
            h_win_lens.data(),
            chunk * sizeof(int),
            cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(
            d_prev2, 0,
            static_cast<size_t>(chunk) * rows * sizeof(int)));
        CUDA_CHECK(cudaMemset(
            d_prev1, 0,
            static_cast<size_t>(chunk) * rows * sizeof(int)));
        CUDA_CHECK(cudaMemset(
            d_curr, 0,
            static_cast<size_t>(chunk) * rows * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_best_scores, 0, chunk * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_best_endcols, 0, chunk * sizeof(int)));

        size_t shmem = static_cast<size_t>(2 * cfg.threads_per_block) * sizeof(int);

        sw_batch_score_kernel<<<chunk, cfg.threads_per_block, shmem>>>(
            d_q,
            d_windows,
            d_win_lens,
            q_len,
            cfg.window_len,
            chunk,
            /*match=*/2,
            /*mismatch=*/-1,
            /*gap=*/-1,
            d_prev2,
            d_prev1,
            d_curr,
            d_best_scores,
            d_best_endcols
        );
        CUDA_CHECK(cudaGetLastError());

        std::vector<int> h_scores(chunk);
        std::vector<int> h_endcols(chunk);

        CUDA_CHECK(cudaMemcpy(h_scores.data(), d_best_scores,
                              chunk * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_endcols.data(), d_best_endcols,
                              chunk * sizeof(int), cudaMemcpyDeviceToHost));

        for (int i = 0; i < chunk; ++i) {
            const auto& w = windows[base + i];
            all_hits.push_back({h_scores[i], w.start, w.len, h_endcols[i]});
        }
    }

    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&coarse_ms, start_event, stop_event));

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_windows));
    CUDA_CHECK(cudaFree(d_win_lens));
    CUDA_CHECK(cudaFree(d_prev2));
    CUDA_CHECK(cudaFree(d_prev1));
    CUDA_CHECK(cudaFree(d_curr));
    CUDA_CHECK(cudaFree(d_best_scores));
    CUDA_CHECK(cudaFree(d_best_endcols));

    if (all_hits.empty()) {
        out_time_ms = coarse_ms;
        return;
    }

    std::sort(all_hits.begin(), all_hits.end(),
              [](const WindowHit& a, const WindowHit& b) {
                  return a.score > b.score;
              });

    if (all_hits[0].score == 0) {
        out_time_ms = coarse_ms;
        return;
    }

    int use_k = std::min<int>(cfg.top_k, static_cast<int>(all_hits.size()));
    std::vector<std::pair<int, int>> intervals;
    intervals.reserve(use_k);

    for (int i = 0; i < use_k; ++i) {
        int s = std::max(0, all_hits[i].start - halo);
        int e = std::min(d_len, all_hits[i].start + all_hits[i].len + halo);
        intervals.push_back({s, e});
    }

    intervals = merge_intervals(intervals);

    float refine_ms = 0.0f;

    for (const auto& iv : intervals) {
        std::string sub = target.substr(iv.first, iv.second - iv.first);

        int score = 0;
        int start = 0;
        int stop = -1;
        std::string aq, ad;
        float single_time_ms = 0.0f;

        run_cuda_smith_waterman_single(
            q,
            sub,
            score,
            start,
            stop,
            aq,
            ad,
            single_time_ms
        );

        refine_ms += single_time_ms;

        if (score > out_score) {
            out_score = score;
            out_start = start + iv.first;
            out_stop = stop + iv.first;
            out_aligned_q = std::move(aq);
            out_aligned_d = std::move(ad);
        }
    }

    out_time_ms = coarse_ms + refine_ms;
}


void run_cuda_smith_waterman(
    const std::string& q,
    const std::string& d,
    int& out_score,
    int& out_start,
    int& out_stop,
    std::string& out_aligned_q,
    std::string& out_aligned_d,
    float& out_time_ms)
{
    SWBatchConfig cfg;
    cfg.window_len = std::max(8 * static_cast<int>(q.size()), 1024);
    cfg.stride = std::max(2 * static_cast<int>(q.size()), 256);
    cfg.top_k = 8;
    cfg.batch_windows = 1024;
    cfg.threads_per_block = 256;
    cfg.refine_extra = 2 * static_cast<int>(q.size());

    run_cuda_smith_waterman_batched(
        q,
        d,
        out_score,
        out_start,
        out_stop,
        out_aligned_q,
        out_aligned_d,
        out_time_ms,
        cfg
    );
}