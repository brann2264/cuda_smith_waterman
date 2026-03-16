#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>

// Define our Tile dimensions. 32 perfectly matches one CUDA Warp.
#define TILE_SIZE 32

__device__ __forceinline__ int sw_max4(int a, int b, int c, int d) {
    return max(max(a, b), max(c, d));
}

__device__ __forceinline__ int diag_len(int k, int q_len, int d_len) {
    return min(min(k, q_len), min(d_len, q_len + d_len - k));
}

__device__ __forceinline__ int diag_start_row(int k, int d_len) {
    return (k <= d_len) ? 1 : (k - d_len + 1);
}



// 2. Host function
void run_cuda_smith_waterman_single(const std::string& q, const std::string& d, 
                             int& out_score, int& out_start, int& out_stop, 
                             std::string& out_aligned_q, std::string& out_aligned_d,
                             float& out_time_ms) {
    int q_len = q.length();
    int d_len = d.length();
    int rows = q_len + 1;
    int cols = d_len + 1;
    size_t matrix_bytes = rows * cols * sizeof(int);

    char *d_q, *d_d;
    int *d_matrix;
    unsigned long long *d_max_info;

    cudaMalloc(&d_q, q_len * sizeof(char));
    cudaMalloc(&d_d, d_len * sizeof(char));
    cudaMalloc(&d_matrix, matrix_bytes);
    cudaMalloc(&d_max_info, sizeof(unsigned long long));

    cudaMemcpy(d_q, q.c_str(), q_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, d.c_str(), d_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(d_matrix, 0, matrix_bytes); 
    cudaMemset(d_max_info, 0, sizeof(unsigned long long)); 

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event);

    int match = 2, mismatch = -1, gap = -1; 
    
    // Calculate how many TILE blocks we need
    int num_tiles_q = (q_len + TILE_SIZE - 1) / TILE_SIZE;
    int num_tiles_d = (d_len + TILE_SIZE - 1) / TILE_SIZE;
    int total_macro_diagonals = num_tiles_q + num_tiles_d - 1;
    
    // Launch Kernel for each MACRO-Diagonal
    for (int mk = 1; mk <= total_macro_diagonals; ++mk) {
        int tiles_in_diag = std::min(std::min(mk, num_tiles_q), std::min(num_tiles_d, num_tiles_q + num_tiles_d - mk));
        
        // 1 Block = 1 Tile. Threads per Block = TILE_SIZE (32)
        sw_tiled_macro_wavefront_kernel<<<tiles_in_diag, TILE_SIZE>>>(
            d_q, d_d, d_matrix, d_max_info, q_len, d_len, match, mismatch, gap, mk
        );
    }

    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event); 
    
    cudaEventElapsedTime(&out_time_ms, start_event, stop_event);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    int* h_matrix;
    cudaHostAlloc(&h_matrix, matrix_bytes, cudaHostAllocDefault);
    cudaMemcpy(h_matrix, d_matrix, matrix_bytes, cudaMemcpyDeviceToHost);

    unsigned long long h_max_info = 0;
    cudaMemcpy(&h_max_info, d_max_info, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    int max_score = (h_max_info >> 42) & 0x3FFFFF;      
    int max_row   = (h_max_info >> 21) & 0x1FFFFF;      
    int max_col   = h_max_info & 0x1FFFFF;              

    cudaFree(d_q);
    cudaFree(d_d);
    cudaFree(d_matrix);
    cudaFree(d_max_info);
    
    out_score = max_score;
    out_stop = max_col - 1; 

    std::string aligned_q = "";
    std::string aligned_d = "";
    int r = max_row;
    int c = max_col;

    if (max_score == 0) {
        out_start = 0;
        out_stop = 0;
        cudaFreeHost(h_matrix);
        return;
    }

    while (r > 0 && c > 0 && h_matrix[r * cols + c] > 0) {
        int current_score = h_matrix[r * cols + c];
        int diag_score = h_matrix[(r - 1) * cols + (c - 1)];
        int up_score = h_matrix[(r - 1) * cols + c];
        int left_score = h_matrix[r * cols + (c - 1)];

        char q_char = q[r - 1];
        char d_char = d[c - 1];
        
        int match_val = (q_char == d_char || q_char == 'N' || d_char == 'N') ? match : mismatch;

        if (current_score == diag_score + match_val) {
            aligned_q = q_char + aligned_q;
            aligned_d = d_char + aligned_d;
            r--; c--;
        } else if (current_score == left_score + gap) {
            aligned_q = "-" + aligned_q;
            aligned_d = d_char + aligned_d;
            c--;
        } else if (current_score == up_score + gap) {
            aligned_q = q_char + aligned_q;
            aligned_d = "-" + aligned_d;
            r--;
        } else {
            break; 
        }
    }

    out_start = c; 
    out_aligned_q = aligned_q;
    out_aligned_d = aligned_d;

    cudaFreeHost(h_matrix);
}

/*
One block = one target window.
Score-only Smith-Waterman with linear gap.
Rolling diagonals are stored per-window in global memory slices.

Inputs:
  d_q           : query chars
  d_windows     : flattened windows, each stored in a fixed-width slot of size window_stride
  d_win_lens    : actual length of each window
  q_len         : query length
  window_stride : fixed storage stride for each window in d_windows
  batch_size    : number of windows in this launch

Per-window rolling buffers:
  d_prev2, d_prev1, d_curr each have size batch_size * (q_len + 1)

Outputs:
  d_best_scores[wid]
  d_best_endcols[wid]
*/
__global__ void sw_batch_score_kernel(
    const char* __restrict__ d_q,
    const char* __restrict__ d_windows,
    const int*  __restrict__ d_win_lens,
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
        int start_row = diag_start_row(k, d_len);
        int elems = diag_len(k, q_len, d_len);

        for (int idx = tid; idx < elems; idx += blockDim.x) {
            int row = start_row + idx;
            int col = k - row + 1;

            char q_char = d_q[row - 1];
            char d_char = win[col - 1];

            int sub = (q_char == d_char || q_char == 'N' || d_char == 'N')
                        ? match_score
                        : mismatch_score;

            int up   = (row > 1)            ? prev1[row - 1] : 0;
            int left = (col > 1)            ? prev1[row]     : 0;
            int diag = (row > 1 && col > 1) ? prev2[row - 1] : 0;

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
    int* s_scores = s_reduce;
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

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)             \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";          \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

struct SWBatchConfig {
    int window_len = 0;      // 0 => auto
    int stride = 0;          // 0 => auto
    int top_k = 8;           // refine top-K windows
    int batch_windows = 1024;
    int threads_per_block = 256;
    int refine_extra = 0;    // extra halo beyond overlap
};

struct WindowSpec {
    int start;
    int len;
};

struct WindowHit {
    int score;
    int start;
    int len;
    int end_col; // inside this window
};

static std::vector<WindowSpec> build_overlapping_windows(
    int target_len,
    int window_len,
    int stride)
{
    std::vector<WindowSpec> out;
    if (target_len <= 0) return out;

    for (int start = 0; ; start += stride) {
        int len = std::min(window_len, target_len - start);
        out.push_back({start, len});
        if (start + len >= target_len) break;
    }

    int tail_start = std::max(0, target_len - window_len);
    if (out.empty() || out.back().start != tail_start) {
        out.push_back({tail_start, target_len - tail_start});
    }

    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        return a.start < b.start;
    });

    out.erase(std::unique(out.begin(), out.end(), [](const auto& a, const auto& b) {
        return a.start == b.start;
    }), out.end());

    return out;
}

static std::vector<std::pair<int,int>> merge_intervals(
    std::vector<std::pair<int,int>> intervals)
{
    if (intervals.empty()) return {};

    std::sort(intervals.begin(), intervals.end());
    std::vector<std::pair<int,int>> merged;
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


void run_cuda_smith_waterman_batched(
    const std::string& q,
    const std::string& d,
    int& out_score,
    int& out_start,
    int& out_stop,
    std::string& out_aligned_q,
    std::string& out_aligned_d,
    float& out_time_ms,
    SWBatchConfig cfg = {})
{
    const int q_len = static_cast<int>(q.size());
    const int d_len = static_cast<int>(d.size());

    if (q_len == 0 || d_len == 0) {
        out_score = 0;
        out_start = 0;
        out_stop = -1;
        out_aligned_q.clear();
        out_aligned_d.clear();
        out_time_ms = 0.0f;
        return;
    }

    // Accuracy-oriented defaults:
    // large overlap, decent batching
    if (cfg.window_len == 0) cfg.window_len = std::max(8 * q_len, 1024);
    if (cfg.stride == 0)     cfg.stride     = std::max(2 * q_len, 256);
    if (cfg.stride >= cfg.window_len) cfg.stride = std::max(1, cfg.window_len / 4);

    const int overlap = cfg.window_len - cfg.stride;
    const int halo = overlap + cfg.refine_extra;

    std::vector<WindowSpec> windows = build_overlapping_windows(d_len, cfg.window_len, cfg.stride);
    if (windows.empty()) {
        out_score = 0;
        out_start = 0;
        out_stop = -1;
        out_aligned_q.clear();
        out_aligned_d.clear();
        out_time_ms = 0.0f;
        return;
    }

    char* d_q = nullptr;
    char* d_windows = nullptr;
    int* d_win_lens = nullptr;
    int* d_prev2 = nullptr;
    int* d_prev1 = nullptr;
    int* d_curr  = nullptr;
    int* d_best_scores = nullptr;
    int* d_best_endcols = nullptr;

    const int rows = q_len + 1;
    const int max_batch = cfg.batch_windows;

    CUDA_CHECK(cudaMalloc(&d_q, q_len * sizeof(char)));
    CUDA_CHECK(cudaMemcpy(d_q, q.data(), q_len * sizeof(char), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_windows, static_cast<size_t>(max_batch) * cfg.window_len * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_win_lens, max_batch * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_prev2, static_cast<size_t>(max_batch) * rows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prev1, static_cast<size_t>(max_batch) * rows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_curr,  static_cast<size_t>(max_batch) * rows * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_best_scores, max_batch * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_endcols, max_batch * sizeof(int)));

    std::vector<WindowHit> all_hits;
    all_hits.reserve(windows.size());

    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    CUDA_CHECK(cudaEventRecord(start_event));

    for (size_t base = 0; base < windows.size(); base += max_batch) {
        int chunk = static_cast<int>(std::min<size_t>(max_batch, windows.size() - base));

        std::vector<char> h_windows(static_cast<size_t>(chunk) * cfg.window_len, 'X');
        std::vector<int> h_win_lens(chunk, 0);

        for (int i = 0; i < chunk; ++i) {
            const auto& w = windows[base + i];
            h_win_lens[i] = w.len;
            std::memcpy(
                h_windows.data() + static_cast<size_t>(i) * cfg.window_len,
                d.data() + w.start,
                w.len * sizeof(char)
            );
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

        CUDA_CHECK(cudaMemset(d_prev2, 0, static_cast<size_t>(chunk) * rows * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_prev1, 0, static_cast<size_t>(chunk) * rows * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_curr,  0, static_cast<size_t>(chunk) * rows * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_best_scores, 0, chunk * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_best_endcols, 0, chunk * sizeof(int)));

        size_t shmem = 2 * cfg.threads_per_block * sizeof(int);

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

        CUDA_CHECK(cudaMemcpy(h_scores.data(), d_best_scores, chunk * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_endcols.data(), d_best_endcols, chunk * sizeof(int), cudaMemcpyDeviceToHost));

        for (int i = 0; i < chunk; ++i) {
            const auto& w = windows[base + i];
            all_hits.push_back({
                h_scores[i],
                w.start,
                w.len,
                h_endcols[i]
            });
        }
    }

    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&out_time_ms, start_event, stop_event));

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

    std::sort(all_hits.begin(), all_hits.end(), [](const WindowHit& a, const WindowHit& b) {
        return a.score > b.score;
    });

    if (all_hits.empty() || all_hits[0].score == 0) {
        out_score = 0;
        out_start = 0;
        out_stop = -1;
        out_aligned_q.clear();
        out_aligned_d.clear();
        return;
    }

    // Refine top-K windows exactly
    int use_k = std::min<int>(cfg.top_k, static_cast<int>(all_hits.size()));
    std::vector<std::pair<int,int>> intervals;
    intervals.reserve(use_k);

    for (int i = 0; i < use_k; ++i) {
        int s = std::max(0, all_hits[i].start - halo);
        int e = std::min(d_len, all_hits[i].start + all_hits[i].len + halo);
        intervals.push_back({s, e});
    }

    intervals = merge_intervals(intervals);

    out_score = 0;
    out_start = 0;
    out_stop = -1;
    out_aligned_q.clear();
    out_aligned_d.clear();

    for (const auto& iv : intervals) {
        std::string sub = d.substr(iv.first, iv.second - iv.first);

        int score = 0, start = 0, stop = -1;
        std::string aq, ad;
        float single_time = 0.0f;

        run_cuda_smith_waterman_single(
            q, sub,
            score, start, stop,
            aq, ad,
            single_time
        );

        if (score > out_score) {
            out_score = score;
            out_start = start + iv.first;
            out_stop  = stop  + iv.first;
            out_aligned_q = std::move(aq);
            out_aligned_d = std::move(ad);
        }
    }
}

