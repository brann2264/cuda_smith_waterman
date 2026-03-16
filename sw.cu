#include <cuda_runtime.h>
#include <stdint.h>
#include <limits.h>

constexpr int WARP_SIZE = 32;
constexpr int MAX_QUERY_LEN = 4096;   // example bound
constexpr int SIGMA = 32;             // alphabet size example

// Query stored in constant memory, as in the paper's design.
__constant__ uint8_t d_query[MAX_QUERY_LEN];

template <int P, int K>
__global__ void smith_waterman_affine_kernel(
    const uint8_t* __restrict__ db_chars,     // concatenated database sequences
    const int* __restrict__ db_offsets,       // start offset of each sequence
    const int* __restrict__ db_lengths,       // length of each sequence
    const int8_t* __restrict__ submat_global, // SIGMA x SIGMA
    int query_len,
    int alpha,   // gap open
    int beta,    // gap extend
    int* __restrict__ out_scores              // one score per alignment
) {
    // Optional: cache substitution matrix in shared memory per block.
    __shared__ int8_t submat[SIGMA * SIGMA];
    for (int idx = threadIdx.x; idx < SIGMA * SIGMA; idx += blockDim.x) {
        submat[idx] = submat_global[idx];
    }
    __syncthreads();

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id    = global_tid / WARP_SIZE;
    int lane       = threadIdx.x & (WARP_SIZE - 1);

    // We only use the first P lanes of each warp as a subwarp.
    int subwarp_id_in_warp = lane / P;
    int lane_in_subwarp    = lane % P;

    // If you want multiple subwarps per warp:
    int alignments_per_warp = WARP_SIZE / P;
    int alignment_id = warp_id * alignments_per_warp + subwarp_id_in_warp;

    int n = db_lengths[alignment_id];
    const uint8_t* S = db_chars + db_offsets[alignment_id];

    // Local K subject letters for this thread.
    uint8_t s_local[K];

    // Register-resident DP state.
    int H_prev[K];   // H from previous wavefront step / antecedent row slice
    int H_cur[K];
    int E[K];
    int F[K];

    // Initialize registers.
    #pragma unroll
    for (int j = 0; j < K; j++) {
        int col = lane_in_subwarp * K + j;
        s_local[j] = (col < n) ? S[col] : 0;

        H_prev[j] = 0;
        H_cur[j]  = 0;
        E[j]      = INT_MIN / 4;
        F[j]      = INT_MIN / 4;
    }

    int best_local = 0;

    // The paper maps one alignment onto a (sub)warp of p threads,
    // each thread handling k columns, proceeding in wavefront order. :contentReference[oaicite:3]{index=3}
    //
    // This simplified skeleton does one "active row" at a time.
    for (int i = 1; i <= query_len + P - 1; i++) {
        // Active logical row for this lane in the wavefront.
        int row = i - lane_in_subwarp;
        bool active = (row >= 1 && row <= query_len);

        // Boundary values received from previous thread.
        int left_H_boundary    = 0;
        int left_F_boundary    = INT_MIN / 4;
        int diag_boundary      = 0;

        // Rightmost values from previous lane via shuffle-up.
        // The paper uses __shfl_up_sync for this exact neighbor communication. :contentReference[oaicite:4]{index=4}
        int prev_lane_right_H = __shfl_up_sync(0xFFFFFFFF, H_prev[K - 1], 1, WARP_SIZE);
        int prev_lane_right_F = __shfl_up_sync(0xFFFFFFFF, F[K - 1],      1, WARP_SIZE);

        // For the diagonal dependency, you typically also need one value from the
        // second antecedent row / previous iteration boundary.
        int prev_lane_diag = __shfl_up_sync(0xFFFFFFFF, H_prev[K - 1], 1, WARP_SIZE);

        if (lane_in_subwarp > 0) {
            left_H_boundary = prev_lane_right_H;
            left_F_boundary = prev_lane_right_F;
            diag_boundary   = prev_lane_diag;
        }

        int q_char = 0;
        if (active) {
            q_char = d_query[row - 1];
        }

        int left_H = left_H_boundary;
        int left_F = left_F_boundary;
        int diag   = diag_boundary;

        #pragma unroll
        for (int j = 0; j < K; j++) {
            int col = lane_in_subwarp * K + j + 1;

            if (!active || col > n) {
                H_cur[j] = 0;
                continue;
            }

            int sub = submat[q_char * SIGMA + s_local[j]];

            // Affine-gap Smith-Waterman recurrence:
            // E(i,j) = max(E(i-1,j)-beta, H(i-1,j)-alpha)
            // F(i,j) = max(F(i,j-1)-beta, H(i,j-1)-alpha)
            // H(i,j) = max(0, H(i-1,j-1)+sub, E(i,j), F(i,j)) :contentReference[oaicite:5]{index=5}
            E[j] = max(E[j] - beta, H_prev[j] - alpha);
            F[j] = max(left_F - beta, left_H - alpha);

            int h = max(0, max(diag + sub, max(E[j], F[j])));

            diag   = H_prev[j];
            left_H = h;
            left_F = F[j];

            H_cur[j] = h;
            best_local = max(best_local, h);
        }

        // Swap buffers for next iteration.
        #pragma unroll
        for (int j = 0; j < K; j++) {
            H_prev[j] = H_cur[j];
        }
    }

    // Warp/subwarp reduction for final max score.
    // The paper tracks a per-thread max and does a warp-level reduction at the end. :contentReference[oaicite:6]{index=6}
    for (int offset = P / 2; offset > 0; offset >>= 1) {
        best_local = max(best_local,
                         __shfl_down_sync(0xFFFFFFFF, best_local, offset, WARP_SIZE));
    }

    if (lane_in_subwarp == 0) {
        out_scores[alignment_id] = best_local;
    }
}