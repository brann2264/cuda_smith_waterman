// #include <cuda_runtime.h>
// #include <iostream>
// #include <algorithm>
// #include <string>

// // 1. GPU Kernel
// __global__ void sw_wavefront_kernel(
//     const char* d_q, const char* d_d, int* d_matrix,            
//     int q_len, int d_len, int match_score, int mismatch_score, int gap_score, int k)                    
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     int elements_in_diag = min(min(k, q_len), min(d_len, q_len + d_len - k));

//     if (tid < elements_in_diag) {
//         int start_row = (k <= d_len) ? 1 : (k - d_len + 1);
//         int row = start_row + tid;
//         int col = k - row + 1;
//         int width = d_len + 1;

//         char q_char = d_q[row - 1];
//         char d_char = d_d[col - 1];

//         int diag_score = d_matrix[(row - 1) * width + (col - 1)];
        
//         // Handle 'N' wildcard matching
//         if (q_char == d_char || q_char == 'N' || d_char == 'N') {
//             diag_score += match_score;
//         } else {
//             diag_score += mismatch_score;
//         }

//         int up_score = d_matrix[(row - 1) * width + col] + gap_score;
//         int left_score = d_matrix[row * width + (col - 1)] + gap_score;

//         int max_val = max(0, max(diag_score, max(up_score, left_score)));
//         d_matrix[row * width + col] = max_val;
//     }
// }

// // 2. Host function (Updated to return out_time_ms)
// void run_cuda_smith_waterman(const std::string& q, const std::string& d, 
//                              int& out_score, int& out_start, int& out_stop, 
//                              std::string& out_aligned_q, std::string& out_aligned_d,
//                              float& out_time_ms) {
//     int q_len = q.length();
//     int d_len = d.length();
//     int rows = q_len + 1;
//     int cols = d_len + 1;
//     size_t matrix_bytes = rows * cols * sizeof(int);

//     char *d_q, *d_d;
//     int *d_matrix;
//     cudaMalloc(&d_q, q_len * sizeof(char));
//     cudaMalloc(&d_d, d_len * sizeof(char));
//     cudaMalloc(&d_matrix, matrix_bytes);

//     cudaMemcpy(d_q, q.c_str(), q_len * sizeof(char), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_d, d.c_str(), d_len * sizeof(char), cudaMemcpyHostToDevice);
//     cudaMemset(d_matrix, 0, matrix_bytes); 

//     // --- START CUDA TIMER ---
//     cudaEvent_t start_event, stop_event;
//     cudaEventCreate(&start_event);
//     cudaEventCreate(&stop_event);
//     cudaEventRecord(start_event);

//     int match = 2, mismatch = -1, gap = -1; 
//     int total_diagonals = q_len + d_len - 1;
    
//     for (int k = 1; k <= total_diagonals; ++k) {
//         int elements = std::min(std::min(k, q_len), std::min(d_len, q_len + d_len - k));
//         int threadsPerBlock = 256;
//         int blocksPerGrid = (elements + threadsPerBlock - 1) / threadsPerBlock;

//         sw_wavefront_kernel<<<blocksPerGrid, threadsPerBlock>>>(
//             d_q, d_d, d_matrix, q_len, d_len, match, mismatch, gap, k
//         );
//     }

//     // --- STOP CUDA TIMER ---
//     cudaEventRecord(stop_event);
//     cudaEventSynchronize(stop_event); 
    
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    
//     // Pass time back to main.cpp instead of printing
//     out_time_ms = milliseconds; 

//     cudaEventDestroy(start_event);
//     cudaEventDestroy(stop_event);

//     int* h_matrix = new int[rows * cols];
//     cudaMemcpy(h_matrix, d_matrix, matrix_bytes, cudaMemcpyDeviceToHost);

//     cudaFree(d_q);
//     cudaFree(d_d);
//     cudaFree(d_matrix);

//     // --- CPU TRACEBACK ---
//     int max_score = 0, max_row = 0, max_col = 0;

//     for (int r = 1; r <= q_len; ++r) {
//         for (int c = 1; c <= d_len; ++c) {
//             if (h_matrix[r * cols + c] > max_score) {
//                 max_score = h_matrix[r * cols + c];
//                 max_row = r;
//                 max_col = c;
//             }
//         }
//     }

//     out_score = max_score;
//     out_stop = max_col - 1; 

//     std::string aligned_q = "";
//     std::string aligned_d = "";
//     int r = max_row;
//     int c = max_col;

//     while (r > 0 && c > 0 && h_matrix[r * cols + c] > 0) {
//         int current_score = h_matrix[r * cols + c];
//         int diag_score = h_matrix[(r - 1) * cols + (c - 1)];
//         int up_score = h_matrix[(r - 1) * cols + c];
//         int left_score = h_matrix[r * cols + (c - 1)];

//         char q_char = q[r - 1];
//         char d_char = d[c - 1];
        
//         int match_val = (q_char == d_char || q_char == 'N' || d_char == 'N') ? match : mismatch;

//         if (current_score == diag_score + match_val) {
//             aligned_q = q_char + aligned_q;
//             aligned_d = d_char + aligned_d;
//             r--; c--;
//         } else if (current_score == left_score + gap) {
//             aligned_q = "-" + aligned_q;
//             aligned_d = d_char + aligned_d;
//             c--;
//         } else if (current_score == up_score + gap) {
//             aligned_q = q_char + aligned_q;
//             aligned_d = "-" + aligned_d;
//             r--;
//         } else {
//             break; 
//         }
//     }

//     out_start = c; 
//     out_aligned_q = aligned_q;
//     out_aligned_d = aligned_d;

//     delete[] h_matrix;
// }

// v2

// #include <cuda_runtime.h>
// #include <iostream>
// #include <algorithm>
// #include <string>

// // Define our Tile dimensions. 32 perfectly matches one CUDA Warp.
// #define TILE_SIZE 32

// // 1. Tiled GPU Kernel
// __global__ void sw_tiled_macro_wavefront_kernel(
//     const char* __restrict__ d_q, 
//     const char* __restrict__ d_d, 
//     int* __restrict__ d_matrix,            
//     unsigned long long* __restrict__ d_max_info, 
//     int q_len, int d_len, int match, int mismatch, int gap, int macro_k)                    
// {
//     // Determine which TILE this block is calculating based on the Macro-Diagonal (macro_k)
//     int num_tiles_d = (d_len + TILE_SIZE - 1) / TILE_SIZE;
    
//     int start_tile_row = (macro_k <= num_tiles_d) ? 1 : (macro_k - num_tiles_d + 1);
//     int tile_row = start_tile_row + blockIdx.x;
//     int tile_col = macro_k - tile_row + 1;

//     // Thread ID handles a specific row/col within the tile
//     int tx = threadIdx.x; 

//     // Global matrix width
//     int width = d_len + 1;

//     // Allocate Ultra-Fast Shared Memory for this specific block
//     __shared__ char s_q[TILE_SIZE];
//     __shared__ char s_d[TILE_SIZE];
//     __shared__ int s_matrix[TILE_SIZE + 1][TILE_SIZE + 1];

//     // Global starting coordinates for this tile
//     int global_q_idx = (tile_row - 1) * TILE_SIZE + tx;
//     int global_d_idx = (tile_col - 1) * TILE_SIZE + tx;

//     // --- PHASE 1: LOAD DATA INTO SHARED MEMORY ---
    
//     // 1a. Load sequence strings (with bounds checking)
//     s_q[tx] = (global_q_idx < q_len) ? __ldg(&d_q[global_q_idx]) : '-';
//     s_d[tx] = (global_d_idx < d_len) ? __ldg(&d_d[global_d_idx]) : '-';

//     // 1b. Load Top and Left boundaries from Global Memory into Shared Memory
//     int global_row = (tile_row - 1) * TILE_SIZE + tx + 1;
//     int global_col = (tile_col - 1) * TILE_SIZE + tx + 1;

//     // Load Top Boundary (from the tile above us)
//     if (global_col <= d_len) {
//         s_matrix[0][tx + 1] = d_matrix[((tile_row - 1) * TILE_SIZE) * width + global_col];
//     } else {
//         s_matrix[0][tx + 1] = 0;
//     }

//     // Load Left Boundary (from the tile to our left)
//     if (global_row <= q_len) {
//         s_matrix[tx + 1][0] = d_matrix[global_row * width + ((tile_col - 1) * TILE_SIZE)];
//     } else {
//         s_matrix[tx + 1][0] = 0;
//     }

//     // Top-Left corner cell of the tile
//     if (tx == 0) {
//         s_matrix[0][0] = d_matrix[((tile_row - 1) * TILE_SIZE) * width + ((tile_col - 1) * TILE_SIZE)];
//     }

//     // Force all 32 threads to wait until Shared Memory is fully loaded
//     __syncthreads();

//     // --- PHASE 2: MICRO-WAVEFRONT (COMPUTE THE TILE) ---
    
//     int local_max = 0;
//     int local_max_r = 0;
//     int local_max_c = 0;

//     // Sweep across the 32x32 tile internally (63 micro-diagonals)
//     for (int mk = 1; mk <= 2 * TILE_SIZE - 1; mk++) {
//         int start_r = max(1, mk - TILE_SIZE + 1);
//         int end_r = min(TILE_SIZE, mk);
//         int elements = end_r - start_r + 1;

//         if (tx < elements) {
//             int r = start_r + tx;
//             int c = mk - r + 1;

//             char q_char = s_q[r - 1];
//             char d_char = s_d[c - 1];
            
//             int diag = s_matrix[r - 1][c - 1] + ((q_char == d_char || q_char == 'N' || d_char == 'N') ? match : mismatch);
//             int up = s_matrix[r - 1][c] + gap;
//             int left = s_matrix[r][c - 1] + gap;
            
//             int cell_score = max(0, max(diag, max(up, left)));
//             s_matrix[r][c] = cell_score;

//             // Track the best score in this specific thread's path
//             if (cell_score > local_max && (global_q_idx - tx + r - 1) < q_len && (global_d_idx - tx + c - 1) < d_len) {
//                 local_max = cell_score;
//                 local_max_r = (tile_row - 1) * TILE_SIZE + r;
//                 local_max_c = (tile_col - 1) * TILE_SIZE + c;
//             }
//         }
//         // Wait for all threads to finish this micro-diagonal before moving to the next
//         __syncthreads(); 
//     }

//     // --- PHASE 3: WRITE RESULTS TO GLOBAL MEMORY ---
    
//     // Write the computed interior of the tile back to the global matrix
//     if (global_row <= q_len) {
//         for (int c = 1; c <= TILE_SIZE; c++) {
//             if (((tile_col - 1) * TILE_SIZE + c) <= d_len) {
//                 d_matrix[global_row * width + ((tile_col - 1) * TILE_SIZE + c)] = s_matrix[tx + 1][c];
//             }
//         }
//     }

//     // OPTIMIZATION: Push highest score to global tracker
//     if (local_max > 0) {
//         unsigned long long packed = ((unsigned long long)local_max << 42) | 
//                                     ((unsigned long long)local_max_r << 21) | 
//                                     (unsigned long long)local_max_c;
//         atomicMax(d_max_info, packed);
//     }
// }

// // 2. Host function
// void run_cuda_smith_waterman(const std::string& q, const std::string& d, 
//                              int& out_score, int& out_start, int& out_stop, 
//                              std::string& out_aligned_q, std::string& out_aligned_d,
//                              float& out_time_ms) {
//     int q_len = q.length();
//     int d_len = d.length();
//     int rows = q_len + 1;
//     int cols = d_len + 1;
//     size_t matrix_bytes = rows * cols * sizeof(int);

//     char *d_q, *d_d;
//     int *d_matrix;
//     unsigned long long *d_max_info;

//     cudaMalloc(&d_q, q_len * sizeof(char));
//     cudaMalloc(&d_d, d_len * sizeof(char));
//     cudaMalloc(&d_matrix, matrix_bytes);
//     cudaMalloc(&d_max_info, sizeof(unsigned long long));

//     cudaMemcpy(d_q, q.c_str(), q_len * sizeof(char), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_d, d.c_str(), d_len * sizeof(char), cudaMemcpyHostToDevice);
//     cudaMemset(d_matrix, 0, matrix_bytes); 
//     cudaMemset(d_max_info, 0, sizeof(unsigned long long)); 

//     cudaEvent_t start_event, stop_event;
//     cudaEventCreate(&start_event);
//     cudaEventCreate(&stop_event);
//     cudaEventRecord(start_event);

//     int match = 2, mismatch = -1, gap = -1; 
    
//     // Calculate how many TILE blocks we need
//     int num_tiles_q = (q_len + TILE_SIZE - 1) / TILE_SIZE;
//     int num_tiles_d = (d_len + TILE_SIZE - 1) / TILE_SIZE;
//     int total_macro_diagonals = num_tiles_q + num_tiles_d - 1;
    
//     // Launch Kernel for each MACRO-Diagonal
//     for (int mk = 1; mk <= total_macro_diagonals; ++mk) {
//         int tiles_in_diag = std::min(std::min(mk, num_tiles_q), std::min(num_tiles_d, num_tiles_q + num_tiles_d - mk));
        
//         // 1 Block = 1 Tile. Threads per Block = TILE_SIZE (32)
//         sw_tiled_macro_wavefront_kernel<<<tiles_in_diag, TILE_SIZE>>>(
//             d_q, d_d, d_matrix, d_max_info, q_len, d_len, match, mismatch, gap, mk
//         );
//     }

//     cudaEventRecord(stop_event);
//     cudaEventSynchronize(stop_event); 
    
//     cudaEventElapsedTime(&out_time_ms, start_event, stop_event);
//     cudaEventDestroy(start_event);
//     cudaEventDestroy(stop_event);

//     int* h_matrix;
//     cudaHostAlloc(&h_matrix, matrix_bytes, cudaHostAllocDefault);
//     cudaMemcpy(h_matrix, d_matrix, matrix_bytes, cudaMemcpyDeviceToHost);

//     unsigned long long h_max_info = 0;
//     cudaMemcpy(&h_max_info, d_max_info, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

//     int max_score = (h_max_info >> 42) & 0x3FFFFF;      
//     int max_row   = (h_max_info >> 21) & 0x1FFFFF;      
//     int max_col   = h_max_info & 0x1FFFFF;              

//     cudaFree(d_q);
//     cudaFree(d_d);
//     cudaFree(d_matrix);
//     cudaFree(d_max_info);
    
//     out_score = max_score;
//     out_stop = max_col - 1; 

//     std::string aligned_q = "";
//     std::string aligned_d = "";
//     int r = max_row;
//     int c = max_col;

//     if (max_score == 0) {
//         out_start = 0;
//         out_stop = 0;
//         cudaFreeHost(h_matrix);
//         return;
//     }

//     while (r > 0 && c > 0 && h_matrix[r * cols + c] > 0) {
//         int current_score = h_matrix[r * cols + c];
//         int diag_score = h_matrix[(r - 1) * cols + (c - 1)];
//         int up_score = h_matrix[(r - 1) * cols + c];
//         int left_score = h_matrix[r * cols + (c - 1)];

//         char q_char = q[r - 1];
//         char d_char = d[c - 1];
        
//         int match_val = (q_char == d_char || q_char == 'N' || d_char == 'N') ? match : mismatch;

//         if (current_score == diag_score + match_val) {
//             aligned_q = q_char + aligned_q;
//             aligned_d = d_char + aligned_d;
//             r--; c--;
//         } else if (current_score == left_score + gap) {
//             aligned_q = "-" + aligned_q;
//             aligned_d = d_char + aligned_d;
//             c--;
//         } else if (current_score == up_score + gap) {
//             aligned_q = q_char + aligned_q;
//             aligned_d = "-" + aligned_d;
//             r--;
//         } else {
//             break; 
//         }
//     }

//     out_start = c; 
//     out_aligned_q = aligned_q;
//     out_aligned_d = aligned_d;

//     cudaFreeHost(h_matrix);
// }

// v3
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <cstdlib>

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)             \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";          \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

__device__ __forceinline__ int sw_max4(int a, int b, int c, int d) {
    return max(max(a, b), max(c, d));
}

__device__ __forceinline__ int diag_len(int k, int q_len, int d_len) {
    return min(min(k, q_len), min(d_len, q_len + d_len - k));
}

__device__ __forceinline__ int diag_start_row(int k, int d_len) {
    return (k <= d_len) ? 1 : (k - d_len + 1);
}

/*
 * Warp-specialized kernel:
 * - one warp computes one whole alignment
 * - q_len must be <= 32
 * - keeps only rolling anti-diagonals in registers
 * - uses warp shuffles to read neighbors from previous diagonals
 */
__global__ void sw_wavefront_warp_kernel(
    const char* __restrict__ d_q,
    const char* __restrict__ d_d,
    int* __restrict__ d_matrix,   // keep full matrix for traceback
    int q_len,
    int d_len,
    int match_score,
    int mismatch_score,
    int gap_score,
    int* __restrict__ d_best_score,
    int* __restrict__ d_best_pos)
{
    const unsigned FULL_MASK = 0xffffffffu;
    int lane = threadIdx.x & 31;

    if (blockIdx.x != 0 || threadIdx.x >= 32 || q_len > 32) return;

    int width = d_len + 1;

    // Each lane stores one element from the previous two diagonals.
    int prev1 = 0; // k-1
    int prev2 = 0; // k-2
    int curr  = 0;

    int local_best_score = 0;
    int local_best_pos   = 0;

    for (int k = 1; k <= q_len + d_len - 1; ++k) {
        int len_k  = diag_len(k, q_len, d_len);
        int len_p  = (k >= 2) ? diag_len(k - 1, q_len, d_len) : 0;
        int len_pp = (k >= 3) ? diag_len(k - 2, q_len, d_len) : 0;

        int s_k  = diag_start_row(k, d_len);
        int s_p  = (k >= 2) ? diag_start_row(k - 1, d_len) : 1;
        int s_pp = (k >= 3) ? diag_start_row(k - 2, d_len) : 1;

        curr = 0;

        if (lane < len_k) {
            int row = s_k + lane;
            int col = k - row + 1;

            char q_char = d_q[row - 1];
            char d_char = d_d[col - 1];
            int sub = (q_char == d_char || q_char == 'N' || d_char == 'N')
                        ? match_score
                        : mismatch_score;

            // Pull neighbors from previous diagonals.
            int up_idx   = (row - 1) - s_p;   // H(row-1, col)
            int left_idx = row - s_p;         // H(row,   col-1)
            int diag_idx = (row - 1) - s_pp;  // H(row-1, col-1)

            int up   = (row > 1 && 0 <= up_idx && up_idx < len_p)
                        ? __shfl_sync(FULL_MASK, prev1, up_idx)
                        : 0;

            int left = (col > 1 && 0 <= left_idx && left_idx < len_p)
                        ? __shfl_sync(FULL_MASK, prev1, left_idx)
                        : 0;

            int diag = (row > 1 && col > 1 && 0 <= diag_idx && diag_idx < len_pp)
                        ? __shfl_sync(FULL_MASK, prev2, diag_idx)
                        : 0;

            curr = sw_max4(0, diag + sub, up + gap_score, left + gap_score);

            d_matrix[row * width + col] = curr;

            if (curr > local_best_score) {
                local_best_score = curr;
                local_best_pos   = row * width + col;
            }
        }

        prev2 = prev1;
        prev1 = curr;
    }

    // Warp reduce best score / best position
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other_score = __shfl_down_sync(FULL_MASK, local_best_score, offset);
        int other_pos   = __shfl_down_sync(FULL_MASK, local_best_pos, offset);
        if (other_score > local_best_score) {
            local_best_score = other_score;
            local_best_pos   = other_pos;
        }
    }

    if (lane == 0) {
        *d_best_score = local_best_score;
        *d_best_pos   = local_best_pos;
    }
}

/*
 * General kernel:
 * - handles arbitrary q_len
 * - one block computes the whole alignment
 * - each thread walks over multiple cells of a diagonal by striding
 * - rolling diagonals are stored in 3 buffers indexed by row number
 *
 * Row-indexed rolling buffers:
 *   prev1[r] stores the value on diagonal k-1 at row r
 *   prev2[r] stores the value on diagonal k-2 at row r
 *
 * Then for current cell (row, col):
 *   up   = prev1[row - 1]   // H(row-1, col)
 *   left = prev1[row]       // H(row,   col-1)
 *   diag = prev2[row - 1]   // H(row-1, col-1)
 */
__global__ void sw_wavefront_block_kernel(
    const char* __restrict__ d_q,
    const char* __restrict__ d_d,
    int* __restrict__ d_matrix,
    int* __restrict__ d_prev2_buf,
    int* __restrict__ d_prev1_buf,
    int* __restrict__ d_curr_buf,
    int q_len,
    int d_len,
    int match_score,
    int mismatch_score,
    int gap_score,
    int* __restrict__ d_best_score,
    int* __restrict__ d_best_pos)
{
    if (blockIdx.x != 0) return;

    int tid = threadIdx.x;
    int width = d_len + 1;

    int* prev2 = d_prev2_buf;
    int* prev1 = d_prev1_buf;
    int* curr  = d_curr_buf;

    int local_best_score = 0;
    int local_best_pos   = 0;

    for (int k = 1; k <= q_len + d_len - 1; ++k) {
        int start_row = diag_start_row(k, d_len);
        int elements  = diag_len(k, q_len, d_len);

        for (int idx = tid; idx < elements; idx += blockDim.x) {
            int row = start_row + idx;
            int col = k - row + 1;

            char q_char = d_q[row - 1];
            char d_char = d_d[col - 1];
            int sub = (q_char == d_char || q_char == 'N' || d_char == 'N')
                        ? match_score
                        : mismatch_score;

            int up   = (row > 1)           ? prev1[row - 1] : 0;
            int left = (col > 1)           ? prev1[row]     : 0;
            int diag = (row > 1 && col > 1)? prev2[row - 1] : 0;

            int score = sw_max4(0, diag + sub, up + gap_score, left + gap_score);

            curr[row] = score;
            d_matrix[row * width + col] = score;

            if (score > local_best_score) {
                local_best_score = score;
                local_best_pos   = row * width + col;
            }
        }

        __syncthreads();

        // Rotate rolling diagonals
        int* tmp = prev2;
        prev2 = prev1;
        prev1 = curr;
        curr  = tmp;

        __syncthreads();
    }

    extern __shared__ int s_reduce[];
    int* s_scores = s_reduce;
    int* s_pos    = s_reduce + blockDim.x;

    s_scores[tid] = local_best_score;
    s_pos[tid]    = local_best_pos;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_scores[tid + stride] > s_scores[tid]) {
                s_scores[tid] = s_scores[tid + stride];
                s_pos[tid]    = s_pos[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *d_best_score = s_scores[0];
        *d_best_pos   = s_pos[0];
    }
}

void run_cuda_smith_waterman(const std::string& q, const std::string& d,
                             int& out_score, int& out_start, int& out_stop,
                             std::string& out_aligned_q, std::string& out_aligned_d,
                             float& out_time_ms)
{
    const int q_len = static_cast<int>(q.size());
    const int d_len = static_cast<int>(d.size());
    const int rows  = q_len + 1;
    const int cols  = d_len + 1;

    const int match = 2;
    const int mismatch = -1;
    const int gap = -1;

    const size_t matrix_bytes = static_cast<size_t>(rows) * cols * sizeof(int);
    const size_t diag_bytes   = static_cast<size_t>(rows) * sizeof(int);

    char* d_q = nullptr;
    char* d_d = nullptr;
    int* d_matrix = nullptr;
    int* d_prev2 = nullptr;
    int* d_prev1 = nullptr;
    int* d_curr  = nullptr;
    int* d_best_score = nullptr;
    int* d_best_pos = nullptr;

    CUDA_CHECK(cudaMalloc(&d_q, q_len * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_d, d_len * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_matrix, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_prev2, diag_bytes));
    CUDA_CHECK(cudaMalloc(&d_prev1, diag_bytes));
    CUDA_CHECK(cudaMalloc(&d_curr,  diag_bytes));
    CUDA_CHECK(cudaMalloc(&d_best_score, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_pos,   sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_q, q.data(), q_len * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d, d.data(), d_len * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_matrix, 0, matrix_bytes));
    CUDA_CHECK(cudaMemset(d_prev2,  0, diag_bytes));
    CUDA_CHECK(cudaMemset(d_prev1,  0, diag_bytes));
    CUDA_CHECK(cudaMemset(d_curr,   0, diag_bytes));
    CUDA_CHECK(cudaMemset(d_best_score, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_best_pos,   0, sizeof(int)));

    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    CUDA_CHECK(cudaEventRecord(start_event));

    if (q_len <= 32) {
        sw_wavefront_warp_kernel<<<1, 32>>>(
            d_q, d_d, d_matrix,
            q_len, d_len,
            match, mismatch, gap,
            d_best_score, d_best_pos
        );
    } else {
        constexpr int TPB = 256;
        size_t shmem = 2 * TPB * sizeof(int);

        sw_wavefront_block_kernel<<<1, TPB, shmem>>>(
            d_q, d_d, d_matrix,
            d_prev2, d_prev1, d_curr,
            q_len, d_len,
            match, mismatch, gap,
            d_best_score, d_best_pos
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&out_time_ms, start_event, stop_event));

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    std::vector<int> h_matrix(rows * cols);
    int best_score = 0;
    int best_pos = 0;

    CUDA_CHECK(cudaMemcpy(h_matrix.data(), d_matrix, matrix_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&best_score, d_best_score, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&best_pos, d_best_pos, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_d));
    CUDA_CHECK(cudaFree(d_matrix));
    CUDA_CHECK(cudaFree(d_prev2));
    CUDA_CHECK(cudaFree(d_prev1));
    CUDA_CHECK(cudaFree(d_curr));
    CUDA_CHECK(cudaFree(d_best_score));
    CUDA_CHECK(cudaFree(d_best_pos));

    out_score = best_score;

    if (best_score == 0 || best_pos == 0) {
        out_start = 0;
        out_stop = -1;
        out_aligned_q.clear();
        out_aligned_d.clear();
        return;
    }

    int max_row = best_pos / cols;
    int max_col = best_pos % cols;

    out_stop = max_col - 1;

    std::string aligned_q;
    std::string aligned_d;
    int r = max_row;
    int c = max_col;

    while (r > 0 && c > 0 && h_matrix[r * cols + c] > 0) {
        int current_score = h_matrix[r * cols + c];
        int diag_score    = h_matrix[(r - 1) * cols + (c - 1)];
        int up_score      = h_matrix[(r - 1) * cols + c];
        int left_score    = h_matrix[r * cols + (c - 1)];

        char q_char = q[r - 1];
        char d_char = d[c - 1];
        int match_val = (q_char == d_char || q_char == 'N' || d_char == 'N') ? match : mismatch;

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