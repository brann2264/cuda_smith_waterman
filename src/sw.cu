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

#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <string>

// Define our Tile dimensions. 32 perfectly matches one CUDA Warp.
#define TILE_SIZE 32

// 1. Tiled GPU Kernel
__global__ void sw_tiled_macro_wavefront_kernel(
    const char* __restrict__ d_q, 
    const char* __restrict__ d_d, 
    int* __restrict__ d_matrix,            
    unsigned long long* __restrict__ d_max_info, 
    int q_len, int d_len, int match, int mismatch, int gap, int macro_k)                    
{
    // Determine which TILE this block is calculating based on the Macro-Diagonal (macro_k)
    int num_tiles_d = (d_len + TILE_SIZE - 1) / TILE_SIZE;
    
    int start_tile_row = (macro_k <= num_tiles_d) ? 1 : (macro_k - num_tiles_d + 1);
    int tile_row = start_tile_row + blockIdx.x;
    int tile_col = macro_k - tile_row + 1;

    // Thread ID handles a specific row/col within the tile
    int tx = threadIdx.x; 

    // Global matrix width
    int width = d_len + 1;

    // Allocate Ultra-Fast Shared Memory for this specific block
    __shared__ char s_q[TILE_SIZE];
    __shared__ char s_d[TILE_SIZE];
    __shared__ int s_matrix[TILE_SIZE + 1][TILE_SIZE + 1];

    // Global starting coordinates for this tile
    int global_q_idx = (tile_row - 1) * TILE_SIZE + tx;
    int global_d_idx = (tile_col - 1) * TILE_SIZE + tx;

    // --- PHASE 1: LOAD DATA INTO SHARED MEMORY ---
    
    // 1a. Load sequence strings (with bounds checking)
    s_q[tx] = (global_q_idx < q_len) ? __ldg(&d_q[global_q_idx]) : '-';
    s_d[tx] = (global_d_idx < d_len) ? __ldg(&d_d[global_d_idx]) : '-';

    // 1b. Load Top and Left boundaries from Global Memory into Shared Memory
    int global_row = (tile_row - 1) * TILE_SIZE + tx + 1;
    int global_col = (tile_col - 1) * TILE_SIZE + tx + 1;

    // Load Top Boundary (from the tile above us)
    if (global_col <= d_len) {
        s_matrix[0][tx + 1] = d_matrix[((tile_row - 1) * TILE_SIZE) * width + global_col];
    } else {
        s_matrix[0][tx + 1] = 0;
    }

    // Load Left Boundary (from the tile to our left)
    if (global_row <= q_len) {
        s_matrix[tx + 1][0] = d_matrix[global_row * width + ((tile_col - 1) * TILE_SIZE)];
    } else {
        s_matrix[tx + 1][0] = 0;
    }

    // Top-Left corner cell of the tile
    if (tx == 0) {
        s_matrix[0][0] = d_matrix[((tile_row - 1) * TILE_SIZE) * width + ((tile_col - 1) * TILE_SIZE)];
    }

    // Force all 32 threads to wait until Shared Memory is fully loaded
    __syncthreads();

    // --- PHASE 2: MICRO-WAVEFRONT (COMPUTE THE TILE) ---
    
    int local_max = 0;
    int local_max_r = 0;
    int local_max_c = 0;

    // Sweep across the 32x32 tile internally (63 micro-diagonals)
    for (int mk = 1; mk <= 2 * TILE_SIZE - 1; mk++) {
        int start_r = max(1, mk - TILE_SIZE + 1);
        int end_r = min(TILE_SIZE, mk);
        int elements = end_r - start_r + 1;

        if (tx < elements) {
            int r = start_r + tx;
            int c = mk - r + 1;

            char q_char = s_q[r - 1];
            char d_char = s_d[c - 1];
            
            int diag = s_matrix[r - 1][c - 1] + ((q_char == d_char || q_char == 'N' || d_char == 'N') ? match : mismatch);
            int up = s_matrix[r - 1][c] + gap;
            int left = s_matrix[r][c - 1] + gap;
            
            int cell_score = max(0, max(diag, max(up, left)));
            s_matrix[r][c] = cell_score;

            // Track the best score in this specific thread's path
            if (cell_score > local_max && (global_q_idx - tx + r - 1) < q_len && (global_d_idx - tx + c - 1) < d_len) {
                local_max = cell_score;
                local_max_r = (tile_row - 1) * TILE_SIZE + r;
                local_max_c = (tile_col - 1) * TILE_SIZE + c;
            }
        }
        // Wait for all threads to finish this micro-diagonal before moving to the next
        __syncthreads(); 
    }

    // --- PHASE 3: WRITE RESULTS TO GLOBAL MEMORY ---
    
    // Write the computed interior of the tile back to the global matrix
    if (global_row <= q_len) {
        for (int c = 1; c <= TILE_SIZE; c++) {
            if (((tile_col - 1) * TILE_SIZE + c) <= d_len) {
                d_matrix[global_row * width + ((tile_col - 1) * TILE_SIZE + c)] = s_matrix[tx + 1][c];
            }
        }
    }

    // OPTIMIZATION: Push highest score to global tracker
    if (local_max > 0) {
        unsigned long long packed = ((unsigned long long)local_max << 42) | 
                                    ((unsigned long long)local_max_r << 21) | 
                                    (unsigned long long)local_max_c;
        atomicMax(d_max_info, packed);
    }
}

// 2. Host function
void run_cuda_smith_waterman(const std::string& q, const std::string& d, 
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