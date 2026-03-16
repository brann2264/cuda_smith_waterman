#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <string>

// 1. GPU Kernel
__global__ void sw_wavefront_kernel(
    const char* d_q, const char* d_d, int* d_matrix,            
    int q_len, int d_len, int match_score, int mismatch_score, int gap_score, int k)                    
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int elements_in_diag = min(min(k, q_len), min(d_len, q_len + d_len - k));

    if (tid < elements_in_diag) {
        int start_row = (k <= d_len) ? 1 : (k - d_len + 1);
        int row = start_row + tid;
        int col = k - row + 1;
        int width = d_len + 1;

        char q_char = d_q[row - 1];
        char d_char = d_d[col - 1];

        int diag_score = d_matrix[(row - 1) * width + (col - 1)];
        
        // Handle 'N' wildcard matching
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

// 2. Host function (Updated to return out_time_ms)
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
    cudaMalloc(&d_q, q_len * sizeof(char));
    cudaMalloc(&d_d, d_len * sizeof(char));
    cudaMalloc(&d_matrix, matrix_bytes);

    cudaMemcpy(d_q, q.c_str(), q_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, d.c_str(), d_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(d_matrix, 0, matrix_bytes); 

    // --- START CUDA TIMER ---
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event);

    int match = 2, mismatch = -1, gap = -1; 
    int total_diagonals = q_len + d_len - 1;
    
    for (int k = 1; k <= total_diagonals; ++k) {
        int elements = std::min(std::min(k, q_len), std::min(d_len, q_len + d_len - k));
        int threadsPerBlock = 256;
        int blocksPerGrid = (elements + threadsPerBlock - 1) / threadsPerBlock;

        sw_wavefront_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_q, d_d, d_matrix, q_len, d_len, match, mismatch, gap, k
        );
    }

    // --- STOP CUDA TIMER ---
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event); 
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    
    // Pass time back to main.cpp instead of printing
    out_time_ms = milliseconds; 

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    int* h_matrix = new int[rows * cols];
    cudaMemcpy(h_matrix, d_matrix, matrix_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_d);
    cudaFree(d_matrix);

    // --- CPU TRACEBACK ---
    int max_score = 0, max_row = 0, max_col = 0;

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
    out_stop = max_col - 1; 

    std::string aligned_q = "";
    std::string aligned_d = "";
    int r = max_row;
    int c = max_col;

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

    delete[] h_matrix;
}