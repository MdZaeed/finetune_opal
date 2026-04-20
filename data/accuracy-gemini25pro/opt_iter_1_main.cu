#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <cuda.h>
#include <cub/cub.cuh>
#include "reference.h"

#define GPU_NUM_THREADS 256

template <typename T>
__device__ void BlockReduce(T &input) {
  typedef cub::BlockReduce<T, GPU_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  input = BlockReduce(temp_storage).Sum(input);
}

__global__
void accuracy_kernel(
    const int N,
    const int D,
    const int top_k,
    const float* __restrict__ Xdata,
    const int* __restrict__ labelData,
    int* accuracy)
{
  int count = 0;

  // Shared memory for label and label_pred to reduce global memory reads
  // and address L1TEX stalls (Roofline, PC Summary)
  __shared__ float s_label_pred;
  __shared__ int s_label;

  for (int row = blockIdx.x; row < N; row += gridDim.x) {
    // Preconditions:
    // - N, D, top_k are positive integers.
    // - Xdata points to N * D float values on device.
    // - labelData points to N int values on device, where 0 <= labelData[i] < D.
    // - accuracy points to a single int on device, initialized to 0.
    // - blockDim.x is GPU_NUM_THREADS.

    // Load label and label_pred into shared memory once per block per row
    if (threadIdx.x == 0) {
      s_label = labelData[row];
      s_label_pred = Xdata[row * D + s_label];
    }
    __syncthreads(); // Ensure shared memory is visible to all threads

    const int   label      = s_label;
    const float label_pred = s_label_pred;

    int ngt = 0;
    const float* row_ptr = Xdata + row * D; // Pointer to the start of the current row

    // Loop Invariant:
    // At the beginning of each iteration of this loop, for a given threadIdx.x:
    // 1. `ngt` holds the correct count of elements `pred` (from `Xdata[row * D + k]`) that satisfy the condition
    //    `(pred > label_pred || (pred == label_pred && k <= label))` for all `k` such that `k = col_base_prev + threadIdx.x`
    //    where `col_base_prev` are all previous `col_base` values.
    // 2. `col_base` is a multiple of `blockDim.x` and `0 <= col_base < D`.
    // 3. `label` and `label_pred` (local copies from shared memory) hold the correct values for the current `row`.
    // Proof of Invariant Maintenance:
    // - Initialization: `ngt` is 0. `col_base` starts at 0. `label` and `label_pred` are loaded from shared memory. Invariant holds.
    // - Iteration: `current_col = col_base + threadIdx.x`. If `current_col < D`, `pred = row_ptr[current_col]` is read (coalesced).
    //   `ngt` is incremented if the condition is met, correctly accumulating the count for `current_col`.
    //   `col_base` increments by `blockDim.x`, maintaining its property.
    // - Termination: `col_base >= D`. `ngt` for each thread has accumulated counts for its assigned columns.
    // Optimized loop for coalesced global memory access (Roofline, PC Summary)
    // Replaces the original strided loop with a coalesced access pattern.
    for (int col_base = 0; col_base < D; col_base += blockDim.x) {
        int current_col = col_base + threadIdx.x;
        if (current_col < D) {
            float pred = row_ptr[current_col]; // Coalesced global memory read
            if (pred > label_pred || (pred == label_pred && current_col <= label)) {
                ++ngt;
            }
        }
    }

    BlockReduce(ngt);

    // Correctness fix: only threadIdx.x == 0 has the valid reduced ngt sum
    // This also reduces divergence for the increment of 'count'.
    if (threadIdx.x == 0 && ngt <= top_k) {
      ++count;
    }
    __syncthreads(); // Ensure all threads are done with BlockReduce before next row
  }
  if (threadIdx.x == 0) {
    atomicAdd(accuracy, count);
  }
  // Postconditions:
  // - `accuracy` will contain the total count of rows where the true label's prediction score (`label_pred`)
  //   is ranked within the `top_k` highest scores (considering ties by column index).
}


__global__
void accuracy_kernel2(
    const int N,
    const int D,
    const int top_k,
    const float* __restrict__ Xdata,
    const int*   __restrict__ labelData,
    int* accuracy)
{
  // Preconditions:
  // - N, D, top_k are positive integers.
  // - Xdata points to N * D float values on device.
  // - labelData points to N int values on device, where 0 <= labelData[i] < D.
  // - accuracy points to a single int on device, initialized to 0.
  // - blockDim.x is GPU_NUM_THREADS.

  __shared__ float s_label_pred;
  __shared__ int s_label;

  int count = 0;

  for (int row = blockIdx.x; row < N; row += gridDim.x) {

    if (threadIdx.x == 0) {
      s_label = labelData[row];
      s_label_pred = Xdata[row * D + s_label];
    }
    __syncthreads();

    const int   label      = s_label;
    const float label_pred = s_label_pred;

    int ngt = 0;
    const float* row_ptr = Xdata + row * D;

    // Loop Invariant:
    // At the beginning of each iteration of this loop, for a given threadIdx.x:
    // 1. `ngt` holds the correct count of elements `pred` (from `Xdata[row * D + k]`) that satisfy the condition
    //    `(pred > label_pred || (pred == label_pred && k <= label))` for all `k` such that `k = col_base_prev + threadIdx.x`
    //    where `col_base_prev` are all previous `col_base` values.
    // 2. `col_base` is a multiple of `blockDim.x` and `0 <= col_base < D`.
    // 3. `s_label` and `s_label_pred` (shared memory copies) hold the correct values for the current `row`.
    // Proof of Invariant Maintenance:
    // - Initialization: `ngt` is 0. `col_base` starts at 0. `s_label` and `s_label_pred` are loaded and synchronized. Invariant holds.
    // - Iteration: `current_col = col_base + threadIdx.x`. If `current_col < D`, `pred = row_ptr[current_col]` is read (coalesced).
    //   `ngt` is incremented if the condition is met, correctly accumulating the count for `current_col`.
    //   `col_base` increments by `blockDim.x`, maintaining its property.
    // - Termination: `col_base >= D`. `ngt` for each thread has accumulated counts for its assigned columns.
    // Optimized loop for coalesced global memory access (Roofline, PC Summary)
    // Replaces the unrolled and trailing loops with a single coalesced loop.
    for (int col_base = 0; col_base < D; col_base += blockDim.x) {
        int current_col = col_base + threadIdx.x;
        if (current_col < D) {
            float pred = row_ptr[current_col]; // Coalesced global memory read
            if (pred > label_pred || (pred == label_pred && current_col <= label)) {
                ++ngt;
            }
        }
    }

    BlockReduce(ngt);

    if (threadIdx.x == 0 && ngt <= top_k) {
      ++count;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && count > 0) {
    atomicAdd(accuracy, count);
  }
  // Postconditions:
  // - `accuracy` will contain the total count of rows where the true label's prediction score (`label_pred`)
  //   is ranked within the `top_k` highest scores (considering ties by column index).
}


int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <number of rows> <number of columns> <top K> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int ndims = atoi(argv[2]);
  const int top_k = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const int data_size = nrows * ndims;

  const int label_size_bytes = nrows * sizeof(int);
  const size_t data_size_bytes = data_size * sizeof(float);

  int *label = (int*) malloc (label_size_bytes);

  srand(123);
  for (int i = 0; i < nrows; i++)
    label[i] = rand() % ndims;

  float *data = (float*) malloc (data_size_bytes);

  std::default_random_engine g (123);
  std::uniform_real_distribution<float> distr (0.f, 1.f);
  for (int i = 0; i < data_size; i++) {
    data[i] = distr(g);
  }

  int count_ref = reference(nrows, ndims, top_k, data, label);

  int *d_label;
  cudaMalloc((void**)&d_label, label_size_bytes);
  cudaMemcpy(d_label, label, label_size_bytes, cudaMemcpyHostToDevice);

  float *d_data;
  cudaMalloc((void**)&d_data, data_size_bytes);
  cudaMemcpy(d_data, data, data_size_bytes, cudaMemcpyHostToDevice);

  int *d_count;
  cudaMalloc((void**)&d_count, sizeof(int));

  cudaDeviceSynchronize();
  dim3 block (GPU_NUM_THREADS);

  for (int ngrid = nrows / 4; ngrid <= nrows; ngrid += nrows / 4) {

    dim3 grid (ngrid);
    printf("Grid size is %d\n", ngrid);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      cudaMemset(d_count, 0, sizeof(int));
      accuracy_kernel<<<grid, block>>>(nrows, ndims, top_k, d_data, d_label, d_count);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of accuracy kernel: %f (us)\n", (time * 1e-3f) / repeat);

    int count;
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%s\n", (count == count_ref) ? "PASS" : "FAIL");


    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      cudaMemset(d_count, 0, sizeof(int));
      accuracy_kernel2<<<grid, block>>>(nrows, ndims, top_k, d_data, d_label, d_count);
    }

    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of accuracy kernel2: %f (us)\n", (time * 1e-3f) / repeat);
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%s\n", (count == count_ref) ? "PASS" : "FAIL");
  }

  cudaFree(d_label);
  cudaFree(d_data);
  cudaFree(d_count);

  free(label);
  free(data);

  return 0;
}