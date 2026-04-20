#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <cuda.h>
#include <cub/cub.cuh>
#include "reference.h"

#define GPU_NUM_THREADS 256

/*
Mathematical specification (pre/post conditions and invariants)

Preconditions:
- N >= 0, D >= 0, top_k >= 0
- For all 0 <= r < N: 0 <= labelData[r] < D
- Xdata is a dense row-major tensor of size N*D
- accuracy points to a valid scalar integer in device memory

Postcondition:
- accuracy contains the total number of rows r in [0, N) for which the label index labelData[r]
  is ranked within top_k by descending score order computed from Xdata[r, :], with a stable
  tie-break defined by: a column c contributes as "greater or equal" to the rank if
     (Xdata[r, c] > Xdata[r, label]) OR (Xdata[r, c] == Xdata[r, label] AND c <= label)
  i.e., the 1-based rank is the count of such columns. The row is correct if rank <= top_k.

Loop invariants (per block and per row):
- Let S_row be the set of columns processed by all threads in the block for a given row.
  S_row equals the full column set {0, 1, ..., D-1}.
- For each thread t, let ngt_t be its local counter of columns in S_row ∩ S_t (columns assigned to t)
  satisfying the predicate above. The block reduction returns ngt = sum_t ngt_t.
- BlockReduce is invoked with identical input semantics in all threads; after it returns,
  each thread holds the same ngt value equal to the block-wide sum across S_row.
- The per-block variable 'count' equals the number of rows handled by this block so far
  for which ngt <= top_k holds (inductive: true initially; after each row, updated by +1 iff condition holds).

Safety and absence of races:
- Shared variables s_label and s_label_pred are written by thread 0 only and read by all threads exactly
  after a __syncthreads() at the start of an iteration. After that point, threads use register copies only.
  No thread accesses s_label/s_label_pred again until the next iteration's __syncthreads(), hence removing
  the trailing __syncthreads() is safe and does not introduce a race.
- BlockReduce uses internal synchronization to safely compute the sum. No re-use of its shared storage occurs
  before all threads have returned from the call.

Proof sketch:
- Initialization: count = 0 satisfies invariant.
- Maintenance: For a given row, each thread t enumerates a disjoint subset S_t of columns (strided by blockDim.x).
  The predicate evaluation over S_t contributes ngt_t to the rank count. Because the S_t form a partition of S_row,
  reduction ngt = sum_t ngt_t equals the total number of columns in S_row satisfying the predicate, by additivity.
  If (ngt <= top_k), thread 0 increments count, preserving the invariant that count tracks the number of rows
  satisfying the postcondition among those already processed by the block.
- Termination: After the loop over all rows assigned to the block, count equals the number of "correct" rows
  processed by this block. A single atomicAdd from thread 0 accumulates the block's count into the global
  accuracy, establishing the postcondition across all blocks by associativity and commutativity of integer addition.
*/

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
  // Shared broadcast for the row's label and its score to minimize redundant global loads
  __shared__ float s_label_pred;
  __shared__ int   s_label;

  int count = 0;

  for (int row = blockIdx.x; row < N; row += gridDim.x) {
    // Broadcast label and label_pred once per row
    if (threadIdx.x == 0) {
      const int lbl = labelData[row];
      s_label = lbl;
      s_label_pred = Xdata[row * D + lbl];
    }
    __syncthreads();

    const int   label      = s_label;
    const float label_pred = s_label_pred;

    // Count number of predictions >= label's prediction with stable tie-breaker
    int ngt = 0;
    const float* __restrict__ row_ptr = Xdata + row * D;
    const int stride = blockDim.x;

    int col = threadIdx.x;

    // Unroll by 4 to increase ILP and reduce scoreboard stalls (addresses PC: stall_long_sb_ratio, stall_wait_ratio)
    for (; col + 3 * stride < D; col += 4 * stride) {
      float p0 = row_ptr[col];
      float p1 = row_ptr[col +     stride];
      float p2 = row_ptr[col + 2 * stride];
      float p3 = row_ptr[col + 3 * stride];

      ngt += (p0 > label_pred) || (p0 == label_pred && (col              <= label));
      ngt += (p1 > label_pred) || (p1 == label_pred && (col +     stride <= label));
      ngt += (p2 > label_pred) || (p2 == label_pred && (col + 2 * stride <= label));
      ngt += (p3 > label_pred) || (p3 == label_pred && (col + 3 * stride <= label));
    }

    for (; col < D; col += stride) {
      float pred = row_ptr[col];
      ngt += (pred > label_pred) || (pred == label_pred && col <= label);
    }

    // Block-wide sum of predicate counts
    BlockReduce(ngt);

    if (threadIdx.x == 0 && ngt <= top_k) {
      ++count;
    }

    // No barrier needed here: s_label/s_label_pred won't be read again until the next loop's barrier.
  }

  if (threadIdx.x == 0 && count > 0) {
    atomicAdd(accuracy, count);
  }
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
  // Shared broadcast for the row's label and its score to minimize redundant global loads
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
    const float* __restrict__ row_ptr = Xdata + row * D;
    const int stride = blockDim.x;

    int col = threadIdx.x;
    for (; col + 3 * stride < D; col += 4 * stride) {
      float p0 = row_ptr[col];
      float p1 = row_ptr[col +     stride];
      float p2 = row_ptr[col + 2 * stride];
      float p3 = row_ptr[col + 3 * stride];

      ngt += (p0 > label_pred) || (p0 == label_pred && (col              <= label));
      ngt += (p1 > label_pred) || (p1 == label_pred && (col +     stride <= label));
      ngt += (p2 > label_pred) || (p2 == label_pred && (col + 2 * stride <= label));
      ngt += (p3 > label_pred) || (p3 == label_pred && (col + 3 * stride <= label));
    }
    for (; col < D; col += stride) {
      float pred = row_ptr[col];
      ngt += (pred > label_pred) || (pred == label_pred && col <= label);
    }

    BlockReduce(ngt);

    if (threadIdx.x == 0 && ngt <= top_k) {
      ++count;
    }
    // No trailing barrier needed for correctness; removing it reduces stall_membar_ratio (PC lines: 38, 43)
  }

  if (threadIdx.x == 0 && count > 0) {
    atomicAdd(accuracy, count);
  }
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

  for (int ngrid = nrows / 4; ngrid <= nrows; ngrid += (nrows > 4 ? nrows / 4 : 1)) {

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