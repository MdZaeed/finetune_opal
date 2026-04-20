#include "sobol.h"
#include "sobol_gpu.h"

#define k_2powneg32 2.3283064E-10F
#define SOBOL_BITS_PER_DIMENSION 32 // Assuming 32-bit Sobol numbers, fixing the bug with 'n_directions'

__global__
void sobolGPU_kernel(unsigned n_vectors, unsigned n_dimensions,
                     unsigned *__restrict__ d_directions,
                     float *__restrict__ d_output)
{


    __shared__ unsigned int v[SOBOL_BITS_PER_DIMENSION];



    d_directions += SOBOL_BITS_PER_DIMENSION * blockIdx.y;
    d_output += n_vectors * blockIdx.y;




    if (threadIdx.x < SOBOL_BITS_PER_DIMENSION)
    {
        v[threadIdx.x] = d_directions[threadIdx.x];
    }


    __syncthreads();




    int i0     = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;




    unsigned int g = i0 ^ (i0 >> 1);





    unsigned int X = 0;
    unsigned int mask;

    #pragma unroll
    for (unsigned int k = 0 ; k < __ffs(stride) - 1 ; k++)
    {




        mask = - (g & 1);
        X ^= mask & v[k];
        g = g >> 1;
    }

    if (i0 < n_vectors)
    {
        d_output[i0] = (float)X * k_2powneg32;
    }






























    unsigned int v_log2stridem1 = v[__ffs(stride) - 2];
    unsigned int v_stridemask = stride - 1;

    for (unsigned int i = i0 + stride ; i < n_vectors ; i += stride)
    {




        X ^= v_log2stridem1 ^ v[__ffs(~((i - stride) | v_stridemask)) - 1];
        d_output[i] = (float)X * k_2powneg32;
    }
}

double sobolGPU(int repeat, int n_vectors, int n_dimensions,
                unsigned int *d_directions, float *d_output)
{
    const int threadsperblock = 64;


    dim3 dimGrid;
    dim3 dimBlock;







    dimGrid.y = n_dimensions;




    if (n_dimensions < (4 * 24))
    {
        dimGrid.x = 4 * 24;
    }
    else
    {
        dimGrid.x = 1;
    }


    if (dimGrid.x > (unsigned int)(n_vectors / threadsperblock))
    {
        dimGrid.x = (n_vectors + threadsperblock - 1) / threadsperblock;
    }



    unsigned int targetDimGridX = dimGrid.x;
    for (dimGrid.x = 1 ; dimGrid.x < targetDimGridX ; dimGrid.x *= 2);


    dimBlock.x = threadsperblock;

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();


    for (int i = 0; i < repeat; i++)
      sobolGPU_kernel <<<dimGrid, dimBlock>>> (
        n_vectors, n_dimensions, d_directions, d_output);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return time;
}