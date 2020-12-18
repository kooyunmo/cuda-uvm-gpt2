#include "mat_mul.h"
#include "util.h"

#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_THREAD_PER_BLOCK    32
#define VECTOR_WIDTH            4


__global__ void sgemm(float4* A, float4* B, float4* C, int M, int N, int K) {
  // blockDim: dimension of a block (the number of threads in one block)   (32, 8)
  // blockIdx: the index of current block
  // threadIdx: the index of current thread inside current block
  const int row = threadIdx.x;
  const int col = threadIdx.y;
  const int global_row = blockDim.x * blockIdx.x + threadIdx.x;    // row ID of C [0: M)
  const int global_col = blockDim.y * blockIdx.y + threadIdx.y;    // col ID of C [0: N/VECTOR_WIDTH]

  __shared__ float4 tileA[NUM_THREAD_PER_BLOCK][NUM_THREAD_PER_BLOCK/VECTOR_WIDTH];
  __shared__ float4 tileB[NUM_THREAD_PER_BLOCK][NUM_THREAD_PER_BLOCK/VECTOR_WIDTH];

  float4 i_val = { 0.0f, 0.0f, 0.0f, 0.0f };

  const int num_tiles = K/NUM_THREAD_PER_BLOCK;  
  for (int t = 0; t < num_tiles; t++) {
    const int t_row = NUM_THREAD_PER_BLOCK * t + row;
    const int t_col = (NUM_THREAD_PER_BLOCK/VECTOR_WIDTH) * t + col;
    tileA[row][col] = A[global_row * (K/VECTOR_WIDTH) + t_col];
    tileB[row][col] = B[t_row * (N/VECTOR_WIDTH) + global_col];

    __syncthreads();

    float4 vecA, vecB;
    float valA;
    for(int k = 0; k < NUM_THREAD_PER_BLOCK/VECTOR_WIDTH; k++) {
      vecA = tileA[row][k];
      for (int w = 0; w < VECTOR_WIDTH; w++) {
        vecB = tileB[VECTOR_WIDTH*k + w][col];

        switch(w) {
          case 0: valA = vecA.x;  break;
          case 1: valA = vecA.y;  break;
          case 2: valA = vecA.z;  break;
          case 3: valA = vecA.w;  break;
        } 

        i_val.x += vecB.x * valA;
        i_val.y += vecB.y * valA;
        i_val.z += vecB.z * valA;
        i_val.w += vecB.w * valA;
      }
    }

    __syncthreads();
  }

  C[global_row*(N/VECTOR_WIDTH) + global_col] = i_val;
}


static float *a_d, *b_d, *c_d;
static float *paddedA, *paddedB, *paddedC;

void add_padding(float *mat, float *padded_mat, int row, int col, int pad_row, int pad_col);
void remove_padding(float *mat, float *padded_mat, int row, int col, int pad_row, int pad_col);

void mat_mul(float *A, float *B, float *C, int M, int N, int K) {
  if (M % NUM_THREAD_PER_BLOCK != 0) {
    M = M + (NUM_THREAD_PER_BLOCK - (M % NUM_THREAD_PER_BLOCK));
  }
  if (K % NUM_THREAD_PER_BLOCK != 0) {
    K = K + (NUM_THREAD_PER_BLOCK - (K % NUM_THREAD_PER_BLOCK));
  }
  if (N % NUM_THREAD_PER_BLOCK != 0) {
    N = N + (NUM_THREAD_PER_BLOCK - (N % NUM_THREAD_PER_BLOCK));
  }

  dim3 gridDim(M/NUM_THREAD_PER_BLOCK, N/NUM_THREAD_PER_BLOCK, 1);  // The number of thread blocks
  dim3 blockDim(NUM_THREAD_PER_BLOCK, NUM_THREAD_PER_BLOCK/VECTOR_WIDTH, 1);     // The number of threads per a thread block
  sgemm<<<gridDim, blockDim>>>((float4 *)a_d, (float4 *)b_d, (float4 *)c_d, M, N, K);

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  cudaDeviceSynchronize();
}

void mat_mul_init(float *A, float *B, float *C, int M, int N, int K) {
  int origin_M = M, origin_K = K, origin_N = N;  
  
  if (M % NUM_THREAD_PER_BLOCK != 0) {
    M = M + (NUM_THREAD_PER_BLOCK - (M % NUM_THREAD_PER_BLOCK));
  }
  if (K % NUM_THREAD_PER_BLOCK != 0) {
    K = K + (NUM_THREAD_PER_BLOCK - (K % NUM_THREAD_PER_BLOCK));
  }
  if (N % NUM_THREAD_PER_BLOCK != 0) {
    N = N + (NUM_THREAD_PER_BLOCK - (N % NUM_THREAD_PER_BLOCK));
  }

  if (M != origin_M || K != origin_K) {
    alloc_mat(&paddedA, M, K);
    add_padding(A, paddedA, origin_M, origin_K, M, K);
  }
  if (K != origin_K || N != origin_N) {
    alloc_mat(&paddedB, K, N);
    add_padding(B, paddedB, origin_K, origin_N, K, N);
  }
  if (M != origin_M || N != origin_N) {
    alloc_mat(&paddedC, M, N);
    add_padding(C, paddedC, origin_M, origin_N, M, N);
  }

  cudaMalloc(&a_d, M * K * sizeof(float));
  cudaMalloc(&b_d, K * N * sizeof(float));
  cudaMalloc(&c_d, M * N * sizeof(float));

  if (M != origin_M || K != origin_K) {
    cudaMemcpy(a_d, paddedA, M * K * sizeof(float), cudaMemcpyHostToDevice);
  }
  else {
    cudaMemcpy(a_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  }
  if (K != origin_K || N != origin_N) {
    cudaMemcpy(b_d, paddedB, K * N * sizeof(float), cudaMemcpyHostToDevice);
  }
  else {
    cudaMemcpy(b_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  cudaDeviceSynchronize();
}

void mat_mul_final(float *A, float *B, float *C, int M, int N, int K) {
  int origin_M = M, origin_N = N;

  if (M % NUM_THREAD_PER_BLOCK != 0) {
    M = M + (NUM_THREAD_PER_BLOCK - (M % NUM_THREAD_PER_BLOCK));
  }
  if (N % NUM_THREAD_PER_BLOCK != 0) {
    N = N + (NUM_THREAD_PER_BLOCK - (N % NUM_THREAD_PER_BLOCK));
  }

  if (M != origin_M || N != origin_N) {
    cudaMemcpy(paddedC, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  }
  else {
    cudaMemcpy(C, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  cudaDeviceSynchronize();

  if (M != origin_M || N != origin_N) {
    remove_padding(C, paddedC, origin_M, origin_N, M, N);
  }
}

void add_padding(float *mat, float *padded_mat, int row, int col, int pad_row, int pad_col) {
  for (int i=0; i<pad_row; i++) {
    for (int j=0; j<pad_col; j++) {
      if (i < row && j < col) {
        padded_mat[i*pad_col + j] = mat[i*col + j];
      }
      else {
        padded_mat[i*pad_col + j] = 0.0f;
      }
    }
  }
}

void remove_padding(float *mat, float *padded_mat, int row, int col, int pad_row, int pad_col) {
  for (int i=0; i<row; i++) {
    for (int j=0; j<col; j++) {
      mat[i*col + j] = padded_mat[i*pad_col + j];
    }
  }
}