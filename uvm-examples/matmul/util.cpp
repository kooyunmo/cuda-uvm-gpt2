#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>

static double start_time[8];

static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void timer_start(int i) {
    start_time[i] = get_time();
}

double timer_stop(int i) {
    return get_time() - start_time[i];
}

void check_mat_mul(float *A, float *B, float *C, int M, int N, int K) {
  printf("Validating...\n");

  float *C_ans;
  alloc_mat(&C_ans, M, N);
  zero_mat(C_ans, M, N);
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < N; ++j) {
        C_ans[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }

  bool is_valid = true;
  int cnt = 0, thr = 10;
  float eps = 1e-3;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float c = C[i * N + j];
      float c_ans = C_ans[i * N + j];
      if (fabsf(c - c_ans) > eps && (c_ans == 0 || fabsf((c - c_ans) / c_ans) > eps)) {
        ++cnt;
        if (cnt <= thr)
          printf("C[%d][%d] : correct_value = %f, your_value = %f\n", i, j, c_ans, c);
        if (cnt == thr + 1)
          printf("Too many error, only first %d values are printed.\n", thr);
        is_valid = false;
      }
    }
  }

  if (is_valid) {
    printf("Result: VALID\n");
  } else {
    printf("Result: INVALID\n");
  }
}

void print_mat(float *m, int R, int C) {
  for (int i = 0; i < R; ++i) { 
    for (int j = 0; j < C; ++j) {
      printf("%+.3f ", m[i * C + j]);
    }
    printf("\n");
  }
}

void alloc_mat(float **m, int R, int C) {
  *m = (float *) malloc(sizeof(float) * R * C);
  if (*m == NULL) {
    printf("Failed to allocate memory for matrix.\n");
    exit(0);
  }
}

void rand_mat(float *m, int R, int C) {
  for (int i = 0; i < R; i++) { 
    for (int j = 0; j < C; j++) {
      m[i * C + j] = (float) rand() / RAND_MAX - 0.5;
    }
  }
}

void zero_mat(float *m, int R, int C) {
  memset(m, 0, sizeof(float) * R * C);
}