#pragma once

void timer_start(int i);

double timer_stop(int i);

void check_mat_mul(float *A, float *B, float *C, int M, int N, int K);

void print_mat(float *m, int R, int C);

void alloc_mat(float **m, int R, int C);

void rand_mat(float *m, int R, int C);

void zero_mat(float *m, int R, int C);
