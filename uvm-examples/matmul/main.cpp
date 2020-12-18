#include <stdio.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdlib.h>

#include "util.h"
#include "mat_mul.h"

static void print_help(const char* prog_name) {
  printf("Usage: %s [-pvh] [-t num_threads] [-n num_iterations] M N K\n", prog_name);
  printf("Options:\n");
  printf("  -p : print matrix data. (default: off)\n");
  printf("  -v : validate matrix multiplication. (default: off)\n");
  printf("  -h : print this page.\n");
  printf("  -t : number of threads (default: 1)\n");
  printf("  -n : number of iterations (default: 1)\n");
  printf("   M : number of rows of matrix A and C. (default: 8)\n");
  printf("   N : number of columns of matrix B and C. (default: 8)\n");
  printf("   K : number of columns of matrix A and rows of B. (default: 8)\n");
}

static bool print_matrix = false;
static bool validation = false;
static int M = 8, N = 8, K = 8;
static int num_threads = 1;
static int num_iterations = 1;

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "pvht:n:")) != -1) {
    switch (c) {
      case 'p':
        print_matrix = true;
        break;
      case 'v':
        validation = true;
        break;
      case 't':
        num_threads = atoi(optarg);
        break;
      case 'n':
        num_iterations = atoi(optarg);
        break;
      case 'h':
      default:
        print_help(argv[0]);
        exit(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j) {
    switch (j) {
      case 0: M = atoi(argv[i]); break;
      case 1: N = atoi(argv[i]); break;
      case 2: K = atoi(argv[i]); break;
      default: break;
    }
  }
  printf("Options:\n");
  printf("  Problem size: M = %d, N = %d, K = %d\n", M, N, K);
  printf("  Number of threads: %d\n", num_threads);
  printf("  Number of iterations: %d\n", num_iterations);
  printf("  Print matrix: %s\n", print_matrix ? "on" : "off");
  printf("  Validation: %s\n", validation ? "on" : "off");
  printf("\n");
}

int main(int argc, char **argv) {
  parse_opt(argc, argv);

  printf("Initializing matrix... "); fflush(stdout);
  float *A, *B, *C;
  alloc_mat(&A, M, K);
  alloc_mat(&B, K, N);
  alloc_mat(&C, M, N);
  rand_mat(A, M, K);
  rand_mat(B, K, N);
  printf("done!\n");

  printf("Initializing OpenCL...\n"); fflush(stdout);
  mat_mul_init(A, B, C, M, N, K);

  double elapsed_time_sum = 0;
  for (int i = -3; i < num_iterations; ++i) {
    if (i < 0) {
      printf("Warming up GPU..."); fflush(stdout);
    } else {
      printf("Calculating...(iter=%d) ", i); fflush(stdout);
    }
    timer_start(0);
    mat_mul(A, B, C, M, N, K);
    double elapsed_time = timer_stop(0);
    printf("%f sec\n", elapsed_time);
    if (i < 0) {
    } else {
      elapsed_time_sum += elapsed_time;
    }
  }

  if (print_matrix) {
    printf("MATRIX A:\n"); print_mat(A, M, K);
    printf("MATRIX B:\n"); print_mat(B, K, N);
    printf("MATRIX C:\n"); print_mat(C, M, N);
  }

  if (validation) {
    mat_mul_final(A, B, C, M, N, K);
    check_mat_mul(A, B, C, M, N, K);
  }

  double elapsed_time_avg = elapsed_time_sum / num_iterations;
  printf("Avg. time: %f sec\n", elapsed_time_avg);
  printf("Avg. throughput: %f GFLOPS\n", 2.0 * M * N * K / elapsed_time_avg / 1e9);

  return 0;
}