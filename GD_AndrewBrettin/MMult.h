#ifndef MMULT_H
#define MMULT_H

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

#define BLOCK_SIZE 16

// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
void MMult0(long m, long n, long k, double *a, double *b, double *c) {
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}

void MMult1(long m, long n, long k, double *a, double *b, double *c) {
  long N = m/BLOCK_SIZE;
  for (long J = 1; J < N+1; J++){
    for (long I = 1; I < N+1; I++){
      for (long P = 1; P < N+1; P++){
        #pragma omp parallel
        #pragma omp for collapse(2)
        for (long j = (J-1)*BLOCK_SIZE; j < J*BLOCK_SIZE; j++) {
          for (long i = (I-1)*BLOCK_SIZE; i < I*BLOCK_SIZE; i++) {
            double C_ij = c[i+j*m];
            for (long p = (P-1)*BLOCK_SIZE; p < P*BLOCK_SIZE; p++) {
              double A_ip = a[i+p*m];
              double B_pj = b[p+j*k];
              C_ij = C_ij + A_ip * B_pj;
            }
            c[i+j*m] = C_ij;
          }
        }
      }
    }
  }
}

#endif