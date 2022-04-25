#ifndef MMULT_H
#define MMULT_H

#include <stdio.h>
#include <math.h>
//#include <omp.h>
#include "utils.h"


#define BLOCK_SIZE 16

// NOTE: matrices are stored in column major order; i.e. the array elements in
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


// NOTE: works only on square matrices of size multiple of BLOCK_SIZE !!
void MMult1(long m, long n, long k, double *a, double *b, double *c) {
   long bs = BLOCK_SIZE;
   long N  = n / bs;   // we assume n = k = m ( = p)
   for (long i = 1; i < N+1; i++){
      for (long j = 1; j < N+1; j++){
         for (long k = 1; k < N+1; k++){
            //#pragma omp parallel
            //#pragma omp for collapse(2)
            for (long p = (i-1)*bs; p < i*bs; p++) {
               for (long q = (j-1)*bs; q < j*bs; q++) {
                  double c_pq = c[p+q*n];
                  for (long r = (k-1)*bs; r < k*bs; r++) {
                     double a_pr = a[p+r*n];
                     double b_rq = b[r+q*n];
                     c_pq = c_pq + a_pr * b_rq;
                  }
                  c[p+q*n] = c_pq;
               }
            }
         }
      }
   }
}


// NOTE: works only on square matrices of size multiple of BLOCK_SIZE !!
void MMult2(long m, long n, long k, double *a, double *b, double *c) {
   long N = m/BLOCK_SIZE;
   for (long J = 1; J < N+1; J++){
      for (long I = 1; I < N+1; I++){
         for (long P = 1; P < N+1; P++){
            //#pragma omp parallel
            //#pragma omp for collapse(2)
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
