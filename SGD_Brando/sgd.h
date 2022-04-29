#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H

#include <stdio.h>
#include <math.h>
//#include <omp.h>
#include <random>         // for the random number generators
#include <stdlib.h>
#include "MMult.h"
using namespace std;


// Takes the transpose of an (n x d) matrix A. Stores the result in At.
void transpose(long n, long d, double* A, double* At) {
    for (long i=0; i<n; i++) {
        for (long j=0; j<d; j++) {
            // At[j, i] = A[i, j]
            At[j + d*i] = A[i + n*j];
        }
    }
}


// Computes r = Ax - b.
void residual(long n, long d, double* A, double* x, double* b, double* r)
{
   double* Ax = (double*) calloc(n, sizeof(double));
   MMult0(n, 1, d, A, x, Ax);
   
   for (long i=0; i<n; i++) {
      r[i] = Ax[i] - b[i];
   }
   
   free(Ax);
}


// Computes the 2-norm of r
double norm(double* r, long n) {
   float mag = 0.0;
   for (long i=0; i<n; i++) {
      mag = mag + r[i]*r[i];
   }
   return sqrt(mag);
}


// Computes grad(F_i(x)) = (a_i * x - b_i) a_i, where
// a_i is i-th row of A and b_i is i-th element of b
void gradFi(long n, long d, long i, double *A, double *x, double *b, double *gradi) {
   
   double* ai  = (double*) malloc( d * sizeof(double));    // (d x 1) vector, i-th row of A
   //double* ai = (double*) calloc(d, sizeof(double));  // (d x 1) vector, i-th row of A
   double sum  = 0.;
   double bi   = b[i];    // i-th element of b
   

   for (long j=0; j<d; j++) {    // compute  a_i * x
      sum += ai[j] * x[j];
   }
   sum = sum - bi;   // compute  a_i * x - b_i
   
   for (long j=0; j<d; j++) {   // compute gradient F_i: (a_i * x - b_i) a_i
      gradi[j] = sum * ai[j];
   }

   free(ai);
}


// performs Stochastic Gradient Descent
void SGD(long n,              // number of columns of A
         long d,              // number of rows    of A
         long T,              // number of iterations for SGD
         double eta,          // learning rate
         double *A,           // data matrix of size (n x d)
         double *x,           // features vector of size (d x 1)
         double *b,           // vector of size (n x 1)
         double *gradi,       // for gradient of F_i, vector of size (d x 1)
         double *r,           // for residual (Ab - x), vector of size (n x 1)
         vector<long> &I,      // vector of size n, contains indices for reshuffling
         mt19937 RG) {         // Marsenne Twister random number generator
   
   
   printf("Iteration | Residual\n");
   
   for (long t=0; t<T; t++){   // do T iterations of SGD step
      shuffle(I.begin(), I.end(), RG);   // first, reshuffle index vector
      
      for (long k=0; k<n; k++){   // swipe through each data point
         double i = I[k];    // get randomly permuted intex
         gradFi(n, d, i, A, x, b, gradi);  // compute gradient of F_i
         
         for (long j=0; j<d; j++) {   // update x = x - eta * gradi
            x[j] = x[j] - eta*gradi[j];
            
         }  // end of x update
         
      }  // end of sweep through n data points
      
      residual(n, d, A, x, b, r);  // Compute residual r = Ax - b
      printf("%3ld       | %f\n", t, norm(r, n));
      
   }  // end of SGD
}


#endif

