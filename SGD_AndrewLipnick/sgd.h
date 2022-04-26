#ifndef GRADIENT_DESCENTS_H
#define GRADIENT_DESCENTS_H


#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <random>         // for the random number generators
#include <stdlib.h>
#include "MMult.h"
#include "gradient_descent.h"
#include <algorithm>    // std::shuffle

using namespace std;


// Computes r = Ax - b.
// void residual(long n, long d, double* A, double* x, double* b, double* r)
// {
//    double* Ax = (double*) calloc(n, sizeof(double));
//    MMult0(n, 1, d, A, x, Ax);
   
//    for (long i=0; i<n; i++) {
//       r[i] = Ax[i] - b[i];
//    }
   
//    free(Ax);
// }


// // Computes the 2-norm of r
// double norm(double* r, long n) {
//    float mag = 0.0;
//    for (long i=0; i<n; i++) {
//       mag = mag + r[i]*r[i];
//    }
//    return sqrt(mag);
// }


// Computes grad(F_i(x)) = (a_i * x - b_i) a_i, where
// a_i is i-th row of A and b_i is i-th element of b
void gradFi(long n, long d, long i, double *A, double *x, double *b, double *gradi) {
   
   //double* ai = (double*) calloc(d, sizeof(double));  // (d x 1) vector, i-th row of A
   double sum  = 0.;
   
   for (long j=0; j<d; j++) {    // compute  a_i * x
      sum += A[i*d+j] * x[j];
   }
   
   sum = sum - b[i];   // compute  a_i * x - b_i
      for (long j=0; j<d; j++) {   // compute gradient F_i: (a_i * x - b_i) a_i
      gradi[j] = sum * A[i*d+j];
   }
}


// performs Stochastic Gradient Descent
void SGD(long n,              // number of columns of A
         long d,              // number of rows    of A
         long T,              // number of iterations for SGD
         double eta,          // learning rate
         double *A,           // data matrix of size (n x d)
         double *x,           // features vector of size (d x 1)
         double *b,           // vector of size (n x 1)
         double *r,           // for residual (Ab - x), vector of size (n x 1)
         // vector<long> &I,      // vector of size n, contains indices for reshuffling
         mt19937 RG,          // Marsenne Twister random number generator
         int num_of_threads) {
   
   double* gradi   = (double*) malloc(d * sizeof(double));        // (d x 1) vector for grad(F_i(x))
   double* x_new   = (double*) malloc(num_of_threads * d * sizeof(double));        // (d x n) vector for x
   double* x_temp = (double*) malloc(d * sizeof(double));
   vector<long> I(n);
   for (long i=0; i<n; i++) {
             I[i] = i;
         }
   for (long i=0; i<d; i++) {
      x_temp[i] = x[i];
   } 
   //printf("Iteration | Residual\n");
   residual(n, d, A, x, b, r);
   double tt = omp_get_wtime();
   printf("%f,%f\n", norm(r, n), omp_get_wtime()-tt);

   for (long t=0; t<T; t++){   // do T iterations of SGD step
      
       #pragma omp parallel num_threads(num_of_threads) firstprivate(I,x_temp,gradi) shared(n,d,A,x_new,b)
         {

         // get thread number
         int ThreadID = omp_get_thread_num();
         //set x_temp to current x

         shuffle(I.begin(), I.end(), RG);   // first, reshuffle index vector
         
         for (long k=0; k<n; k++){   // swipe through each data point
            
            long i = I[k];    // get randomly permuted intex
            gradFi(n, d, i, A, x_temp, b, gradi);  // compute gradient of F_i
            
            for (long j=0; j<d; j++) {   // update x = x - eta * gradi
               x_temp[j] = x_temp[j] - eta*gradi[j];   
            }  // end of x update
         }

         // log vector into grid of vectors
         for (long i=0; i<d; i++) {
            x_new[i+ThreadID*d] = x_temp[i];
         }

      }  // end of sweep through n data points

      //update x
      for (long i = 0; i < d; i++) {
         x[i] = 0;
         for (long j = 0; j < num_of_threads; j++) {
            x[i] += x_new[i+j*d];
         }
         x[i] = x[i]/num_of_threads;
         x_temp[i] = x[i];
      }

      residual(n, d, A, x, b, r);  // Compute residual r = Ax - b
      printf("%f,%f\n", norm(r, n), omp_get_wtime()-tt);  
   }  // end of SG

   free(x_new);
   free(gradi);
   free(x_temp);

}


#endif

