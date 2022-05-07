#ifndef GRADIENT_DESCENTS_s_H
#define GRADIENT_DESCENTS_s_H


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
// void gradFi(long n, long d, long i, double *A, double *x, double *b, double *gradi) {
   
//    // double* ai = (double*) calloc(d, sizeof(double));  // (d x 1) vector, i-th row of A
//    double sum  = 0.;
   

//    for (long j=0; j<d; j++) {    // compute  a_i * x
//       sum += A[i*d+j] * x[j];  //2 flops -> 2d flops
//    }
   
//    sum = sum - b[i];   // compute  a_i * x - b_i // one flop
//       for (long j=0; j<d; j++) {   // compute gradient F_i: (a_i * x - b_i) a_i
//       gradi[j] = sum * A[i*d+j];  //1 flop -> d flops
//    }
// }


// performs Stochastic Gradient Descent
void SGD_s(long n,              // number of columns of A
         long d,              // number of rows    of A
         long T,              // number of iterations for SGD
         double eta,          // learning rate
         double *A,           // data matrix of size (n x d)
         double *x,           // features vector of size (d x 1)
         double *b,           // vector of size (n x 1)
         double *r,           // for residual (Ab - x), vector of size (n x 1)
         // vector<long> &I,      // vector of size n, contains indices for reshuffling
         // mt19937 RG,          // Marsenne Twister random number generator
         int num_of_threads, int s, double sf) {
   
   double* gradi   = (double*) malloc(d * sizeof(double));        // (d x 1) vector for grad(F_i(x))
   double* x_new   = (double*) malloc(num_of_threads * d * sizeof(double));        // (d x n) vector for x
   double* grad   = (double*) malloc(d * sizeof(double));        // (d x 1) vector for grad(F_i(x))
   // vector<long> I(n);
   uniform_int_distribution<long> SU(0,n-1);


   //printf("Iteration | Residual\n");
   residual(n, d, A, x, b, r);
   double res = norm(r, n);
   double tol = sf * res;
   double tt2;
   double update_time;
   double tt = omp_get_wtime();
   printf("%f,%f\n", res, omp_get_wtime()-tt);

   for (long t=0; t<T; t++){   // do T iterations of SGD step
      double tt2 = omp_get_wtime();
      #pragma omp parallel num_threads(num_of_threads) firstprivate(x,gradi) shared(n,d,A,x_new,b)
         {
         // get thread number
         int ThreadID = omp_get_thread_num();
         
         unsigned seed = (17*ThreadID);     // seed for the random number generator -- 17 is the most random number?
         mt19937 RG(seed);        // Mersenne twister random number generator

         // shuffle(I.begin(), I.end(), RG);   // first, reshuffle index vector

         // if (ThreadID == 0) {
         //    tt2 = omp_get_wtime();
         // }

         for (long k=0; k<s; k++){   // swipe through each data point
               long i = SU(RG);    // get randomly permuted intex
               // if (k==0){
               //    printf("Thread %d's first index is %ld\n",ThreadID, i);
               // }
               gradFi(n, d, i, A, x, b, gradi);  // compute gradient of F_i // 3d+1 flops
       
            for (long j=0; j<d; j++) {   // update x = x - eta * gradi 
               x[j] = x[j] - eta*gradi[j];   //2 flops -> 2d flops
            }  // end of x update

         }
         // if (ThreadID == 0) {
         //    update_time = omp_get_wtime()-tt2;
         // }
         // log vector into grid of vectors
         for (long i=0; i<d; i++) {
            x_new[i+ThreadID*d] = x[i];
         }
         // each thread does s*(5d+1) flops
      }  // end of sweep through s data points
      
      double update_time = omp_get_wtime()-tt2;
      
      //update x
      // printf("update time = %f\n", omp_get_wtime()-tt);
      #pragma omp parallel for schedule(static)
      for (long i = 0; i < d; i++) {
         x[i] = 0;
         for (long j = 0; j < num_of_threads; j++) {
            x[i] += x_new[i+j*d]; //2 flops -> 2d*num_of_threads flops
         }
         x[i] = x[i]/num_of_threads; //d flops
      }
      // printf("average time = %f\n", omp_get_wtime()-tt);

      residual(n, d, A, x, b, r);  // Compute residual r = Ax - b //2*d*n+n flops
      double res = norm(r, n);
      printf("%f, %f , %f, %f\n", res, omp_get_wtime()-tt, update_time, (double((5*d+1)*s*num_of_threads))/(update_time*1e9));
      if (res < tol){
         break;
      }
   }  // end of SG

   free(x_new);
   free(gradi);
   free(grad);

}


#endif