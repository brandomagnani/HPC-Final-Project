#ifndef GRADIENT_DESCENTS2_H
#define GRADIENT_DESCENTS2_H


#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <random>         // for the random number generators
#include <stdlib.h>
#include "MMult.h"
#include "gradient_descent.h"
#include <algorithm>    // std::shuffle

using namespace std;

// performs Stochastic Gradient Descent
void SGD2(long n,              // number of columns of A
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
   double* x_temp   = (double*) malloc(d * sizeof(double));        // (d x 1) vector for grad(F_i(x))
   
   vector<long> I(n);

   for (long i=0; i<n; i++) {
             I[i] = i;
         }

   //printf("Iteration | Residual\n");



   residual(n, d, A, x, b, r);
   double tt = omp_get_wtime();
   printf("%f,%f\n", norm(r, n), omp_get_wtime()-tt);

   #pragma omp parallel num_threads(num_of_threads) firstprivate(I,x_temp,gradi) shared(n,d,A,x_new,b,x)
   {
      // get thread number
      int ThreadID = omp_get_thread_num();
      for (long t=0; t<T; t++){   // do T iterations of SGD step
         printf("thread %d started iteration %ld\n", ThreadID,t);
         //set x_temp to current x
         for(long i = 0; i <d; i++){
            x_temp[i] = x[i];
         }

         shuffle(I.begin(), I.end(), RG);   // first, reshuffle index vector
         
         for (long k=0; k<n; k++){   // swipe through each data point
            
            long i = I[k];    // get randomly permuted intex
            gradFi(n, d, i, A, x_temp, b, gradi);  // compute gradient of F_i // 3d+1 flops
            
            for (long j=0; j<d; j++) {   // update x = x - eta * gradi 
               x_temp[j] = x_temp[j] - eta*gradi[j];   //2 flops -> 2d flops
            }  // end of x update
         }

         // log vector into grid of vectors
         for (long i=0; i<d; i++) {
            x_new[i+ThreadID*d] = x_temp[i];
         }
         // each thread does n*(5d+1) flops

      //update x
      // printf("update time = %f\n", omp_get_wtime()-tt);
      //#pragma omp parallel for schedule(static)
      printf("thread %d got to barrier %ld \n", ThreadID, t);
      #pragma omp barrier
      printf("thread %d got past barrier %ld \n", ThreadID, t);

      #pragma omp parallel for schedule(static)
      for (long i = 0; i < d; i++) {
            x[i] = 0;
            for (long j = 0; j < num_of_threads; j++) {
               x[i] += x_new[i+j*d]; //2 flops -> 2d*num_of_threads flops
            }
         x[i] = x[i]/num_of_threads; //d flops
      }
         
      // printf("average time = %f\n", omp_get_wtime()-tt);
      if (ThreadID == 0){
         residual(n, d, A, x, b, r);  // Compute residual r = Ax - b //2*d*n+n flops
         printf("%f,%f,%f\n", norm(r, n), omp_get_wtime()-tt, ((5*d+1)*n*num_of_threads +2*d*num_of_threads+d +2*d*n+n)/(omp_get_wtime()-tt));    
      }
      printf("thread %d got to barrier %ld \n", ThreadID, t);
      #pragma omp barrier
      printf("thread %d got past barrier %ld \n", ThreadID, t);
    }  // end of SG
 }

   free(x_new);
   free(gradi);

}


#endif