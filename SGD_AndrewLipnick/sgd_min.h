#ifndef GRADIENT_DESCENTSM_H
#define GRADIENT_DESCENTSM_H


#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <random>         // for the random number generators
#include <stdlib.h>
#include "MMult.h"
#include "gradient_descent.h"
#include <algorithm>    // std::shuffle

using namespace std;
void indexofSmallestElement(double*array, int size, int index)
{
   index = 0;
    for(int i = 1; i < size; i++)
    {
        if(array[i] < array[index])
            index = i;              
    }

}

// performs Stochastic Gradient Descent
void SGD_min(long n,              // number of columns of A
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
   double* r_list = (double*) malloc(num_of_threads * sizeof(double));
   int min_index = 0;

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
      
       #pragma omp parallel num_threads(num_of_threads) firstprivate(I,x_temp,gradi,r) shared(n,d,A,x_new,b,r_list)
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

         residual(n, d, A, x_temp, b, r);
         r_list[ThreadID] =  r[0];

      }  // end of sweep through n data points

      //update x
      indexofSmallestElement(r, num_of_threads, min_index);
      #pragma omp parallel for schedule(static)
      for (long i = 0; i < d; i++) {
         x[i] = x_new[i+d*min_index];
      }

      residual(n, d, A, x, b, r);  // Compute residual r = Ax - b
      printf("%f,%f\n", norm(r, n), omp_get_wtime()-tt);  
   }  // end of SG

   free(x_new);
   free(gradi);
   free(x_temp);

}


#endif

