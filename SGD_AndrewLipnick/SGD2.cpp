#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H

#include <stdio.h>
#include <math.h>
//#include <omp.h>
#include <random>         // for the random number generators
#include <stdlib.h>
#include "MMult.h"
using namespace std;



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

void gradFs(set<unsigned int> indicies, double *A, double *b, double *x ){ // calculate gradient of subset all at once, perhaps faster than doing each direction separately

}

void SGD2_update(long s, long *d, double *A, double *b, double *x)
         // calc gradient

         // if s << d
         std::set<unsigned int> indices;
         while (indices.size() < s)
         indices.insert(RandInt(0,d-1))
            //update, maybe write fuction that doesn't require a loop
         for (long i = 0; i<s, i++){
            long index = indices[i];
            gradFi(n, d, index, A, x, b, gradi);    // get gradient
            gradFi =  gradFi + gradi;            // add up gradients
         }

         //else shuffle
         // consider writing code for when s is closer to d
         // end of calculating gradient

         // update x
         for (long j=0; j<d; j++) {   // update x = x - eta * gradi
            x[j] = x[j] - eta*gradi[j];
         }  // end of x update



// performs Stochastic Gradient Descent
void SGD(long n,              // number of columns of A
         long d,              // number of rows    of A
         long T,              // number of iterations for SGD
         double eta,          // learning rate
         double *A,           // data matrix of size (n x d)
         double *x,           // features vector of size (d x 1)
         double *x_new
         double *b,           // vector of size (n x 1)
         // double *gradi,       // for gradient of F_i, vector of size (d x 1)
         double *r,           // for residual (Ab - x), vector of size (n x 1)

         long T2,             // number of sub iterations
         long s,              // minibatch size
         int numthreads) {   // number of threads          
         
   
   double *x_zeros; // temp var for quicker reseting of x_new
   for (long i = 0; i<n; i++){
      x_zeros[i] = 0.0;
   }

   printf("Iteration | Residual\n");
   for (long t=0; t<T; t++){   // do T iterations of SGD step
      #pragma omp parallel num_threads(numthreads) reduction (+:x_new)
      // give each thread a copy of x
      double* x_temp = (double*) malloc(n * sizeof(double));
      x_temp = x;
      for (long t2 = 0; t2<T2; t2++){ // do T2 iteraterations for each step

         // SGD_update
         GD2_update(s,d,x_temp,A,b);

         //combine results
         if (t2 == T2-1){
            x_new = x_new + x_temp;
         }
      }
      free(x_temp);
      x_new = x_new/s; // average
      x = x_new;       // replace x
      x_new = x_zeros; // reset x_new to zeros

      residual(n, d, A, x, b, r);  // Compute residual r = Ax - b
      printf("%3ld       | %f\n", t, norm(r, n));
   }
}

#endif

