#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H


#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include "tools.h"

using namespace std;


void gradientIteration(long n,
                       long d,
                       double* A,
                       double* At,
                       double* x,
                       double* b,
                       double* r,
                       double* grad,
                       double eta,
                       double* Ax) {
   
   // Performs a single step of gradient descent
   // Gradient is given by A^T(Ax - b)

   // Update residual r = Ax - b
   residual(n, d, A, x, b, r, Ax);

   // Update grad = A^T(Ax-b) (multiply by 1/n later)
   Matvec0(d, n, At, r, grad);
   // MMult0(d, 1, n, At, r, grad);

   // Perform iteration
   for (long i=0; i<d; i++) {
      x[i] = x[i] - eta * grad[i];
   }
}


void gradientDescent(long n,
                     long d,
                     double* A,
                     double* At,
                     double* x,
                     double* b,
                     double* r,
                     double eta,
                     long n_iter,
                     double sf) {

   double* grad   = (double*) malloc(d * sizeof(double));        // (d x 1) vector for grad(F_i(x))
   double* Ax = (double*) malloc(n * sizeof(double));
   double tt = omp_get_wtime();
   residual(n, d, A, x, b, r, Ax);
   double tol = sf * norm(r, n);
   
   for (long i=0; i<n_iter; i++){
      gradientIteration(n, d, A, At, x, b, r, grad, eta, Ax);
      double res = norm(r, n);
      printf("%f, %f\n", res, omp_get_wtime()-tt);
      if (res < tol) {
         break;
      }
   }
   free(grad);
   free(Ax);
}

#endif
