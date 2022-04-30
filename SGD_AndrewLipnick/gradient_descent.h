#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H


#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

#include "MMult.h"
// #include "debug.h"

using namespace std;

void transpose(long m, long n, double* A, double* At)
{
    // Takes the transpose of an mxn matrix A. Stores the result in At.
    for (long i=0; i<m; i++) {
        for (long j=0; j<n; j++) {
            // At[j, i] = A[i, j]
            At[j + n*i] = A[i + m*j];
        }
    }
}

void residual(long n, long d, double* A, double* x, double* b, double* r)
{
    // Computes r = Ax - b.
    double* Ax = (double*) malloc(n * sizeof(double));
    Matvec0(d, n, A, x, Ax);
    // MMult0(n, 1, d, A, x, Ax);

    for (long i=0; i<n; i++) {
        r[i] = Ax[i] - b[i]; //1 flop -> n flops
    }

    free(Ax);
}

double norm(double* r, long n)
{
    // Computes the 2-norm of r
    float mag = 0.0;
    for (long i=0; i<n; i++) {
        mag = mag + r[i]*r[i];
    }
    return sqrt(mag);
}

void gradientIteration(long n, long d, double* A, double* At, 
                       double* x, double* b, double* r, double* grad,
                       double eta)
{
    // Performs a single step of gradient descent
    // Gradient is given by A^T(Ax - b)

    // Update residual r = Ax - b
    residual(n, d, A, x, b, r);

    // Reset gradient
    // for (long i=0; i<d; i++)
    //     grad[i] = 0.0;
    // Update grad = A^T(Ax-b) (multiply by 1/n later)
     Matvec0(d, n, At, r, grad); 
    // MMult0(d, 1, n, At, r, grad); 

    // Perform iteration
    for (long i=0; i<d; i++) {
        x[i] = x[i] - eta * grad[i];
    }
}

void gradientDescent(long n, long d, double* A, double* At, 
                     double* x, double* b, double* r,
                     double eta, long n_iter)
{
    //printf("Iteration | Residual\n");
    double* grad   = (double*) malloc(n * sizeof(double));        // (d x 1) vector for grad(F_i(x))
    double tt = omp_get_wtime();
    for (long i=0; i<n_iter; i++) 
    {
        gradientIteration(n, d, A, At, x, b, r, grad, eta);
        printf("%f, %f\n", norm(r, n), omp_get_wtime()-tt);
    }
    free(grad);
}

#endif
