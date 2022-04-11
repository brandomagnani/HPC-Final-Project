#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

#include "MMult.h"

using namespace std;

void residual(long n, long d, double* A, double* x, double* b, double* r)
{
    // Computes r = Ax - b.
    double* Ax = (double*) malloc(n * sizeof(double));
    MMult0(n, 1, d, A, x, Ax);

    for (long i=0; i<n; i++) {
        r[i] = Ax[i] - b[i];
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

// void gradient(long n, long d, double* A, double* x, double* b, 
//     double* r, double* grad)
// {
//     // Computes the gradient A^T(Ax - b) of g(x) = ||Ax - b||^2/2
// }

void gradientIteration(long n, long d, double* A, double* x, double* b,
    double eta)
{
    // Performs a single step of gradient descent
    double* r = (double*) malloc(n * sizeof(double));
    residual(n, d, A, x, b, r);
    
    for (long i=0; i<d; i++) {
        x[i] = x[i] - eta * grad[i];
    }
}

void gradientDescent(long n, long d, double* A, double* At, 
                     double* x, double* b, double*r,
                     double eta, long n_iter)
{
    for (long i=0; i<n_iter; i++) 
    {
        gradientIteration(n, d, A, x, b, eta);
    }
}

#endif