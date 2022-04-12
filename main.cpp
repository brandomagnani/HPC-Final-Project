#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

#include "MMult.h"
#include "gradient_descent.h"
#include "debug.h"

using namespace std;

int main(int argc, char** argv)
{
    // Set random seed:
    srand48(0);

    // Initialize parameters
    long n = 1000;      // Number of rows in A
    long d = 10;        // Number of cols in A
    double eta = 0.1;  // Learning rate
    long n_iter = 1000;  // Number of iterations of gradient descent

    // Initialize matrices
    double* A = (double*) malloc(n * d * sizeof(double));
    double* b = (double*) malloc(n * sizeof(double));
    double* x = (double*) malloc(d * sizeof(double));

    for (long i=0; i<n*d; i++)
        A[i] = drand48();
    for (long i=0; i<n; i++)
        b[i] = drand48();
    for (long i=0; i<d; i++)
        x[i] = drand48();

    double* At = (double*) malloc(d * n * sizeof(double));
    transpose(n, d, A, At);     // Set At

    double* r = (double*) malloc(n * sizeof(double));
    residual(n, d, A, x, b, r); // Set r
    
    double* grad = (double*) calloc(d, sizeof(double));
    
    // Perform Gradient descent
    gradientDescent(n, d, A, At, x, b, r, grad, eta, n_iter);

    // Free memory
    free(A);
    free(At);
    free(b);
    free(x);
    free(r);
    free(grad);
    
    printf("Process completed\n");
    return 0;
}