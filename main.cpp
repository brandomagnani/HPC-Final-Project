#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

#include "MMult.h"
#include "gradient_descent.h"

using namespace std;

int main(int argc, char** argv)
{
    // Initialize parameters
    long n = 1000;     // Number of rows in A
    long d = 10;       // Number of cols in A
    double eta = 0.1;  // Learning rate
    long n_iter = 100; // Number of iterations of gradient descent

    // Initialize matrices
    double* A = (double*) malloc(n * d * sizeof(double));
    double* b = (double*) malloc(n * sizeof(double));
    double* x = (double*) malloc(d * sizeof(double));
    for (long i=0; i<n*d; i++)
        A[i] = drand48();
    for (long i=0; i<n; i++)
        b[i] = 1.0;
    for (long i=0; i<d; i++)
        x[i] = drand48();

    // Test computation:
    double* r = (double*) malloc(n * sizeof(double));
    residual(n, d, A, x, b, r);
    double mag = norm(r, n);
    printf("%f \n", mag);
    free(r);
    
    // Perform Gradient descent
    gradientDescent(n, d, A, x, b, eta, n_iter);

    // Free memory
    free(A);
    free(b);
    free(x);
    
    printf("Process completed\n");
    return 0;
}