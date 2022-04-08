#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

using namespace std;


int main(int argc, char** argv)
{
    long n = strtol(argv[1], nullptr, 0);    // Number of rows in A
    long d = strtol(argv[1], nullptr, 0);    // Number of cols in A

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

    // Perform Gradient descent

    return 0;
}