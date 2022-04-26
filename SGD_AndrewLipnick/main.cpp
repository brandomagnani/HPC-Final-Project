#include <stdio.h>
#include <math.h>
//#include <blaze/Math.h>
#include <omp.h>
#include <stdlib.h>
#include <random>                 // contains the random number generator
#include "MMult.h"
#include "sgd.h"
#include "gradient_descent.h"
using namespace std;

int main(int argc, char** argv)
{
   
    unsigned seed = 17U;     // seed for the random number generator -- 17 is the most random number?
    mt19937 RG(seed);        // Mersenne twister random number generator
   
    // Set random seed:
    srand48(0);

   // Initialize parameters
   long n     = 400;        // Number of rows in A
   long d     = 300;         // Number of cols in A
   double eta = 0.00001;       // Learning rate
   long T     = 10;       // Number of iterations of stochastic gradient descent
   
   int num_of_threads = 2;    //number of threads

   // Initialize matrices
   double* A       = (double*) malloc(n * d * sizeof(double));    // (n x d) data matrix
   double* b       = (double*) malloc(n * sizeof(double));        // (n x 1) vector
   double* x       = (double*) malloc(d * sizeof(double));        // (d x 1) vector
   double* x_store = (double*) malloc(d * sizeof(double));        // (d x 1) vector
   double* At      = (double*) malloc(d * n * sizeof(double));    // (d x n) data matrix (transpose of A)
   double* r       = (double*) malloc(n * sizeof(double));        // (n x 1) vector (r = Ax-b)
   // vector<long> I(n);       // contains numbers from 0 to n-1, used for reshuffling
   
   // set up data points
   for (long i=0; i<n*d; i++)
       A[i] = drand48();
   for (long i=0; i<n; i++) {
       b[i] = drand48();
       // I[i] = i;
   }
   for (long i=0; i<d; i++) {
       x[i] = drand48();
       x_store[i] = x[i];
   }
    transpose(d, n, A, At); // create transposed matrix for gradient descent

    gradientDescent(n, d, A, At, x, b, r, eta, T); //run gradient descent

   for (long i=0; i<d; i++) //reset x
       x[i] = x_store[i];

   eta = 0.0005;
   SGD(n, d, T, eta, A, x, b, r, RG, num_of_threads); //run stochastic gradient descent
   
    
    // Free memory
   free(A);
   free(b);
   free(x);
   free(At);
   free(r);   
    
   return 0;
}
