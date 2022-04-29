#include <stdio.h>
#include <math.h>
//#include <blaze/Math.h>
//#include <omp.h>
#include <stdlib.h>
#include <random>                 // contains the random number generator
#include "MMult.h"
#include "sgd.h"
using namespace std;

int main(int argc, char** argv)
{
   
    unsigned seed = 17U;     // seed for the random number generator -- 17 is the most random number?
    mt19937 RG(seed);        // Mersenne twister random number generator
   
    // Set random seed:
    srand48(0);

   // Initialize parameters
   long n     = 4;        // Number of rows in A
   long d     = 3;         // Number of cols in A
   double eta = 0.1;       // Learning rate
   long T     = 100;       // Number of iterations of stochastic gradient descent
   

   // Initialize matrices
   double* A      = (double*) malloc(n * d * sizeof(double));    // (n x d) data matrix
   double* b      = (double*) malloc(n * sizeof(double));        // (n x 1) vector
   double* x      = (double*) malloc(d * sizeof(double));        // (d x 1) vector
   double* At     = (double*) malloc(d * n * sizeof(double));    // (d x n) data matrix (transpose of A)
   double* r      = (double*) malloc(n * sizeof(double));        // (n x 1) vector (r = Ax-b)
   double* gradi  = (double*) malloc(d * sizeof(double));        // (d x 1) vector for grad(F_i(x))
   vector<long> I(n);       // contains numbers from 0 to n-1, used for reshuffling
   

   for (long i=0; i<n*d; i++)
       A[i] = drand48();
   for (long i=0; i<n; i++) {
       b[i] = drand48();
       I[i] = i;
   }
   for (long i=0; i<d; i++)
       x[i] = drand48();

   
   SGD(n, d, T, eta, A, x, b, gradi, r, I, RG);
   
    
    // Free memory
   free(A);
   free(b);
   free(x);
   free(At);
   free(r);
   free(gradi);
   
    
   printf("Process completed\n");
   return 0;
}
