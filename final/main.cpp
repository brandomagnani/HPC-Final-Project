#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <random>                 // contains the random number generator
#include "sgd_s.h"
#include "gradient_descent.h"
#include "tools.h"
using namespace std;

int main(int argc, char *argv[])
{
    int num_of_threads = stoi(argv[1], nullptr, 0);    //number of threads

   // Initialize parameters - These are the only ones to train
   long n     = 1000000;            // Number of rows in A
   long d     = 1000;                   // Number of cols in A
   int s      = d;                      //mini batch size
   double sf  = 0.5;                    //stopping factor
   long T     = 100;                    // Number of iterations of stochastic gradient descent
   double eta = 1.0/double (n*d);       // Learning rate

   
    unsigned seed = 17U;     // seed for the random number generator -- 17 is the most random number?
    mt19937 RG(seed);        // Mersenne twister random number generator

    normal_distribution<double> SN(0.,5.);          // random normal
    uniform_real_distribution<double> SU(0.,1.);    // random uniform
   

   // Initialize matrices and vectors
   double* A       = (double*) malloc(n * d * sizeof(double));    // (n x d) data matrix
   double* b       = (double*) malloc(n * sizeof(double));        // (n x 1) vector
   double* x       = (double*) malloc(d * sizeof(double));        // (d x 1) vector
   double* x_store = (double*) malloc(d * sizeof(double));        // (d x 1) vector
   double* At      = (double*) malloc(d * n * sizeof(double));    // (d x n) data matrix (transpose of A)
   double* r       = (double*) malloc(n * sizeof(double));        // (n x 1) vector (r = Ax-b)

   // set up data points
   // set up A
   for (long i=0; i<n*d; i++) {
        A[i] = SN(RG);
    }

    // create x* solution
    for (long i=0; i<d; i++) {
       x[i] = SN(RG)+10*SU(RG);
   }   

   // create b
   Matvec0(d, n, A, x, b); //b = Ax
   for (long i=0; i<n; i++) {
       b[i] += SN(RG); // add noise
   }

   // set starting x
   for (long i=0; i<d; i++) {
       // x[i] += SU(RG);   // random varible
       x[i] = 0;            // 0
       x_store[i] = x[i];   // store starting x incase want to compare multiple methods in one run 
   }

   // Run gradient descent
   //  transpose(n, d, A, At);                              // create transposed matrix for gradient descent
   //  gradientDescent(n, d, A, At, x, b, r, eta, T, sf);   //run gradient descent

   // reset x
   //  for (long i=0; i<d; i++) 
   //     x[i] = x_store[i];

   //run stochastic gradient descent
    SGD_s(n, d, T, eta*double(n/s), A, x, b, r, 2, s, sf); //run stochastic gradient descent

    // Free memory
   free(A);
   free(b);
   free(x);
   free(At);
   free(r);
   free(x_store); 
    
   return 0;
}
