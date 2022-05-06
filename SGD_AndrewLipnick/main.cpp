#include <stdio.h>
#include <math.h>
//#include <blaze/Math.h>
#include <omp.h>
#include <stdlib.h>
#include <random>                 // contains the random number generator
#include "MMult.h"
#include "sgd.h"
#include "sgd_min.h"
#include "sgd_s.h"
#include "sgd_r.h"
#include "sgd2.h"
#include "gradient_descent.h"
using namespace std;

int main(int argc, char *argv[])
{
    int num_of_threads = stoi(argv[1], nullptr, 0);    //number of threads

   // Initialize parameters - These are the only ones to train
   long n     = 1000000;        // Number of rows in A
   long d     = 1000;         // Number of cols in A
   int s      = n/1000.0;              //mini batch size
   double sf  = 0.1;         //stopping factor

   long T     = 100;       // Number of iterations of stochastic gradient descent
   double eta = 1/double (n*d);       // Learning rate

   
    unsigned seed = 17U;     // seed for the random number generator -- 17 is the most random number?
    mt19937 RG(seed);        // Mersenne twister random number generator

    normal_distribution<double> SN(0.,5.);
    uniform_real_distribution<double> SU(0.,1.);
   
    // Set random seed:
    srand48(0);

   // Initialize matrices
   double* A       = (double*) malloc(n * d * sizeof(double));    // (n x d) data matrix
   double* b       = (double*) malloc(n * sizeof(double));        // (n x 1) vector
   double* x       = (double*) malloc(d * sizeof(double));        // (d x 1) vector
   double* x_store = (double*) malloc(d * sizeof(double));        // (d x 1) vector
   double* At      = (double*) malloc(d * n * sizeof(double));    // (d x n) data matrix (transpose of A)
   double* r       = (double*) malloc(n * sizeof(double));        // (n x 1) vector (r = Ax-b)
   // vector<long> I(n);       // contains numbers from 0 to n-1, used for reshuffling
   

   // set up data points
   for (long i=0; i<n*d; i++) {
        A[i] = SN(RG);
    }

    for (long i=0; i<d; i++) {
       x[i] = SN(RG);
       // I[i] = i;
   }   

   Matvec0(d, n, A, x, b);

   for (long i=0; i<n; i++) {
       b[i] += SN(RG);
   }

   // for (long i =0; i<n; i++ ){
   //  for (long j = 0; j<d; j++){
   //      if (i==j){
   //          A[i+d*j]=1;
   //      }
   //      else{
   //          A[i+d*j] = 0;
   //      }
   //  }
   // }

   for (long i=0; i<d; i++) {
       // x[i] += SU(RG);
        x[i] = 0;
       x_store[i] = x[i];
   }

    transpose(n, d, A, At); // create transposed matrix for gradient descent
    gradientDescent(n, d, A, At, x, b, r, eta, T, sf); //run gradient descent

    for (long i=0; i<d; i++) //reset x
       x[i] = x_store[i];


    // SGD_s(n, d, T, eta*double(n/s), A, x, b, r, RG, num_of_threads, s, sf); //run stochastic gradient descent


    // Free memory
   free(A);
   free(b);
   free(x);
   free(At);
   free(r);
   free(x_store); 
    
   return 0;
}
