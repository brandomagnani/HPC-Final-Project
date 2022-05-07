#ifndef Tools_H
#define Tools_H

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>


using namespace std;


// parallel matrix vector multiplication: A*x
void Matvec0(  long n,       // rows of A
               long d,       // cols of A
               double *A,    // matrix to be multiplied, size (n x d)
               double *x,    // vector to be multiplied, size (d x 1)
               double* Ax) { // storage for result

   #pragma omp for schedule(static) // parallelize over rows
   for (long i = 0; i < n; i++){
      double a = 0.0;
      for (long j = 0; j < d; j++) {
         double A_ij = A[i*d+j];
         double x_j = x[j];
         a += A_ij*x_j; //dot product //2flops -> 2dn flops
      }
      Ax[i] = a;
   }
}

// computes transpose of matrix A
void transpose(long n,           // rows of A
               long d,           // cols of A
               double* A,        // matrix to be transposed, size (n x d)
               double* At) {     // sorage for transposed matrix
   
    // Takes the transpose of an ncd matrix A. Stores the result in At.
    for (long i=0; i<n; i++) {
        for (long j=0; j<d; j++) {
            // At[j, i] = A[i, j]
            At[i+n*j] = A[i*d+j];
        }
    }
}

// computes residual r = Ax - b
void residual(long n,         // rows of A
              long d,         // cols of A
              double* A,      // data matrix of size (n x d)
              double* x,      // coefficient vector, size (d x 1)
              double* b,      // label vector, size (n x 1)
              double* r,      // residual vector, size (n x 1)
              double* Ax) {   // storage for matrix-vec multiplocation, size (n x 1)

    Matvec0(n, d, A, x, Ax);
    for (long i=0; i<n; i++) {
        r[i] = Ax[i] - b[i];     //1 flop -> n flops
    }
}

// computes 2-norm of given vector r, of size (n x 1)
double norm(double* r, long n)
{
    // Computes the 2-norm of r
    float mag = 0.0;
    for (long i=0; i<n; i++) {
        mag = mag + r[i]*r[i];
    }
    return sqrt(mag);
}


#endif
