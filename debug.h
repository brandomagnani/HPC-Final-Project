#ifndef DEBUG_H
#define DEBUG_H

void printMatrix(long m, long n, double* A)
{
    for (long i=0; i<m; i++) {
        for (long j=0; j<n; j++)
            printf("%7.5f ", A[i+m*j]);
        printf("\n");
    }
    printf("\n");
}

#endif