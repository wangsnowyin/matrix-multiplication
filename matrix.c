/**
 *l1 cache: 32k, sqrt(32*1024/4/3)=52.25
 *l2 cache: 256k,
 *l3 cache: 20480k, sqrt(20480*1024/4/3)=1322
 */
#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include "mkl.h"

double **allocMatrix(int m, int n){
  int i;
  double **matrix = (double **)malloc(sizeof(double*) * m);
  *matrix = (double*)malloc(sizeof(double)*n*m);

  for(i=0; i<m; i++){
    *(matrix+i) = *matrix + (i*n);
  } 
  return matrix;
}

void freeM(double **matrix){
  free(*matrix);
  free(matrix);
}

void init(double **a, double **b, double **c, int size){
	int i;
	for(i=0; i<size; i++){
		*(*a+i) = i % 30;
		*(*b+i) = i % 60;
		*(*c+i) = 0;
	}
}

/*naive method*/
void mmx1(double **a, double **b, double **c, int n, int bs){
	int i, j, k;
	for(i=0; i<n; i++){
		for(j=0; j<n; j++){
			for(k=0; k<n; k++){
				*(*(c+i)+j) = *(*(c+i)+j) + *(*(a+i)+k) * *(*(b+k)+j);
			}
		}
	}
}

/*tiling method*/
void mmx2(double **a, double **b, double **c, int n, int bs){
	int step = n/bs;
	int i,j,k,p,q,r;
	for(p=0; p<step; p++){
		for(q=0; q<step; q++){
			for(r=0; r<step; r++){
				for(i=p*bs; i<(p+1)*bs; i++){
					for(j=q*bs; j<(q+1)*bs; j++){
						for(k=r*bs; k<(r+1)*bs; k++){
							*(*(c+i)+j) = *(*(c+i)+j) + *(*(a+i)+k) * *(*(b+k)+j);
						}
					}
				}
			}
		}
	}
}

/*recursion method*/
void recurseMMX3(double **a, double **b, double **c, int p, int q, int r, int size, int min){
	if(size==min) {
		int i, j, k;
		for(i=p; i<p+size; i++){
			for(j=q; j<q+size; j++){
				for(k=r; k<r+size; k++){
					*(*(c+i)+j) = *(*(c+i)+j) + *(*(a+i)+k) * *(*(b+k)+j);
				}
			}
		}
		return;
	}
	else{
		recurseMMX3(a, b, c, p, q, r, size/2, min); recurseMMX3(a, b, c, p, q, r+size/2, size/2, min);
		recurseMMX3(a, b, c, p+size/2, q, r, size/2, min); recurseMMX3(a, b, c, p+size/2, q, r+size/2, size/2, min);
		recurseMMX3(a, b, c, p, q+size/2, r, size/2, min); recurseMMX3(a, b, c, p, q+size/2, r+size/2, size/2, min);
		recurseMMX3(a, b, c, p+size/2, q+size/2, r, size/2, min); recurseMMX3(a, b, c, p+size/2, q+size/2, r+size/2, size/2, min);
	}
}

void mmx3(double **a, double **b, double **c, int n, int bs){
	recurseMMX3(a, b, c, 0, 0, 0, n, bs);
}

int main(){
	clock_t t1, t2; //for recording the running time
	double interval;
	int i,j;

	printf("------------------------------------------\n");
	printf("          TIME OF FOUR METHODS           \n");
	printf("------------------------------------------\n");
	
	printf("------------------naive-------------------\n");
	for(i=32; i<=2048; i*=2){
		double **A = allocMatrix(i, i);
		double **B = allocMatrix(i, i);
		double **C = allocMatrix(i, i);
		init(A, B, C, i*i);
		t1 = clock();
		mmx1(A, B, C, i, i);
		t2 = clock();
		interval = ((double)(t2-t1))/CLOCKS_PER_SEC;
		printf("Running time is: %f\n", interval);
		freeM(A);
		freeM(B);
		freeM(C);
	}

	printf("--------------------tiling-------------------\n");
	for(i=32; i<=2048; i*=2){
		double **D = allocMatrix(i, i);
		double **E = allocMatrix(i, i);
		double **F = allocMatrix(i, i);
		init(D, E, F, i*i);
		t1 = clock();
		mmx2(D, E, F, i, 16);
		t2 = clock();
		interval = ((double)(t2-t1))/CLOCKS_PER_SEC;
		printf("Running time is: %f\n", interval);
		freeM(D);
		freeM(E);
		freeM(F);
	}

	printf("-------------------recursion------------------\n");
	for(i=32; i<=2048; i*=2){
		double **D = allocMatrix(i, i);
		double **E = allocMatrix(i, i);
		double **F = allocMatrix(i, i);
		init(D, E, F, i*i);
		t1 = clock();
		mmx3(D, E, F, i, 2);
		t2 = clock();
		interval = ((double)(t2-t1))/CLOCKS_PER_SEC;
		printf("Running time is: %f\n", interval);
		freeM(D);
		freeM(E);
		freeM(F);
	}

	printf("-------------------netlib---------------------\n");
	for(i=32; i<=2048; i*=2){
		double *A = (double *)mkl_malloc( i*i*sizeof( double ), 64 );
    	double *B = (double *)mkl_malloc( i*i*sizeof( double ), 64 );
    	double *C = (double *)mkl_malloc( i*i*sizeof( double ), 64 );
    	for (j = 0; j < (i*i); j++) A[j] = (double)(i+1);
    	for (j = 0; j < (i*i); j++) B[j] = (double)(-i-1);
    	for (j = 0; j < (i*i); j++) C[j] = 0.0;
    	t1 = clock();
    	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                i, i, i, 1.0, A, i, B, i, 0.0, C, i);
    	t2 = clock();
		interval = ((double)(t2-t1))/CLOCKS_PER_SEC;
		printf("Running time is: %f\n", interval);
		mkl_free(A);
    	mkl_free(B);
    	mkl_free(C);
	}

	printf("------------------------------------------\n");
	printf("       TILING: VARIABLE BLOCK SIZE        \n");
	printf("------------------------------------------\n");
	int n = 1024;
	for(i=2; i<=1024; i*=2){
		double **A = allocMatrix(n, n);
		double **B = allocMatrix(n, n);
		double **C = allocMatrix(n, n);
		init(A, B, C, n*n);
		t1 = clock();
		mmx2(A, B, C, n, i);
		t2 = clock();
		interval = ((double)(t2-t1))/CLOCKS_PER_SEC;
		printf("Running time is: %f\n", interval);
		freeM(A);
		freeM(B);
		freeM(C);
	}

	printf("------------------------------------------\n");
	printf("      RECURSION: VARIABLE MIN SIZE        \n");
	printf("------------------------------------------\n");
	for(i=1; i<1024; i*=2){
		double **A = allocMatrix(n, n);
		double **B = allocMatrix(n, n);
		double **C = allocMatrix(n, n);
		init(A, B, C, n*n);
		t1 = clock();
		mmx3(A, B, C, n, i);
		t2 = clock();
		interval = ((double)(t2-t1))/CLOCKS_PER_SEC;
		printf("Running time is: %f\n", interval);
		freeM(A);
		freeM(B);
		freeM(C);
	}
}
