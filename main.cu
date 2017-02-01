#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "la.cuh"	
//#include "la.c"

// argv = [name_of_file, sequential/parallel, dim_matrix, max_iterations, tol_error, print_errors]

int main(int argc, char *argv[]){
	
	int n = 10000, niter = 100, print = 0, status;
	double error = 0.0001;
	
	if (!(argc+1))
		return 0;
		
	if (argc >= 5) {
		n = atoi(argv[2]);
		niter = atoi(argv[3]);
		error = atof(argv[4]);
	}
	
	if (argc >= 6)
		if (atoi(argv[5]) == 1)
			print = 1;
	
	
	Matrix A;
	Vector B, X, X0, X1;
	
	Malloc(&A, n, n);
	Valloc(&X1, n);
	Valloc(&X0, n);
	Valloc(&B, n);
	/*
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			A.A[i*n+j] = 1;
	for (int i = 0; i < n-1; i++)
		A.A[i*n+i] = 2;
	A.A[n*n-1] = -2;*/
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			A.A[i*n+j] = 1;
	for (int i = 0; i < n; i++)
		A.A[i*n+i] = n;
	for (int i = 0; i < n; i++)
		X1.V[i] = 1;
	 /*
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			A.A[i*n+j] = 1/(1000/n*i-1000/n*j-1.1);
	*/
	
	/*
	for (int i = 0; i < 10; i++)
		for (int j = 0; j < 10; j++)
			printf("i = %d j = %d A[i][j] = %lf\n",i,j,A.A[i*n+j]);
			*/
	
	
	MdotV(A, X1, &B);
	
	/*
	X0.V[0]=1.0;X0.V[1]=1.0;
	A.A[0] = 2.0;A.A[1] = 1.0;A.A[2]=5.0;A.A[3]=7.0;
	B.V[0]=11.0;B.V[1]=13.0;
	*/
	
		
	if (!strcmp(argv[1],"seq")) {
		if (print)
			printf("\n\nSEQUENTIAL EXECUTION\n\n");
		status = gauss_jacobi_seq(A,B,X0,&X,niter,error, print);
		if (print) {
			if (status)
				printf("fine\n");
			else
				printf("not fine\n");
		}
	}
	
	else if (!strcmp(argv[1],"par")) {
		if (print)
			printf("PARALLEL EXECUTION\n\n");
		status = gauss_jacobi_cuda(A,B,X0,&X,niter,error, print);
		if (print) {
			if (status)
				printf("fine\n");
			else
				printf("not fine\n");
		}
	}
	
	
	
	/*Mfree(&A);
	Vfree(&B);
	Vfree(&X0);
	Vfree(&X);*/
	return 0;
}
