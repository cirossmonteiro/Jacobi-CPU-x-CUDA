#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Falta verificar porque o free tem dado problema!!!

typedef struct Matrix {
	double *A;
	unsigned int m;
	unsigned int n;
} Matrix;

typedef struct Vector {
	double *V;
	unsigned int n;
} Vector;

void Malloc (Matrix *M, unsigned int m, unsigned int n) {
	M->A = (double *) calloc(m*n, sizeof(double));
	M->m = m;
	M->n = n;
}

void Valloc(Vector *V, unsigned int n) {
	V->V = (double *) calloc(n, sizeof(double));
	V->n = n;
}

void Mprint (Matrix M) {
	for (int i = 0; i < M.m; i++) {
		for (int j = 0; j < M.n; j++)
			printf("%lf ", M.A[i*M.n+j]);
		printf("\n");
	}
}

void Massign(Matrix *M, unsigned int i, unsigned int j, double value){
	if (i < M->m && j < M->n)
		M->A[i*M->n+j] = value;
	else
		printf("Bad index.\n %d %d", M->m, M->n);
}

double Mget(Matrix M, unsigned int i, unsigned int j) {
	if (i < M.m && j < M.n)
		return M.A[i*M.n+j];
	else
		printf("Bad index.\n");
		return 0;
}

void Mfree (Matrix *M) {
	free(M->A);
	//M = NULL;
}

void Vfree(Vector *V) {
	free(V->V);
	//V = NULL;
}

void Vprint(Vector V) {
	printf("(");
	for (int i = 0; i < V.n-1; i++)
		printf("%lf, ", V.V[i]);
	printf("%lf)\n",V.V[V.n-1]);
}

void Mdotd (Matrix A, double a, Matrix *C) {
	Mfree(C);
	Malloc(C, A.m, A.n);
	for (int i = 0; i < A.m; i++)
		for (int j = 0; j < A.n; j++)
			C->A[i*A.n+j] = A.A[i*A.n+j]*a;
}

void MsumM (Matrix A, Matrix B, Matrix *C) {
	Mfree(C);
	if (A.m != B.m || A.n != B.n)
		return;
	Malloc(C, A.m, A.n);
	for (int i = 0; i < A.m; i++)
		for (int j = 0; j < A.m; j++)
			C->A[i*A.n+j] = A.A[i*A.n+j] + B.A[i*A.n+j];
}

void MsubM (Matrix A, Matrix B, Matrix *C) {
	Mfree(C);
	if (A.m != B.m || A.n != B.n)
		return;
	Malloc(C, A.m, A.n);
	for (int i = 0; i < A.m; i++)
		for (int j = 0; j < A.m; j++)
			C->A[i*A.n+j] = A.A[i*A.n+j] - B.A[i*A.n+j];
}

void MdotM (Matrix A, Matrix B, Matrix *C) {
	Mfree(C);
	if (A.n != B.m)
		return;
	Malloc(C, A.m, B.n);
	for (int i = 0; i < A.m; i++)
		for (int j = 0; j < A.n; j++)
			for (int k = 0; k < A.n; k++)
				C->A[i*A.n+j] += A.A[i*A.n+k] * B.A[k*A.n+j];
}

void MdotV (Matrix A, Vector V, Vector *Z) {
	//Vfree(Z);
	if (A.n != V.n)
		return;
	Valloc(Z, V.n);
	for (int i = 0; i < A.m; i++)
		for (int j = 0; j < A.n; j++)
			Z->V[i] += A.A[i*A.n+j]*V.V[j];
}

void VsumV(Vector V1, Vector V2, Vector *V3) {
	if (V1.n != V2.n)
		return;
	Valloc(V3,V1.n);
	for (int i = 0; i < V1.n; i++)
		V3->V[i] = V1.V[i]+V2.V[i];
}

void VsubV(Vector V1, Vector V2, Vector *V3) {
	if (V1.n != V2.n)
		return;
	Valloc(V3,V1.n);
	for (int i = 0; i < V1.n; i++)
		V3->V[i] = V1.V[i]-V2.V[i];
}

double Vnorm(Vector V, int p = 2) {
	double s = 0;
	for (int i = 0; i < V.n; i++, s += pow(fabs(V.V[i]), p)) ;
	s = pow(s,1/p);
	return s;
}

__global__ void gj_kernel(double *A, double *B, double *xold, double *xnew, int n) {
	int ind = blockIdx.x*blockDim.x+threadIdx.x;
	double s;// xk1;
	if (ind >= n)
		return;
	s = B[ind];
	for (int j = 0; j < n; j++)
		if (ind != j) {
			s -= A[ind*n+j]*xold[j];
			/*xk1 = B[j];
			for (int l = 0; l < n; l++)
				if (j != l)
					xk1 -= A[j*n+l]*xold[l];
			xk1 /= A[j*n+j];
			s -= A[ind*n+j]*xk1;*/
		}
					
	xnew[ind] = s / A[ind*n+ind];
}


// adicionar iteracoes de jacobi realizadas no nucleo
int gauss_jacobi_cuda(Matrix A, Vector B, Vector X0, Vector *X, int nlim, double eps, int print) {
	int num = A.n, size = num*sizeof(double);
	int thpb; // max: 1024
	int blpg; // max: 65536
	double *dXold, *dXnew, *dA, *dB, *hXold, *hXnew;
	double err;
	
	if (A.m != A.n || A.n != B.n)
		return 0;
	
	if (num <= 65536*32) {
		thpb = 32;
		blpg = num/32+1;
		if (num == 65536*32)
			blpg--;
	}
	else {
		thpb = num/65536+1;
		blpg = 65536;
	}
	
	hXold = (double *) calloc(num, size);
	hXnew = (double *) calloc(num, size);
	Valloc(X, num);
	cudaMalloc(&dXold, size);
	cudaMalloc(&dXnew, size);
	cudaMalloc(&dA, num*size);
	cudaMalloc(&dB, size);
	
	cudaMemcpy(dA, A.A, num*size, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B.V, size, cudaMemcpyHostToDevice);
	
	// first iteration
	cudaMemcpy(dXold, X0.V, size, cudaMemcpyHostToDevice);
	gj_kernel<<<blpg,thpb>>>(dA, dB, dXold, dXnew, num);
	cudaMemcpy(hXnew, dXnew, size, cudaMemcpyDeviceToHost);
	
	// compute error
	err = 0;
	for (int i = 0; i < num; i++)
		err += pow(fabs(hXnew[i] - X0.V[i]),2);
	err = pow(err,0.5);
	
	// other iterations
	int i;
	for (i = 0; i < nlim-1 && err > eps; i++) {
		cudaMemcpy(dXold, dXnew, size, cudaMemcpyDeviceToDevice);
		gj_kernel<<<blpg,thpb>>>(dA, dB, dXold, dXnew, num);
		cudaMemcpy(hXold, dXold, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(hXnew, dXnew, size, cudaMemcpyDeviceToHost);
		
		// compute error
		err = 0;
		for (int j = 0; j < num; j++)
			err += pow(fabs(hXnew[j] - hXold[j]),2);
		err = pow(err,0.5);
		if (print)
			printf("err par: %lf\n", err);
	}
	for (i = 0; i < num; i++)
		X->V[i] = hXnew[i];
	
	cudaFree(dXold);
	cudaFree(dXnew);
	cudaFree(dA);
	cudaFree(dB);
	free(hXold);
	free(hXnew);
	return 1;
}

int gauss_jacobi_seq(Matrix A, Vector B, Vector X0, Vector *X, int nlim, double eps, int print) {
	int num = A.n, size = num*sizeof(double);
	double *xold, *xnew;
	double err = eps+1;
	
	if (A.m != A.n || A.n != B.n)
		return 0;
	
	xold = (double *) malloc(size);
	xnew = (double *) malloc(size);
	for (int i = 0; i < num; i++)
		xold[i] = X0.V[i];
	
	
	for (int n = 0; n < nlim && err > eps; n++) {
		for (int i = 0; i < num; i++) {
			xnew[i] = B.V[i];
			for (int j = 0; j < num; j++)
				if (i != j)
					xnew[i] -= A.A[i*num+j]*xold[j];
			xnew[i] /= A.A[i*num+i];
		}
		err = 0;
		for (int i = 0; i < num; i++)
			err += pow(fabs(xnew[i] - xold[i]), 2);
		err = pow(err,0.5);
		for (int i = 0; i < num; i++)
			xold[i] = xnew[i];
		if (print)
			printf("err seq: %lf\n", err);
	}
	
	free(xold);
	X->V = xnew;
	X->n = num;
	return 1;
}
