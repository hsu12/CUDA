#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

/*
#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)
*/
#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiply(float * A, float * B, float * C,
  		       int numARows, int numAColumns,
			       int numBRows, int numBColumns,
			       int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x, by = blockIdx.y,
       tx = threadIdx.x, ty = threadIdx.y,
       Row = by * TILE_WIDTH + ty,
       Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;

    for (int m = 0; m < (numAColumns-1)/TILE_WIDTH+1; ++m) {
       if (Row < numARows && m*TILE_WIDTH+tx < numAColumns)
          ds_M[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
       else
          ds_M[ty][tx] = 0;
       if (Col < numBColumns && m*TILE_WIDTH+ty < numBRows)
          ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
       else
          ds_N[ty][tx] = 0;

       __syncthreads();
       for (int k = 0; k < TILE_WIDTH; ++k)
          Pvalue += ds_M[ty][k] * ds_N[k][tx];
       __syncthreads();
    }
    if (Row < numCRows && Col < numCColumns)
       C[Row*numCColumns+Col] = Pvalue;
}

void initialize_K(float **K,int n)
{//checked
int i,j,idx;
double h ;
//h = 1/(m+1);

//debugprintf("initialize_K\n");
	for (i = 0; i < n; i++)
	{
      for  (j = 0; j < n; j++)
	  {

			K[i][j] = j;
      //printf("Kij is%f\n",K[i][j] );
      //printf("i is%d,j is %d\n",i,j);
      }
      //printf("\n");
	}

}


void load_value(int n,float **host_K,float *K)
{
	int i,j;
	for (i = 0; i < n; i++)
	{
    for(j = 0; j < n; j++)
		{

				K[i*n+j]=host_K[i][j];

		}
  }
}


int main(int argc, char ** argv) {
  int n=16;
    float ** hostA; // The A matrix
    float ** hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows=n; // number of rows in the matrix A
    int numAColumns=n; // number of columns in the matrix A
    int numBRows=n; // number of rows in the matrix B
    int numBColumns=n; // number of columns in the matrix B
    int numCRows=n; // number of rows in the matrix C (you have to set this)
    int numCColumns=n; // number of columns in the matrix C (you have to set this)
    hostA=(float **)malloc(sizeof(float *) * n);
    for (int i = 0; i < n; i++) {
    hostA[i]=(float *)malloc( sizeof(float) * n);
      }

    hostB=(float **)malloc(sizeof(float *) * n);
    for (int i = 0; i < n; i++) {
        hostB[i]=(float *)malloc( sizeof(float) * n);
        }
   deviceB=(float *)malloc( sizeof(float) * n*n);
   deviceA=(float *)malloc( sizeof(float) * n*n);
   initialize_K(hostA, n);
   initialize_K(hostB, n);
   printf("reached here\n");
    //@@ Set numCRows and numCColumns
    load_value(n,hostB,deviceB);
    load_value(n,hostA,deviceA);
    printf("reached here\n");
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
    hostC = (float *)malloc(sizeof(float) * numCRows * numCColumns);

    //@@ Allocate GPU memory here
    cudaMalloc(&deviceA, sizeof(float) * numARows * numAColumns);
    cudaMalloc(&deviceB, sizeof(float) * numBRows * numBColumns);
    cudaMalloc(&deviceC, sizeof(float) * numCRows * numCColumns);


    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns, cudaMemcpyHostToDevice);


    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid((numCColumns-1)/TILE_WIDTH+1, (numCRows-1)/TILE_WIDTH+1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);


    //@@ Launch the GPU Kernel here
    matrixMultiply<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC,
                                          numARows, numAColumns,
                                          numBRows, numBColumns,
                                          numCRows, numCColumns);

    cudaThreadSynchronize();

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns, cudaMemcpyDeviceToHost);

    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);



    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
