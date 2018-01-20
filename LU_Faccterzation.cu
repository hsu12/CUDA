//v9
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#define MAXSIZE 8000
#define nCYCLIC
//#define BLOCK_SIZE 1024

int num_threads;
double rstar[2]={0.5,0.5};

__global__ void LU(double* a,int size, int index, int b_size)
{
    extern __shared__ float pivot[];
	   int i;
    int tid=threadIdx.x;
    int bid=blockIdx.x;
  	int block_size=b_size;
    //debugprintf("num_rows is %d\n",num_rows);
    //debugprintf("blockDim.x is %d\n",blockDim.x);
    int pivot_start=(index*size+index);
  	int pivot_end=(index*size+size);

  	int start;
  	int end;
  	int pivot_row;
  	int my_row;
  //  if(threadIdx.x==1)
  //  {
    //debug  printf("Begin of LU\n");
    //  for(int index=0;index<n*n;index++){
      //debugprintf("%f ",A[index]);
      //if((index+1)%(n)==0&&index!=0)
      //debugprintf("\n");
      //}
  //  }
    //printf("num_rows is%d\n",num_rows);
    if(tid==0){
     for(i=index;i<size;i++) pivot[i]=a[(index*size)+i];
    }
    __syncthreads();
    if(my_row >pivot_row){
        for(i=start+1;i<end;i++){
                 // a[i]=a[i]-(a[start]*a[(index*size)+i]);
		// a[i]=a[i]-(a[start]*a[(index*size)+(index+(i-start))]);
		 a[i]=a[i]-(a[start]*pivot[(i-my_row)]);

             }
        }

  //if(threadIdx.x==1)
  //{
    //debugprintf("End of LU!!!\n");
    //for(int index=0;index<n*n;index++){
    //debugprintf("%f ",A[index]);
    //if((index+1)%(n)==0&&index!=0)
    //debugprintf("\n");
    //}
  //}

}

__global__ void scale(double *a, int size, int index){
 	int i;
	int start=(index*size+index);
	int end=(index*size+size);

	for(i=start+1;i<end;i++){
		a[i]=(a[i]/a[start]);
	}

}


/* nrm: Compute 2-norm of x
 */

__global__ void solve_L(int n, double *x, double *L)
 {
     //printf("enter solveL\n");
     int i, j,l,ji_1;

     //int num_rows = n / blockDim.x;
      int num_rows = n/blockDim.x;
#ifdef  CYCLIC
if(threadIdx.x==0)
printf("solve_L Cyclic parallel\n");
    for (i=1; i< n; i++)
     {
       //i=l*blockDim.x+threadIdx.x;


       for(l= 0; l< num_rows; l++)
          {
            j=l*blockDim.x+threadIdx.x;
              ji_1=j*n+i-1;
              if((i<=j)&&(j<n))
              x[j] = x[j] - L[ji_1]*x[i-1];
            }
            __syncthreads();
        }

        //}



#else
if(threadIdx.x==0)
printf("solve_L non Cyclic parallel\n");
      //code for uncyclic

      for (i = 1; i < n; i++) {

          for (l= 0; l< num_rows; l++) {
            j=threadIdx.x*num_rows + l;
            ji_1=j*n+i-1;
            if((i<=j)&&(j<n))
              x[j] = x[j] - L[ji_1]*x[i-1];
            }
            __syncthreads();
      }
#endif


/*
    int i, j;
    if(threadIdx.x==1){
    //int num_row = n / num_threads;
    for (i = 1; i < n; i++) {

        for (j = i; j < n; j++) {
            x[j] = x[j] - L[j*n+i-1]*x[i-1];
          }
    }
  }
*/
 }
 /* solve_U: Solve upper triangular system Ux = b
  * Routine is called with x=b
  * (Lower triangle of U)L is ignored
  */
__global__ void solve_U(int n, double *x, double *U)
 {

#ifdef  CYCLIC
if(threadIdx.x==0)
printf("solve_U Cyclic parallel\n");
     int l,i, k,j,ji;
     int num_rows = n/blockDim.x;

     for(i=n-1; i>0;i--){
    if(threadIdx.x==0)
      x[i]=x[i]/ U[i*n+i];
        __syncthreads();

     for (l = 0; l < num_rows; l++)
     {
        j = l*blockDim.x+threadIdx.x;
        ji = j*n + i;
           if(j>=0&&j<i)
             x[j] = x[j] - U[ji]*x[i];

 	     }
       __syncthreads();
   }
   if(threadIdx.x==0)
     x[0] = x[0]/U[0];

#else
if(threadIdx.x==0)
printf("solve_U non Cyclic parallel\n");
      int i, k;
      int num_rows = n/blockDim.x;

      for (i = n-1; i > 0; i--) {
        if(threadIdx.x==0)
          x[i] = x[i]/U[i*n+i];
          __syncthreads();

      for(k=0;k<num_rows;k++){
      int j = threadIdx.x*num_rows + k;
      int UIdx = j*n + i;
      if(j>=0&&j<i)
      x[j] = x[j] - U[UIdx]*x[i];
      }
      __syncthreads();
    }
    if(threadIdx.x==0)
      x[0] = x[0]/U[0];


#endif

   /*
     int i, j;

    if(threadIdx.x==1){
     for (i = n-1; i > 0; i--) {
       x[i] = x[i]/U[i*n+i];

       for (j = 0; j < i; j++) {
         x[j] = x[j] - U[j*n+i]*x[i];
        }
      }
      x[0] = x[0]/U[0];
      }
*/
 }


void initialize_XY(int m,int n, double **XY)
{//checked
    int i, j;
	double h = 1/((double)m+1);

	int idx = 0;
    for (i = 0; i < m; i++)
	{
      for (j = 0; j < m; j++)
		{
		  idx = idx+1;
          XY[idx-1][0] = (double)(i+1)*h;
          XY[idx-1][1] = (double)(j+1)*h;
          //printf("XY%d0 is %f\n",idx-1,XY[idx-1][0]);
          //printf("XY%d1 is %f\n",idx-1,XY[idx-1][1]);
		}
	}
//printf("i reached end of init_XY\n");
}
void initialize_f(int m,int n,double *f,double **XY)
{//
	int i,j;
  double tempx,tempy;
	double x_y[n];
	for (i = 0; i < m; i++) {
			x_y[i] = 0;
	}
	for (i = 0; i < n; i++) {
			//f[i] = 0.1*(((double)rand()%(double)(1))-0.5);
      f[i]=((double)rand()/RAND_MAX*2.0-1.0)*0.5*0.1;
      //(double)rand()/RAND_MAX*2.0-1.0;//float in range -1 to 1
      //printf("fi is%f\n",f[i]);
	}

	for(i=0;i<n;i++)
	{
		tempx=(XY[i][0]-0.5)*(XY[i][0]-0.5);
    tempy=(XY[i][1]-0.5)*(XY[i][1]-0.5);
		 //x_y[i]=(XY[j][0]-0.5)*(XY[i][0]-0.5)+(XY[j][1]-0.5)*(XY[j][1]-0.5);

		f[i] = f[i] + 1.0 -tempx-tempy;
	}
  for (i = 0; i < n; i++) {

      //debugprintf("fi is%f\n",f[i]);
	}
}

void initialize_K(int m, double **K,int n, double **XY)
{//checked
int i,j,idx;
double d[2];
double h ;
h = 1/(m+1);

//debugprintf("initialize_K\n");
	for (i = 0; i < n; i++)
	{
      for  (j = 0; j < n; j++)
	  {

			d[0] = XY[i][0]-XY[j][0];
			d[1] = XY[i][1]-XY[j][1];
			K[i][j] = exp(-(d[0]*d[0])-(d[1]*d[1]));
      //printf("Kij is%f\n",K[i][j] );
      //printf("i is%d,j is %d\n",i,j);
      }
      //printf("\n");
	}

}


void Initialize_ksmall(int n,double *k,double **XY)
{ int j;
  double d[2];
  //k = zeros(n,1);
  for(j = 0; j < n; j++)
    {  d[0] = rstar[0]-XY[j][0];
      d[1] = rstar[1]-XY[j][1];
      k[j] = exp(-(d[0]*d[0])-(d[1]*d[1]));
      //printf("k_small is%f\n",k[j] );
    }

}


void load_value(int n,double **host_K,double *K)
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
  //debugprintf("end of load data\n");
//  for(int index=0;index<n*n;index++){
//debug  printf("%f  ",K[index]);
  //if(index%16==0&&index!=0)
  //debugprintf("\n");
  //}
//}

__global__ void compute_k(int n ,double *device_K)
{
  //double t=0.01;
//  t*eye(n)+K
  double **eye;
/*  for(int i=0;i<n;i++)
  {   for(int j=0;j<n;j++)
      {
        eye[i][j]=0;
        if(i==j){
          eye[i][j]=0.01;
          device_K[i*n+j]+=0.01;
          }
      }
  }*/
  int l,i,z,iz;
  int num_rows = n / blockDim.x;
  for (l= 0; l< num_rows; l++)
  {
    i=l*blockDim.x+threadIdx.x;
//printf("thread id is %d, my row is%d\n",threadIdx.x,i);
      for (z = 0; z < n; z++)
      {
      iz=i*n+z;
        if(i==z)
        device_K[iz]+=0.01;
      }
  }
}



int main(int argc, char *argv[])
{int i1,i2;
  i1= atoi(argv[1]);
  double a1;
  double a2;
  a1=atof(argv[2]);
    a2=atof(argv[3]);
      i2=atoi(argv[4]);
  //  printf(" argv1 is%d, argv2 is%f, arv3 is %f",i1,a1,a2);
  int num_of_threads;
	double **host_XY, **host_K;
	double **XY, *K, *z,*k;
	double *device_XY, *device_K,*device_f;
  double *f;
  double t = 0.01;
  int blocks;
	//int **arr = (int **)malloc(r * sizeof(int *));
  int n,m;
  double result=0;
if(argc==5&&((i1*i1)%i2==0)&&i1>1&&i2>0&&i2<=1024){
    m=i1;
    num_of_threads=i2;
rstar[0]=a1;
rstar[1]=a2;

 printf(" you have input m is%d, rstar is%f %f, threads is %d\n",m, rstar[0],rstar[1],num_of_threads);
}
else{
  num_of_threads =1024;
  m=64;

rstar[0]=0.5;
rstar[1]=0.5;
    printf(" your input is 'ilegal'\n");
  printf("  input is m is%d, rstar is%f %f, threads is %d\n",m, rstar[0],rstar[1],num_of_threads);

}
    /*= atoi(argv[0]);
    if (n > MAXSIZE) {
        printf("n must be less than %d ... aborting\n", MAXSIZE);
        exit(0);
    }
    if (n <= 0) {
        printf("n must be a positive integer ... aborting\n");
        exit(0);
    }*/
	n=m*m;
//debugprintf("n is%d\n",n);
blocks=((n/512));
  //debugprintf("number of threads is%d\n",num_of_threads);
	//host_XY=(double **)malloc(sizeof(double)*n*2);
	//XY=(double **)malloc(sizeof(double)*n*2);
   //cudaMallocHost((void **) &h_a, sizeof(int)*m*n);
   XY= new double* [n];
  for(int setIndex=0; setIndex<n; setIndex++)
  {
    XY[ setIndex ]  = new double[ n ];
    for(int way=0; way<2; way++)
    {
        // initialize stack position (for true LRU)
        XY[ setIndex ][ way ] = 0;
      }
  }

	//K=(double *)malloc(sizeof(double)*n*n);
  K= new double [n*n];
 for(int setIndex=0; setIndex<n*n; setIndex++)
 {
       K[ setIndex ] = 0;

 }
	//host_K=(double **)malloc(sizeof(double)*n*n);
  host_K= new double* [n];
 for(int setIndex=0; setIndex<n; setIndex++)
 {
   host_K[ setIndex ]  = new double[ n ];
   for(int way=0; way<n; way++)
   {
       // initialize stack position (for true LRU)
       host_K[ setIndex ][ way ] = 0;
     }
 }
 //f=(double *)malloc(sizeof(double)*n);
 f= new double [n];
for(int setIndex=0; setIndex<n; setIndex++)
{
      f[ setIndex ] = 0;

}
 //k=(double *)malloc(sizeof(double)*n);
  k= new double [n];
 for(int setIndex=0; setIndex<n; setIndex++)
 {
       k[setIndex ] = 0;

 }
//  z=(double *)malloc(sizeof(double)*n);
  z= new double [n];
 for(int setIndex=0; setIndex<n; setIndex++)
 {
       z[setIndex ] = 0;

 }
 //printf("I had reached end of all for loop\n");
	initialize_XY(m, n, XY);

  initialize_f(m, n,f, XY);

  initialize_K(m, host_K, n, XY);
//printf("I had reached here1\n");
  Initialize_ksmall(n,k,XY);

	load_value(n,host_K,K);
//printf("I had reached here2\n");
	cudaMalloc( (void**)&device_K, n*n* sizeof (double) );
	cudaMemcpy(device_K, K, n*n* sizeof (double) ,cudaMemcpyHostToDevice);
  cudaMalloc( (void**)&device_f, n*sizeof (double) );
  cudaMemcpy(device_f, f, n*sizeof (double) ,cudaMemcpyHostToDevice);
//  for(int setIndex=0; setIndex<n*n; setIndex++)
//  {
  //      printf("K[ setIndex ] is %f\n",K[ setIndex ]);

//  }
	//cudaMalloc ( (void**)&dev_b, N*N* sizeof (double) );
//cudaMalloc( (void**)&device_K, n*n* sizeof (double) );
    //dim3 dimBlock(num_of_threads, 1);
    //dim3 dimGrid(m / dimBlock.x, m / dimBlock.y);
    //t*eye(n)+K
    compute_k<<<1, num_of_threads>>>(n,device_K);

    cudaEvent_t start, stop;
float time1,time2;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord( start, 0 );

for(int i=0;i<n;i++){
       scale<<<1,1>>>(device_K,n,i);
// blocks= ((N-i-1)/512)+1;
blocks=((n/512));
//	printf("Number of blocks rxd : %d \n",blocks);
LU<<<blocks,512,n*sizeof(double)>>>(device_K,n,i,512);

     }


//	  LU<<<1, num_of_threads>>>(n,device_K);
cudaEventRecord( stop, 0 );
cudaEventSynchronize( stop );
cudaEventElapsedTime( &time1, start, stop );
cudaEventDestroy( start );
cudaEventDestroy( stop );

    /*for(int setIndex=0; setIndex<n*n; setIndex++)
    {
    //      printf("device_K[ setIndex ] is %f\n",device_K[ setIndex ]);

  }*/

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );
    solve_L<<<1, num_of_threads>>>(n,device_f,device_K);
    solve_U<<<1, num_of_threads>>>(n,device_f,device_K);

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time2, start, stop );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

      cudaMemcpy(z, device_f, n*sizeof (double),cudaMemcpyDeviceToHost);
      cudaFree(device_K);
      cudaFree(device_f);
	//cudaMemcpy(d_K, K, size, cudaMemcpyDeviceToHost);

  for(int l=0;l<n;l++)
      {
      result+=z[l]*k[l];
      //debugprintf("z[%d] is %f, k[l] is %f\n",l, z[l],k[l]);
      }
      printf("f(x.y) is %f\n",result );
      printf("LU time is %f\n",time1 );
      printf("solve_L_U time is %f\n",time2 );

      return 1;
}
