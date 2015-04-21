#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#ifdef MKL
  #include "mkl.h"
  #ifdef NOGPU
    #include "mkl_vsl.h"
  #endif
#else
  #include "cblas.h"
#endif

#ifndef NOGPU
  #include <cuda.h>
  #include <curand.h>
  #include <cuda_runtime.h>
  #include "cublas_v2.h"

  /* CUDA VERSION 6.5 or HIGHER */
  #define CUDA_CALL(x) \
      do \
      {  \
        if((x)==cudaErrorInvalidDevicePointer) \
        { \
          printf("Error CUDA Error Invalid Device Pointer at %s:%d\n",__FILE__,__LINE__); \
          return EXIT_FAILURE; \
        } \
        if((x)==cudaErrorInitializationError ) \
        { \
          printf("Error CUDA Initialization at %s:%d\n",__FILE__,__LINE__); \
          return EXIT_FAILURE; \
        } \
        if((x)==cudaErrorMemoryAllocation) \
        { \
          printf("Error CUDA Memory Allocation at %s:%d\n",__FILE__,__LINE__); \
          return EXIT_FAILURE; \
        } \
        if((x)!=cudaSuccess) \
        { \
          printf("Error at %s:%d\n",__FILE__,__LINE__); \
          return EXIT_FAILURE; \
        } \
      } while(0)

  #define CUBLAS_CALL(x) \
      do \
      {  \
        if((x)==CUBLAS_STATUS_NOT_INITIALIZED) \
        { \
          printf("Missing call to cublasCreate() at %s:%d\n",__FILE__,__LINE__); \
          return EXIT_FAILURE; \
        } \
        if((x)!=CUBLAS_STATUS_SUCCESS) \
        { \
          printf("Error at %s:%d\n",__FILE__,__LINE__); \
          return EXIT_FAILURE; \
        } \
      } while(0)

  #define CURAND_CALL(x) \
      do \
      {  \
        if((x)!=CURAND_STATUS_SUCCESS) \
        { \
          printf("Error at %s:%d\n",__FILE__,__LINE__); \
          return EXIT_FAILURE; \
        } \
      } while(0)
#endif

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

//#define m  500
//#define k 1000
//#define n 1500

void printMatrixFlat(double matrix[],int rows, int columns);
void printMatrixRow(double matrix[],int rows, int columns);
void printMatrixCol(double matrix[],int rows, int columns);
void matrixMultiplyCPU(const int MM, const int NN, const int KK, 
                       double alpha, const double *A, const double *B, 
                       double beta, double *C);
double dseconds(void);
double getBandwidth(const int MM, const int NN, const float microSeconds);
double getGFlops(const double nFPOperations, const float microSeconds);
double matrixMultiplyError(const int MM, const int NN, 
                         const double *C_test, const double *C_ref);
long long int resizeColumnsMatrixB(const long long int columnsMatrixB, 
                                   const long long int maxMatrixSize);

int main(int argc, char *argv[])
{
/* TURN OFF BUFFERING */
  setbuf(stdout, NULL); 

#ifndef NOGPU
/* CUDA */
  cudaError_t cudaStat;
/* CUBLAS */
  cublasHandle_t handle;
/* CURAND */
  curandGenerator_t gen;
/* TIMINGS */
  cudaEvent_t start, stop;
#endif

#ifdef NOGPU
  #ifdef MKL
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 777777);
  #endif
#endif

  long long int m=500;
  long long int k=1000;
  long long int n=1500;
  
  int i, j, l, index, ratio;
  int iterations=10;
//int ratioArray[12]={110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 1};
  int ratioArray[12]={1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110};
  
  long long int memoryRequired;
  long long int elements;
  long long int maxRows;
  long long int maxMatrixSize;
  long long int nRows,nRowsA,nRowsB,nRowsC;
  long long int nCols,nColsA,nColsB,nColsC; 
  long long int sizeAinBs,sizeBinBs,sizeCinBs;
  
  double matrixSizeMB;
  double sizeAinMBs,sizeBinMBs,sizeCinMBs;
  double sizeAinGBs,sizeBinGBs,sizeCinGBs;
  
  double* a;
  double* b;
  double* c_no_blas;
  double* c_cblas;
  double* c_cublas;
  
  double* rand_a;
  double* rand_b;

  double* d_a;
  double* d_b;
  double* d_c;
  
  double alpha=1.0f;
  double beta=0.0f;
  
  float msecTotal=0.0;
  float msecTime=0.0;
  
  double errorCheckCBLAS;
  double errorCheckCUBLAS;
  
  double clockIn;
  double clockOut;
  
  size_t freeGPUMemoryInBytes=0;
  size_t totalGPUMemoryInBytes=0;
  
  double nFPOperations=2*m*k*n;

#ifndef NOGPU
  CUBLAS_CALL(cublasCreate(&handle));

/* CREATE CUDA EVENTS FOR TIMING */
  CUDA_CALL(cudaEventCreate(&start));
  CUDA_CALL(cudaEventCreate(&stop));

/* CREATE GENERATOR FOR CURAND */
  CURAND_CALL(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT));

/* SEED */
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen,1234ULL));

/* GET THE AMOUNT OF FREE MEMORY */
  CUDA_CALL(cudaMemGetInfo(&freeGPUMemoryInBytes,&totalGPUMemoryInBytes));
#endif
  
/* READ THE NUMBER OF MATRIX ELEMENTS IN MATRIX [A] FROM ARGUMENT LIST */
  if (sscanf (argv[1], "%lf", &matrixSizeMB)!=1) 
  { 
    printf("Usage: ./dgemm.x [matrix size of A in MB]\n"); 
    return EXIT_FAILURE;
  }

  maxRows=matrixSizeMB*1024*1024/8;

#ifndef NOGPU
  printf("+----------------------+\n"); 
  printf("| CUDA VERSION : %d  |\n",CUDA_VERSION);
#endif
  printf("+----------------------+\n"); 
  printf("| Size of Matrix [A]  : %.2f MB\n",matrixSizeMB);
  printf("+----------------------+\n"); 
  printf("| %-20s: %lld\n","Maxium value of m",maxRows);
  printf("+----------------------+\n");

  for(ratio=0;ratio<(sizeof(ratioArray)/4);ratio++)
  {
//  printf("ratioArray(%2d)=%d\n",ratio,ratioArray[ratio]);
    nCols=sqrt(maxRows*ratioArray[ratio]);
    nRows=maxRows/nCols;
    nRowsA=nRows;
    nColsA=nCols;
    nRowsB=nColsA;
    nColsB=1;
    nRowsC=nRowsA;
    nColsC=nColsB;

#ifndef NOGPU
    maxMatrixSize=(0.9*freeGPUMemoryInBytes/8.0-nRows*nCols)/(nRows+nCols);
#else
    maxMatrixSize=(0.9*16.0*1024.0*1024.0*1024.0/8.0-nRows*nCols)/(nRows+nCols);
#endif

#ifdef PINNED
    printf("| USE OF PINNED MEMORY: \n");
#else
    printf("| NO PINNED MEMORY    : \n");
#endif
    printf("+----------------------+\n");
    printf("| %-20s: %lld\n","Max Row/Column Size",maxMatrixSize);

    if(nRows <= nCols)
    {
      printf("| %-20s: 1 : %.0f\n","Ratio",(double)(nCols/nRows));
    }
    else
    {
      printf("| %-20s: %.0f : 1\n","Ratio",(double)(nCols/nRows));
    }  

    printf("+----------------------+\n");

    l=1;
    
#ifndef NOGPU
    printf("  %3s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %13s, %13s, %13s, %13s, %12s, %12s\n",
           "#","M","K","N","[A] (MB)","[B] (MB)","[C] (MB)","Total (MB)","GPU (MB)","REQ (MB)",
           "D2H[A] (GB/s)","D2H[B] (GB/s)","H2D[A] (GB/s)","H2D[B] (GB/s)","D2H[C] (GB/s)",
           "CBLAS (GF/s)","CUBLAS (GF/s)","CUBLAS+DT (GF/s)","CPU (GF/s)","CBLAS Error","CUBLAS Error");
#else
    printf("  %3s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %13s\n",
           "#","M","K","N","[A] (MB)","[B] (MB)","[C] (MB)","Total (MB)","REQ (MB)",
           "CBLAS (GF/s)");
//         "CBLAS (GF/s)","CPU (GF/s)","CBLAS Error");
#endif
    do
    {
      m=nRowsA;
      k=nColsA;
      n=nColsB;
     
      sizeAinBs=m*k*sizeof(double);
      sizeBinBs=k*n*sizeof(double);
      sizeCinBs=m*n*sizeof(double);
     
      sizeAinMBs=(double)(sizeAinBs/1048576.0);
      sizeBinMBs=(double)(sizeBinBs/1048576.0);
      sizeCinMBs=(double)(sizeCinBs/1048576.0);
      
      sizeAinGBs=sizeAinMBs/1024.0;
      sizeBinGBs=sizeBinMBs/1024.0;
      sizeCinGBs=sizeCinMBs/1024.0;
      
      nFPOperations=2*m*k*n;

      /* MEMORY INFORMATION */
      memoryRequired=0;
      memoryRequired+=m*k*sizeof(double); //Matrix A on device
      memoryRequired+=k*n*sizeof(double); //Matrix B on device
      memoryRequired+=m*n*sizeof(double); //Matrix C on device

#if DEBUG
      printf("+----------------------+\n");
      printf("+ M=%lld K=%lld N=%lld  \n",m,k,n);
      printf("+----------------------+\n");
      printf("+ MEMORY USAGE \n");
      printf("+----------------------+\n");
  #ifndef NOGPU
      printf("| Free     memory  (B):   %15zu\n",freeGPUMemoryInBytes);
      printf("| Total    memory  (B):   %15zu\n",totalGPUMemoryInBytes);
      printf("| Free     memory (MB):   %15.2f\n",(double)freeGPUMemoryInBytes/1048576);
      printf("| Total    memory (MB):   %15.2f\n",(double)totalGPUMemoryInBytes/1048576);
      printf("+----------------------+\n");
  #endif
      printf("| Required memory (MB):   %15.2f\n",(double)memoryRequired/1048576);
      printf("+----------------------+\n");
#endif

#ifdef PINNED
/* IF  PINNED HOST MEMORY IS USED */
      cudaMallocHost((void **)&a,m*k*sizeof(double));
      cudaMallocHost((void **)&b,k*n*sizeof(double));
      cudaMallocHost((void **)&c_cublas,m*n*sizeof(double));
#else     
/* NON-PINNED HOST MEMORY */
      a=(double*)malloc(m*k*sizeof(double));
      b=(double*)malloc(k*n*sizeof(double));
      c_cublas=(double*)malloc(m*n*sizeof(double));
#endif

/* HOST MEMORY */
      c_no_blas=(double*)malloc(m*n*sizeof(double));
      c_cblas=(double*)malloc(m*n*sizeof(double));

#ifndef NOGPU
/* DEVICE MEMORY */
      cudaMalloc((void **)&rand_a,m*k*sizeof(double));
      cudaMalloc((void **)&rand_b,k*n*sizeof(double));

/* POPULATE MATRIX A AND B ON DEVICE */
      CURAND_CALL(curandGenerateUniformDouble(gen,rand_a,m*k));
      CURAND_CALL(curandGenerateUniformDouble(gen,rand_b,k*n));

/* GET MATRIX A FROM DEVICE */
      CUDA_CALL(cudaEventRecord(start, 0));
      CUDA_CALL(cudaMemcpy(a,rand_a,m*k*sizeof(double),cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaEventRecord(stop, 0));
      CUDA_CALL(cudaEventSynchronize(stop));
      CUDA_CALL(cudaEventElapsedTime(&msecTime, start, stop));
      float msecD2H_A=msecTime;
      double performanceD2H_A=getBandwidth(m,k,msecD2H_A);

/* GET MATRIX B FROM DEVICE */
      CUDA_CALL(cudaEventRecord(start, 0));
      CUDA_CALL(cudaMemcpy(b,rand_b,k*n*sizeof(double),cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaEventRecord(stop, 0));
      CUDA_CALL(cudaEventSynchronize(stop));
      CUDA_CALL(cudaEventElapsedTime(&msecTime, start, stop));
      float msecD2H_B=msecTime;
      double performanceD2H_B=getBandwidth(k,n,msecD2H_B);

/* DELETE DEVICE MEMORY FOR CURAND */
      cudaFree(rand_a);
      cudaFree(rand_b);

/* CREATE DEVICE MEMORY FOR CUBLAS */ 
      cudaMalloc((void **)&d_a,m*k*sizeof(double));
      cudaMalloc((void **)&d_b,k*n*sizeof(double));
      cudaMalloc((void **)&d_c,m*n*sizeof(double));
#else
/* POPULATE MATRIX [A] & [B] */
  #ifdef MKL
// See: https://software.intel.com/en-us/node/521851
      vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF,stream,m*k, a, 0.0, 1.0);
      vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF,stream,n*k, b, 0.0, 1.0);

  #else
      if(m>n)
      {
        for(i=0;i<(n*k);i++)
        {
          a[i]=(double)rand() / (double)RAND_MAX;
          b[i]=(double)rand() / (double)RAND_MAX;
        }
        for(j=(n*k);j<(m*k);j++)
        {
          a[j]=(double)rand() / (double)RAND_MAX;
        }
      }
      if(m<n)
      {
        for(i=0;i<(m*k);i++)
        {
          a[i]=(double)rand() / (double)RAND_MAX;
          b[i]=(double)rand() / (double)RAND_MAX;
        }
        for(j=(m*k);j<(n*k);j++)
        {
          b[j]=(double)rand() / (double)RAND_MAX;
        }        
      }
      if(m==n)
      {
        for(i=0;i<(m*k);i++)
        {
          a[i]=(double)rand() / (double)RAND_MAX;
          b[i]=(double)rand() / (double)RAND_MAX;
        }
      }
  #endif
#endif

/* PERFORM A BLAS ASSUMING MATRIX [A] & [B] IN COLUMN MAJOR ORDER */
      clockIn=dseconds();
      clockIn=dseconds();
      for (i=0; i<iterations; i++)
      { 
      cblas_dgemm(CblasColMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m,n,k,alpha,a,m,b,k,beta,c_cblas,m);
      }
      clockOut=dseconds();
      msecTime=(float)((clockOut-clockIn)/iterations*1000.0);
      float msecCBLAS=msecTime;
      double performanceCBLAS=getGFlops(nFPOperations,msecCBLAS);

#ifndef NOGPU
/* TRANSFER MATRIX A TO DEVICE */
      CUDA_CALL(cudaEventRecord(start, 0));
      CUDA_CALL(cudaMemcpy(d_a,a,m*k*sizeof(double),cudaMemcpyHostToDevice));
      CUDA_CALL(cudaEventRecord(stop, 0));
      CUDA_CALL(cudaEventSynchronize(stop));
      CUDA_CALL(cudaEventElapsedTime(&msecTime, start, stop));
      float msecH2D_A=msecTime;
      double performanceH2D_A=getBandwidth(m,k,msecH2D_A);

/* TRANSFER MATRIX B TO DEVICE */
      CUDA_CALL(cudaEventRecord(start, 0));
      CUDA_CALL(cudaMemcpy(d_b,b,k*n*sizeof(double),cudaMemcpyHostToDevice));
      CUDA_CALL(cudaEventRecord(stop, 0));
      CUDA_CALL(cudaEventSynchronize(stop));
      CUDA_CALL(cudaEventElapsedTime(&msecTime, start, stop));
      float msecH2D_B=msecTime;
      double performanceH2D_B=getBandwidth(k,n,msecH2D_B);

/* CUBLAS WITH COLUMN-MAJOR ORDERING */
      CUDA_CALL(cudaEventRecord(start, 0));
      for (i=0; i<iterations; i++)
      {
        CUBLAS_CALL(cublasDgemm(handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                m,n,k,&alpha,d_a,m,d_b,k,&beta,d_c,m));
      }
      CUDA_CALL(cudaEventRecord(stop, 0));
      CUDA_CALL(cudaEventSynchronize(stop));
      CUDA_CALL(cudaEventElapsedTime(&msecTime, start, stop));
      float msecCUBLAS=msecTime;
      double performanceCUBLAS=getGFlops(nFPOperations,msecCUBLAS/iterations);

/* TRANSFER MATRIX C TO HOST */
      CUDA_CALL(cudaEventRecord(start, 0));
      CUDA_CALL(cudaMemcpy(c_cublas,d_c,m*n*sizeof(double),cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaEventRecord(stop, 0));
      CUDA_CALL(cudaEventSynchronize(stop));
      CUDA_CALL(cudaEventElapsedTime(&msecTime, start, stop));
      float msecD2H_C=msecTime;
      double performanceD2H_C=getBandwidth(m,n,msecD2H_C);
      float msecTotal=(msecCUBLAS/iterations)+msecH2D_A+msecH2D_B+msecD2H_C;
      double performanceCUBLAS_wDT=getGFlops(nFPOperations,msecTotal);
#endif

#ifdef CPU_DGEMM
/* MATRIX-MATRIX MULTIPLY ON CPU WITH [A] & [B] IN COLUMN-MAJOR ORDERING */
      clockIn=dseconds();
      clockIn=dseconds();
      for(i=0; i<iterations; i++)
      {
        matrixMultiplyCPU(m, n, k, alpha, a, b, beta, c_no_blas);
      }
      clockOut=dseconds();
      msecTime=(float)((clockOut-clockIn)/iterations*1000.0);
      float msecNoBLAS=msecTime;
      double performanceNoBLAS=getGFlops(nFPOperations,msecNoBLAS);
#endif

#ifndef NOGPU
  #ifdef VERIFY
      errorCheckCBLAS=0.0;
      errorCheckCBLAS=matrixMultiplyError(m, n, c_cblas, c_no_blas);
  #endif

  #ifdef VERIFY
      errorCheckCUBLAS=0.0;
      errorCheckCUBLAS=matrixMultiplyError(m, n, c_cublas, c_no_blas);
  #endif
#endif

/* PRINT OUT RESULTS */
#ifndef NOGPU
  #ifdef CPU_DGEMM
      printf("  %3d, %10lld, %10lld, %10lld, %10.4f, %10.4f, %10.4f, %10.4f, %10.2f, %10.4f, %13.4f, %13.4f, %13.4f, %13.4f, %13.4f, %13.4f, %13.4f, %16.4f, %13.4f",
             l,m,k,n,sizeAinMBs,sizeBinMBs,sizeCinMBs,sizeAinMBs+sizeBinMBs+sizeCinMBs,
             (double)freeGPUMemoryInBytes/1048576.0,(double)memoryRequired/1048576.0,
             performanceD2H_A,performanceD2H_B,performanceH2D_A,performanceH2D_B,performanceD2H_C,
             performanceCBLAS,performanceCUBLAS,performanceCUBLAS_wDT,
             performanceNoBLAS);
  #else
      printf("  %3d, %10lld, %10lld, %10lld, %10.4f, %10.4f, %10.4f, %10.4f, %10.2f, %10.4f, %13.4f, %13.4f, %13.4f, %13.4f, %13.4f, %13.4f, %13.4f, %16.4f",
             l,m,k,n,sizeAinMBs,sizeBinMBs,sizeCinMBs,sizeAinMBs+sizeBinMBs+sizeCinMBs,
             (double)freeGPUMemoryInBytes/1048576.0,(double)memoryRequired/1048576.0,
             performanceD2H_A,performanceD2H_B,performanceH2D_A,performanceH2D_B,performanceD2H_C,
             performanceCBLAS,performanceCUBLAS,performanceCUBLAS_wDT);
  #endif
#else
      printf("  %3d, %10lld, %10lld, %10lld, %10.4f, %10.4f, %10.4f, %10.4f, %10.2f, %13.4f",
             l,m,k,n,sizeAinMBs,sizeBinMBs,sizeCinMBs,sizeAinMBs+sizeBinMBs+sizeCinMBs,
             (double)memoryRequired/1048576.0,
             performanceCBLAS);
#endif

/* EITHER PRINT OUT MORE STUFF OR A LINE BREAK */
#ifndef NOGPU
  #ifdef VERIFY
      printf(", %.10f, %.10f\n",errorCheckCBLAS,errorCheckCUBLAS);
  #else
      printf("\n");
  #endif
#else
      printf("\n");
#endif

/* ITERATE VALUES */     
      nColsB=resizeColumnsMatrixB(nColsB,maxMatrixSize);
      nColsC=nColsB;
      l=l+1;
      
/* DEALLOCATION */
      
/* HOST MEMORY */
#ifdef PINNED
/* IF  PINNED HOST MEMORY IS USED */
      cudaFreeHost(a);
      cudaFreeHost(b);
      cudaFreeHost(c_cublas);
#else
/* IF NON-PINNED MEMORY IS USED */
      free(a);
      free(b);
      free(c_cublas);
#endif

      free(c_no_blas);
      free(c_cblas);

#ifndef NOGPU
/* DEVICE MEMORY */
      cudaFree(d_a);
      cudaFree(d_b);
      cudaFree(d_c);
#endif

    } while(nColsB <= maxMatrixSize);
    
    printf("+----------------------+\n");
  
  }

#ifdef NOGPU
  #ifdef MKL
    vslDeleteStream(&stream);
  #endif
#endif

#ifndef NOGPU
/* CUDA HANDLES */
  curandDestroyGenerator(gen);

  cublasDestroy(handle);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
#endif

  return EXIT_SUCCESS;
}

double dseconds(void)
{
  static double base;
  struct timeval t;
  double td;
  gettimeofday(&t, NULL);
  td = t.tv_sec+t.tv_usec*1.0e-6;
  if (!base)
    base = td;
  return td-base;
}

void printMatrixFlat(double matrix[],int rows, int columns)
{
  int i;
  for(i=0;i<rows*columns;i++)
  {
    printf("%15.10f\n",matrix[i]);
  }
  printf("\n");
  return;
}

void printMatrixRow(double matrix[],int rows, int columns)
{
  int i, j;
  for(i=0;i<rows;i++)
  {
    printf("{ ");
    for(j=0;j<columns-1;j++)
    {
      printf("%15.10f,",matrix[IDX2C(j,i,columns)]);
    }
    printf("%15.10f",matrix[IDX2C(j,i,columns)]);
    printf(" }\n");
  }
  printf("\n");
  return;
}

void printMatrixCol(double matrix[],int rows, int columns)
{
  int i, j;
  for(i=0;i<rows;i++)
  {
    printf("{ ");
    for(j=0;j<columns-1;j++)
    {
      printf("%15.10f,",matrix[IDX2C(i,j,rows)]);
    }
    printf("%15.10f",matrix[IDX2C(i,j,rows)]);
    printf(" }\n");
  }
  printf("\n");
  return;
}

void matrixMultiplyCPU(const int MM, const int NN, const int KK, double alpha, 
                       const double *A, const double *B, double beta, double *C)
{
  int i, j, index;
  for (i=0; i<MM; i++)
  {
    for (j=0; j<NN; j++)
    {
      double product=0.0;
      for(index=0; index<KK; index++)
      {
        product += A[IDX2C(i,index,MM)]*B[IDX2C(index,j,KK)];
      }
      C[IDX2C(i,j,MM)]=alpha*product + beta*C[IDX2C(i,j,MM)];
    }
  }
}

double matrixMultiplyError(const int MM, const int NN, 
                           const double *C_test, const double *C_ref)
{
  double error_norm=0.0;
  double ref_norm=0.0;
  double diff;
  int i;
  for (i=0; i<MM*NN; i++)
  {
    diff = C_test[i]-C_ref[i];
    error_norm += diff*diff;
      ref_norm += C_ref[i]*C_ref[i];
  }

  if(error_norm > 0)
  {
    error_norm = sqrt(error_norm);
  }
  else
  {
    error_norm = -1.0;
  }

  if(error_norm > 0)
  {
    ref_norm = sqrt(ref_norm);
  }
  else
  {
    ref_norm = -1.0;
  }

  if (fabs(ref_norm) < 1.0e-7)
  {
    printf("**** Reference Norm is ZERO ****\n");
  }

#ifdef DEBUG
/* PRINT ERROR ANALYSIS */
  printf("+------------------------------+\n");
  printf("| Error Norm                  : %25.15f \n",error_norm);
  printf("| Reference Norm              : %25.15f \n",ref_norm);
  printf("| Error Norm / Reference Norm : %25.15f \n",error_norm/ref_norm);
  printf("+------------------------------+\n");
#endif
  
//return error_norm/ref_norm;
  return error_norm;
}

double getBandwidth(const int MM, const int NN, const float milliSeconds)
{
  double Bytes=(double)(MM*NN*sizeof(double));
  double GBs=(double)(Bytes/1073741824.0);
  return (double)(GBs/(milliSeconds/1000.0));
}

double getGFlops(const double nFPOperations, const float milliSeconds)
{
  double seconds=(double)(milliSeconds/1000.0);
  return (double)((nFPOperations/1000.0/1000.0/1000.0/seconds));
}

long long int resizeColumnsMatrixB(const long long int columnsMatrixB, 
                                   const long long int maxMatrixSize)
{
  long long int newColumnsMatrixB=columnsMatrixB*2;
  
  if (newColumnsMatrixB>maxMatrixSize) 
  {
    newColumnsMatrixB=newColumnsMatrixB/2+4096;
    if (newColumnsMatrixB>maxMatrixSize) 
    {
      newColumnsMatrixB=newColumnsMatrixB-4096+2048;
      if (newColumnsMatrixB>maxMatrixSize) 
      {
        newColumnsMatrixB=newColumnsMatrixB-2048+1024;
        if (newColumnsMatrixB>maxMatrixSize) 
        {
          newColumnsMatrixB=newColumnsMatrixB-1024+512;
          if (newColumnsMatrixB>maxMatrixSize) 
          {
            newColumnsMatrixB=newColumnsMatrixB-512+256;
            if (newColumnsMatrixB>maxMatrixSize) 
            {
              newColumnsMatrixB=newColumnsMatrixB-256+128;
              if (newColumnsMatrixB>maxMatrixSize) 
              {
                newColumnsMatrixB=newColumnsMatrixB-128+64;
                if (newColumnsMatrixB>maxMatrixSize) 
                {
                  newColumnsMatrixB=newColumnsMatrixB-64+32;
                  if (newColumnsMatrixB>maxMatrixSize) 
                  {
                    newColumnsMatrixB=newColumnsMatrixB-32+16;
                    if (newColumnsMatrixB>maxMatrixSize) 
                    {
                      newColumnsMatrixB=newColumnsMatrixB-16+8;
                      if (newColumnsMatrixB>maxMatrixSize) 
                      {
                        newColumnsMatrixB=newColumnsMatrixB-8+4;
                        if (newColumnsMatrixB>maxMatrixSize) 
                        {
                          newColumnsMatrixB=newColumnsMatrixB-4+2;
                          if (newColumnsMatrixB>maxMatrixSize) 
                          {
                            newColumnsMatrixB=newColumnsMatrixB-2+1;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  
  return newColumnsMatrixB;
}
