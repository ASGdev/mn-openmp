#include "mnblas.h"

#include <nmmintrin.h>

typedef float float4 [4] __attribute__ ((aligned (16))) ;

void mncblas_sgemv (const MNCBLAS_LAYOUT layout,
		    const MNCBLAS_TRANSPOSE TransA, const int M, const int N,
		    const float alpha, const float *A, const int lda,
		    const float *X, const int incX, const float beta,
		    float *Y, const int incY)
{
  register unsigned int i ;
  register unsigned int j ;
  register float r ;
  register float x ;
  register unsigned int indice ;

  float4 x4, r4 ;
  __m128 xv4, a1, dot ;
  
  for (i = 0; i < M; i += incX)
    {
      r = 0.0 ;
      indice = i * M ;

      x4 [0] = X [i] ;
      x4 [1] = X [i] ;
      x4 [2] = X [i] ;
      x4 [3] = X [i] ;

      xv4 = _mm_load_ps (x4) ;
      
      for (j = 0 ; j < M; j += 4)
	{
	  a1 = _mm_load_ps (A+indice+j) ;
	  dot = _mm_dp_ps (a1, xv4, 0xFF) ;
	  _mm_store_ps (r4, dot) ;
	  r += r4 [0] ;
	}
      
      Y [i] = (beta * Y[i])  + (alpha * r) ;

    }
  
  return ;
}

void mncblas_sgemv_1 (const MNCBLAS_LAYOUT layout,
		      const MNCBLAS_TRANSPOSE TransA, const int M, const int N,
		      const float alpha, const float *A, const int lda,
		      const float *X, const int incX, const float beta,
		      float *Y, const int incY
		      )
{
  register unsigned int i ;
  register unsigned int j ;
  register float r ;
  register float x ;
  register unsigned int indice ;
  
  for (i = 0; i < M; i += incX)
    {
      r = 0.0 ;
      x = X [i] ;
      indice = i * M ;
      
      for (j = 0 ; j < M; j += incY)
	{
	 r += A[indice+j] * x ;
	}
      
      Y [i] = (beta * Y[i])  + (alpha * r) ;

    }
  return ;
}



void mncblas_dgemv (MNCBLAS_LAYOUT layout,
		    MNCBLAS_TRANSPOSE TransA, const int M, const int N,
		    const double alpha, const double *A, const int lda,
		    const double *X, const int incX, const double beta,
		    double *Y, const int incY)
{

  return ;
}


void mncblas_cgemv (MNCBLAS_LAYOUT layout,
		    MNCBLAS_TRANSPOSE TransA, const int M, const int N,
		    const void *alpha, const void *A, const int lda,
		    const void *X, const int incX, const void *beta,
		    void *Y, const int incY)
{

  return ;
}


void mncblas_zgemv (MNCBLAS_LAYOUT layout,
		    MNCBLAS_TRANSPOSE TransA, const int M, const int N,
		    const void *alpha, const void *A, const int lda,
		    const void *X, const int incX, const void *beta,
		    void *Y, const int incY)
{

  return ;
}

