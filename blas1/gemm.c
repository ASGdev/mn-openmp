#include <stdlib.h>

#include "mnblas.h"

#include <nmmintrin.h>

typedef float *floatM;

typedef float float4 [4]  __attribute__ ((aligned (16))) ;

void mncblas_sgemm (
		    MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
		    MNCBLAS_TRANSPOSE TransB, const int M, const int N,
		    const int K, const float alpha, const float *A,
		    const int lda, const float *B, const int ldb,
		    const float beta, float *C, const int ldc
		   )
{
  /*
    vectorized implementation
  */
  
  register unsigned int i ;
  register unsigned int j ;
  register unsigned int k ;
  register unsigned int l ;

  register unsigned int indice_ligne ;
  
  register float r ;
  floatM   Bcol ;
  float4   R4 ;
  int      err ;
  
  __m128 av4 ;
  __m128 bv4 ;
  __m128 dot ;

  /*
    Bcol = (floatM) malloc (M * sizeof (float)) ;
    err = posix_memalign ((void **) &Bcol, 16, M*sizeof(float)) ;
  */

  if (TransB == MNCblasNoTrans)
    {
      Bcol = aligned_alloc (16, M * sizeof (float)) ;
  
      for (i = 0 ; i < M; i = i + 1)
	{
	  
	  for (j = 0 ; j < M; j ++)
	    {
	      

	      /*
		load a B column (j)
	      */

	      for (l = 0 ; l < M ; l = l + 4)
		{
		  Bcol [l]     = B [l        * M + j ] ;
		  Bcol [l + 1] = B [(l + 1)  * M + j ] ;
		  Bcol [l + 2] = B [(l + 2)  * M + j ] ;
		  Bcol [l + 3] = B [(l + 3)  * M + j ] ;	      
		}

	      r = 0.0 ;	  
	      indice_ligne = i * M ;
	  
	      for (k = 0; k < M; k = k + 4)
		{

		  av4 = _mm_load_ps (A+indice_ligne + k);
		  bv4 = _mm_load_ps (Bcol+ k) ;
		  
		  dot = _mm_dp_ps (av4, bv4, 0xFF) ;
	      
		  _mm_store_ps (R4, dot) ;

		  r = r + R4 [0] ;
		}
	  
	      C [indice_ligne + j] = (alpha * r) + (beta * C [indice_ligne + j]) ;
	    }
	}
    }
  else
    {
      for (i = 0 ; i < M; i = i + 1)
	{
	  
	  for (j = 0 ; j < M; j ++)
	    {
	      
	      r = 0.0 ;	  
	      indice_ligne = i * M ;
	  
	      for (k = 0; k < M; k = k + 4)
		{

		  av4 = _mm_load_ps (A + indice_ligne + k);
		  bv4 = _mm_load_ps (B + indice_ligne + k) ;
		  
		  dot = _mm_dp_ps (av4, bv4, 0xFF) ;
	      
		  _mm_store_ps (R4, dot) ;

		  r = r + R4 [0] ;
		}
	  
	      C [indice_ligne + j] = (alpha * r) + (beta * C [indice_ligne + j]) ;
	    }
	}
    }
  return ;
}


void mncblas_sgemm_1 (
		      MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
		      MNCBLAS_TRANSPOSE TransB, const int M, const int N,
		      const int K, const float alpha, const float *A,
		      const int lda, const float *B, const int ldb,
		      const float beta, float *C, const int ldc
		      )
{
  /* 
     scalar implementation
  */
  
  register unsigned int i ;
  register unsigned int j ;
  register unsigned int k ;
  register float r ;

  for (i = 0 ; i < M; i = i + 4)
    {
      /* i */
      for (j = 0 ; j < M; j ++)
	{
	  r = 0.0 ;
	  for (k = 0; k < M; k=k+4)
	    {
	      r = r + A [(i * M) + k    ] * B [(k * M)       + j] ;
	      r = r + A [(i * M) + k + 1] * B [(k + 1) * M   + j] ;
	      r = r + A [(i * M) + k + 2] * B [(k + 2) * M   + j] ;
	      r = r + A [(i * M) + k + 3] * B [(k + 3) * M   + j] ;
	    }
	  C [(i*M) + j] = (alpha * r) + (beta * C [(i*M) + j]) ;
	}

      /* i + 1 */
      for (j = 0 ; j < M; j ++)
	{
	  r = 0.0 ;
	  for (k = 0; k < M; k=k+4)
	    {
	      r = r + A [((i + 1) * M) + k    ] * B [(k * M)     + j] ;
	      r = r + A [((i + 1) * M) + k + 1] * B [(k + 1) * M + j] ;
	      r = r + A [((i + 1) * M) + k + 2] * B [(k + 2) * M + j] ;
	      r = r + A [((i + 1) * M) + k + 3] * B [(k + 3) * M + j] ;
	    }
	  C [((i + 1) * M) + j] = (alpha * r) + (beta * C [((i + 1) * M) + j]) ;
	}

       /* i + 2 */
      for (j = 0 ; j < M; j ++)
	{
	  r = 0.0 ;
	  for (k = 0; k < M; k = k + 4)
	    {
	      r = r + A [((i + 2) * M) + k    ] * B [(k * M)     + j] ;
	      r = r + A [((i + 2) * M) + k + 1] * B [(k + 1) * M + j] ;
	      r = r + A [((i + 2) * M) + k + 2] * B [(k + 2) * M + j] ;
	      r = r + A [((i + 2) * M) + k + 3] * B [(k + 3) * M + j] ;
	    }
	  C [((i + 2) * M) + j] = (alpha * r) + (beta * C [((i + 2) * M) + j]) ;
	}

      /* i + 3 */
      for (j = 0 ; j < M; j ++)
	{
	  r = 0.0 ;
	  for (k = 0; k < M; k = k + 4)
	    {
	      r = r + A [((i + 3) * M) + k    ] * B [(k * M)     + j] ;
	      r = r + A [((i + 3) * M) + k + 1] * B [(k + 1) * M + j] ;
	      r = r + A [((i + 3) * M) + k + 2] * B [(k + 2) * M + j] ;
	      r = r + A [((i + 3) * M) + k + 3] * B [(k + 3) * M + j] ;
	    }
	  C [((i + 3) * M) + j] = (alpha * r) + (beta * C [((i + 3) * M) + j]) ;
	}

    }
  return ;
}


void mncblas_dgemm(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
		   MNCBLAS_TRANSPOSE TransB, const int M, const int N,
		   const int K, const double alpha, const double *A,
		   const int lda, const double *B, const int ldb,
		   const double beta, double *C, const int ldc)
{

  return ;
}


void mncblas_cgemm (
		    MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
		    MNCBLAS_TRANSPOSE TransB, const int M, const int N,
		    const int K, const void *alpha, const void *A,
		    const int lda, const void *B, const int ldb,
		    const void *beta, void *C, const int ldc
		   )
{

  return ;
}

void mncblas_zgemm (
		    MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
		    MNCBLAS_TRANSPOSE TransB, const int M, const int N,
		    const int K, const void *alpha, const void *A,
		    const int lda, const void *B, const int ldb,
		    const void *beta, void *C, const int ldc
		   )
{

  return ;
}

