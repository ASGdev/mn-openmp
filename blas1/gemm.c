#include <stdlib.h>

#include "mnblas.h"

#include <nmmintrin.h>
#include <stdio.h>

#define VEC_SIZE 5

typedef float *floatM;
typedef double *doubleM;

//typedef double matrix [4][4] ;

typedef float float4 [4]  __attribute__ ((aligned (16))) ;
typedef double double2 [2] __attribute__ ((aligned (16))) ;

typedef struct {
  float REEL;
  float IMAG;
}vcomplexe;

typedef vcomplexe VCOMP [VEC_SIZE] ;

typedef struct {
  double REEL;
  double IMAG;
}dcomplexe;

typedef dcomplexe DCOMP [VEC_SIZE] ;

typedef vcomplexe matrix[2][2];

void print_matrix (matrix M, int N)
{
  register unsigned int i, j ;

  for (i = 0 ; i < N; i++)
    {
      for (j = 0 ; j < N; j++)
	{
	  printf (" %3.2f ", M[i][j]) ;
	}
      printf ("\n") ;
    }
  printf ("\n") ;
  return ;
}

void print_(double *m){
	for(int i = 0; i< 16; i++)
		printf("%f ", *(m+i));
}

void mncblas_sgemm (
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

      //printf("\n-> %f\n", r);
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

void mncblas_sgemm_omp (
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

  #pragma omp parallel for schedule(static) private(i, j, k)
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

      //printf("\n-> %f\n", r);
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

void mncblas_sgemm_vec (
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

void mncblas_dgemm (
		      MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
		      MNCBLAS_TRANSPOSE TransB, const int M, const int N,
		      const int K, const double alpha, const double *A,
		      const int lda, const double *B, const int ldb,
		      const double beta, double *C, const int ldc
		      )
{
  /* 
     scalar implementation
  */
  register unsigned int i ;
  register unsigned int j ;
  register unsigned int k ;
  register double r ;

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

	  //printf("\n-> %f\n", r);
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

void mncblas_dgemm_omp (
              MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
              MNCBLAS_TRANSPOSE TransB, const int M, const int N,
              const int K, const double alpha, const double *A,
              const int lda, const double *B, const int ldb,
              const double beta, double *C, const int ldc
              )
{
  /* 
     scalar implementation
  */
  register unsigned int i ;
  register unsigned int j ;
  register unsigned int k ;
  register double r ;

  #pragma omp parallel for schedule(static) private(i, j, k)
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

      //printf("\n-> %f\n", r);
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

void mncblas_dgemm_vec(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
		   MNCBLAS_TRANSPOSE TransB, const int M, const int N,
		   const int K, const double alpha, const double *A,
		   const int lda, const double *B, const int ldb,
		   const double beta, double *C, const int ldc)
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
  doubleM   Bcol ;
  double2   R4 ;
  int      err ;
  
  __m128d av4 ;
  __m128d bv4 ;
  __m128d dot ;

  /*
    Bcol = (floatM) malloc (M * sizeof (float)) ;
    err = posix_memalign ((void **) &Bcol, 16, M*sizeof(float)) ;
  */

  if (TransB == MNCblasNoTrans)
    {
      Bcol = aligned_alloc (16, M * sizeof (double)) ;
  
      for (i = 0 ; i < M; i = i + 1)
	{
	  
	  for (j = 0 ; j < M; j ++)
	    {
	      

	      /*
		load a B column (j)
	      */

	      for (l = 0 ; l < M*2 ; l = l + 4)
		{
		  Bcol [l]     = B [l        * M + j ] ;
		  Bcol [l + 1] = B [(l + 1)  * M + j ] ;
		  Bcol [l + 2] = B [(l + 2)  * M + j ] ;
		  Bcol [l + 3] = B [(l + 3)  * M + j ] ;	      
		}

	      r = 0.0 ;	  
	      indice_ligne = i * M ;
	  
	      for (k = 0; k < M; k = k + 2)
		{

		  av4 = _mm_load_pd (A+indice_ligne + k);
		  bv4 = _mm_load_pd (Bcol+ k) ;
		  
		  dot = _mm_dp_pd (av4, bv4, 0xFF) ;
	      
		  _mm_store_pd (R4, dot) ;

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

		  av4 = _mm_load_pd (A + indice_ligne + k);
		  bv4 = _mm_load_pd (B + indice_ligne + k) ;
		  
		  dot = _mm_dp_pd (av4, bv4, 0xFF) ;
	      
		  _mm_store_pd (R4, dot) ;

		  r = r + R4 [0] ;
		}
	  
	      C [indice_ligne + j] = (alpha * r) + (beta * C [indice_ligne + j]) ;
	    }
	}
    }
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


void mncblas_cgemm_vec(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
       MNCBLAS_TRANSPOSE TransB, const int M, const int N,
       const int K, const void *alpha, const void *A,
       const int lda, const void *B, const int ldb,
       const void *beta, void *C, const int ldc)
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
  float4 tres;

  float *AP = (float *)A;
  float *BP = (float *)B;
  float *CP = (float *)C;
  float *bv = (float *)beta;
  float *av = (float *)alpha;
  
  __m128 av4 ;
  __m128 bv4 ;
  __m128 dot ;
  __m128 alphav;
  __m128 betav;
  __m128 rv;

  // load alpha & beta
 alphav = _mm_set_ps(*(av), *(av), *(av+1), *(av+1));
 betav = _mm_set_ps(*(bv), *(bv), *(bv+1), *(bv+1));


  /* NoTrans only */

  Bcol = aligned_alloc (16, M * sizeof (double)) ;
  printf("M = %d\n", M);
  for (i = 0 ; i < M; i = i + 1) {

    for (j = 0 ; j < M; j ++) {
          /*
      load a B column (j)
          */
        for (l = 0 ; l < M ; l = l + 4)
        {
          // complexe 1
          Bcol [l]     = BP [l        * M + j ] ;
          Bcol [l + 1] = BP [(l + 1)  * M + j ] ;
          // complexe 2
          Bcol [l + 2] = BP [(l + 2)  * M + j ] ;
          Bcol [l + 3] = BP [(l + 3)  * M + j ] ;
        }
        for(int e = 0; e<M; e++){
          printf("%f ", Bcol[i]);
        }

        r = 0.0 ;
        indice_ligne = i * M ;

        // for (k = 0; k < M; k = k + 2)
        // {

        //   av4 = _mm_load_ps ((AP+indice_ligne + k));
        //   bv4 = _mm_load_ps (Bcol+ k) ;

        //   dot = _mm_dp_ps (av4, bv4, 0xFF) ;

        //   _mm_store_ps (R4, dot) ;

        //   r = r + R4 [0] ;
        // }

        // alpha * r
        __m128 ar = _mm_mul_ps(alphav, rv);
        ar = _mm_addsub_ps(ar, _mm_shuffle_ps(ar, ar, _MM_SHUFFLE(0, 0, 3, 2)));

        // beta * C
        float4 cl;
        cl[0] = cl[1] = CP[indice_ligne + j];
        cl[2] = cl[3] = CP[indice_ligne + j + 1];
        __m128 cv = _mm_load_ps(cl);
        __m128 bc = _mm_mul_ps(betav, cv);
        bc = _mm_addsub_ps(bc, _mm_shuffle_ps(bc, bc, _MM_SHUFFLE(0, 0, 3, 2)));


        CP [indice_ligne + j] = ar[0] + bc[0];
        CP [indice_ligne + j + 1] = ar[1] + bc[1];
      }
  }
  return ;
}

int main(){
	// matrix A = {
	//     {1, 1, 1, 1},
	//     {1, 1, 1, 1},
	//     {1, 1, 1, 1},
	//     {1, 1, 1, 1}
 //  	};
 //  	matrix B = {
	//     {2, 2, 2, 2},
	//     {2, 2, 2, 2},
	//     {2, 2, 2, 2},
	//     {2, 2, 2, 2}
 //  	};
 //  	matrix C1 = {
	//     {2, 2, 2, 2},
	//     {2, 2, 2, 2},
	//     {2, 2, 2, 2},
	//     {2, 2, 2, 2}
 //  	};

	// float alpha = 1;
	// float beta = 1;

	// mncblas_sgemm_omp (101, 111, 111, 4, 4, 4, alpha, *A, 1, *B, 1, beta, *C1, 1);
	// print_matrix(C1, 4);

	// matrix C2 = {
	//     {2, 2, 2, 2},
	//     {2, 2, 2, 2},
	//     {2, 2, 2, 2},
	//     {2, 2, 2, 2}
 //  	};
 //  	mncblas_sgemm_vec (101, 111, 111, 4, 4, 4, alpha, *A, 1, *B, 1, beta, *C2, 1);
	// print_matrix(C2, 4);

	// double alpha = 1;
	// double beta = 1;

	// mncblas_dgemm_omp (101, 111, 111, 4, 4, 4, alpha, *A, 1, *B, 1, beta, *C1, 1);
	// // mncblas_dgemm_vec(101, 111, 111, 4, 4, 4, alpha, *A, 1, *B, 1, beta, *C1, 1);

	// print_matrix(C1, 4);

    matrix A = {
      {{1,2}, {1,2}},
      {{1,2}, {1,2}}
    };
    matrix B = {
      {{1,2}, {1,2}},
      {{1,2}, {1,2}}
    };
    matrix C1 = {
      {{1,2}, {1,2}},
      {{1,2}, {1,2}}
    };

  vcomplexe alpha = {1,2};
  vcomplexe beta = {1,2};

  mncblas_cgemm_vec (101, 111, 111, 2, 2, 2, &alpha, *A, 1, *B, 1, &beta, *C1, 1);

	return 0;
}