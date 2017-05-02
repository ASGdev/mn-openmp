#include "mnblas.h"
#include <x86intrin.h>
#include <emmintrin.h>
#include <stdio.h>

#define XMM_NUMBER 8
typedef float float4 [4] __attribute__ ((aligned (16))) ;
typedef double double2 [2] __attribute__ ((aligned (16))) ;


float mncblas_sdot(const int N, const float *X, const int incX, 
                 const float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register float dot = 0.0 ;

  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
      dot = dot + X [i] * Y [j] ;
    }

  return dot ;
}

float mncblas_sdot_vec(const int N, const float *X, const int incX, 
                 const float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  __m128 dot = _mm_set1_ps(0.0);

  float4 fdot;

  for (; ((i < N) && (j < N)) ; i += incX + 4, j+=incY + 4)
    {
      dot = _mm_add_ps(dot, _mm_mul_ps (_mm_load_ps (X+i), _mm_load_ps (Y+i)));
    }

  _mm_store_ps(fdot, dot);
  return (fdot[0] + fdot[1] + fdot[2] + fdot[3]) ;
}


float mncblas_sdot_omp(const int N, const float *X, const int incX, 
                 const float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register float dot = 0.0 ;
  
  #pragma omp parallel for schedule(static) reduction(+:dot) private(i, j)
  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
      dot = dot + X [i] * Y [j] ;
    }

  return dot ;
}

double mncblas_ddot(const int N, const double *X, const int incX, 
                 const double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register double dot = 0.0 ;

  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
      dot = dot + X [i] * Y [j] ;
    }

  return dot ;
}

double mncblas_ddot_omp(const int N, const double *X, const int incX, 
                 const double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register double dot = 0.0 ;

  #pragma omp parallel for schedule(static) reduction(+:dot) private(i, j)
  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
      dot = dot + X [i] * Y [j] ;
    }

  return dot ;
}

double mncblas_ddot_vec(const int N, const double *X, const int incX, 
                 const double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  __m128d dot = _mm_set1_pd(0.0);

  double2 fdot;

  for (; ((i < N) && (j < N)) ; i += incX + 2, j+=incY + 2)
    {
      dot = _mm_add_pd(dot, _mm_mul_pd (_mm_load_pd (X+i), _mm_load_pd (Y+i)));
    }

  _mm_store_pd(fdot, dot);
  return (fdot[0] + fdot[1]) ;
}

void   mncblas_cdotu_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu)
{
  /* a completer */
  
  return ;
}

void   mncblas_cdotc_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
  /* a completer */
  
  return ;
}

void   mncblas_zdotu_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu)
{
  /* a completer */
  
  return ;
}
  
void   mncblas_zdotc_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
  /* a completer */
  
  return ;
}

/* FOR TEST PURPOSES */
void printvec(double v[], int size){
  for(int i = 0 ; i<size; i++)
    printf("%f ", v[i]);

  printf("\n");
}

int main(){
  double v1[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
  double v2[5] = {6.0, 7.0, 8.0, 9.0, 10.0};

  printf("Dot d seq : %f\n", mncblas_ddot(5, v1, 1, v2, 1));
  printf("Dot d p : %f\n", mncblas_ddot_omp(5, v1, 1, v2, 1));
  printf("Dot d vec : %f\n", mncblas_ddot_vec(5, v1, 0, v2, 0));

}