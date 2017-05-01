#include "mnblas.h"
#include <stdio.h>
#include <x86intrin.h>
#include <emmintrin.h>

void mncblas_scopy(const int N, const float *X, const int incX, 
                 float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  for (; ((i < N) && (j < N)) ; i = i + incX + 4, j = j + incY + 4)
    {
      Y [j] = X [i] ;
      Y [j+1] = X [i+1] ;
      Y [j+2] = X [i+2] ;
      Y [j+3] = X [i+3] ;
    }
  return ;
}

void mncblas_scopy_omp(const int N, const float *X, const int incX, 
                 float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  #pragma omp for schedule(static)
  for (; ((i < N) && (j < N)) ; i = i + incX + 4, j = j + incY + 4)
    {
      Y [j] = X [i] ;
      Y [j+1] = X [i+1] ;
      Y [j+2] = X [i+2] ;
      Y [j+3] = X [i+3] ;
    }

  return ;
}

void mncblas_scopy_vec(const int N, const float *X, const int incX, 
                 float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  __m128 v;

  for (; ((i < N) && (j < N)) ; i = i + incX + 4, j = j + incY + 4)
    {
      v = _mm_load_ps(X+i);
      _mm_store_ps (Y+i, v) ;
    }

  return ;
}

void mncblas_dcopy(const int N, const double *X, const int incX, 
                 double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  for (; ((i < N) && (j < N)) ; i = i * incX + 2, j = j * incY + 2)
    {
      Y [j] = X [i] ;
      Y [j+1] = X [i+1] ;
    }
  return ;
}

void mncblas_dcopy_omp(const int N, const double *X, const int incX, 
                 double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  #pragma omp for schedule(static) private(i, j)
  for (; ((i < N) && (j < N)) ; i = i + incX + 2, j = j + incY + 2)
    {
      Y [j] = X [i] ;
      Y [j+1] = X [i+1] ;
    }
  return ;
}

void mncblas_dcopy_vec(const int N, const double *X, const int incX, 
                 double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  __m128d v;

  for (; ((i < N) && (j < N)) ; i = i + incX + 2, j = j + incY + 2)
    {
      v = _mm_load_pd(X+i);
      _mm_store_pd(Y+i, v) ;
    }

  return ;
}

void mncblas_ccopy(const int N, const void *X, const int incX, 
		                    void *Y, const int incY)
{

}

void mncblas_zcopy(const int N, const void *X, const int incX, 
		                    void *Y, const int incY)
{

}

/* FOR TEST PURPOSES */
void printvec(double v[], int size){
  for(int i = 0 ; i<size; i++)
    printf("%f ", v[i]);

  printf("\n");
}

int main(){
  double v1[5] = {1.0, 2.0, 3.0, 4.0, 5.0};

  // double v2[5];
  // printvec(v2, 5);
  // mncblas_dcopy(5, v1, 1, v2, 1);
  // printvec(v2, 5);
  // printvec(v1, 5);

  // double v3[5];
  // printvec(v3, 5);
  // mncblas_dcopy_vec(5, v1, 0, v3, 0);
  // printvec(v3, 5);

  double v4[5];
  printvec(v4, 5);
  mncblas_dcopy_omp(5, v1, 0, v4, 0);
  printvec(v4, 5);
}