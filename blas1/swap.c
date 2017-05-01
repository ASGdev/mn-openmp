#include "mnblas.h"
#include <x86intrin.h>
#include <stdio.h>


void mncblas_sswap(const int N, float *X, const int incX, 
                 float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register float save ;
  
  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
      save = Y [j] ;
      Y [j] = X [i] ;
      X [i] = save ;
    }

  return ;
}

void mncblas_sswap_omp(const int N, float *X, const int incX, 
                 float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register float save ;
  
  #pragma omp for
  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
      save = Y [j] ;
      Y [j] = X [i] ;
      X [i] = save ;
    }

  return ;
}

void mncblas_sswap_vec(const int N, float *X, const int incX, 
                 float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  __m128 save, x, y;

  for (; ((i < N) && (j < N)) ; i += incX * 4, j+=incY * 4)
    {
      save = _mm_load_ps(Y+i) ;
      _mm_store_ps(Y+i, _mm_load_ps(X+i)) ;
      _mm_store_ps(X+i, save) ;
    }

  return ;
}

void mncblas_dswap_vec(const int N, double *X, const int incX, 
                 double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  
  __m128d save, x, y;

  for (; ((i < N) && (j < N)) ; i += incX * 4, j+=incY * 4)
    {
      save = _mm_load_pd(Y+i) ;
      _mm_store_pd(Y+i, _mm_load_pd(X+i)) ;
      _mm_store_pd(X+i, save) ;
    }

  return ;
}

void mncblas_dswap_omp(const int N, double *X, const int incX, 
                 double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register double save ;

  #pragma omp for schedule(static)
  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
      save = Y [j] ;
      Y [j] = X [i] ;
      X [i] = save ;
    }

  return ;
}

void mncblas_cswap(const int N, void *X, const int incX, 
		                    void *Y, const int incY)
{

  return ;
}

void mncblas_zswap(const int N, void *X, const int incX, 
		                    void *Y, const int incY)
{

  return ;
}

/* FOR TEST PURPOSES */
void printvec(float v[], int size){
  for(int i = 0 ; i<size; i++)
    printf("%f ", v[i]);

  printf("\n");
}

int main(){
  double v1[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
  double v2[5] = {6.0, 7.0, 8.0, 9.0, 10.0};

  printvec(v1, 5);
  printvec(v2, 5);
  mncblas_dswap_vec(5, v1, 1, v2, 1);
  printvec(v1, 5);
  printvec(v2, 5);

  // float v3[5];
  // mncblas_scopy_vec(5, v1, 0, v3, 0);
  // printvec(v3, 5);
}