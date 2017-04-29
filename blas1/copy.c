#include "mnblas.h"

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

void mncblas_dcopy(const int N, const double *X, const int incX, 
                 double *Y, const int incY)
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

void mncblas_ccopy(const int N, const void *X, const int incX, 
		                    void *Y, const int incY)
{

}

void mncblas_zcopy(const int N, const void *X, const int incX, 
		                    void *Y, const int incY)
{

}

