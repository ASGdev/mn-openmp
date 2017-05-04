#include "mnblas.h"
#include <x86intrin.h>
#include <emmintrin.h>
#include <stdio.h>

#define XMM_NUMBER 8
#define VEC_SIZE 5

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

typedef float float4 [4] __attribute__ ((aligned (16))) ;
typedef double double2 [2] __attribute__ ((aligned (16))) ;

void printvec(float v[], int size){
  for(int i = 0 ; i<size; i++)
    printf("%f ", v[i]);

  printf("\n");
}

float mncblas_sdot(const int N, const float *X, const int incX, 
                 const float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register float dot = 0.0 ;

  for (; i < N ; i += incX)
    {
      dot = dot + X [i] * Y [i] ;
    }

  return dot ;
}

float mncblas_sdot_omp(const int N, const float *X, const int incX, 
                 const float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register float dot = 0.0 ;

  #pragma omp parallel for schedule(static) private(i) reduction(+:dot) 
  for (i = 0; i < N ; i += incX)
    {
      dot = dot + X [i] * Y [i] ;
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

double mncblas_ddot(const int N, const double *X, const int incX, 
                 const double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register double dot = 0.0 ;

  for (; i < N ; i += incX)
    {
      dot = dot + X [i] * Y [i] ;
    }

  return dot ;
}

double mncblas_ddot_omp(const int N, const double *X, const int incX, 
                 const double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register double dot = 0.0 ;

  #pragma omp parallel for schedule(static) reduction(+:dot) private(i)
  for (i=0 ; i < N ; i += incX)
    {
      dot = dot + X [i] * Y [i] ;
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

void mncblas_cdotu_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  float *XP = (float *)X;
  float *YP = (float *)Y;
  vcomplexe *temp = (vcomplexe *)dotu;
  float re, im;
  temp->REEL = 0;
  re = im = 0;

  for (; ((i < N*2)) ; i += incX + 2){
    printf("X = %f\n", *(XP+i));
    for(; j < N*2; j+=incY + 2){
      printf("\t%f\n", *(YP+j));
      re = re + ((*(XP+i) * *(YP+j)) - (*(XP+i+1) * *(YP+j+1)));
    }
    j = 0;
    temp->REEL += re;
    printf("re = %f\n", re);
    //im = im + XP[i] * YP[j]; 
  }

  printf("vcomplexe : %f %f \n", temp->REEL, temp->IMAG);
}

void   mncblas_cdotc_sub_vec(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
  /* conj(X)*Y */
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  float *XP = (float *) X;
  float *YP = (float *) Y;

  float v3[5], reel, imag;
  float reelContainer[4];

  __m128 dot = _mm_set1_ps(0.0);
  __m128 conj, R;

  double2 fdot;

  for (; ((i < N) && (j < N)) ; i += incX + 4, j+=incY + 4)
    {
      // conjugé
      __m128 temp = _mm_load_ps(XP+i);
      conj = _mm_sub_ps(_mm_set1_ps(0.0), temp);
      _mm_store_ps(v3+i, conj);

      //temp
      R = _mm_mul_ps(conj, _mm_load_ps(Y+i));

      // reel

    }

  printvec(v3, 5);
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


int main(){
  // vcomplexe v1[5] = {{1.0, 2.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}, {5.0, 5.0}};
  // vcomplexe v2[5] = {{1.0, 2.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}, {5.0, 5.0}};
  double v1[5] = {6.0, 7.0, 8.0, 9.0, 10.0};
  double v2[5] = {6.0, 7.0, 8.0, 9.0, 10.0};
  vcomplexe r;

  printf("Dot d seq : %f\n", mncblas_ddot(5, v1, 1, v2, 1));
  printf("Dot d p : %f\n", mncblas_ddot_omp(5, v1, 1, v2, 1));
  printf("Dot d vec : %f\n", mncblas_ddot_vec(5, v1, 0, v2, 0));

  float v3[5] = {6.0, 7.0, 8.0, 9.0, 10.0};
  float v4[5] = {6.0, 7.0, 8.0, 9.0, 10.0};

  printf("Dot s seq : %f\n", mncblas_sdot(5, v3, 1, v4, 1));
  printf("Dot s p : %f\n", mncblas_sdot_omp(5, v3, 1, v4, 1));
  printf("Dot s vec : %f\n", mncblas_sdot_vec(5, v3, 0, v4, 0));
  // mncblas_cdotu_sub(5, v1, 0, v2, 0, &r);

}