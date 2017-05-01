#include <stdio.h>
#include <cblas.h>

/*
  Mesure des cycles
*/

#include <x86intrin.h>

#define NBEXPERIMENTS    102
static long long unsigned int experiments [NBEXPERIMENTS] ;

// #define VECSIZE    32
#define VECSIZE    1048576

typedef float vfloat  [VECSIZE] __attribute__ ((aligned (16))) ;
typedef float vdouble [VECSIZE] __attribute__ ((aligned (16))) ;

vfloat vec1, vec2 ;

long long unsigned int average (long long unsigned int *exps)
{
  unsigned int i ;
  long long unsigned int s = 0 ;

  for (i = 2; i < (NBEXPERIMENTS-2); i++)
    {
      s = s + exps [i] ;
    }

  return s / (NBEXPERIMENTS-2) ;
}


void vector_init (vfloat V, float x)
{
  register unsigned int i ;

  for (i = 0; i < VECSIZE; i++)
    V [i] = x ;

  return ;
}

void vector_print (vfloat V)
{
  register unsigned int i ;

  for (i = 0; i < VECSIZE; i++)
    printf ("%f ", V[i]) ;
  printf ("\n") ;
  
  return ;
}

int main (int argc, char **argv)
{
  unsigned long long int start, end ;
  unsigned long long int residu ;
  unsigned long long int av ;
  int exp ;
  
 /* Calcul du residu de la mesure */
  start = _rdtsc () ;
  end = _rdtsc () ;
  residu = end - start ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      vector_init (vec1, 1.0) ;
      
      start = _rdtsc () ;

         cblas_scopy (VECSIZE, vec1, 1, vec2, 1) ;
     
      end = _rdtsc () ;
      
      experiments [exp] = end - start ;
    }
  
  av = average (experiments) ;

  printf ("cblas_scopy nombre de cycles cblas: \t %Ld \n", av-residu) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      vector_init (vec1, 1.0) ;
      vector_init (vec2, 2.0) ;
      
      start = _rdtsc () ;

          cblas_saxpy (VECSIZE, 3.0, vec1, 1, vec2, 1) ;
     
      end = _rdtsc () ;
      
      experiments [exp] = end - start ;
    }
  
  av = average (experiments) ;

  // vector_print (vec2) ;
  printf ("cblas_saxpy nombre de cycles cblas: \t %Ld \n", av-residu) ;  
}