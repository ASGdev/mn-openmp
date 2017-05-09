#include <stdio.h>
#include <omp.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "mnblas.h"

#include <x86intrin.h>

#define NBEXPERIMENTS    22
static long long unsigned int experiments [NBEXPERIMENTS] ;


#define N              512
#define TILE           16

typedef float floatVector [N] ;
typedef double doubleVector [N] ;
typedef vcomplexe vcompVector [N] ;
typedef dcomplexe dcompVector [N] ;

vcomplexe cvalue = {1.0, 1.0};
dcomplexe zvalue = {1.0, 1.0};

typedef float floatMatrix [N][N] ;
typedef double doubleMatrix [N][N];
typedef vcomplexe vcompMatrix [N][N];
typedef dcomplexe dcompMatrix [N][N];


/* =============== Utilitaires ===================*/
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


void init_vector_double (doubleVector X, const double val)
{
  register unsigned int i ;

  for (i = 0 ; i < N; i++)
    X [i] = val ;

  return ;
}

void init_vector_float (floatVector X, const float val)
{
  register unsigned int i ;

  for (i = 0 ; i < N; i++)
    X [i] = val ;

  return ;
}

void init_vector_vcomplexe (vcompVector X, const vcomplexe val)
{
  register unsigned int i ;

  for (i = 0 ; i < N; i++)
    X [i] = val ;

  return ;
}

void init_vector_dcomplexe (dcompVector X, const dcomplexe val)
{
  register unsigned int i ;

  for (i = 0 ; i < N; i++)
    X [i] = val ;

  return ;
}

void init_matrix_float (floatMatrix X, const float val)
{
  register unsigned int i, j;

  for (i = 0; i < N; i++)
    {
      for (j = 0 ;j < N; j++)
  {
    X [i][j] = val ;
  }
    }
}

void init_matrix_double (doubleMatrix X, const double val)
{
  register unsigned int i, j;

  for (i = 0; i < N; i++)
    {
      for (j = 0 ;j < N; j++)
	{
	  X [i][j] = val ;
	}
    }
}

void init_matrix_vcomplexe (vcompMatrix X, const vcomplexe val)
{
  register unsigned int i, j;

  for (i = 0; i < N; i++)
    {
      for (j = 0 ;j < N; j++)
  {
    X [i][j] = val ;
  }
    }
}

void init_matrix_dcomplexe (dcompMatrix X, const dcomplexe val)
{
  register unsigned int i, j;

  for (i = 0; i < N; i++)
    {
      for (j = 0 ;j < N; j++)
  {
    X [i][j] = val ;
  }
    }
}

void print_vectors_float (floatVector X, floatVector Y)
{
  register unsigned int i ;

  for (i = 0 ; i < N; i++)
    printf (" X [%d] = %le Y [%d] = %le\n", i, X[i], i,Y [i]) ;

  return ;
}

void print_vectors_double (doubleVector X, doubleVector Y)
{
  register unsigned int i ;

  for (i = 0 ; i < N; i++)
    printf (" X [%d] = %le Y [%d] = %le\n", i, X[i], i,Y [i]) ;

  return ;
}

int main ()
{
  int nthreads, maxnthreads ;

  int tid;

  unsigned long long int start, end ;
  unsigned long long int residu ;

  unsigned long long int av ;

  int exp ;

  /*
     rdtsc: read the cycle counter
  */

  start = _rdtsc () ;
  end = _rdtsc () ;
  residu = end - start ;

  /*
     Sequential code executed only by the master thread
  */

  nthreads = omp_get_num_threads();
  maxnthreads = omp_get_max_threads () ;
  printf("Sequential execution: \n# threads %d\nmax threads %d \n", nthreads, maxnthreads);


  floatVector a, b;
  doubleVector c, d;
  vcompVector e, f;
  dcompVector g, h;

  floatMatrix am, bm;
  doubleMatrix cm, dm;
  vcompMatrix *em, *fm;
  dcompMatrix *gm, *hm;


  printf ("=============================== BLAS ===============================\n") ;
  printf ("=============== AXPY ===============\n") ;

  init_vector_float (a, 1.0) ;
  init_vector_float (b, 2.0) ;
  float alphaf = 1.0;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_saxpy(N, alphaf, a, 1, b, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Simple Precision - sequential\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_float (a, 1.0) ;
  init_vector_float (b, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_saxpy_omp(N, alphaf, a, 1, b, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Simple Precision - OMP\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_float (a, 1.0) ;
  init_vector_float (b, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_saxpy_vec(N, alphaf, a, 1, b, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Simple Precision - SSE\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_double (c, 1.0) ;
  init_vector_double (d, 2.0) ;
  double alphad = 1.0;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_daxpy(N, alphad, c, 1, d, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Double Precision - sequential\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_double (c, 1.0) ;
  init_vector_double (d, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_daxpy_omp(N, alphad, c, 1, d, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Double Precision - OMP\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_double (c, 1.0) ;
  init_vector_double (d, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_daxpy_vec(N, alphad, c, 1, d, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Double Precision - SSE\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_vcomplexe (e, cvalue) ;
  init_vector_vcomplexe (f, cvalue) ;
  vcomplexe alphac = {1.0, 1.0};

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_caxpy(N, &alphac, e, 1, f, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Simple Precision - sequential\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_vcomplexe (e, cvalue) ;
  init_vector_vcomplexe (f, cvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_caxpy_omp(N, &alphac, e, 1, f, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Simple Precision - OMP\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_vcomplexe (e, cvalue) ;
  init_vector_vcomplexe (f, cvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_caxpy_vec(N, &alphac, e, 1, f, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Simple Precision - SSE\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;
  dcomplexe alphaz = {1.0, 1.0};

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_zaxpy(N, &alphac, e, 1, f, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("\nComplex Double Precision - sequential\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_zaxpy_omp(N, &alphac, e, 1, f, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Double Precision - OMP\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_zaxpy_vec(N, &alphac, e, 1, f, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Double Precision - SSE\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  printf ("\n=============== DOT ===============\n") ;

  init_vector_float (a, 1.0) ;
  init_vector_float (b, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_sdot(N, a, 1, b, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Simple Precision - sequential\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_float (a, 1.0) ;
  init_vector_float (b, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_sdot_omp(N, a, 1, b, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Simple Precision - OMP\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_float (a, 1.0) ;
  init_vector_float (b, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      //mncblas_sdot_vec(N, a, 1, b, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Simple Precision - SSE\n\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_double (c, 1.0) ;
  init_vector_double (d, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_ddot(N, c, 1, d, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Double Precision - sequential\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_double (c, 1.0) ;
  init_vector_double (d, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      //mncblas_ddot_omp(N, c, 1, d, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Double Precision - OMP\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_double (c, 1.0) ;
  init_vector_double (d, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      //mncblas_ddot_vec(N, c, 1, d, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Double Precision - SSE\n\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_vcomplexe (e, cvalue) ;
  init_vector_vcomplexe (f, cvalue) ;
  vcomplexe rdotu;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      //mncblas_cdotu_sub(N, &c, 1, &d, 1, &rdotu);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Simple Precision - sequential\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_vcomplexe (e, cvalue) ;
  init_vector_vcomplexe (f, cvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;


      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Simple Precision - OMP\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_vcomplexe (e, cvalue) ;
  init_vector_vcomplexe (f, cvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;


      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Simple Precision - SSE");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;


      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("\nComplex Double Precision - sequential\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;


      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Double Precision - OMP\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;


      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Double Precision - SSE\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  printf ("\n=============== COPY ===============\n") ;

  clock_t clock_start, clock_end;

  init_vector_float (a, 1.0) ;
  init_vector_float (b, 0.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_scopy(N, a, 0, b, 0);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("\nSimple Precision - sequential\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  init_vector_float (a, 1.0) ;
  init_vector_float (b, 0.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_scopy_omp(N, a, 0, b, 0);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Simple Precision - OMP\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  init_vector_float (a, 1.0) ;
  init_vector_float (b, 0.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_scopy_vec(N, a, 0, b, 0);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Simple Precision - SSE\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  init_vector_double (c, 1.0) ;
  init_vector_double (d, 0.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_dcopy(N, c, 1, d, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("\nDouble Precision - sequential\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  init_vector_double (c, 1.0) ;
  init_vector_double (d, 0.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_dcopy_omp(N, c, 1, d, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Double Precision - OMP\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  init_vector_double (c, 1.0) ;
  init_vector_double (d, 0.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_dcopy(N, c, 1, d, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Double Precision - SSE\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  init_vector_vcomplexe (e, cvalue) ;
  init_vector_vcomplexe (f, cvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_ccopy(N, e, 0, f, 0);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("\nComplex Simple Precision - sequential\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  init_vector_vcomplexe (e, cvalue) ;
  init_vector_vcomplexe (f, cvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_ccopy_omp(N, e, 0, f, 0);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Simple Precision - OMP\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  init_vector_vcomplexe (e, cvalue) ;
  init_vector_vcomplexe (f, cvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_ccopy_omp(N, e, 0, f, 0);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Simple Precision - SSE\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_zcopy(N, e, 0, f, 0);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("\nComplex Double Precision - sequential\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_zcopy_omp(N, e, 0, f, 0);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Double Precision - OMP\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_zcopy_vec(N, e, 0, f, 0);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Double Precision - SSE\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  printf ("\n=============== SWAP ===============\n") ;

  init_vector_float (a, 1.0) ;
  init_vector_float (b, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_sswap(N, a, 1, b, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("\nSimple Precision - sequential\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  init_vector_float (a, 1.0) ;
  init_vector_float (b, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_sswap_omp(N, a, 1, b, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Simple Precision - OMP\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  init_vector_float (a, 1.0) ;
  init_vector_float (b, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_sswap_vec(N, a, 1, b, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Simple Precision - SSE\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  init_vector_double (c, 1.0) ;
  init_vector_double (d, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_dswap(N, c, 1, d, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("\nDouble Precision - sequential\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  init_vector_double (c, 1.0) ;
  init_vector_double (d, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_dswap_omp(N, c, 1, d, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Double Precision - OMP\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  init_vector_double (c, 1.0) ;
  init_vector_double (d, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_dswap_vec(N, c, 1, d, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Double Precision - SSE\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  init_vector_vcomplexe (e, cvalue) ;
  init_vector_vcomplexe (f, cvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_cswap(N, e, 1, f, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("\nComplex Simple Precision - sequential\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  init_vector_vcomplexe (e, cvalue) ;
  init_vector_vcomplexe (f, cvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_cswap_omp(N, e, 1, f, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Simple Precision - OMP\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  init_vector_vcomplexe (e, cvalue) ;
  init_vector_vcomplexe (f, cvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_cswap_vec(N, e, 1, f, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Simple Precision - SSE\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_zswap(N, g, 1, h, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("\nComplex Double Precision - sequential\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_zswap_omp(N, g, 1, h, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Double Precision - OMP\n");
  printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  // init_vector_dcomplexe (g, zvalue) ;
  // init_vector_dcomplexe (h, zvalue) ;

  // for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  //   {
  //     start = _rdtsc () ;

  //     //mncblas_zswap_vec(N, g, 1, h, 1);

  //     end = _rdtsc () ;
  //     experiments [exp] = end - start ;
  //   }

  // av = average (experiments) ;

  // printf ("Complex Double Precision - SSE\n");
  // printf ("\t%3.6f Go/s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  printf ("=============== GEMV ===============\n") ;

  init_vector_float (a, 1.0) ;
  init_vector_float (b, 1.0) ;
  init_matrix_float (bm, 2.0) ;
  float alphagmv = 1.0, betagmv = 2.0;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_sgemv(101, 111, N, N, alphagmv, *bm, 0, a, 1, betagmv, b, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("\nSimple Precision - sequential");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_float (a, 1.0) ;
  init_matrix_float (bm, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;


      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Simple Precision - OMP");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  init_vector_float (a, 1.0) ;
  init_matrix_float (bm, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;


      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Simple Precision - SSE");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  /**/

  // init_vector_double (c, 1.0) ;
  // init_vector_double (d, 2.0) ;

  // for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  //   {
  //     start = _rdtsc () ;


  //     end = _rdtsc () ;
  //     experiments [exp] = end - start ;
  //   }

  // av = average (experiments) ;

  // printf ("Double Precision - sequential");
  // printf ("\t%Ld cycles\n", av-residu) ;
  // printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  // /**/

  // init_vector_double (c, 1.0) ;
  // init_vector_double (d, 2.0) ;

  // for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  //   {
  //     start = _rdtsc () ;


  //     end = _rdtsc () ;
  //     experiments [exp] = end - start ;
  //   }

  // av = average (experiments) ;

  // printf ("Double Precision - OMP");
  // printf ("\t%Ld cycles\n", av-residu) ;
  // printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  // /**/

  // init_vector_double (c, 1.0) ;
  // init_vector_double (d, 2.0) ;

  // for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  //   {
  //     start = _rdtsc () ;


  //     end = _rdtsc () ;
  //     experiments [exp] = end - start ;
  //   }

  // av = average (experiments) ;

  // printf ("Double Precision - SSE");
  // printf ("\t%Ld cycles\n", av-residu) ;
  // printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  // /**/

  // init_vector_vcomplexe (e, cvalue) ;
  // init_vector_vcomplexe (f, cvalue) ;

  // for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  //   {
  //     start = _rdtsc () ;


  //     end = _rdtsc () ;
  //     experiments [exp] = end - start ;
  //   }

  // av = average (experiments) ;

  // printf ("Complex Simple Precision - sequential");
  // printf ("\t%Ld cycles\n", av-residu) ;
  // printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  // /**/

  // init_vector_vcomplexe (e, cvalue) ;
  // init_vector_vcomplexe (f, cvalue) ;

  // for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  //   {
  //     start = _rdtsc () ;


  //     end = _rdtsc () ;
  //     experiments [exp] = end - start ;
  //   }

  // av = average (experiments) ;

  // printf ("Complex Simple Precision - OMP");
  // printf ("\t%Ld cycles\n", av-residu) ;
  // printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  // /**/

  // init_vector_vcomplexe (e, cvalue) ;
  // init_vector_vcomplexe (f, cvalue) ;

  // for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  //   {
  //     start = _rdtsc () ;


  //     end = _rdtsc () ;
  //     experiments [exp] = end - start ;
  //   }

  // av = average (experiments) ;

  // printf ("Complex Simple Precision - SSE");
  // printf ("\t%Ld cycles\n", av-residu) ;
  // printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  // /**/

  // init_vector_dcomplexe (g, zvalue) ;
  // init_vector_dcomplexe (h, zvalue) ;

  // for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  //   {
  //     start = _rdtsc () ;


  //     end = _rdtsc () ;
  //     experiments [exp] = end - start ;
  //   }

  // av = average (experiments) ;

  // printf ("Complex Double Precision - sequential");
  // printf ("\t%Ld cycles\n", av-residu) ;
  // printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  // /**/

  // init_vector_dcomplexe (g, zvalue) ;
  // init_vector_dcomplexe (h, zvalue) ;

  // for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  //   {
  //     start = _rdtsc () ;


  //     end = _rdtsc () ;
  //     experiments [exp] = end - start ;
  //   }

  // av = average (experiments) ;

  // printf ("Complex Double Precision - OMP");
  // printf ("\t%Ld cycles\n", av-residu) ;
  // printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  // /**/

  // init_vector_dcomplexe (g, zvalue) ;
  // init_vector_dcomplexe (h, zvalue) ;

  // for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  //   {
  //     start = _rdtsc () ;


  //     end = _rdtsc () ;
  //     experiments [exp] = end - start ;
  //   }

  // av = average (experiments) ;

  // printf ("Complex Double Precision - SSE");
  // printf ("\t%Ld cycles\n", av-residu) ;
  // printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  // printf ("=============== GEMM ===============\n") ;

  // init_vector_float (a, 1.0) ;
  // init_vector_float (b, 2.0) ;

  // for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  //   {
  //     start = _rdtsc () ;


  //     end = _rdtsc () ;
  //     experiments [exp] = end - start ;
  //   }

  // av = average (experiments) ;

  // printf ("Simple Precision - sequential");
  // printf ("\t%Ld cycles\n", av-residu) ;
  // printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  // /**/

  // init_vector_float (a, 1.0) ;
  // init_vector_float (b, 2.0) ;

  // for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  //   {
  //     start = _rdtsc () ;


  //     end = _rdtsc () ;
  //     experiments [exp] = end - start ;
  //   }

  // av = average (experiments) ;

  // printf ("Simple Precision - OMP");
  // printf ("\t%Ld cycles\n", av-residu) ;
  // printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  // /**/

  // init_vector_float (a, 1.0) ;
  // init_vector_float (b, 2.0) ;

  // for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  //   {
  //     start = _rdtsc () ;


  //     end = _rdtsc () ;
  //     experiments [exp] = end - start ;
  //   }

  // av = average (experiments) ;

  // printf ("Simple Precision - SSE");
  // printf ("\t%Ld cycles\n", av-residu) ;
  // printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  // /**/

  // init_vector_double (c, 1.0) ;
  // init_vector_double (d, 2.0) ;

  // for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  //   {
  //     start = _rdtsc () ;


  //     end = _rdtsc () ;
  //     experiments [exp] = end - start ;
  //   }

  // av = average (experiments) ;

  // printf ("Double Precision - sequential");
  // printf ("\t%Ld cycles\n", av-residu) ;
  // printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  // /**/

  // init_vector_double (c, 1.0) ;
  // init_vector_double (d, 2.0) ;

  // for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  //   {
  //     start = _rdtsc () ;


  //     end = _rdtsc () ;
  //     experiments [exp] = end - start ;
  //   }

  // av = average (experiments) ;

  // printf ("Double Precision - OMP");
  // printf ("\t%Ld cycles\n", av-residu) ;
  // printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  // /**/

  // init_vector_double (c, 1.0) ;
  // init_vector_double (d, 2.0) ;

  // for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  //   {
  //     start = _rdtsc () ;


  //     end = _rdtsc () ;
  //     experiments [exp] = end - start ;
  //   }

  // av = average (experiments) ;

  // printf ("Double Precision - SSE");
  // printf ("\t%Ld cycles\n", av-residu) ;
  // printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  // /**/

  // init_vector_vcomplexe (e, cvalue) ;
  // init_vector_vcomplexe (f, cvalue) ;

  // for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  //   {
  //     start = _rdtsc () ;


  //     end = _rdtsc () ;
  //     experiments [exp] = end - start ;
  //   }

  // av = average (experiments) ;

  // printf ("Complex Simple Precision - sequential");
  // printf ("\t%Ld cycles\n", av-residu) ;
  // printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  // /**/

  // init_vector_vcomplexe (e, cvalue) ;
  // init_vector_vcomplexe (f, cvalue) ;

  // for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  //   {
  //     start = _rdtsc () ;


  //     end = _rdtsc () ;
  //     experiments [exp] = end - start ;
  //   }

  // av = average (experiments) ;

  // printf ("Complex Simple Precision - OMP");
  // printf ("\t%Ld cycles\n", av-residu) ;
  // printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  // /**/

  // init_vector_vcomplexe (e, cvalue) ;
  // init_vector_vcomplexe (f, cvalue) ;

  // for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  //   {
  //     start = _rdtsc () ;


  //     end = _rdtsc () ;
  //     experiments [exp] = end - start ;
  //   }

  // av = average (experiments) ;

  // printf ("Complex Simple Precision - SSE");
  // printf ("\t%Ld cycles\n", av-residu) ;
  // printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  // /**/

  // init_vector_dcomplexe (g, zvalue) ;
  // init_vector_dcomplexe (h, zvalue) ;

  // for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  //   {
  //     start = _rdtsc () ;


  //     end = _rdtsc () ;
  //     experiments [exp] = end - start ;
  //   }

  // av = average (experiments) ;

  // printf ("Complex Double Precision - sequential");
  // printf ("\t%Ld cycles\n", av-residu) ;
  // printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  // /**/

  // init_matrix_dcomplexe (gm, zvalue) ;
  // init_matrix_dcomplexe (hm, zvalue) ;

  // for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  //   {
  //     start = _rdtsc () ;


  //     end = _rdtsc () ;
  //     experiments [exp] = end - start ;
  //   }

  // av = average (experiments) ;

  // printf ("Complex Double Precision - OMP");
  // printf ("\t%Ld cycles\n", av-residu) ;
  // printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  // /**/

  // init_matrix_dcomplexe (gm, zvalue) ;
  // init_matrix_dcomplexe (hm, zvalue) ;

  // for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  //   {
  //     start = _rdtsc () ;


  //     end = _rdtsc () ;
  //     experiments [exp] = end - start ;
  //   }

  // av = average (experiments) ;

  // printf ("Complex Double Precision - SSE");
  // printf ("\t%Ld cycles\n", av-residu) ;
  // printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));


  // printf ("=============================== TRIS ===============================\n") ;

  // struct timeval t1;
  // struct timeval t2;

  // unsigned long long tresidu, t ;
  // gettimeofday(&t1, NULL);
  // gettimeofday(&t2, NULL);
  // tresidu = (t2.tv_sec-t1.tv_sec)*1000000LL + t2.tv_usec-t1.tv_usec;
  // //printf("Residu time = %llu ms\n", tresidu);

  // printf ("=============== QUICKSORT ===============\n") ;

  // gettimeofday(&t1, NULL);
  // // do
  // gettimeofday(&t2, NULL);
  // t = ((t2.tv_sec-t1.tv_sec)*1000000LL + t2.tv_usec-t1.tv_usec) - tresidu;
  // printf ("Sequential : %llu ms\n", t);

  // /**/

  // gettimeofday(&t1, NULL);
  // // do
  // gettimeofday(&t2, NULL);
  // t = ((t2.tv_sec-t1.tv_sec)*1000000LL + t2.tv_usec-t1.tv_usec) - tresidu;
  // printf ("OMP : %llu ms\n", t);


  // printf ("=============== BULLE ===============\n") ;

  // gettimeofday(&t1, NULL);
  // // do
  // gettimeofday(&t2, NULL);
  // t = ((t2.tv_sec-t1.tv_sec)*1000000LL + t2.tv_usec-t1.tv_usec) - tresidu;
  // printf ("Sequential : %llu ms\n", t);

  // /**/

  // gettimeofday(&t1, NULL);
  // // do
  // gettimeofday(&t2, NULL);
  // t = ((t2.tv_sec-t1.tv_sec)*1000000LL + t2.tv_usec-t1.tv_usec) - tresidu;
  // printf ("OMP : %llu ms\n", t);


  // printf ("=============== FUSION ===============\n") ;

  // gettimeofday(&t1, NULL);
  // // do
  // gettimeofday(&t2, NULL);
  // t = ((t2.tv_sec-t1.tv_sec)*1000000LL + t2.tv_usec-t1.tv_usec) - tresidu;
  // printf ("Sequential : %llu ms\n", t);

  // /**/

  // gettimeofday(&t1, NULL);
  // // do
  // gettimeofday(&t2, NULL);
  // t = ((t2.tv_sec-t1.tv_sec)*1000000LL + t2.tv_usec-t1.tv_usec) - tresidu;
  // printf ("OMP : %llu ms\n", t);


  return 0;
}

