#include <stdio.h>
#include <omp.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>

#include "mnblas.h"
#include "tris.h"

#define NBEXPERIMENTS    22
static long long unsigned int experiments [NBEXPERIMENTS] ;

#define N              512
#define TILE           16
#define SORT_ARRAY_SIZE 512
#define FREQ_GHZ 0.22

typedef float floatVector [N] ;
typedef double doubleVector [N] ;
typedef vcomplexe vcompVector [N] ;
typedef dcomplexe dcompVector [N] ;

vcomplexe cvalue = {1.0, 1.0};
dcomplexe zvalue = {1.0, 1.0};

typedef float floatMatrix [N][N] ;
typedef double doubleMatrix [N][N];
typedef vcomplexe vcompMatrix [N*N];
typedef dcomplexe dcompMatrix [N*N];


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
  register unsigned int i;

  for (i = 0; i < N*N; i+=2){
    *(X+i) = val;
  }
}

void init_matrix_dcomplexe (dcompMatrix X, const dcomplexe val)
{
  register unsigned int i;

  for (i = 0; i < N; i++){
    for (i = 0; i < N*N; i+=2){
      *(X+i) = val;
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

void random_fill(int tab[], int size, int max){
  srand(time(NULL));
  for(int i = 0; i<size; i++)
    tab[i] = rand()%max;
}


/* =============== Tests ===================*/
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
  vcompMatrix *em = malloc(N * N * sizeof(vcomplexe));
  vcompMatrix *fm = malloc(N * N * sizeof(vcomplexe));
  dcompMatrix *gm = malloc(N * N * sizeof(dcomplexe));
  dcompMatrix *hm = malloc(N * N * sizeof(dcomplexe));


  printf ("=============================== BLAS ===============================\n") ;
  printf ("\n=============== AXPY ===============\n") ;

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
  printf ("\t%3.6f GFLOP/s\n", (double) N / ((double) (av - residu) * (double) FREQ_GHZ));

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
  printf ("\t%3.6f GFLOP/s\n", (double) N / ((double) (av - residu) * (double) FREQ_GHZ));

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
  printf ("\t%3.6f GFLOP/s\n", (double) N / ((double) (av - residu) * (double) FREQ_GHZ));

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

  printf ("\nDouble Precision - sequential\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (double) N / ((double) (av - residu) * (double) FREQ_GHZ));

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
  printf ("\t%3.6f GFLOP/s\n", (double) N / ((double) (av - residu) * (double) FREQ_GHZ));

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
  printf ("\t%3.6f GFLOP/s\n", (double) N / ((double) (av - residu) * (double) FREQ_GHZ));

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

  printf ("\nComplex Simple Precision - sequential\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (double) N / ((double) (av - residu) * (double) FREQ_GHZ));

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
  printf ("\t%3.6f GFLOP/s\n", (double) N / ((double) (av - residu) * (double) FREQ_GHZ));

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
  printf ("\t%3.6f GFLOP/s\n", (double) N / ((double) (av - residu) * (double) FREQ_GHZ));

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
  printf ("\t%3.6f GFLOP/s\n", (double) N / ((double) (av - residu) * (double) FREQ_GHZ));

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
  printf ("\t%3.6f GFLOP/s\n", (double) N / ((double) (av - residu) * (double) FREQ_GHZ));

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
  printf ("\t%3.6f GFLOP/s\n", (double) N / ((double) (av - residu) * (double) FREQ_GHZ));

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
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) N) / ((double) (av- residu) * (double) FREQ_GHZ));

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
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) N) / ((double) (av- residu) * (double) FREQ_GHZ));

  /**/

  init_vector_float (a, 1.0) ;
  init_vector_float (b, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_sdot_vec(N, a, 0, b, 0);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Simple Precision - SSE\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) N) / ((double) (av- residu) * (double) FREQ_GHZ));

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

  printf ("\nDouble Precision - sequential\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) N) / ((double) (av- residu) * (double) FREQ_GHZ));

  /**/

  init_vector_double (c, 1.0) ;
  init_vector_double (d, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;
      // return arithmetic exception
      //mncblas_ddot_omp(N, c, 0, d, 0);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Double Precision - OMP\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) N) / ((double) (av- residu) * (double) FREQ_GHZ));

  /**/

  init_vector_double (c, 1.0) ;
  init_vector_double (d, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_ddot_vec(N, c, 0, d, 0);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Double Precision - SSE\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) N) / ((double) (av- residu) * (double) FREQ_GHZ));

  /**/

  init_vector_vcomplexe (e, cvalue) ;
  init_vector_vcomplexe (f, cvalue) ;
  vcomplexe rdotu;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_cdotu_sub(N, &e, 0, &f, 0, &rdotu);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("\nComplex Simple Precision - sequential\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) N) / ((double) (av- residu) * (double) FREQ_GHZ));

  /**/

  init_vector_vcomplexe (e, cvalue) ;
  init_vector_vcomplexe (f, cvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_cdotu_sub_omp(N, &e, 0, &f, 0, &rdotu);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Simple Precision - OMP\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) N) / ((double) (av- residu) * (double) FREQ_GHZ));

  /**/

  init_vector_vcomplexe (e, cvalue) ;
  init_vector_vcomplexe (f, cvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_cdotu_sub_vec(N, &e, 0, &f, 0, &rdotu);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Simple Precision - SSE\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) N) / ((double) (av- residu) * (double) FREQ_GHZ));

  /**/
  dcomplexe zdotu;

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_zdotu_sub(N, &g, 0, &h, 0, &zdotu);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("\nComplex Double Precision - sequential\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) N) / ((double) (av- residu) * (double) FREQ_GHZ));

  /**/

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_zdotu_sub_omp(N, &g, 0, &h, 0, &zdotu);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Double Precision - OMP\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) N) / ((double) (av- residu) * (double) FREQ_GHZ));

  /**/

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_zdotu_sub_vec(N, &g, 0, &h, 0, &zdotu);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Double Precision - SSE\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", ((double) 2 * (double) N) / ((double) (av- residu) * (double) FREQ_GHZ));

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

  printf ("Simple Precision - sequential\n");
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(float) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

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
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(float) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

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
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(float) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

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
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(double) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

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
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(double) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

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
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(double) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

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
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(vcomplexe) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

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
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(vcomplexe) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

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
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(vcomplexe) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

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
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(dcomplexe) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

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
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(dcomplexe) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

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
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(dcomplexe) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

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

  printf ("Simple Precision - sequential\n");
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(float) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

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
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(float) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

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
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(float) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

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
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(double) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

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
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(double) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

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
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(double) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

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
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(vcomplexe) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

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
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(vcomplexe) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

  /**/

  init_vector_vcomplexe (e, cvalue) ;
  init_vector_vcomplexe (f, cvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_cswap_vec(N, e, 0, f, 0);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Simple Precision - SSE\n");
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(vcomplexe) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

  /**/

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_zswap(N, g, 0, h, 0);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("\nComplex Double Precision - sequential\n");
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(dcomplexe) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

  /**/

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_zswap_omp(N, g, 0, h, 0);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Double Precision - OMP\n");
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(dcomplexe) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

  /**/

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_zswap_vec(N, g, 0, h, 0);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Double Precision - SSE\n");
  printf ("\t%3.6f Go/s\n", (float)((((float)N * (float)sizeof(dcomplexe) *  (float)NBEXPERIMENTS ) / (1024.0 * 1024.0 * 1024.0)) / ((float)(end - start) / (float)CLOCKS_PER_SEC)));

  printf ("\n=============== GEMV ===============\n") ;

  init_vector_float (a, 1.0) ;
  init_vector_float (b, 1.0) ;
  init_matrix_float (bm, 2.0) ;
  float alphafgmv = 1.0, betafgmv = 2.0;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_sgemv(101, 111, N, N, alphafgmv, *bm, 0, a, 1, betafgmv, b, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("\nSimple Precision - sequential\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  /**/

  init_vector_float (a, 1.0) ;
  init_matrix_float (bm, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_sgemv_omp(101, 111, N, N, alphafgmv, *bm, 0, a, 1, betafgmv, b, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Simple Precision - OMP\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  /**/

  init_vector_float (a, 1.0) ;
  init_matrix_float (bm, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_sgemv_vec(101, 111, N, N, alphafgmv, *bm, 0, a, 1, betafgmv, b, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Simple Precision - SSE\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  /**/

  init_vector_double (c, 1.0) ;
  init_vector_double (d, 1.0) ;
  init_matrix_double (cm, 2.0) ;
  double alphadgmv = 1.0, betadgmv = 2.0;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_dgemv_vec(101, 111, N, N, alphadgmv, *cm, 0, c, 1, betadgmv, d, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("\nDouble Precision - sequential\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  /**/

  init_vector_double (c, 1.0) ;
  init_vector_double (d, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_dgemv_vec(101, 111, N, N, alphadgmv, *cm, 0, c, 1, betadgmv, d, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Double Precision - OMP\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  /**/

  init_vector_double (c, 1.0) ;
  init_vector_double (d, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_dgemv_vec(101, 111, N, N, alphadgmv, *cm, 0, c, 1, betadgmv, d, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Double Precision - SSE\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  /**/

  init_vector_vcomplexe (e, cvalue) ;
  init_vector_vcomplexe (f, cvalue) ;
  init_matrix_vcomplexe (*em, cvalue);
  vcomplexe alphacgmv = {1.0, 2.0};
  vcomplexe betacgmv = {1.0, 2.0};

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_cgemv(101, 111, N, N, &alphacgmv, &em, 0, &e, 1, &betacgmv, &f, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("\nComplex Simple Precision - sequential\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  /**/

  init_vector_vcomplexe (e, cvalue) ;
  init_vector_vcomplexe (f, cvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_cgemv_omp(101, 111, N, N, &alphacgmv, &em, 0, &e, 1, &betacgmv, &f, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Simple Precision - OMP\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  /**/

  init_vector_vcomplexe (e, cvalue) ;
  init_vector_vcomplexe (f, cvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_cgemv_vec(101, 111, N, N, &alphacgmv, &em, 0, &e, 1, &betacgmv, &f, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Simple Precision - SSE\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  /**/

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;
  init_matrix_dcomplexe (*gm, zvalue);
  dcomplexe alphazgmv = {1.0, 2.0};
  dcomplexe betazgmv = {1.0, 2.0};

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_zgemv(101, 111, N, N, &alphazgmv, &gm, 0, &g, 1, &betazgmv, &h, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("\nComplex Double Precision - sequential\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  /**/

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_zgemv(101, 111, N, N, &alphazgmv, &gm, 0, &g, 1, &betazgmv, &h, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Double Precision - OMP\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  /**/

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_zgemv(101, 111, N, N, &alphazgmv, &gm, 0, &g, 1, &betazgmv, &h, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Double Precision - SSE\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  printf ("\n=============== GEMM ===============\n") ;

  floatMatrix cfgemm;

  init_vector_float (a, 1.0) ;
  init_vector_float (b, 2.0) ;
  init_matrix_float (am, 1.0);
  init_matrix_float (bm, 1.0);
  init_matrix_float (cfgemm, 1.0);
  alphaf = 1.0;
  float betaf = 1.0;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_sgemm (101, 111, 111, N, N, N, alphaf, *am, 1, *bm, 1, betaf, *cfgemm, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Simple Precision - sequential\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  /**/

  init_vector_float (a, 1.0) ;
  init_vector_float (b, 2.0) ;
  init_matrix_float (am, 1.0);
  init_matrix_float (bm, 1.0);
  init_matrix_float (cfgemm, 1.0);

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_sgemm_omp (101, 111, 111, N, N, N, alphaf, *am, 1, *bm, 1, betaf, *cfgemm, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Simple Precision - OMP\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  /**/

  init_vector_float (a, 1.0) ;
  init_vector_float (b, 2.0) ;
  init_matrix_float (am, 1.0);
  init_matrix_float (bm, 1.0);
  init_matrix_float (cfgemm, 1.0);

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_sgemm_vec (101, 111, 111, N, N, N, alphaf, *am, 1, *bm, 1, betaf, *cfgemm, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Simple Precision - SSE\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  /**/
  double alphadgmm = 1.0;
  double betadgmm = 2.0;
  doubleMatrix *dgemm = malloc(N * N * sizeof(double));

  init_vector_double (c, 1.0) ;
  init_vector_double (d, 2.0) ;
  init_matrix_double (cm, 1.0);
  init_matrix_double (dm, 1.0);
  init_matrix_double (*dgemm, 1.0);

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_dgemm (101, 111, 111, N, N, N, alphadgmm, *cm, 1, *dm, 1, betadgmm, (double *)dgemm, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("\nDouble Precision - sequential\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  /**/

  init_vector_double (c, 1.0) ;
  init_vector_double (d, 2.0) ;
  init_matrix_double (cm, 1.0);
  init_matrix_double (dm, 1.0);
  init_matrix_double (*dgemm, 1.0);

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_dgemm_omp (101, 111, 111, N, N, N, alphadgmm, *cm, 1, *dm, 1, betadgmm, (double *)dgemm, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Double Precision - OMP\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  /**/

  init_vector_double (c, 1.0) ;
  init_vector_double (d, 2.0) ;
  init_matrix_double (cm, 1.0);
  init_matrix_double (dm, 1.0);
  init_matrix_double (*dgemm, 1.0);

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      // take a long time + segfault - malloc error
      //mncblas_dgemm_vec (101, 111, 111, N, N, N, alphadgmm, *cm, 1, *dm, 1, betadgmm, (double *)dgemm, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Double Precision - SSE\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  /**/

  vcomplexe alphacgmm = {1.0, 1.0};
  vcomplexe betacgmm = {2.0, 2.0};
  vcompMatrix *ccgemm = malloc(N * N * sizeof(vcomplexe));

  init_vector_vcomplexe (e, cvalue) ;
  init_vector_vcomplexe (f, cvalue) ;
  init_matrix_vcomplexe (*em, cvalue);
  init_matrix_vcomplexe (*fm, cvalue);
  init_matrix_vcomplexe (*ccgemm, cvalue);

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;
      // segfault
      //mncblas_cgemm (101, 111, 111, N, N, N, &alphacgmm, *em, 1, *fm, 1, &betacgmm, (float *)ccgemm, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("\nComplex Simple Precision - sequential\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  /**/

  init_vector_vcomplexe (e, cvalue) ;
  init_vector_vcomplexe (f, cvalue) ;
  init_matrix_vcomplexe (*em, cvalue);
  init_matrix_vcomplexe (*fm, cvalue);
  init_matrix_vcomplexe (*ccgemm, cvalue);

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_cgemm_omp (101, 111, 111, N, N, N, &alphacgmm, *em, 1, *fm, 1, &betacgmm, (float *)ccgemm, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Simple Precision - OMP\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  /**/

  init_vector_vcomplexe (e, cvalue) ;
  init_vector_vcomplexe (f, cvalue) ;
  init_matrix_vcomplexe (*em, cvalue);
  init_matrix_vcomplexe (*fm, cvalue);
  init_matrix_vcomplexe (*ccgemm, cvalue);

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_cgemm_vec (101, 111, 111, N, N, N, &alphacgmm, *em, 1, *fm, 1, &betacgmm, (float *)ccgemm, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Simple Precision - SSE\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  /**/

  dcomplexe alphazgmm = {1.0, 1.0};
  dcomplexe betazgmm = {2.0, 2.0};
  dcompMatrix *czgemm = malloc(N * N* sizeof(dcomplexe));

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;
  init_matrix_dcomplexe (*gm, zvalue);
  init_matrix_dcomplexe (*hm, zvalue);
  init_matrix_dcomplexe (*czgemm, zvalue);

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_zgemm(101, 111, 111, N, N, N, &alphazgmm, *gm, 1, *hm, 1, &betazgmm, (float *)czgemm, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("\nComplex Double Precision - sequential\n");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  /**/

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;
  init_matrix_dcomplexe (*gm, zvalue);
  init_matrix_dcomplexe (*hm, zvalue);
  init_matrix_dcomplexe (*czgemm, zvalue);

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_zgemm_omp(101, 111, 111, N, N, N, &alphazgmm, *gm, 1, *hm, 1, &betazgmm, (float *)czgemm, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Double Precision - OMP");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));

  /**/

  init_vector_dcomplexe (g, zvalue) ;
  init_vector_dcomplexe (h, zvalue) ;
  init_matrix_dcomplexe (*gm, zvalue);
  init_matrix_dcomplexe (*hm, zvalue);
  init_matrix_dcomplexe (*czgemm, zvalue);

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mncblas_zgemm_vec(101, 111, 111, N, N, N, &alphazgmm, *gm, 1, *hm, 1, &betazgmm, (float *)czgemm, 1);

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Complex Double Precision - SSE");
  printf ("\t%Ld cycles\n", av-residu) ;
  printf ("\t%3.6f GFLOP/s\n", (((double) 2 * (double) N * (double) N * (double) N)) / ((double) (av - residu) * (double) FREQ_GHZ));


  printf ("\n=============================== TRIS ===============================\n") ;

  int tab[SORT_ARRAY_SIZE];
  random_fill(tab, SORT_ARRAY_SIZE, 20);

  printf ("=============== QUICKSORT ===============\n") ;

  start = _rdtsc () ;

  quick_sort(tab, SORT_ARRAY_SIZE);

  end = _rdtsc () ;
  printf ("Sequential : %3.6f s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  start = _rdtsc () ;

  quick_sort_omp(tab, SORT_ARRAY_SIZE);

  end = _rdtsc () ;
  printf ("Parallel : %3.6f s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);


  printf ("\n=============== BULLE ===============\n") ;

  start = _rdtsc () ;

  bubble_sort(tab, SORT_ARRAY_SIZE);

  end = _rdtsc () ;
  printf ("Sequential : %3.6f s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  start = _rdtsc () ;

  bubble_sort_omp(tab, SORT_ARRAY_SIZE);

  end = _rdtsc () ;
  printf ("Parallel : %3.6f s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);


  printf ("\n=============== FUSION ===============\n") ;

  start = _rdtsc () ;

  merge_sort(tab, 0, SORT_ARRAY_SIZE-1);

  end = _rdtsc () ;
  printf ("Sequential : %3.6f s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  /**/

  start = _rdtsc () ;

  merge_sort_omp(tab, 0, SORT_ARRAY_SIZE-1);

  end = _rdtsc () ;
  printf ("Parallel : %3.6f s\n", (float)(end - start) / (float)CLOCKS_PER_SEC);

  return 0;
}

