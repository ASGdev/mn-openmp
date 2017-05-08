#include <stdio.h>
#include <omp.h>

#include <x86intrin.h>

#define NBEXPERIMENTS    22
static long long unsigned int experiments [NBEXPERIMENTS] ;


#define N              512
#define TILE           16

typedef double vector [N] ;

typedef double matrix [N][N] ;

static vector a, b, c ;
static matrix M1, M2 ;


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


void init_vector (vector X, const double val)
{
  register unsigned int i ;

  for (i = 0 ; i < N; i++)
    X [i] = val ;

  return ;
}

void init_matrix (matrix X, const double val)
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

  
void print_vectors (vector X, vector Y)
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
  
  double r ;

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
	
  /*
    Vector Initialization
  */

  init_vector (a, 1.0) ;
  init_vector (b, 2.0) ;

  /*
    print_vectors (a, b) ;
  */
    

  printf ("=============== AXPY ===============\n") ;

  init_vector (a, 1.0) ;
  init_vector (b, 2.0) ;

  /*
    print_vectors (a, b) ;
  */
  
  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

         add_vectors1 (c, a, b) ;
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("OpenMP static loop %Ld cycles\n", av-residu) ;
  printf ("%3.6f GFLOP/s\n", ((double) 2 * (double) 0.22)/ ((double) (av - residu)));

  init_vector (a, 1.0) ;
  init_vector (b, 2.0) ;
	
  /*
    print_vectors (a, b) ;
  */
  
  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

         add_vectors2 (c, a, b) ;
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("OpenMP dynamic loop %Ld cycles\n", av-residu) ;

  printf ("==============================================================\n") ;

  printf ("=============== DOT ===============\n") ;

  init_vector (a, 1.0) ;
  init_vector (b, 2.0) ;
  
  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      r = dot1 (a, b) ;
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("OpenMP static loop dot %e: %Ld cycles\n", r, av-residu) ;

  init_vector (a, 1.0) ;
  init_vector (b, 2.0) ;

   for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

         r = dot2 (a, b) ;
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;
  
  printf ("OpenMP dynamic loop dot %e: %Ld cycles\n", r, av-residu) ;

  init_vector (a, 1.0) ;
  init_vector (b, 2.0) ;

   for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

         r = dot3 (a, b) ;
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;
  
  printf ("OpenMP static unrolled loop dot %e: %Ld cycles\n", r, av-residu) ;
  
  printf ("=============================================================\n") ;

  printf ("======================== Mult Mat Vector =====================================\n") ;
  
  init_matrix (M1, 1.0) ;
  init_vector (b, 2.0) ;

   for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

           mult_mat_vect0 (M1, b, a) ; 
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;
  
  printf ("OpenMP static loop MultMatVect0: %Ld cycles\n", av-residu) ;

  init_matrix (M1, 1.0) ;
  init_vector (b, 2.0) ;

   for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

           mult_mat_vect1 (M1, b, a) ; 
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;
 
  printf ("OpenMP static loop MultMatVect1: %Ld cycles\n", av-residu) ;

  init_matrix (M1, 1.0) ;
  init_vector (b, 2.0) ;

   for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

           mult_mat_vect2 (M1, b, a) ; 
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("OpenMP static loop MultMatVect2: %Ld cycles\n", av-residu) ;

  init_matrix (M1, 1.0) ;
  init_vector (b, 2.0) ;

   for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

           mult_mat_vect3 (M1, b, a) ; 
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;
 
  printf ("OpenMP static loop MultMatVect3: %Ld cycles\n", av-residu) ;

  printf ("==============================================================\n") ;

  printf ("======================== Mult Mat Mat =====================================\n") ;
  
  init_matrix (M1, 1.0) ;
  init_matrix (M2, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mult_mat_mat0 (M1, M2, M2) ; 
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Sequential matrice vector multiplication:\t %Ld cycles\n", av-residu) ;
  
  init_matrix (M1, 1.0) ;
  init_matrix (M2, 2.0) ;


  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mult_mat_mat1 (M1, M2, M2) ; 
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("OpenMP static loop MultMatVect1:\t\t %Ld cycles\n", av-residu) ;

  init_matrix (M1, 1.0) ;
  init_matrix (M2, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mult_mat_mat2 (M1, M2, M2) ; 
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;
    

  printf ("OpenMP unrolled loop MultMatMat2: %Ld cycles\n", av-residu) ;

  init_matrix (M1, 1.0) ;
  init_matrix (M2, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mult_mat_mat3 (M1, M2, M2) ; 
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("OpenMP Tiled loop MultMatMat3: %Ld cycles\n", av-residu) ;
  
  return 0;
  
}

