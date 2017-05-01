#include <stdio.h>
#include <time.h>

/*
  Mesure des cycles
*/

#include <x86intrin.h>


#define NBEXPERIMENTS    102
#define VECSIZE 50 //taille du tableau à trier


static long long unsigned int experiments [NBEXPERIMENTS] ;

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

void vector_print (int V[])
{
  register unsigned int i ;

  for (i = 0; i < VECSIZE; i++)
    printf ("%f ", V[i]) ;
  printf ("\n") ;
  
  return ;
}

int* generate_random_vector(int size){
  int v = malloc(size * sizeof(int));

  srand(time(null));

  for(int i=0; i<size; i++){
    v[i] = rand();
  }

  return v;
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

  printf ("<==== Tri Quick séquentiel ====>\n") ;
  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

         qsort_n();

      end = _rdtsc () ;
      
      experiments [exp] = end - start ;
    }
  
  av = average (experiments) ;

  printf ("Résultats : \t %Ld \n", av-residu) ;

  printf ("<==== Tri Quick parallèl ====>\n") ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      
      start = _rdtsc () ;

          qsort_parallel();
     
      end = _rdtsc () ;
      
      experiments [exp] = end - start ;
    }
  
  av = average (experiments) ;

  printf ("Résultats: \t %Ld \n", av-residu) ;  
}