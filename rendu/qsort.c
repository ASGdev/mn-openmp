#include <math.h>
#include <stdlib.h>
#include <omp.h>

#include "tris.h"

#define CHUNK_SIZE 6
#define THREAD_NUM 2
#define ARRAY_SIZE 10
#define DEBUG 1

//fonction de comparaison
int compare(const void *a, const void *b){
	return (*(int*)b - *(int*)a);
}

void quick_sort_omp(int tab[], int tab_size){

	// if(DEBUG)
	// 	printf("Num threads : %d\n", (int)ceil((double)ARRAY_SIZE/CHUNK_SIZE));

	//int num_t = (int)ceil((double)ARRAY_SIZE/CHUNK_SIZE);
	int num_t = THREAD_NUM;
	omp_set_dynamic(0);
	omp_set_num_threads(num_t);

	#pragma omp for schedule(static)
	for(int i=0; i<num_t; i++){
		// if(DEBUG)
		// 	printf("OMP num threads : %d\n", omp_get_num_threads());

		qsort((tab + CHUNK_SIZE * i), CHUNK_SIZE, sizeof(int), compare);
	}

}

void quick_sort(int tab[], int tab_size){
	qsort(tab, tab_size, sizeof(int), compare);
}
