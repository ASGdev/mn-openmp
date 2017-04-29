#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define CHUNK_SIZE 6
#define THREAD_NUM 0
#define ARRAY_SIZE 10
#define DEBUG 1

int main(){
	int a[ARRAY_SIZE] = {8, 5, 2, 6, 8, 7, 9, 2, 1, 4};
	// to optimize
	int sorted;

	// séquentiel
	for(int i = ARRAY_SIZE - 1; i>1; i--){
		for(int j=0; j<=i-1; j++){
			if(a[j+1] < a[j]){
				int t = a[j+1];
				a[j+1] = a[j];
				a[j] = t;
			}
		}
	}

	for(int i = 0; i<ARRAY_SIZE; i++)
		printf("%d ", a[i]);

	// parallel
	int b[ARRAY_SIZE] = {8, 5, 2, 6, 8, 7, 9, 2, 1, 4};
	int num_t = 2;
	omp_set_dynamic(0);
	omp_set_num_threads(2);

	int base;
	#pragma omp parallel for private(base)
	for(int o=0; o<num_t; o++){
		if(DEBUG)
			printf("OMP num threads : %d\n", omp_get_num_threads());

		base = (CHUNK_SIZE * o);
		printf("Base is %d\n", base);
		for(int i = CHUNK_SIZE - 1; i>1; i--){
			for(int j=0; j<=i-1; j++){
				// exceed array boundaries
				if(j+base+1 >= ARRAY_SIZE)
					break;
				if(b[j+base+1] < b[j+base]){
					int t = b[j+base+1];
					b[j+base+1] = b[j+base];
					b[j+base] = t;
				}
			}
		}
	}

	for(int i = 0; i<ARRAY_SIZE; i++)
		printf("%d ", b[i]);

	// smooth it
	// à finir -> regarder les frontières pour vérifier que c'est trié, sinon on retrie sur le tableau b pré-trié(goto)


}