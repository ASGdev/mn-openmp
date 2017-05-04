#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define CHUNK_SIZE 6
#define THREAD_NUM 0
#define ARRAY_SIZE 10
#define DEBUG 1

void bulle_seq (int * tab){
    // séquentiel
    for(int i = ARRAY_SIZE - 1; i>1; i--){
        for(int j=0; j<=i-1; j++){
            if(tab[j+1] < tab[j]){
                int t = tab[j+1];
                tab[j+1] = tab[j];
                tab[j] = t;
            }
        }
    }
}

void bulle_omp (int * tab, int num_t){
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
                if(tab[j+base+1] < tab[j+base]){
                    int t = tab[j+base+1];
                    tab[j+base+1] = tab[j+base];
                    tab[j+base] = t;
                }
            }
        }
    }
}

int main(){
	int a[ARRAY_SIZE] = {8, 5, 2, 6, 8, 7, 9, 2, 1, 4};
	// to optimize

    bulle_seq(a);

	for(int i = 0; i<ARRAY_SIZE; i++)
		printf("%d ", a[i]);

	// parallel
	int b[ARRAY_SIZE] = {8, 5, 2, 6, 8, 7, 9, 2, 1, 4};
	int num_t = 2;
	omp_set_dynamic(0);
	omp_set_num_threads(2);

    bulle_omp(b, num_t);

    for(int i = 0; i<ARRAY_SIZE-1; i++){
        if(b[i]>b[i+1]){
            bulle_seq(b);
        }
    }

    printf("TABLEAU TRIÉ : ");
    for(int i = 0; i<ARRAY_SIZE; i++){
        printf("%d ", b[i]);
    }

	// smooth it
	// à finir -> regarder les frontières pour vérifier que c'est trié, sinon on retrie sur le tableau b pré-trié(goto)


}