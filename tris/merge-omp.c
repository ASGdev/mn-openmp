#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <limits.h>

#define CHUNK_SIZE 6
#define THREAD_NUM 0
#define ARRAY_SIZE 10
#define DEBUG 1

int* merge (int* tab){
    int* tempArray;
    int j = CHUNK_SIZE;
    int i = 0;
    int cpt = 0;
    for(; i != CHUNK_SIZE; i++){
        while(j != ARRAY_SIZE){
            if(tab[i] == INT_MAX) {
                tempArray[cpt] = tab[j];
                tab[j] = INT_MAX;
                printf("INT_MAX\n");
            }else if(tab[i] > tab[j]){
                tempArray[cpt] = tab[j];
                tab[j] = INT_MAX;
                printf("I > J\n");
            } else {
                tempArray[cpt] = tab[i];
                tab[i] = INT_MAX;
                printf("SINON\n");
            }
            cpt++;
            j++;
            break;
        }
    }
    return tempArray;
}



int main(){
    int a[ARRAY_SIZE] = {2, 5, 6, 7, 8, 8, 1, 2, 4, 9};
    // to optimize

    // bulle_seq(a);
    int * b;
    b = merge(a);

    for(int i = 0; i<ARRAY_SIZE; i++)
        printf("%d ", b[i]);

    // // parallel
    // int b[ARRAY_SIZE] = {8, 5, 2, 6, 8, 7, 9, 2, 1, 4};
    // int num_t = 2;
    // omp_set_dynamic(0);
    // omp_set_num_threads(2);

    // bulle_omp(b, num_t);

    // printf("TABLEAU TRIÉ : ");
    // for(int i = 0; i<ARRAY_SIZE; i++){
    //     printf("%d ", b[i]);
    // }
}