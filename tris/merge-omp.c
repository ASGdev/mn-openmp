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
    int *tempArray = malloc(ARRAY_SIZE * sizeof(int));
    int j = CHUNK_SIZE;
    int i = 0;
    int cpt = 0;
    while(cpt!=ARRAY_SIZE){
        if(tab[i] == INT_MAX) {
            printf("INT_MAX\n");
            tempArray[cpt] = tab[j];
            tab[j] = INT_MAX;
        }else if(tab[i] > tab[j]){
            printf("I > J\n");
            tempArray[cpt] = tab[j];
            tab[j] = INT_MAX;
        } else {
            printf("SINON\n");
            tempArray[cpt] = tab[i];
            tab[i] = INT_MAX;
        }
        cpt++;
        i++;
        j++;
        printf("i = %d j = %d\n", i , j);
        printf("ARRAY_SIZE %d\n", ARRAY_SIZE);
        printf("CHUNK_SIZE %d\n", CHUNK_SIZE);

        if (i+1 == ARRAY_SIZE){
            return tempArray;
        }
        if (j == ARRAY_SIZE){
            j = i+1;
        }

        // if (i == CHUNK_SIZE && j != ARRAY_SIZE){
        // // if (cpt != ARRAY_SIZE){
        //     printf("PAS FINI \n");
        //     i = j;
        //     j = j + CHUNK_SIZE;
        // }
    }
    return tempArray;
}


// int* merge (int* tab){
//     int *tempArray = malloc(ARRAY_SIZE * sizeof(int));
//     int j = CHUNK_SIZE;
//     int i = 0;
//     int cpt = 0;
//     while( i != CHUNK_SIZE && j != ARRAY_SIZE){
//         if(tab[i] == INT_MAX) {
//             printf("INT_MAX\n");
//             tempArray[cpt] = tab[j];
//             tab[j] = INT_MAX;
//         }else if(tab[i] > tab[j]){
//             printf("I > J\n");
//             tempArray[cpt] = tab[j];
//             tab[j] = INT_MAX;
//         } else {
//             printf("SINON\n");
//             tempArray[cpt] = tab[i];
//             tab[i] = INT_MAX;
//         }
//         cpt++;
//         i++;
//         j++;
//         printf("i = %d j = %d\n", i , j);
//         printf("ARRAY_SIZE %d\n", ARRAY_SIZE);
//         printf("CHUNK_SIZE %d\n", CHUNK_SIZE);
//         if (i == CHUNK_SIZE && j != ARRAY_SIZE){
//         // if (cpt != ARRAY_SIZE){
//             printf("IF \n");
//             i = j;
//             j = j + CHUNK_SIZE;
//         }
//     }
//     return tempArray;
// }


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
