#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define CHUNK_SIZE 6
#define TASKS_NUM 0
#define ARRAY_SIZE 10
#define DEBUG 1

void fusion(int a[], int ig, int id, int m){
	int agSize = m-ig+1;
	int adSize = id - m;

	int ad[adSize];
	int ag[agSize];

	// fill ad
	for(int i=0; i<(adSize); i++)
		ad[i] = a[m+1+i];

	for(int i=0; i<(agSize); i++)
		ag[i] = a[ig+i];

	// merge
	int i1 = 0;
	int i2 = 0;
	int ia = ig;
	while(i1<adSize && i2<agSize){
		if(ag[i2] <= ad[i1]){
			a[ia] = ag[i2];
			i2++;
		} else {
			a[ia] = ad[i1];
			i1++;
		}
		ia++;
	}

	// remaining
	while(i2<agSize){
		a[ia] = ag[i2];
		i2++;
		ia++;
	}

	while(i1<adSize){
		a[ia] = ad[i1];
		i1++;
		ia++;
	}

	

}

void tri(int a[], int ig, int id){

	if(ig < id){
		int m = (ig + id)/2;
		printf("Middle : %d\n", m);

		tri(a, ig, m);
		tri(a, m+1, id);

		fusion(a, ig, id, m);
	}

}

int main(){
	int a[] = {8, 5, 2, 6, 8, 7, 9, 2, 1, 4};

	tri(a, 0, ARRAY_SIZE-1);

	printf("\n");

	for(int i = 0; i<ARRAY_SIZE; i++)
		printf("%d ", a[i]);
}


