#include <stdio.h>

int main(){

	int a[10];
	int ad[5] = {1, 2, 3, 4, 5};
	int ag[5] = {1, 2, 3, 4, 5};

	int adSize = 5;
	int agSize = 5;
	register int ig = 0;
	
		register int i2 = 0;
		register int i1 = 0;
		register int ia = ig;
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

	for(int i = 0; i<10; i++)
		printf("%d ", a[i]);
}
