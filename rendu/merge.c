void merge_tabs(int sorted[], int left[], int left_size, int right[], int right_size){

	register int ig = 0;
	
	register int i2 = 0;
	register int i1 = 0;
	register int ia = ig;
		
	while(i1<right_size && i2<left_size){
			if(left[i2] <= right[i1]){
				sorted[ia] = left[i2];
				i2++;
			} else {
				sorted[ia] = right[i1];
				i1++;
			}
			ia++;
		}

		// remaining
		while(i2<left_size){
			sorted[ia] = left[i2];
			i2++;
			ia++;
		}

		while(i1<right_size){
			sorted[ia] = right[i1];
			i1++;
			ia++;
		}

}


void merge(int a[], int ig, int id, int m){
	int agSize = m-ig+1;
	int adSize = id - m;

	int ad[adSize];
	int ag[agSize];

	for(int i=0; i<(adSize); i++)
		ad[i] = a[m+1+i];

	for(int i=0; i<(agSize); i++)
		ag[i] = a[ig+i];

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