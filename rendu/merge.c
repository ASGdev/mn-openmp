void merge(int sorted[], int left[], int left_size, int right[], int right_size){

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
