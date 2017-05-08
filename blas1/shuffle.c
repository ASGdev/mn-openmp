#include "mnblas.h"
#include <x86intrin.h>
#include <stdio.h>

typedef float float4 [4] __attribute__ ((aligned (16))) ;
typedef double double2 [2] __attribute__ ((aligned (16))) ;

int main(){
	float4 f = {0.0, 3.0, 0.0, 4.0};
	__m128 t = _mm_load_ps(f);

	__m128 ts = _mm_shuffle_ps(t, t, _MM_SHUFFLE(0, 3, 0, 1));
	float4 sf;
	_mm_store_ps(sf, ts);

	for(int i =0; i<4; i++){
		printf("%f ", sf[i]);
	}
	printf("\n");
}