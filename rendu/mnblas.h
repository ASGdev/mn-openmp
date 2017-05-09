#define VEC_SIZE 4
#define NUM_PROC 2
#define NUM_THREADS 2


typedef struct {
  float REEL;
  float IMAG;
} vcomplexe;

typedef vcomplexe VCOMP [VEC_SIZE] ;

typedef struct {
  double REEL;
  double IMAG;
} dcomplexe;

typedef dcomplexe DCOMP [VEC_SIZE] ;

typedef float float4 [4] __attribute__ ((aligned (16))) ;
typedef double double2 [2] __attribute__ ((aligned (16))) ;


/* */

typedef enum {MNCblasRowMajor=101, MNCblasColMajor=102} MNCBLAS_LAYOUT ;
typedef enum {MNCblasNoTrans=111, MNCblasTrans=112, MNCblasConjTrans=113} MNCBLAS_TRANSPOSE ;
typedef enum {MNCblasUpper=121, MNCblasLower=122} MNCBLAS_UPLO ;
typedef enum {MNCblasNonUnit=131, MNCblasUnit=132} MNCBLAS_DIAG ;
typedef enum {MNCblasLeft=141, MNCblasRight=142} MNCBLAS_SIDE ;

/*

  BEGIN BLAS1 copy

*/


void mncblas_scopy (
		    const int N, const float *X, const int incX, 
		    float *Y, const int incY
		   ) ;

void mncblas_scopy_omp (
		    const int N, const float *X, const int incX, 
		    float *Y, const int incY
		   ) ;

void mncblas_scopy_vec (
		    const int N, const float *X, const int incX, 
		    float *Y, const int incY
		   ) ;

void mncblas_dcopy (
		    const int N, const double *X, const int incX, 
		    double *Y, const int incY
		   ) ;

void mncblas_dcopy_omp (
		    const int N, const double *X, const int incX, 
		    double *Y, const int incY
		   ) ;

void mncblas_dcopy_vec (
		    const int N, const double *X, const int incX, 
		    double *Y, const int incY
		   ) ;


void mncblas_ccopy (
		    const int N, const void *X, const int incX, 
		    void *Y, const int incY
		   ) ;

void mncblas_ccopy_omp (
		    const int N, const void *X, const int incX, 
		    void *Y, const int incY
		   ) ;

void mncblas_ccopy_vec (
		    const int N, const void *X, const int incX, 
		    void *Y, const int incY
		   ) ;

void mncblas_zcopy (
		    const int N, const void *X, const int incX, 
		    void *Y, const int incY
		   ) ;

void mncblas_zcopy_omp (
		    const int N, const void *X, const int incX, 
		    void *Y, const int incY
		   ) ;

void mncblas_zcopy_vec (
		    const int N, const void *X, const int incX, 
		    void *Y, const int incY
		   ) ;


/*

  end BLAS 1 COPY

*/


/*

  BEGIN BLAS1 SWAP

*/

void mncblas_sswap (
		    const int N, float *X, const int incX, 
		    float *Y, const int incY
		   ) ;

void mncblas_sswap_omp (
		    const int N, float *X, const int incX, 
		    float *Y, const int incY
		   ) ;

void mncblas_sswap_vec (
		    const int N, float *X, const int incX, 
		    float *Y, const int incY
		   ) ;

void mncblas_dswap (
		    const int N, double *X, const int incX, 
		    double *Y, const int incY
		   ) ;

void mncblas_dswap_omp (
		    const int N, double *X, const int incX, 
		    double *Y, const int incY
		   ) ;

void mncblas_dswap_vec (
		    const int N, double *X, const int incX, 
		    double *Y, const int incY
		   ) ;

void mncblas_cswap (
		    const int N, void *X, const int incX, 
		    void *Y, const int incY
		   ) ;

void mncblas_cswap_omp (
		    const int N, void *X, const int incX, 
		    void *Y, const int incY
		   ) ;

void mncblas_cswap_vec (
		    const int N, void *X, const int incX, 
		    void *Y, const int incY
		   ) ;

void mncblas_zswap (
		    const int N, void *X, const int incX, 
		    void *Y, const int incY
		   ) ;

void mncblas_zswap_omp (
		    const int N, void *X, const int incX, 
		    void *Y, const int incY
		   ) ;

void mncblas_zswap_vec (
		    const int N, void *X, const int incX, 
		    void *Y, const int incY
		   ) ;
/*

  END BLAS1 SWAP

*/


/*

  BEGIN BLAS1 DOT

*/

float  mncblas_sdot (
		     const int N, const float  *X, const int incX,
		     const float  *Y, const int incY
		    ) ;

float  mncblas_sdot_omp (
		     const int N, const float  *X, const int incX,
		     const float  *Y, const int incY
		    ) ;

float mncblas_sdot_vec (
		     const int N, const float *X, const int incX,
		     const float *Y, const int incY
		    ) ;

double mncblas_ddot (
		     const int N, const double *X, const int incX,
		     const double *Y, const int incY
		    ) ;

double mncblas_ddot_omp (
		     const int N, const double *X, const int incX,
		     const double *Y, const int incY
		    ) ;

double mncblas_ddot_vec (
		     const int N, const double *X, const int incX,
		     const double *Y, const int incY
		    ) ;

void   mncblas_cdotu_sub (
			  const int N, const void *X, const int incX,
			  const void *Y, const int incY, void *dotu
			 ) ;

void   mncblas_cdotu_sub_omp (
			  const int N, const void *X, const int incX,
			  const void *Y, const int incY, void *dotu
			 ) ;

void   mncblas_cdotu_sub_vec (
			  const int N, const void *X, const int incX,
			  const void *Y, const int incY, void *dotu
			 ) ;

void   mncblas_cdotc_sub (
			  const int N, const void *X, const int incX,
			  const void *Y, const int incY, void *dotc
			 ) ;

void   mncblas_cdotc_sub_omp (
			  const int N, const void *X, const int incX,
			  const void *Y, const int incY, void *dotc
			 ) ;

void   mncblas_cdotc_sub_vec (
			  const int N, const void *X, const int incX,
			  const void *Y, const int incY, void *dotc
			 ) ;

void   mncblas_zdotu_sub (
			  const int N, const void *X, const int incX,
			  const void *Y, const int incY, void *dotu
			 );

void   mncblas_zdotu_sub_omp (
			  const int N, const void *X, const int incX,
			  const void *Y, const int incY, void *dotu
			 );

void   mncblas_zdotu_sub_vec (
			  const int N, const void *X, const int incX,
			  const void *Y, const int incY, void *dotu
			 );

void   mncblas_zdotc_sub (
			  const int N, const void *X, const int incX,
			  const void *Y, const int incY, void *dotc
			 ) ;

void   mncblas_zdotc_sub_omp (
			  const int N, const void *X, const int incX,
			  const void *Y, const int incY, void *dotc
			 ) ;

void   mncblas_zdotc_sub_vec (
			  const int N, const void *X, const int incX,
			  const void *Y, const int incY, void *dotc
			 ) ;

/*

  END BLAS1 DOT

*/

/*

  BEGIN BLAS1 AXPY

*/

void mncblas_saxpy (
		    const int N, const float alpha, const float *X,
		    const int incX, float *Y, const int incY
		   ) ;

void mncblas_saxpy_omp (
		    const int N, const float alpha, const float *X,
		    const int incX, float *Y, const int incY
		   ) ;

void mncblas_saxpy_vec (
		    const int N, const float alpha, const float *X,
		    const int incX, float *Y, const int incY
		   ) ;

void mncblas_daxpy (
		    const int N, const double alpha, const double *X,
		    const int incX, double *Y, const int incY
		   ) ;

void mncblas_daxpy_omp (
		    const int N, const double alpha, const double *X,
		    const int incX, double *Y, const int incY
		   ) ;

void mncblas_daxpy_vec (
		    const int N, const double alpha, const double *X,
		    const int incX, double *Y, const int incY
		   ) ;

void mncblas_caxpy (
		    const int N, const void *alpha, const void *X,
		    const int incX, void *Y, const int incY
		   ) ;

void mncblas_caxpy_omp (
		    const int N, const void *alpha, const void *X,
		    const int incX, void *Y, const int incY
		   ) ;

void mncblas_caxpy_vec (
		    const int N, const void *alpha, const void *X,
		    const int incX, void *Y, const int incY
		   ) ;

void mncblas_zaxpy (
		    const int N, const void *alpha, const void *X,
		    const int incX, void *Y, const int incY
		   ) ;

void mncblas_zaxpy_omp (
		    const int N, const void *alpha, const void *X,
		    const int incX, void *Y, const int incY
		   ) ;

void mncblas_zaxpy_vec (
		    const int N, const void *alpha, const void *X,
		    const int incX, void *Y, const int incY
		   ) ;
/*

  END BLAS1 AXPY

*/


/*

  BEGIN BLAS2 GEMV

*/

void mncblas_sgemv (
		    const MNCBLAS_LAYOUT layout,
		    const MNCBLAS_TRANSPOSE TransA, const int M, const int N,
		    const float alpha, const float *A, const int lda,
		    const float *X, const int incX, const float beta,
		    float *Y, const int incY
		   ) ;

void mncblas_sgemv_omp (
		    const MNCBLAS_LAYOUT layout,
		    const MNCBLAS_TRANSPOSE TransA, const int M, const int N,
		    const float alpha, const float *A, const int lda,
		    const float *X, const int incX, const float beta,
		    float *Y, const int incY
		   ) ;

void mncblas_sgemv_vec (
		    const MNCBLAS_LAYOUT layout,
		    const MNCBLAS_TRANSPOSE TransA, const int M, const int N,
		    const float alpha, const float *A, const int lda,
		    const float *X, const int incX, const float beta,
		    float *Y, const int incY
		   ) ;

void mncblas_dgemv (
		    MNCBLAS_LAYOUT layout,
		    MNCBLAS_TRANSPOSE TransA, const int M, const int N,
		    const double alpha, const double *A, const int lda,
		    const double *X, const int incX, const double beta,
		    double *Y, const int incY
		   ) ;

void mncblas_dgemv_omp (
		    MNCBLAS_LAYOUT layout,
		    MNCBLAS_TRANSPOSE TransA, const int M, const int N,
		    const double alpha, const double *A, const int lda,
		    const double *X, const int incX, const double beta,
		    double *Y, const int incY
		   ) ;

void mncblas_dgemv_vec (
		    MNCBLAS_LAYOUT layout,
		    MNCBLAS_TRANSPOSE TransA, const int M, const int N,
		    const double alpha, const double *A, const int lda,
		    const double *X, const int incX, const double beta,
		    double *Y, const int incY
		   ) ;

void mncblas_cgemv (
		    MNCBLAS_LAYOUT layout,
		    MNCBLAS_TRANSPOSE TransA, const int M, const int N,
		    const void *alpha, const void *A, const int lda,
		    const void *X, const int incX, const void *beta,
		    void *Y, const int incY
		   ) ;

void mncblas_cgemv_omp (
		    MNCBLAS_LAYOUT layout,
		    MNCBLAS_TRANSPOSE TransA, const int M, const int N,
		    const void *alpha, const void *A, const int lda,
		    const void *X, const int incX, const void *beta,
		    void *Y, const int incY
		   ) ;

void mncblas_cgemv_vec (
		    MNCBLAS_LAYOUT layout,
		    MNCBLAS_TRANSPOSE TransA, const int M, const int N,
		    const void *alpha, const void *A, const int lda,
		    const void *X, const int incX, const void *beta,
		    void *Y, const int incY
		   ) ;

void mncblas_zgemv (
		    MNCBLAS_LAYOUT layout,
		    MNCBLAS_TRANSPOSE TransA, const int M, const int N,
		    const void *alpha, const void *A, const int lda,
		    const void *X, const int incX, const void *beta,
		    void *Y, const int incY
		   ) ;

void mncblas_zgemv_omp (
		    MNCBLAS_LAYOUT layout,
		    MNCBLAS_TRANSPOSE TransA, const int M, const int N,
		    const void *alpha, const void *A, const int lda,
		    const void *X, const int incX, const void *beta,
		    void *Y, const int incY
		   ) ;

void mncblas_zgemv_vec (
		    MNCBLAS_LAYOUT layout,
		    MNCBLAS_TRANSPOSE TransA, const int M, const int N,
		    const void *alpha, const void *A, const int lda,
		    const void *X, const int incX, const void *beta,
		    void *Y, const int incY
		   ) ;

/*

  END BLAS2 GEMV

*/


/*

  BEGIN BLAS3 GEMM

*/

void mncblas_sgemm (
		    MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
		    MNCBLAS_TRANSPOSE TransB, const int M, const int N,
		    const int K, const float alpha, const float *A,
		    const int lda, const float *B, const int ldb,
		    const float beta, float *C, const int ldc
		   ) ;

void mncblas_sgemm_omp (
		    MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
		    MNCBLAS_TRANSPOSE TransB, const int M, const int N,
		    const int K, const float alpha, const float *A,
		    const int lda, const float *B, const int ldb,
		    const float beta, float *C, const int ldc
		   ) ;

void mncblas_sgemm_vec (
		    MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
		    MNCBLAS_TRANSPOSE TransB, const int M, const int N,
		    const int K, const float alpha, const float *A,
		    const int lda, const float *B, const int ldb,
		    const float beta, float *C, const int ldc
		   ) ;


void mncblas_dgemm (
		    MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
		    MNCBLAS_TRANSPOSE TransB, const int M, const int N,
		    const int K, const double alpha, const double *A,
		    const int lda, const double *B, const int ldb,
		    const double beta, double *C, const int ldc
		    ) ;

void mncblas_dgemm_omp (
		    MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
		    MNCBLAS_TRANSPOSE TransB, const int M, const int N,
		    const int K, const double alpha, const double *A,
		    const int lda, const double *B, const int ldb,
		    const double beta, double *C, const int ldc
		    ) ;

void mncblas_dgemm_vec (
		    MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
		    MNCBLAS_TRANSPOSE TransB, const int M, const int N,
		    const int K, const double alpha, const double *A,
		    const int lda, const double *B, const int ldb,
		    const double beta, double *C, const int ldc
		    ) ;


void mncblas_cgemm (
		    MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
		    MNCBLAS_TRANSPOSE TransB, const int M, const int N,
		    const int K, const void *alpha, const void *A,
		    const int lda, const void *B, const int ldb,
		    const void *beta, void *C, const int ldc
		   ) ;

void mncblas_cgemm_omp (
		    MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
		    MNCBLAS_TRANSPOSE TransB, const int M, const int N,
		    const int K, const void *alpha, const void *A,
		    const int lda, const void *B, const int ldb,
		    const void *beta, void *C, const int ldc
		   ) ;

void mncblas_cgemm_vec (
		    MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
		    MNCBLAS_TRANSPOSE TransB, const int M, const int N,
		    const int K, const void *alpha, const void *A,
		    const int lda, const void *B, const int ldb,
		    const void *beta, void *C, const int ldc
		   ) ;

void mncblas_zgemm (
		    MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
		    MNCBLAS_TRANSPOSE TransB, const int M, const int N,
		    const int K, const void *alpha, const void *A,
		    const int lda, const void *B, const int ldb,
		    const void *beta, void *C, const int ldc
		   ) ;

void mncblas_zgemm_omp (
		    MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
		    MNCBLAS_TRANSPOSE TransB, const int M, const int N,
		    const int K, const void *alpha, const void *A,
		    const int lda, const void *B, const int ldb,
		    const void *beta, void *C, const int ldc
		   ) ;

void mncblas_zgemm_vec (
		    MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
		    MNCBLAS_TRANSPOSE TransB, const int M, const int N,
		    const int K, const void *alpha, const void *A,
		    const int lda, const void *B, const int ldb,
		    const void *beta, void *C, const int ldc
		   ) ;

/*

  END BLAS3 GEMM

*/




