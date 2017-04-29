# une bibliotheque statique (libmnblas.a)
# une bibliotheque dynamique (libmnblasdyn.so)

all: libmnblas.a libmnblasdyn.so

LIB_DIR=../lib
INC_DIR=../include

OPTIM=-O2
OPTIONS=-fPIC -msse4.2 $(OPTIM) -I$(INC_DIR)

OBJS= copy.o swap.o dot.o axpy.o gemv.o gemm.o

# Bibliothèque statique

libmnblas.a: $(OBJS)
	rm -f libmnblas.a $(LIB_DIR)/libmnblas.a
	ar -r libmnblas.a $(OBJS)
	cp libmnblas.a $(LIB_DIR)

# Bibliothèque dynamique

libmnblasdyn.so: $(OBJS)
	rm -f libmnblasdyn.so $(LIB_DIR)/libmnblasdyn.so
	gcc -shared -o libmnblasdyn.so $(OBJS)
	cp libmnblasdyn.so $(LIB_DIR)

# BLAS 1

copy.o : copy.c $(INC_DIR)/mnblas.h
	gcc $(OPTIONS) -c copy.c

swap.o : swap.c $(INC_DIR)/mnblas.h
	gcc  $(OPTIONS) -c swap.c

dot.o : dot.c $(INC_DIR)/mnblas.h
	gcc  $(OPTIONS) -c dot.c

axpy.o : axpy.c $(INC_DIR)/mnblas.h
	gcc  $(OPTIONS) -c axpy.c


# BLAS 2

gemv.o : gemv.c $(INC_DIR)/mnblas.h
	gcc  $(OPTIONS) -c gemv.c

# BLAS 3

gemm.o : gemm.c $(INC_DIR)/mnblas.h
	gcc  $(OPTIONS) -c gemm.c


clean:
	rm -f *.o *.a *.so *~  *backup $(INC_DIR)/*~ $(LIB_DIR)/libmnblasdyn.so  $(LIB_DIR)/libmnblas.a 


