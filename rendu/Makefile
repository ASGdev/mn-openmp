all: main

CC=gcc -Wall -fopenmp -msse4.2

main:
	$(CC) main.c axpy.c copy.c -o main


clean:
	rm -f *.o *~ *.backup
