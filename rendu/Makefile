all: main

CC=gcc -Wno-unused-variable -fopenmp -msse4.2

main:
	$(CC) *.c -g -o main


clean:
	rm -f *.o *~ *.backup
