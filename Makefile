all: exo1 exo2 exo3 clean

CC=gcc -Wall

exo1:
	$(CC) exo1.c -o exo1 -fopenmp

exo2:
	$(CC) exo2.c -o exo2 -fopenmp

exo3:
	$(CC) exo3.c -o exo3 -fopenmp

clean:
	rm -f *.o *~ *.backup