all: exo1 exo2 exo3 clean

exo1: exo1.o
	gcc -Wall -o exo1 exo1.o -fopenmp

exo1.o: exo1.c
	gcc -c exo1.c

exo2: exo2.o
	gcc -Wall -o exo2 exo2.o -fopenmp

exo2.o: exo2.c
	gcc -c exo2.c

exo3: exo3.o
	gcc -Wall -o exo3 exo3.o -fopenmp

exo3.o: exo3.c
	gcc -c exo3.c

clean:
	rm -f *.o *~ *.backup