all: a.out b.out

a.out: matrix.c
	icc -O0 -mkl matrix.c 

b.out: pro2.c
	gcc -fopenmp -O0 -o b.out pro2.c

clean:
	@rm -f *.o *.out