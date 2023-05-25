#!/bin/bash
nvcc -Xcompiler -fopenmp -o program2 TSPSequence.c
./program2 10 adjPascal