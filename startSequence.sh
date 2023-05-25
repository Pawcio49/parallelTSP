#!/bin/bash
nvcc -Xcompiler -fopenmp -o program2 TSPSequence.c
./program2 4 adj