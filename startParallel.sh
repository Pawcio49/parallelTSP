#!/bin/bash
nvcc -Xcompiler -fopenmp -o program TSPParallel.cu
./program 10 adjPascal