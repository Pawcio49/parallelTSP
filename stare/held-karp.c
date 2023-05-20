#include <stdio.h>
#include <limits.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

int readMatrix(size_t size, int (*a)[size], const char* filename)
{
    FILE *pf;
    pf = fopen (filename, "r");
    if (pf == NULL)
        return 0;

    for(size_t i = 0; i < size; ++i)
    {
        for(size_t j = 0; j < size; ++j)
            fscanf(pf, "%d", a[i] + j);
    }


    fclose (pf);
    return 1;
}

// Driver code
int main(int argc, char *argv[])
{
    if(argc<3){
        printf("Type ./filename matrix_size file_with_matrix");
        return -1;
    }

    const int N = atoi(argv[1]);
    // final_path[] stores the final solution ie, the
    // path of the salesman.
    int final_path[N+1];

    // visited[] keeps track of the already visited nodes
    // in a particular path
    bool visited[N];

    // Adjacency matrix for the given graph
    int adj[N][N];
    readMatrix(N, adj, argv[2]);

    clock_t start, end;
    double cpu_time_used;

    start = clock();
    TSP(N, adj, visited, final_path);
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Time : %f\n", cpu_time_used);
    printf("Minimum cost : %d\n", final_res);
    printf("Path Taken : ");
    for (int i=0; i<=N; i++)
        printf("%d ", final_path[i]);

    return 0;
}
