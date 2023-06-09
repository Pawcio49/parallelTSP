#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>


typedef struct FillCData {
    int subsetSize;
    int ***C;
    int **dist;
} fillCData;


typedef struct REC {
    int v;
    struct REC * prev;
} rec;


int myPow(int base, int power){
    int result = 1;
    int i;
    for(i = 0; i<power; i++){
        result *= base;
    }
    return result;
}


void print_subset(int *subset, int size) {
    printf("{");
    for (int i = 0; i < size; i++) {
        printf("%d", subset[i]);
        if (i < size - 1) {
            printf(", ");
        }
    }
    printf("}\n");
}

     
void fillC (rec * x, fillCData data) {
    int *subset = (int *)malloc(data.subsetSize * sizeof(int));
    int i = 0;
    while (x) { 
        if(x->v > 0){
            subset[i] = x->v;
            i++;
        }
        x = x -> prev;
    }
    
    int bits = 0;
    for(int i=0; i<data.subsetSize; i++){ //CUDA lub OpenMP
        bits |= 1 << subset[i];
    }

    for(int k=0; k<data.subsetSize; k++) { // tu bez zrownoleglenia, bo musi byc po kolei
        int prev = bits & ~(1 << subset[k]);

        int res[2] = {-1,-1};
        for(int m=0; m<data.subsetSize; m++) { //OpenMP
            if(subset[m]==subset[k])
                continue;
            int cost = data.C[prev][subset[m]][0] + data.dist[subset[m]][subset[k]];
            if(res[0] == -1 || cost<res[0]){
                res[0] = cost;
                res[1] = subset[m];
            }
        }
        memcpy(data.C[bits][subset[k]], res, 2*sizeof(int));
    }
    free(subset);
}
     
void generateCombinationsAndFillC(rec * x, int level, int k, fillCData data) {
    rec X1,X2;
    if (level==0) {
        fillC(x, data);
    } 
    else 
    {
        { 
            if (level>k) {
                X1.prev = x;
                X1.v = 0;
                generateCombinationsAndFillC(&X1,level-1, k, data);
            } 
        }
        {
            if (k>0) {
                X2.prev = x;
                X2.v = level;
                generateCombinationsAndFillC(&X2,level-1, k-1, data);
            }
        }
    }
}


int TSP(int n, int **dist, int *path)
{
    int dimension0C = myPow(2, n) - 1;
    int ***C;
    C = (int ***)malloc(dimension0C * n * 2 * sizeof(int));
    for(int i=0; i<dimension0C;i++) {
        C[i] = (int **)malloc(n * 2 * sizeof(int));
        for(int j=0; j<n; j++) {
            C[i][j] = (int *)malloc(2 * sizeof(int));
        }
    }

    for(int i=1; i<n; i++) { //CUDA - sprobowac  - raczej nie ma sensu
        C[1<<i][i][0] = dist[0][i];
        C[1<<i][i][1] = 0;
    }

    fillCData data;
    data.C = C;
    data.dist = dist;
    for(int subsetSize=2; subsetSize<n; subsetSize++){
        data.subsetSize = subsetSize;
        generateCombinationsAndFillC(NULL, n-1, subsetSize, data);
    }

    // We're interested in all bits but the least significant (the start state)
    int bits = (myPow(2, n) - 1) - 1;

    // Calculate optimal cost
    int opt = -1;
    int parent;
    for(int k=1; k<n; k++)  { //CUDA lub OpenMP
        int cost = C[bits][k][0] + dist[k][0];
        if(opt== -1 || cost<opt){
            opt = cost;
            parent = k;
        }
    }

    for(int i=n-1; i>0; i--) { // tu bez zrownoleglenia, bo musi byc po kolei
        path[i] = parent;
        int new_bits = bits & ~(1 << parent);
        parent = C[bits][parent][1];
        bits = new_bits;
    }

    path[0] = 0;
    free(C);
    return opt;
}

int readMatrix(int size, int **a, const char* filename)
{
    FILE *pf;
    pf = fopen (filename, "r");
    if (pf == NULL)
        return 0;

    for(int i = 0; i < size; ++i)
    {
        for(int j = 0; j < size; ++j) {
            
            fscanf(pf, "%d", a[i] + j);
        }
    }


    fclose (pf);
    return 1;
}

int generateMatrix(int size, int **a)
{
    srand(time(NULL)); 
    int min = 1;
    int max = 1000;
    for(int i = 0; i < size; ++i)
    {
        for(int j = 0; j < size; ++j) {
            if(i==j)
                a[i][j] = 0;
            else
                a[i][j] = rand() % (max - min + 1) + min;
        }
    }
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
    int final_path[N];

    // Adjacency matrix for the given graph
    int **adj;
    adj = (int **)malloc(N * N * sizeof(int));
    for(int i=0;i<N;i++){
        adj[i] = (int *)malloc(N * sizeof(int));
    }
    readMatrix(N, adj, argv[2]);
    // generateMatrix(N,adj);
    clock_t start, end;
    double cpu_time_used;

    start = clock();
    int final_res = TSP(N, adj, final_path);
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Time : %f\n", cpu_time_used);
    printf("Minimum cost : %d\n", final_res);
    printf("Path Taken : ");
    for (int i=0; i<N; i++)
        printf("%d ", final_path[i]);
    printf("\n");
    free(adj);
    return 0;
}