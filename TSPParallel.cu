#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>

struct Subset {
    int *value;
    struct Subset *next;
};

int myPow(int base, int power){
    int result = 1;
    int i;
    #pragma omp parallel for reduction(*:result) private(i)
    for(i = 0; i<power; i++){
        result *= base;
    }
    return result;
}

//generowanie kombinacji doku - przyjrzec się ifom zeby mocniej obciac
struct Subset generateSubsets(int n, int subset_size) {
    int *subset = (int *) malloc(subset_size * sizeof(int));
    for (int i = 0; i < subset_size; i++) { //CUDA
        subset[i] = i + 1;
    }

    struct Subset rootSubset = {};
    rootSubset.value = (int *)malloc(subset_size * sizeof(int));
    memcpy(rootSubset.value, subset, subset_size * sizeof(int));
    rootSubset.next = (Subset *)malloc(sizeof(rootSubset));
    struct Subset *nextSubset = rootSubset.next;

    while (1) {
        int i;
        for (i = subset_size - 1; i >= 0; i--) {
            if (subset[i] != i + n - subset_size + 1) {
                break;
            }
        }

        if (i < 0) {
            break;
        }

        subset[i]++;
        for (int j = i + 1; j < subset_size; j++) {
            subset[j] = subset[j - 1] + 1;
        }

        nextSubset->value = (int *)malloc(subset_size * sizeof(int));
        memcpy(nextSubset->value, subset, subset_size * sizeof(int));

        nextSubset->next = (Subset *)malloc(subset_size * sizeof(int));
        nextSubset = nextSubset->next;
    }

    return rootSubset;
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

typedef struct REC { int v; struct REC * prev; } rec;
     
void drukuj (rec * x, int n, int subsetSize, int ***C) {
    int *subset = (int *)malloc(subsetSize * sizeof(int));
    int i = 0;
    while (x) {
        
        if(x->v > 0){
            // printf("%d, %d\n", *i, subsetSize );
            subset[i] = x->v;
            i++;
        }
        x = x -> prev;
    }
    // printf("\n");
    free(subset);
    // printf("%d: %s\n",omp_get_thread_num(),buf);
    // free(buf);
}
     
void genkomb(rec * x, int level, int n, int k, int subsetSize, int ***C) {
    rec X1,X2;
    if (level==n) {
        drukuj(x,n, subsetSize, C);
    } 
    else 
    {
        #pragma omp task
        { 
            X1.prev = x;
            X1.v = 0;
            if (n-level>k) genkomb(&X1,level+1,n, k, subsetSize, C);
        }
        #pragma omp task  
        {
            X2.prev = x;
            X2.v = level;
            if (k>0) genkomb(&X2,level+1,n, k-1, subsetSize, C);
        }
        #pragma omp taskwait  
    }
}

int TSP(int n, int **dist, int *path)
{
    // int C[myPow(2, n) - 1][n][2];
    int dimension0C = myPow(2, n) - 1;
    int ***C;
    C = (int ***)malloc(dimension0C * n * 2 * sizeof(int));
    for(int i=0; i<dimension0C;i++) {
        C[i] = (int **)malloc(n * 2 * sizeof(int));
        for(int j=0; j<n; j++) {
            C[i][j] = (int *)malloc(2 * sizeof(int));
        }
    }

    for(int i=1; i<n; i++) { //CUDA - sprobowac
        C[1<<i][i][0] = dist[0][i];
        C[1<<i][i][1] = 0;
    }


    for(int subsetSize=2; subsetSize<n; subsetSize++){
        #pragma omp parallel
        #pragma omp single
        genkomb(NULL,0,n-1,subsetSize, subsetSize, C);
        struct Subset rootSubset = generateSubsets(n-1, subsetSize);
        struct Subset *subset = &rootSubset;
        while (subset->next){
            int bits = 0;
            for(int i=0; i<subsetSize; i++){ //CUDA lub OpenMP
                bits |= 1 << subset->value[i];
            }

            for(int k=0; k<subsetSize; k++) { // tu bez zrownoleglenia, bo musi byc po kolei
                int prev = bits & ~(1 << subset->value[k]);

                int res[2] = {-1,-1};
                for(int m=0; m<subsetSize; m++) { //OpenMP
                    if(subset->value[m]==subset->value[k])
                        continue;
                    int cost = C[prev][subset->value[m]][0] + dist[subset->value[m]][subset->value[k]];
                    if(res[0] == -1 || cost<res[0]){
                        res[0] = cost;
                        res[1] = subset->value[m];
                    }
                }
                memcpy(C[bits][subset->value[k]], res, 2*sizeof(int));
            }

            subset = subset->next;
        }
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
    return 0;
}
