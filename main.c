#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

struct Subset {
    int *value;
    struct Subset *next;
};

int myPow(int base, int power){
    int result = 1;
    for(int i = 0; i<power; i++){ //OpenMP
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
    rootSubset.value = malloc(subset_size * sizeof(int));
    memcpy(rootSubset.value, subset, subset_size * sizeof(int));
    rootSubset.next = malloc(sizeof(rootSubset));
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

        nextSubset->value = malloc(subset_size * sizeof(int));
        memcpy(nextSubset->value, subset, subset_size * sizeof(int));

        nextSubset->next = malloc(subset_size * sizeof(int));
        nextSubset = nextSubset->next;
    }

    return rootSubset;
}

void print_subset(struct Subset *subset, int size) {
    printf("{");
    for (int i = 0; i < size; i++) {
        printf("%d", subset->value[i]);
        if (i < size - 1) {
            printf(", ");
        }
    }
    printf("}\n");
}

// This function sets up final_path[]
int TSP(int n, int dist[n][n], int path[n])
{
    int C[myPow(2, n) - 1][n][2];

    for(int i=1; i<n; i++) { //CUDA - sprobowac
        C[1<<i][i][0] = dist[0][i];
        C[1<<i][i][1] = 0;
    }


    for(int subsetSize=2; subsetSize<n; subsetSize++){
        //jednoczesnie generowac kombinacje i sprawdzac wartosci dla nich. Pan mowil o przesylaniu kombinacji paczkami
        struct Subset rootSubset = generateSubsets(n-1, subsetSize);
        struct Subset *subset = &rootSubset;
        while (subset->next){
            print_subset(subset,subsetSize);
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
                C[bits][subset->value[k]][0] = res[0];
                C[bits][subset->value[k]][1] = res[1];
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
    int final_path[N];

    // Adjacency matrix for the given graph
    int adj[N][N];
    readMatrix(N, adj, argv[2]);

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

    return 0;
}
