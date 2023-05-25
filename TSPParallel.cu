#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>


typedef struct FillCData {
    int subsetSize;
    int ***C;
    int CSize;
    int n;
} fillCData;


typedef struct REC {
    int v;
    struct REC * prev;
} rec;


int myPow(int base, int power){
    int result = 1;
    int i;
    #pragma omp parallel for reduction(*:result) private(i)
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


__device__ int global_dist[1000]; 


__global__ void process_data(int* input_data, int data_size) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (thread_id < data_size) {
        global_dist[thread_id] = input_data[thread_id];
    }
}


__global__ void calculateCosts(int* subset, int subsetSize, int* res, int k, int* CPrev0, int n) {
    int m = threadIdx.x;

    if (m < subsetSize && m != k) {
        int cost = CPrev0[subset[m]] + global_dist[subset[m] * n + subset[k]];
        atomicMin(&res[0], cost);
        if (cost == res[0]) {
            res[1] = subset[m];
        }
    }
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
    for(int i=0; i<data.subsetSize; i++){
        bits |= 1 << subset[i];
    }

    for(int k=0; k<data.subsetSize; k++) {
        int prev = bits & ~(1 << subset[k]);

        int res[2] = {999999,0};
        
        int* d_subset;
        int* d_res;
        int* d_C;

        
        cudaMalloc((void**)&d_subset, data.subsetSize * sizeof(int));
        cudaMalloc((void**)&d_res, 2 * sizeof(int));
        cudaMalloc((void**)&d_C, data.n * sizeof(int));
        
        cudaMemcpy(d_subset, subset, data.subsetSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_res, res, 2 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, data.C[prev][0], data.n * sizeof(int), cudaMemcpyHostToDevice);
        
        int threadsPerBlock = data.subsetSize;
        int gridSize = 1;

        calculateCosts<<<gridSize, threadsPerBlock>>>(d_subset, data.subsetSize, d_res, k, d_C, data.n);

        cudaMemcpy(res, d_res, 2 * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_subset);
        cudaFree(d_res);
        cudaFree(d_C);

        data.C[bits][0][subset[k]] = res[0];
        data.C[bits][1][subset[k]] = res[1];
    }
    // printf("%d\n", omp_get_thread_num());
    free(subset);
}
     
void generateCombinationsAndFillC(rec * x, int level, int k, fillCData data) {
    rec X1,X2;
    if (level==0) {
        fillC(x, data);
    } 
    else 
    {
        #pragma omp task
        { 
            if (level>k) {
                X1.prev = x;
                X1.v = 0;
                generateCombinationsAndFillC(&X1,level-1, k, data);
            } 
        }
        #pragma omp task 
        {
            if (k>0) {
                X2.prev = x;
                X2.v = level;
                generateCombinationsAndFillC(&X2,level-1, k-1, data);
            }
        }
        #pragma omp taskwait  
    }
}


int TSP(int n, int *dist, int *path)
{
    int data_size = n*n;   

    int* d_input_data;
    cudaMalloc((void**)&d_input_data, data_size * sizeof(int));
    
    cudaMemcpy(d_input_data, dist, data_size * sizeof(int), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = data_size;
    int grid_size = 1;
    process_data<<<grid_size, threadsPerBlock>>>(d_input_data, data_size);

    int dimension0C = myPow(2, n) - 1;
    int CSize = dimension0C * n * 2;
    int ***C;
    C = (int ***)malloc(CSize * sizeof(int));
    for(int i=0; i<dimension0C;i++) {
        C[i] = (int **)malloc(n * 2 * sizeof(int));
        for(int j=0; j<2; j++) {
            C[i][j] = (int *)malloc(n * sizeof(int));
        }
    }

    for(int i=1; i<n; i++) {
        C[1<<i][0][i] = dist[0*n+i];
        C[1<<i][1][i] = 0;
    }

    fillCData data;
    data.C = C;
    data.CSize = dimension0C * n * 2;
    data.n = n;
    for(int subsetSize=2; subsetSize<n; subsetSize++){
        data.subsetSize = subsetSize;
        #pragma omp parallel
        #pragma omp single
        generateCombinationsAndFillC(NULL, n-1, subsetSize, data);
    }

    // We're interested in all bits but the least significant (the start state)
    int bits = (myPow(2, n) - 1) - 1;

    // Calculate optimal cost
    int opt = -1;
    int parent;
    for(int k=1; k<n; k++)  {
        int cost = C[bits][0][k] + dist[k*n + 0];
        if(opt== -1 || cost<opt){
            opt = cost;
            parent = k;
        }
    }

    for(int i=n-1; i>0; i--) {
        path[i] = parent;
        int new_bits = bits & ~(1 << parent);
        parent = C[bits][1][parent];
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


int main(int argc, char *argv[])
{
    if(argc<3){
        printf("Type ./filename matrix_size file_with_matrix");
        return -1;
    }

    const int N = atoi(argv[1]);
    int final_path[N];

    // Adjacency matrix for the given graph
    int **adj;
    adj = (int **)malloc(N * N * sizeof(int));
    for(int i=0;i<N;i++){
        adj[i] = (int *)malloc(N * sizeof(int));
    }
    readMatrix(N, adj, argv[2]);
    // generateMatrix(N,adj);

    int* flatAdj = new int[N*N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            flatAdj[i * N + j] = adj[i][j];
        }
    }

    clock_t start, end;
    double cpu_time_used;

    start = clock();
    int final_res = TSP(N, flatAdj, final_path);
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