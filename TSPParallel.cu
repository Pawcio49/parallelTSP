#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define maxLength 1000

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


// Obliczanie liczby (base) podniesionej do danej potegi (power)
int myPow(int base, int power){
    int result = 1;
    int i;

    // Rownolegle obliczanie potegi
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


__device__ int global_dist[maxLength]; 


__global__ void saveGlobalDist(int* input_data, int data_size) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (thread_id < data_size) {
        global_dist[thread_id] = input_data[thread_id];
    }
}

// Argumenty wejściowe:
// subset - kombinacja (tablica jednowymiarowa)
// subsetSize - rozmiar kombinacji
// res - dwuwymiarowa tablica zawierajaca minimalny koszt sciezki i odpowiadajacy mu ostatni odwiedzony wierzcholek
// k - indeks odwiedzanego wierzcholka
// CPrev0 - tablica jednowymiarowa zawierajaca informacje jakie sa obliczone koszty sciezek
// n - liczba wierzcholkow
__global__ void calculateCosts(int* subset, int subsetSize, int* res, int k, int* CPrev0, int n) {
    int m = threadIdx.x;

    // Przetwarzane sa indeksy mieszczace sie w kombinacji i rozne od indeksu aktualnie odwiedzanego wierzcholka
    if (m < subsetSize && m != k) {
        // Obliczenie kosztu sciezki na podstawie wygenerowanej wczesniej sciezki
        // i dodania do niej kosztu przejscia do analizowanego wierzcholka
        int cost = CPrev0[subset[m]] + global_dist[subset[m] * n + subset[k]];

        // Wybranie miedzy watkami minimalnego, obliczonego kosztu sciezki
        atomicMin(&res[0], cost);
        if (cost == res[0]) {
            // Zapisanie ostatniego odwiedzanego wierzcholka
            res[1] = subset[m];
        }
    }
}

// Uzupelnienie macierzy C
// x - Linked list, ktore zawiera kombinacje
// data - dane potrzebne do wypelnienia macierzy C
void fillC (rec * x, fillCData data) {
    // Odczytanie kombinacji z Linked List i zapisanie ich jako tablica jednowymiarowa
    int *subset = (int *)malloc(data.subsetSize * sizeof(int));
    int i = 0;
    while (x) { 
        if(x->v > 0){
            subset[i] = x->v;
            i++;
        }
        x = x -> prev;
    }
    
    // Zapisanie w postaci bitowej, jakie liczby sa w kombinacji
    int bits = 0;
    for(int i=0; i<data.subsetSize; i++){
        bits |= 1 << subset[i];
    }

    // Rownolegle uzupelnianie macierzy C dla kolejnych wierzcholkow z kombinacji rozpatrywanych jako aktualnie odwiedzany wierzcholek
    #pragma omp parallel for
    for(int k=0; k<data.subsetSize; k++) {
        // Zapisanie w postaci bitowej, jakie liczby sa w kombinacji bez uwzglednienia aktualnie odwiedzanego wierzcholka
        // Jest to informacja, jakie wierzcholki byly juz odwiedzane w rozpatrywanym przypadku
        int prev = bits & ~(1 << subset[k]);

        // Zadeklarowanie tablicy rezultatow
        // Indeks 0 zawiera informacje jaki jest koszt sciezki
        // Indeks 1 zawiera informacje jaki jest ostatni odwiedzony wierzcholek 
        int res[2] = {999999,0};
         
        int* d_subset;
        int* d_res;
        int* d_C;

        // Zadeklarowanie pamieci dla danych przesylanych do CUDA
        cudaMalloc((void**)&d_subset, data.subsetSize * sizeof(int));
        cudaMalloc((void**)&d_res, 2 * sizeof(int));
        cudaMalloc((void**)&d_C, data.n * sizeof(int));
        
        // Skopiowanie danych do zmiennych CUDA
        cudaMemcpy(d_subset, subset, data.subsetSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_res, res, 2 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, data.C[prev][0], data.n * sizeof(int), cudaMemcpyHostToDevice);
        
        // Wywolanie funkcji calculateCosts na GPU. Ilosc watkow jest rowna rozmiarowi kombinacji.
        // Kazda liczba w kombinacji bedzie przetwarzana przez osobny watek
        int threadsPerBlock = data.subsetSize;
        int gridSize = 1;
        calculateCosts<<<gridSize, threadsPerBlock>>>(d_subset, data.subsetSize, d_res, k, d_C, data.n);

        // Skopiowanie wyniku funkcji z GPU do CPU (kosztu sciezki i odpowiadajemu mu ostatniemu odwiedzonemu wierzcholkowi)
        cudaMemcpy(res, d_res, 2 * sizeof(int), cudaMemcpyDeviceToHost);

        // Zwolnienie pamieci w CUDA
        cudaFree(d_subset);
        cudaFree(d_res);
        cudaFree(d_C);

        // Wypelnienie macierzy C obliczonym kosztem sciezki i odpowiadajacym mu ostatnim odwiedzonym wierzcholkiem
        data.C[bits][0][subset[k]] = res[0];
        data.C[bits][1][subset[k]] = res[1];
    }
    // printf("%d\n", omp_get_thread_num());
    // print_subset(subset, data.subsetSize);
    free(subset);
}
     
// Argumenty wejsciowe:
// x - Linked list, w ktorej zapisywana jest generowana kombinacja
// level - ilosc poziomow w ktorych mozna zapisac liczbe
// k - ilosc liczb wiekszych od 0 ktore musza byc jeszcze zapisane w kombinacji
// data - zbior danych potrzebnych funkcji fillC, ktora jest wywolywana
void generateCombinationsAndFillC(rec * x, int level, int k, fillCData data) {
    rec X1,X2;
    // jesli poziom jest rowny 0, to uzupelnij macierz C z wykorzystaniem wygenerowanej kombinacji zapisanej w zmiennej x
    if (level==0) {
        fillC(x, data);
    }
    // w przeciwnym wypadku kontynuuj rekurencyjne generowanie kombinacji
    else 
    {
        // Niezapisanie danego poziomu w kombinacji
        #pragma omp task
        { 
            // Jesli poziom bylby mniejszy lub rowny ilosci liczb ktore trzeba uwzglednic w kombinacji,
            // to nie zmiescily by sie one w generowanej kombinacji, wiec nastapilo by odciecie 
            if (level>k) {
                X1.prev = x;
                // Wpisanie 0 jako liczby. Wszystkie 0 sa pomijane w kombinacjach
                X1.v = 0;
                // Wywolanie rekurencyjnie funkcji ze zaktualizowana Linked list i zmiejszonym poziomem
                generateCombinationsAndFillC(&X1,level-1, k, data);
            } 
        }
        // Zapisanie danego poziomu w kombinacji
        #pragma omp task 
        {
            // Jesli ilosc liczb do wpisania bylaby mniejsza lub rowna 0, to znaczyloby ze nie mozemy wpisac
            // juz zadnej liczby do kombinacji, wiec nastapiloby odciecie
            if (k>0) {
                X2.prev = x;
                // Wpisanie liczby (poziomu) do kombinacji
                X2.v = level;
                // Wywolanie rekurencyjnie funkcji ze zaktualizowana Linked list, zmiejszonym poziomem
                // i zmniejszona iloscia liczb, ktore trzeba jeszcze wpisac do kombinacji
                generateCombinationsAndFillC(&X2,level-1, k-1, data);
            }
        }
        #pragma omp taskwait  
    }
}

// Argumenty wejsciowe:
// n - liczba wierzcholkow
// dist - wskaznik do splaszczonej dwuwymiarowej tablicy zawierajacej koszty przejsc miedzy wierzcholkami
// path - wskaznik do pustej, jednowymiarowej tablicy, w ktorej bedzie zapisana optymalna sciezka
int TSP(int n, int *dist, int *path)
{
    int data_size = n*n;   

    // Przekopiowanie macierzy zawierającej dystanse między punktami do CUDA
    int* d_input_data;
    cudaMalloc((void**)&d_input_data, data_size * sizeof(int));
    
    cudaMemcpy(d_input_data, dist, data_size * sizeof(int), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 128;
    int gridSize = (threadsPerBlock + data_size - 1) / threadsPerBlock;
    saveGlobalDist<<<gridSize, threadsPerBlock>>>(d_input_data, data_size);

    // Zadeklarowanie trójwymiarowej macierzy sciezek i ich kosztow:
    // - pierwszy wymiar oznacza jakie wierzcholki zostaly odwiedzone w postaci bitowej
    // - drugi wymiar ma zawsze rozmiar 2:
    //    a) w indeksach 0 macierz przechowuje informacje jaki jest laczny koszt sciezki
    //    b) w indeksach 1 macierz przechowuje informacje jaki jest ostatni odwiedzony wierzcholek
    // - trzeci wymiar oznacza jaki jest aktualnie odwiedzany wierzcholek
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

    // Zapisanie odleglosci z wierzcholka 0 (od ktorego zaczynamy i na ktorym konczymy) do pozostalych wierzcholkow
    for(int i=1; i<n; i++) {
        C[1<<i][0][i] = dist[0*n+i];
        C[1<<i][1][i] = 0;
    }

    // Zapisanie danych w stworzonej strukturze
    fillCData data;
    data.C = C;
    data.CSize = dimension0C * n * 2;
    data.n = n;

    // Obliczenie kosztow sciezek dla kolejnych rozmiarow kombinacji
    for(int subsetSize=2; subsetSize<n; subsetSize++){
        data.subsetSize = subsetSize;

        // Wywolanie rekurencyjnej funkcji generowania kombinacji
        // i wypełniania macierzy C w sposób rownolegly
        #pragma omp parallel
        #pragma omp single
        generateCombinationsAndFillC(NULL, n-1, subsetSize, data);
    }

    // Obliczenie wartosci, ktora odpowiada wszystkim odwiedzonym wierzcholkom
    int bits = (myPow(2, n) - 1) - 1;

    // Obliczenie optymalnej sciezki
    int opt = -1;
    int parent;
    for(int k=1; k<n; k++)  {
        int cost = C[bits][0][k] + dist[k*n + 0];
        if(opt== -1 || cost<opt){
            opt = cost;
            parent = k;
        }
    }

    // Zapisanie optymalnej sciezki do tablicy path
    for(int i=n-1; i>0; i--) {
        path[i] = parent;
        int new_bits = bits & ~(1 << parent);
        parent = C[bits][1][parent];
        bits = new_bits;
    }

    // Zapisanie poczatkowego wierzcholka (0)
    path[0] = 0;
    
    free(C);

    // Zwrocenie kosztu optymalnej sciezki
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