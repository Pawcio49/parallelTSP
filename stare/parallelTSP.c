// C++ program to solve Traveling Salesman Problem
// using Branch and Bound.
#include <stdio.h>
#include <limits.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

// Stores the final minimum weight of shortest tour.
int final_res = INT_MAX;

// Function to copy temporary solution to
// the final solution
void copyToFinal(int N, int curr_path[], int final_path[N+1])
{
    for (int i=0; i<N; i++)
        final_path[i] = curr_path[i];
    final_path[N] = curr_path[0];
}

// Function to find the minimum edge cost
// having an end at the vertex i
int firstMin(int N, int adj[N][N], int i)
{
    int min = INT_MAX;
    for (int k=0; k<N; k++)
        if (adj[i][k]<min && i != k)
            min = adj[i][k];
    return min;
}

// function to find the second minimum edge cost
// having an end at the vertex i
int secondMin(int N, int adj[N][N], int i)
{
    int first = INT_MAX, second = INT_MAX;
    for (int j=0; j<N; j++)
    {
        if (i == j)
            continue;

        if (adj[i][j] <= first)
        {
            second = first;
            first = adj[i][j];
        }
        else if (adj[i][j] <= second &&
                 adj[i][j] != first)
            second = adj[i][j];
    }
    return second;
}

// function that takes as arguments:
// curr_bound -> lower bound of the root node
// curr_weight-> stores the weight of the path so far
// level-> current level while moving in the search
//		 space tree
// curr_path[] -> where the solution is being stored which
//			 would later be copied to final_path[]
void TSPRec(int N, int adj[N][N], int curr_bound, int curr_weight,
            int level, int curr_path[], bool visited[N], int final_path[N+1])
{
    // base case is when we have reached level N which
    // means we have covered all the nodes once
    if (level==N)
    {
        // check if there is an edge from last vertex in
        // path back to the first vertex
        if (adj[curr_path[level-1]][curr_path[0]] != 0)
        {
            // curr_res has the total weight of the
            // solution we got
            int curr_res = curr_weight +
                           adj[curr_path[level-1]][curr_path[0]];

            // Update final result and final path if
            // current result is better.
            if (curr_res < final_res)
            {
                copyToFinal(N, curr_path, final_path);
                final_res = curr_res;
            }
        }
        return;
    }

    // for any other level iterate for all vertices to
    // build the search space tree recursively
    #pragma omp parallel for
    for (int i=0; i<N; i++)
    {
        // Consider next vertex if it is not same (diagonal
        // entry in adjacency matrix and not visited
        // already)
        if (adj[curr_path[level-1]][i] != 0 &&
            visited[i] == false)
        {
            int temp = curr_bound;
            curr_weight += adj[curr_path[level-1]][i];

            // different computation of curr_bound for
            // level 2 from the other levels
            if (level==1)
                curr_bound -= ((firstMin(N, adj, curr_path[level-1]) +
                                firstMin(N, adj, i))/2);
            else
                curr_bound -= ((secondMin(N, adj, curr_path[level-1]) +
                                firstMin(N, adj, i))/2);

            // curr_bound + curr_weight is the actual lower bound
            // for the node that we have arrived on
            // If current lower bound < final_res, we need to explore
            // the node further
            if (curr_bound + curr_weight < final_res)
            {
                curr_path[level] = i;
                visited[i] = true;

                // call TSPRec for the next level
                TSPRec(N, adj, curr_bound, curr_weight, level+1,
                       curr_path, visited, final_path);

                // Also reset the visited array
                visited[i] = false;
            }

            // Else we have to prune the node by resetting
            // all changes to curr_weight and curr_bound
            curr_weight -= adj[curr_path[level-1]][i];
            curr_bound = temp;
        }
    }
}

// This function sets up final_path[]
void TSP(int N, int adj[N][N], bool visited[N], int final_path[N+1])
{
    int curr_path[N+1];

    // Calculate initial lower bound for the root node
    // using the formula 1/2 * (sum of first min +
    // second min) for all edges.
    // Also initialize the curr_path and visited array
    int curr_bound = 0;
    memset(curr_path, -1, sizeof(curr_path));
    memset(visited, 0, sizeof(curr_path));

    // Compute initial bound
    #pragma omp parallel for
    for (int i=0; i<N; i++)
        curr_bound += (firstMin(N, adj, i) +
                       secondMin(N, adj, i));

    // Rounding off the lower bound to an integer
    curr_bound = (curr_bound%2==0)? curr_bound/2:
                 curr_bound/2 + 1;

    // We start at vertex 1 so the first vertex
    // in curr_path[] is 0
    visited[0] = true;
    curr_path[0] = 0;

    // Call to TSPRec for curr_weight equal to
    // 0 and level 1
    TSPRec(N, adj, curr_bound, 0, 1, curr_path, visited, final_path);
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
