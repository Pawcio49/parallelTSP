#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define N 5 // liczba wierzchołków w grafie

// przykładowy graf ważony w postaci macierzy sąsiedztwa
int graph[N][N] = {
    {0, 2, 9, 10, 4},
    {1, 0, 6, 4, 8},
    {15, 7, 0, 8, 12},
    {6, 3, 12, 0, 5},
    {9, 1, 7, 3, 0}
};

int visited[N] = {0};
int best_path[N];
int best_distance = INT_MAX;

// funkcja obliczająca długość trasy dla danego rozwiązania
int calculate_distance(int *solution) {
    int distance = 0;
    for (int i = 0; i < N - 1; i++) {
        distance += graph[solution[i]][solution[i+1]];
    }
    distance += graph[solution[N-1]][solution[0]];
    return distance;
}

// funkcja przeglądania drzewa w głąb
void dfs(int current_vertex, int current_distance, int *path) {
    if (current_vertex < 0 || current_vertex >= N) {
        return;
    }
    if (current_vertex == 0 && current_distance != 0) {
        return;
    }
    if (current_distance >= best_distance) {
        return;
    }
    if (visited[current_vertex]) {
        return;
    }
    visited[current_vertex] = 1;
    path[N - visited[current_vertex]] = current_vertex;
    if (visited[0] == 1) {
        int distance = calculate_distance(path);
        if (distance < best_distance) {
            best_distance = distance;
            for (int i = 0; i < N; i++) {
                best_path[i] = path[i];
            }
        }
    } else {
        for (int i = 0; i < N; i++) {
            dfs(i, current_distance + graph[current_vertex][i], path);
        }
    }
    visited[current_vertex] = 0;
}

int main() {
    int path[N];
    dfs(0, 0, path);
    printf("Najkrótsza droga to: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", best_path[i]);
    }
    printf("%d\n", best_path[0]);
    printf("Długość drogi: %d\n", best_distance);
    return 0;
}