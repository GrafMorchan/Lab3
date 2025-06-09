#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define SIZE 100000

int main(int argc, char* argv[]) {
    int rank, size;
    int* array = NULL;
    long long local_sum = 0, total_sum = 0;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Инициализация массива (только на процессе 0)
    if (rank == 0) {
        array = (int*)malloc(SIZE * sizeof(int));
        srand(time(NULL));
        for (int i = 0; i < SIZE; i++) {
            array[i] = rand() % 100;
        }
    }

    int chunk_size = SIZE / size;
    int* sub_array = (int*)malloc(chunk_size * sizeof(int));

    // Распределение данных
    MPI_Scatter(array, chunk_size, MPI_INT, sub_array, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Замер времени
    start_time = MPI_Wtime();

    // Вычисление локальной суммы
    local_sum = 0;
    for (int i = 0; i < chunk_size; i++) {
        local_sum += sub_array[i];
    }

    // Сбор результатов
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Total time: %.6f seconds\n", end_time - start_time);
        printf("Total sum: %lld\n", total_sum);
    }

    free(sub_array);
    if (rank == 0) {
        free(array);
    }

    MPI_Finalize();
    return 0;
}
