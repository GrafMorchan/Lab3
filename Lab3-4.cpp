#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

// Поэлементная операция над двумя массивами
void performOperation(const vector<double>& a, const vector<double>& b, vector<double>& result, char op) {
    size_t size = a.size();
    for (size_t i = 0; i < size; ++i) {
        switch (op) {
        case '+': result[i] = a[i] + b[i]; break;
        case '-': result[i] = a[i] - b[i]; break;
        case '*': result[i] = a[i] * b[i]; break;
        case '/':
            if (b[i] != 0)
                result[i] = a[i] / b[i];
            else {
                cerr << "Warning: Division by zero at index " << i << endl;
                result[i] = 0;
            }
            break;
        default:
            cerr << "Error: Invalid operation!" << endl;
            return;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(nullptr, nullptr);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Размеры двумерного массива (например 400x250 = 100000 элементов)
    const int M = 400;
    const int N = 250;
    const int total_size = M * N;

    char operation = '+'; // Можно менять на '+', '-', '*', '/'

    bool runParallel = (size > 1);

    if (runParallel) {
        // Параллельный вариант

        vector<double> a, b, result;
        if (rank == 0) {
            a.resize(total_size, 1.0);
            b.resize(total_size, 2.0);
            result.resize(total_size);
        }

        // Рассчёт размеров для scatter/gather
        vector<int> sendcounts(size);
        vector<int> displs(size);

        int base_size = total_size / size;
        int remainder = total_size % size;

        int offset = 0;
        for (int i = 0; i < size; ++i) {
            sendcounts[i] = base_size + (i < remainder ? 1 : 0);
            displs[i] = offset;
            offset += sendcounts[i];
        }

        int local_size = sendcounts[rank];
        vector<double> local_a(local_size);
        vector<double> local_b(local_size);
        vector<double> local_result(local_size);

        // Рассылаем части массивов
        MPI_Scatterv(rank == 0 ? a.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
            local_a.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(rank == 0 ? b.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
            local_b.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD); // Синхронизация перед замером времени
        auto start = chrono::high_resolution_clock::now();

        performOperation(local_a, local_b, local_result, operation);

        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

        // Собираем результаты
        MPI_Gatherv(local_result.data(), local_size, MPI_DOUBLE,
            rank == 0 ? result.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        if (rank == 0) {
            cout << "Parallel " << operation << " operation with " << size << " processes took "
                << duration.count() << " microseconds." << endl;

            cout << "First 10 elements of result: ";
            for (int i = 0; i < min(10, total_size); ++i)
                cout << result[i] << " ";
            cout << endl;
        }
    }
    else {
        // Последовательный вариант
        vector<double> a(total_size, 1.0);
        vector<double> b(total_size, 2.0);
        vector<double> result(total_size);

        auto start = chrono::high_resolution_clock::now();

        performOperation(a, b, result, operation);

        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

        cout << "Sequential " << operation << " operation took " << duration.count() << " microseconds." << endl;

        cout << "First 10 elements of result: ";
        for (int i = 0; i < min(10, total_size); ++i)
            cout << result[i] << " ";
        cout << endl;
    }

    MPI_Finalize();
    return 0;
}
