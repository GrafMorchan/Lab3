#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>// Include MPI header

using namespace std;
using namespace std::chrono;

// Функция выполнения операции над массивами
vector<double> performOperation(const vector<double>& a, const vector<double>& b, char op) {
    if (a.size() != b.size()) {
        cerr << "Error: Arrays must have the same size." << endl;
        return {}; // Возвращаем пустой вектор в случае ошибки
    }

    vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        switch (op) {
        case '+': result[i] = a[i] + b[i]; break;
        case '-': result[i] = a[i] - b[i]; break;
        case '*': result[i] = a[i] * b[i]; break;
        case '/':
            if (b[i] != 0) {
                result[i] = a[i] / b[i];
            }
            else {
                cerr << "Warning: Division by zero at index " << i << endl;
                result[i] = 0; // Или другое разумное значение по умолчанию
            }
            break;
        default: cerr << "Error: Invalid operation." << endl; return {};
        }
    }
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(nullptr, nullptr); // Initialize MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processes

    const int N = 100000;
    char operation = '+'; // Выберите операцию: +, -, *, /
    bool runParallel = (size > 1); // Use parallel version only if multiple processes are running

    if (runParallel) {
        // Parallel version

        // Master process initializes the data
        vector<double> a, b, result;
        if (rank == 0) {
            a.resize(N, 1.0);
            b.resize(N, 2.0);
            result.resize(N);
        }

        // Calculate local array size
        int local_size = N / size;
        int remainder = N % size;
        int local_start = rank * local_size;
        if (rank < remainder) {
            local_size++;
            local_start += rank;
        }
        else {
            local_start += remainder;
        }
        vector<double> local_a(local_size);
        vector<double> local_b(local_size);
        vector<double> local_result(local_size);

        // Create displacements and recvcounts for Scatterv and Gatherv
        vector<int> displacements(size);
        vector<int> recvcounts(size);
        int current_displacement = 0;
        for (int i = 0; i < size; ++i) {
            recvcounts[i] = N / size;
            if (i < remainder) {
                recvcounts[i]++;
            }
            displacements[i] = current_displacement;
            current_displacement += recvcounts[i];
        }


        // Scatter data to all processes
        MPI_Scatterv(rank == 0 ? a.data() : NULL,
            recvcounts.data(),
            displacements.data(),
            MPI_DOUBLE,
            local_a.data(), local_size, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        MPI_Scatterv(rank == 0 ? b.data() : NULL,
            recvcounts.data(),
            displacements.data(),
            MPI_DOUBLE,
            local_b.data(), local_size, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        // Perform local operation
        auto start = high_resolution_clock::now();
        local_result = performOperation(local_a, local_b, operation);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);

        // Gather results to master process
        MPI_Gatherv(local_result.data(), local_size, MPI_DOUBLE,
            rank == 0 ? result.data() : NULL,
            recvcounts.data(),
            displacements.data(),
            MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        if (rank == 0) {
            cout << "Parallel " << operation << " with " << size << " processes Duration: " << duration.count() << " microseconds" << endl;

            // Optional: Print first 10 elements of the result
            cout << "First 10 elements of the result: ";
            for (int i = 0; i < min((int)result.size(), 10); ++i) {
                cout << result[i] << " ";
            }
            cout << endl;
        }
    }
    else {
        // Sequential version (executed when only one process is running)
        vector<double> a(N, 1.0);
        vector<double> b(N, 2.0);

        auto start = high_resolution_clock::now();
        vector<double> result = performOperation(a, b, operation);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);

        cout << "Sequential " << operation << " Duration: " << duration.count() << " microseconds" << endl;

        // Optional: Print first 10 elements of the result
        cout << "First 10 elements of the result: ";
        for (int i = 0; i < min((int)result.size(), 10); ++i) {
            cout << result[i] << " ";
        }
        cout << endl;
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}
