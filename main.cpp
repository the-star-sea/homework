#include <intrin.h>
#include <chrono>
#include "iostream"
#include "cblas.h"
#include "omp.h"
#include <memory.h>

#define BLOCKSIZE 16
struct matrix {
    unsigned int column, row;
    float *data;
};

void mode1();

void mode2();

void write();

void test2();

void test1();

void test3();

void multi();

void dgemm_avx();


void normalcaculate();

using namespace std;
float *ans;
matrix matrix1, matrix2;
long u;

int main() {
    cout << "please input the mode:" << endl
         << "1.caculation" << endl
         << "2.test" << endl;
    int t;
    cin >> t;
    if (t == 1) {
        mode1();
    }
    if (t == 2) {
        mode2();

    }

}

void mode2() {
    cout << "Now it is the fast mode.please input the size of the data: ";

    cin >> u;

    write();
    using namespace std::literals; // enables the usage of 24h instead of std::chrono::hours(24)
    auto now = std::chrono::system_clock::now();

    auto start = std::chrono::steady_clock::now();
    test2();
    auto end = std::chrono::steady_clock::now();
    test3();
    auto end2 = std::chrono::steady_clock::now();
    std::cout
            << "fast calculations took "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "µs ≈ "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms ≈ "
            << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s.\n"
            << "openblas took "
            << std::chrono::duration_cast<std::chrono::microseconds>(end2 - end).count() << "µs ≈ "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - end).count() << "ms ≈ "
            << std::chrono::duration_cast<std::chrono::seconds>(end2 - end).count() << "s.\n";

    cout << "Do u wanna try the slow mode?maybe ten minutes....Y/N" << endl;
    char ye;
    cin >> ye;

    if (ye == 'Y') {
        auto start1 = std::chrono::steady_clock::now();
        test1();
        auto end1 = std::chrono::steady_clock::now();
        std::cout
                << "Slow calculations took "
                << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() << "µs ≈ "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << "ms ≈ "
                << std::chrono::duration_cast<std::chrono::seconds>(end1 - start1).count() << "s.\n";
    } else { exit(0); }
}

void test3() {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, u, u, u, 1.0, matrix1.data, u, matrix2.data, u, 0.0, ans, 1);

}

void test1() {
    normalcaculate();
}

void test2() {
    multi();

}

void write() {
    matrix1.row = u;
    matrix1.column = u;
    matrix2.row = u;
    matrix2.column = u;
    matrix1.data = new float[u * u];
    matrix2.data = new float[u * u];
    for (long i = 0; i < u * u; i++) {
        matrix1.data[i] = (rand() % 2001) - 1000 + rand() / double(RAND_MAX);
        matrix2.data[i] = (rand() % 2001) - 1000 + rand() / double(RAND_MAX);
    }
    ans = new float[u * u];
}

void mode1() {
    cout << "please input two matrixes:" << endl << "the first matrix: " << endl << "how many columns?";
    cin >> matrix1.column;
    while (cin.fail()) {
        cout << "please check your type." << endl << "retype:";
        cin >> matrix1.column;
    }
    cout << "how many rows?";
    cin >> matrix1.row;
    while (cin.fail()) {
        cout << "please check your type." << endl << "retype:";
        cin >> matrix1.row;
    }
    cout << "the second matrix: " << endl << "how many columns?";
    cin >> matrix2.column;
    while (cin.fail()) {
        cout << "please check your type." << endl << "retype:";
        cin >> matrix2.column;
    }
    cout << "how many rows?";
    cin >> matrix2.row;
    while (cin.fail()) {
        cout << "please check your type." << endl << "retype";
        cin >> matrix2.row;
    }
    if (matrix1.column != matrix2.row) {
        cout << "size donot match";
        mode1();
    }
    cout << "please input the first matrix row by row: " << endl;
    matrix1.data = new float[matrix1.column * matrix1.row];
    matrix2.data = new float[matrix2.column * matrix2.row];
    for (int i = 0; i < matrix1.column * matrix1.row; i++) {
        cin >> matrix1.data[i];
        while (cin.fail()) {
            cout << "please check your type." << endl << "retype from local position:";
            cin >> matrix1.data[i];
        }
    }
    cout << "please input the second matrix column by column: " << endl;
    for (int i = 0; i < matrix2.column * matrix2.row; i++) {
        cin >> matrix2.data[i];
        while (cin.fail()) {
            cout << "please check your type." << endl << "retype from local position:";
            cin >> matrix2.data[i];
        }
    }
    ans = new float[matrix1.row * matrix2.column];


    multi();
    for (int i = 0; i < matrix1.row; i++) {
        for (int j = 0; j < matrix2.column; j++) {
            cout << ans[j + i * matrix2.column] << " ";
        }
        cout << endl;
    }

}


void multi() {
#pragma omp parallel for
    for (int sj = 0; sj < matrix2.column; sj += BLOCKSIZE) {
        for (int si = 0; si < matrix1.row; si += BLOCKSIZE) {
#pragma UNROLL 4
            for (int i = si; i < matrix1.row && i < si + BLOCKSIZE; i++) {
                for (int j = sj; j < matrix2.column && j < sj + BLOCKSIZE; ++j) {
                    __m256 acc;
                    int k;
                    float temp[8];
                    float inner_prod;

                    for (k = 0; k + 8 < matrix1.column; k += 8) {

                        acc = _mm256_add_ps(acc,
                                            _mm256_mul_ps(_mm256_loadu_ps(matrix1.data + k + i * matrix1.column),
                                                          _mm256_loadu_ps(matrix2.data + k + j * matrix1.column)));
                    }
                    _mm256_storeu_ps(&temp[0], acc);
                    inner_prod = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] +
                                 temp[6] + temp[7] + temp[8];
                    for (; k < matrix1.column; ++k) {
                        inner_prod += matrix1.data[k + i * matrix1.column] * matrix2.data[k + j * matrix1.column];
                    }
                    ans[j + i * matrix2.column] = inner_prod;
                }

            }
        }
    }

}

void normalcaculate() {
    memset(ans, 0, sizeof(float) * matrix1.row * matrix2.column);
    for (int i = 0; i < matrix1.row; i++) {
        for (int j = 0; j < matrix2.column; j++) {
            for (int k = 0; k < matrix1.column; k++) {
                ans[j + i * matrix2.column] +=
                        matrix1.data[k + i * matrix1.column] * matrix2.data[k + j * matrix1.column];

            }
        }
    }
}


