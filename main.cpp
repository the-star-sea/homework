#include <intrin.h>
#include <chrono>
#include "iostream"
#include "cblas.h"
#include <memory.h>

#define BLOCKSIZE 1024
struct matrix {
    unsigned int column, row;
    float *data;
};

void mode1();

void mode2();

void write();

void test1();

void test4();

void test2();

void O3blockompavx();


void ompcaculation();

void test3();

void degmm();

void test5();

void withoutblock();

void doblock(int si, int sj, int sk, int i, int i1, int i2);

using namespace std;
float *ans;
matrix matrix1, matrix2;
long u;
#pragma GCC optimize(3, "unroll-loops", "Ofast")

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
    memset(ans, 0, sizeof(float) * matrix1.row * matrix2.column);
    using namespace std::literals; // enables the usage of 24h instead of std::chrono::hours(24)
    auto now = std::chrono::system_clock::now();

    auto start = std::chrono::steady_clock::now();
    test1();
    auto start1 = std::chrono::steady_clock::now();
    test2();
    auto end = std::chrono::steady_clock::now();


    std::cout
            << "omp+avx+block+o3 calculations took "
            << std::chrono::duration_cast<std::chrono::microseconds>(start1 - start).count() << "µs ≈ "
            << std::chrono::duration_cast<std::chrono::milliseconds>(start1 - start).count() << "ms ≈ "
            << std::chrono::duration_cast<std::chrono::seconds>(start1 - start).count() << "s.\n"
            << "openblas took "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start1).count() << "µs ≈ "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start1).count() << "ms ≈ "
            << std::chrono::duration_cast<std::chrono::seconds>(end - start1).count() << "s.\n";
    cout << "Do u wanna try the other function to have a comparision?Y/the other";
    char h;
    cin >> h;
    if (h != 'Y') { exit(0); }
    if (h == 'Y') {
        auto start2 = std::chrono::steady_clock::now();
        test3();

        auto start3 = std::chrono::steady_clock::now();

        test4();
        auto start4 = std::chrono::steady_clock::now();
        test5();

        auto end1 = std::chrono::steady_clock::now();
        cout << "omp+avx+o3 calculations took "
             << std::chrono::duration_cast<std::chrono::microseconds>(start3 - start2).count() << "µs ≈ "
             << std::chrono::duration_cast<std::chrono::milliseconds>(start3 - start2).count() << "ms ≈ "
             << std::chrono::duration_cast<std::chrono::seconds>(start3 - start2).count() << "s.\n"

             << "omp calculations took "
             << std::chrono::duration_cast<std::chrono::microseconds>(start4 - start3).count() << "µs ≈ "
             << std::chrono::duration_cast<std::chrono::milliseconds>(start4 - start3).count() << "ms ≈ "
             << std::chrono::duration_cast<std::chrono::seconds>(start4 - start3).count() << "s.\n"
             << "avx took "
             << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start4).count() << "µs ≈ "
             << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start4).count() << "ms ≈ "
             << std::chrono::duration_cast<std::chrono::seconds>(end1 - start4).count() << "s.\n";


    }
}

void test5() {
    degmm();
}

void innerproduct(int i, int j) {
    __m256 acc = _mm256_setzero_ps();
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

void withoutblock() {

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < matrix1.row; ++i) {
        for (int j = 0; j < matrix2.column; ++j) {

            innerproduct(i, j);
        }

    }
}

void test3() {
    withoutblock();
}

void degmm() {

    for (int i = 0; i < matrix1.row; ++i) {
        for (int j = 0; j < matrix2.column; ++j) {
            __m256 acc = _mm256_setzero_ps();
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

void test2() {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, u, u, u, 1.0, matrix1.data, u, matrix2.data, u, 0.0, ans, u);

}

void test4() {
    ompcaculation();
}

void test1() {
    O3blockompavx();

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
        cin.clear();
        cin.sync();
        cout << "please check your type." << endl << "retype:";
        cin >> matrix1.column;
    }
    cout << "how many rows?";
    cin >> matrix1.row;
    while (cin.fail()) {
        cin.clear();
        cin.sync();
        cout << "please check your type." << endl << "retype:";
        cin >> matrix1.row;
    }
    cout << "the second matrix: " << endl << "how many columns?";
    cin >> matrix2.column;
    while (cin.fail()) {
        cin.clear();
        cin.sync();
        cout << "please check your type." << endl << "retype:";
        cin >> matrix2.column;
    }
    cout << "how many rows?";
    cin >> matrix2.row;
    while (cin.fail()) {
        cin.clear();
        cin.sync();
        cout << "please check your type." << endl << "retype";
        cin >> matrix2.row;
    }
    if (matrix1.column != matrix2.row) {
        cout << "size donot match";
        mode1();
    }
    char g;

    cout << "please input the first matrix : " << endl;
    matrix1.data = new float[matrix1.column * matrix1.row];
    matrix2.data = new float[matrix2.column * matrix2.row];
    for (int i = 0; i < matrix1.column * matrix1.row; i++) {
        cin >> matrix1.data[i];
        while (cin.fail()) {
            cin.clear();
            cin.sync();
            cout << "please check your type." << endl << "retype from local position:";
            cin >> matrix1.data[i];
        }
    }
    cout << "Is it  row-major order?N/the other";
    cin >> g;
    if (g == 'N') {
        float s[matrix1.column * matrix1.row];
        for (int i = 0; i < matrix1.column; i++) {
            for (int j = 0; j < matrix1.row; j++) {
                s[i + j * matrix1.column] = matrix1.data[j + i * matrix1.row];
            }
        }
        matrix1.data = s;
    }
    cout << "please input the second matrix : " << endl;
    for (int i = 0; i < matrix2.column * matrix2.row; i++) {
        cin >> matrix2.data[i];
        while (cin.fail()) {
            cin.clear();
            cin.sync();
            cout << "please check your type." << endl << "retype from local position:";
            cin >> matrix2.data[i];
        }
    }
    cout << "Is it  column-major order?N/the other";
    cin >> g;
    if (g == 'N') {
        float s[matrix2.column * matrix2.row];
        for (int i = 0; i < matrix2.column; i++) {
            for (int j = 0; j < matrix2.row; j++) {
                s[j + i * matrix2.row] = matrix2.data[i + j * matrix2.column];
            }
        }
        matrix2.data = s;
    }
    ans = new float[matrix1.row * matrix2.column];
    memset(ans, 0, sizeof(float) * matrix1.row * matrix2.column);

    O3blockompavx();
    for (int i = 0; i < matrix1.row; i++) {
        for (int j = 0; j < matrix2.column; j++) {
            cout << ans[j + i * matrix2.column] << " ";
        }
        cout << endl;
    }

}

void O3blockompavx() {
    int m, n, p, si, sj, sk;
#pragma omp parallel for schedule(dynamic)
    for (sj = 0; sj < matrix2.column; sj += BLOCKSIZE) {
        for (si = 0; si < matrix1.row; si += BLOCKSIZE) {
            for (sk = 0; sk < matrix1.column; sk += BLOCKSIZE) {
                m = matrix1.row < si + BLOCKSIZE ? matrix1.row : si + BLOCKSIZE;
                n = matrix2.column < sj + BLOCKSIZE ? matrix2.column : sj + BLOCKSIZE;
                p = sk + BLOCKSIZE < matrix1.column ? sk + BLOCKSIZE : matrix1.column;
                doblock(si, sj, sk, m, n, p);

            }

        }

    }
}

void doblock(int si, int sj, int sk, int m, int n, int p) {
    for (int i = si; i < m; i++) {
        for (int j = sj; j < n; j++) {
            __m256 acc = _mm256_setzero_ps();
            float temp[8];
            float inner_prod;
            int k;
            for (k = sk; k + 8 < p; k += 8) {

                acc = _mm256_add_ps(acc,
                                    _mm256_mul_ps(_mm256_loadu_ps(matrix1.data + k + i * matrix1.column),
                                                  _mm256_loadu_ps(matrix2.data + k + j * matrix1.column)));
            }
            _mm256_storeu_ps(&temp[0], acc);
            inner_prod = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] +
                         temp[6] + temp[7] + temp[8];
            for (; k < p; k++) {
                inner_prod += matrix1.data[k + i * matrix1.column] * matrix2.data[k + j * matrix1.column];
            }
            ans[j + i * matrix2.column] += inner_prod;
        }

    }
}


void ompcaculation() {

    int i, j, k;
#pragma omp parallel for shared(ans)private(i, j, k)
    for (i = 0; i < matrix1.row; i++) {
        for (j = 0; j < matrix2.column; j++) {
            for (k = 0; k < matrix1.column; k++) {

                ans[j + i * matrix2.column] +=
                        matrix1.data[k + i * matrix1.column] * matrix2.data[k + j * matrix1.column];

            }
        }
    }
}
