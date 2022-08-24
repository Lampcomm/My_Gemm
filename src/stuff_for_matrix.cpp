#include "../include/stuff_for_matrix .h"
#include <random>
#include <iostream>
#include <assert.h>

#define dabs(x) ( (x) < 0 ? -(x) : x )
#define mtEl(i, j) a[(j) * (ld) + (i)]

void generate_random_matrix(int size, double* a) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dis(0.0, 1e3);

    for (int i = 0; i < size; ++i) {
        *a++ = dis(mt);
        // *a++ = static_cast<double>(1);
    }
}

void print_matrix(int n, int m, int ld, double* a) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            std::cout << mtEl(i, j) << '\t';
        }
        std::cout << '\n';
    }
}

double max_diff(int n, int m, double* c_ref, double* c) {
    double ans = 0.0;
    for (int i = 0; i < n * m; ++i, ++c_ref, ++c) {
        if (dabs(*c_ref - *c) > ans) {
            ans = dabs(*c_ref - *c);
        }
    }

    return ans;
}