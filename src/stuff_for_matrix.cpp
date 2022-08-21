#include "../include/stuff_for_matrix .h"
#include <random>
#include <iostream>
#include <limits>
#include <assert.h>

#define mtEl(i, j) a[(j) * (ld) + (i)]
#define CRef(i, j) c_ref[(j) * (ldCRef) + (i)]
#define C(i, j) c_ref[(j) * (ldC) + (i)]

void generate_random_matrix(int size, double* a) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dis(0.0, 1e9);

    for (int i = 0; i < size; ++i) {
        *a++ = dis(mt);
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

void check(int n, int m, double* c_ref, int ldCRef, double* c, int ldC) {
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < n; ++i) {
            assert((std::abs(CRef(i, j) - C(i, j)) < std::numeric_limits<double>::epsilon()) && "Wrong answer");
        }
    }
}