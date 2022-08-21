#include "../include/gemm.h"

#define gamma(i, j) c[(j) * (ldC) + (i)]
#define alpha(i, j) a[(j) * (ldA) + (i)]
#define beta(i, j) a[(j) * (ldB) + (i)]


void JPI_Loop(int m, int n, int k, double* a, int ldA, double* b, int ldB, double* c, int ldC) {
    for (int j = 0; j < n; ++j) {
        for (int p = 0; p < k; ++p) {
            for (int i = 0; i < m; ++i) {
                gamma(i, j) += alpha(i, p) * beta(p, j);
            }
        }
    }
}