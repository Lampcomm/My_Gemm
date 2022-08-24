#include "../include/gemm.h"
#include "../include/stuff_for_matrix .h"
#include <iostream>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>

#define gamma(i, j) C[(j) * (ldC) + (i)]
#define alpha(i, j) A[(j) * (ldA) + (i)]
#define beta(i, j) B[(j) * (ldB) + (i)]

constexpr int MC = 192,
              NC = 1024,
              KC = 2048,
              MR = 12,
              NR = 4;

void JPI_Loop(int m, int n, int k, double* A, int ldA, double* B, int ldB, double* C, int ldC) {
    for (int j = 0; j < n; ++j) {
        for (int p = 0; p < k; ++p) {
            for (int i = 0; i < m; ++i) {
                gamma(i, j) += alpha(i, p) * beta(p, j);
            }
        }
    }
}

void kernel(int m, int n, int k, double* A, int ldA, double* B, int ldB, double* C, int ldC) {
    for (int j = 0; j < n; ++j) {
        for (int p = 0; p < k; ++p) {
            for (int i = 0; i < m; ++i) {
                gamma(i, j) += alpha(i, p) * beta(p, j);
            }
        }
    }
}

// 12x4
void kernel_packed(int k, double* ap, double* bp, double* C, int ldC) {
    __m256d gamma_0123_0 = _mm256_loadu_pd(&gamma(0, 0));
    __m256d gamma_4567_0 = _mm256_loadu_pd(&gamma(4, 0));
    __m256d gamma_89AB_0 = _mm256_loadu_pd(&gamma(8, 0));
    __m256d gamma_0123_1 = _mm256_loadu_pd(&gamma(0, 1));
    __m256d gamma_4567_1 = _mm256_loadu_pd(&gamma(4, 1));
    __m256d gamma_89AB_1 = _mm256_loadu_pd(&gamma(8, 1));
    __m256d gamma_0123_2 = _mm256_loadu_pd(&gamma(0, 2));
    __m256d gamma_4567_2 = _mm256_loadu_pd(&gamma(4, 2));
    __m256d gamma_89AB_2 = _mm256_loadu_pd(&gamma(8, 2));
    __m256d gamma_0123_3 = _mm256_loadu_pd(&gamma(0, 3));
    __m256d gamma_4567_3 = _mm256_loadu_pd(&gamma(4, 3));
    __m256d gamma_89AB_3 = _mm256_loadu_pd(&gamma(8, 3));
    __m256d beta_j;

    for (int p = 0; p < k; ++p) {
        __m256d alpha_0123_p = _mm256_loadu_pd(ap);
        __m256d alpha_4567_p = _mm256_loadu_pd(ap + 4);
        __m256d alpha_89AB_p = _mm256_loadu_pd(ap + 8);

        beta_j = _mm256_broadcast_sd(bp);
        gamma_0123_0 = _mm256_fmadd_pd(alpha_0123_p, beta_j, gamma_0123_0);
        gamma_4567_0 = _mm256_fmadd_pd(alpha_4567_p, beta_j, gamma_4567_0);
        gamma_89AB_0 = _mm256_fmadd_pd(alpha_89AB_p, beta_j, gamma_89AB_0);

        beta_j = _mm256_broadcast_sd(bp + 1);
        gamma_0123_1 = _mm256_fmadd_pd(alpha_0123_p, beta_j, gamma_0123_1);
        gamma_4567_1 = _mm256_fmadd_pd(alpha_4567_p, beta_j, gamma_4567_1);
        gamma_89AB_1 = _mm256_fmadd_pd(alpha_89AB_p, beta_j, gamma_89AB_1);

        beta_j = _mm256_broadcast_sd(bp + 2);
        gamma_0123_2 = _mm256_fmadd_pd(alpha_0123_p, beta_j, gamma_0123_2);
        gamma_4567_2 = _mm256_fmadd_pd(alpha_4567_p, beta_j, gamma_4567_2);
        gamma_89AB_2 = _mm256_fmadd_pd(alpha_89AB_p, beta_j, gamma_89AB_2);

        beta_j = _mm256_broadcast_sd(bp + 3);
        gamma_0123_3 = _mm256_fmadd_pd(alpha_0123_p, beta_j, gamma_0123_3);
        gamma_4567_3 = _mm256_fmadd_pd(alpha_4567_p, beta_j, gamma_4567_3);
        gamma_89AB_3 = _mm256_fmadd_pd(alpha_89AB_p, beta_j, gamma_89AB_3);

        ap += MR;
        bp += NR;
    }

    _mm256_storeu_pd(&gamma(0, 0), gamma_0123_0);
    _mm256_storeu_pd(&gamma(4, 0), gamma_4567_0);
    _mm256_storeu_pd(&gamma(8, 0), gamma_89AB_0);
    _mm256_storeu_pd(&gamma(0, 1), gamma_0123_1);
    _mm256_storeu_pd(&gamma(4, 1), gamma_4567_1);
    _mm256_storeu_pd(&gamma(8, 1), gamma_89AB_1);
    _mm256_storeu_pd(&gamma(0, 2), gamma_0123_2);
    _mm256_storeu_pd(&gamma(4, 2), gamma_4567_2);
    _mm256_storeu_pd(&gamma(8, 2), gamma_89AB_2);
    _mm256_storeu_pd(&gamma(0, 3), gamma_0123_3);
    _mm256_storeu_pd(&gamma(4, 3), gamma_4567_3);
    _mm256_storeu_pd(&gamma(8, 3), gamma_89AB_3);
}

// 8x6
// void kernel_packed(int k, double* ap, double* bp, double* C, int ldC) {
//     __m256d gamma_0123_0 = _mm256_loadu_pd(&gamma(0, 0));
//     __m256d gamma_4567_0 = _mm256_loadu_pd(&gamma(4, 0));
//     __m256d gamma_0123_1 = _mm256_loadu_pd(&gamma(0, 1));
//     __m256d gamma_4567_1 = _mm256_loadu_pd(&gamma(4, 1));
//     __m256d gamma_0123_2 = _mm256_loadu_pd(&gamma(0, 2));
//     __m256d gamma_4567_2 = _mm256_loadu_pd(&gamma(4, 2));
//     __m256d gamma_0123_3 = _mm256_loadu_pd(&gamma(0, 3));
//     __m256d gamma_4567_3 = _mm256_loadu_pd(&gamma(4, 3));
//     __m256d gamma_0123_4 = _mm256_loadu_pd(&gamma(0, 4));
//     __m256d gamma_4567_4 = _mm256_loadu_pd(&gamma(4, 4));
//     __m256d gamma_0123_5 = _mm256_loadu_pd(&gamma(0, 5));
//     __m256d gamma_4567_5 = _mm256_loadu_pd(&gamma(4, 5));
//     __m256d beta_j;

//     for (int p = 0; p < k; ++p) {
//         __m256d alpha_0123_p = _mm256_loadu_pd(ap);
//         __m256d alpha_4567_p = _mm256_loadu_pd(ap + 4);

//         beta_j = _mm256_broadcast_sd(bp);
//         gamma_0123_0 = _mm256_fmadd_pd(alpha_0123_p, beta_j, gamma_0123_0);
//         gamma_4567_0 = _mm256_fmadd_pd(alpha_4567_p, beta_j, gamma_4567_0);

//         beta_j = _mm256_broadcast_sd(bp + 1);
//         gamma_0123_1 = _mm256_fmadd_pd(alpha_0123_p, beta_j, gamma_0123_1);
//         gamma_4567_1 = _mm256_fmadd_pd(alpha_4567_p, beta_j, gamma_4567_1);

//         beta_j = _mm256_broadcast_sd(bp + 2);
//         gamma_0123_2 = _mm256_fmadd_pd(alpha_0123_p, beta_j, gamma_0123_2);
//         gamma_4567_2 = _mm256_fmadd_pd(alpha_4567_p, beta_j, gamma_4567_2);

//         beta_j = _mm256_broadcast_sd(bp + 3);
//         gamma_0123_3 = _mm256_fmadd_pd(alpha_0123_p, beta_j, gamma_0123_3);
//         gamma_4567_3 = _mm256_fmadd_pd(alpha_4567_p, beta_j, gamma_4567_3);

//         beta_j = _mm256_broadcast_sd(bp + 4);
//         gamma_0123_4 = _mm256_fmadd_pd(alpha_0123_p, beta_j, gamma_0123_4);
//         gamma_4567_4 = _mm256_fmadd_pd(alpha_4567_p, beta_j, gamma_4567_4);

//         beta_j = _mm256_broadcast_sd(bp + 5);
//         gamma_0123_5 = _mm256_fmadd_pd(alpha_0123_p, beta_j, gamma_0123_5);
//         gamma_4567_5 = _mm256_fmadd_pd(alpha_4567_p, beta_j, gamma_4567_5);

//         ap += MR;
//         bp += NR;
//     }

//     _mm256_storeu_pd(&gamma(0, 0), gamma_0123_0);
//     _mm256_storeu_pd(&gamma(4, 0), gamma_4567_0);
//     _mm256_storeu_pd(&gamma(0, 1), gamma_0123_1);
//     _mm256_storeu_pd(&gamma(4, 1), gamma_4567_1);
//     _mm256_storeu_pd(&gamma(0, 2), gamma_0123_2);
//     _mm256_storeu_pd(&gamma(4, 2), gamma_4567_2);
//     _mm256_storeu_pd(&gamma(0, 3), gamma_0123_3);
//     _mm256_storeu_pd(&gamma(4, 3), gamma_4567_3);
//     _mm256_storeu_pd(&gamma(0, 4), gamma_0123_4);
//     _mm256_storeu_pd(&gamma(4, 4), gamma_4567_4);
//     _mm256_storeu_pd(&gamma(0, 5), gamma_0123_5);
//     _mm256_storeu_pd(&gamma(4, 5), gamma_4567_5);
// }


void JPI_JI_Loop(int m, int n, int k, double* a, int ldA, double* b, int ldB, double* c, int ldC) {
    for (int J = 0; J < n; J += NC) {
        for (int P = 0; P < k; P += KC) {
            for (int I = 0; I < m; I += MC) {
                for (int j = 0; j < std::min(NC, n - J); j += NR) {
                    for (int i = 0; i < std::min(MC, m - I); i += MR) {
                        double *A = a + P * ldA + I + i;
                        double *B = b + J * ldB + P + j * ldB;
                        double *C = c + J * ldC + I + j * ldC + i;

                        kernel(std::min(MR, std::min(MC, m - I) - i), 
                               std::min(NR, std::min(NC, n - J) - j), 
                               std::min(KC, k - P), A, ldA, B, ldB, C, ldC);
                    }
                }
            }
        }
    }
}

void pack_panelB(int k, int n, double *B, int ldB, double* btilde) {
    if (NR == n) {
        for (int p = 0; p < k; ++p) {
            for (int j = 0; j < n; ++j) {
                *btilde++ = beta(p, j);
            }
        }
    }
    else {
        for (int p = 0; p < k; ++p) {
            for (int j = 0; j < n; ++j) {
                *btilde++ = beta(p, j);
            }
            for (int j = n; j < NR; ++j) {
                *btilde++ = 0.0;
            }
        }
    }
}

void packB(int k, int n, double *B, int ldB, double* btilde) {
    for (int j = 0; j < n; j += NR) {
        pack_panelB(k, std::min(NR, n - j), &beta(0, j), ldB, &btilde[j * k]);
    }   
}

void pack_panelA(int m, int k, double *A, int ldA, double* atilde) {
    if (MR == m) {
        for (int p = 0; p < k; ++p) {
            for (int i = 0; i < m; ++i) {
                *atilde++ = alpha(i, p);
            }
        }
    }
    else {
        for (int p = 0; p < k; ++p) {
            for (int i = 0; i < m; ++i) {
                *atilde++ = alpha(i, p);
            }
            for (int i = m; i < MR; ++i) {
                *atilde++ = 0.0;
            }
        }
    }
}

void packA(int m, int k, double *A, int ldA, double* atilde) {
    for (int i = 0; i < m; i += MR) {
        pack_panelA(std::min(MR, m - i), k, &alpha(i, 0), ldA, &atilde[i * k]);
    }
}

void JPI_JI_Loop_Packed(int m, int n, int k, double* a, int ldA, double* b, int ldB, double* c, int ldC) {
    auto *atilde = new double[MC * KC];
    auto *btilde = new double[NC * KC];

    for (int J = 0; J < n; J += NC) {
        for (int P = 0; P < k; P += KC) {
            packB(std::min(KC, k - P), std::min(NC, n - J), b + J * ldB + P, ldB, btilde);

            for (int I = 0; I < m; I += MC) {
                packA(std::min(MC, m - I), std::min(KC, k - P), a + P * ldA + I, ldA, atilde);

                for (int j = 0; j < std::min(NC, n - J); j += NR) {
                    for (int i = 0; i < std::min(MC, m - I); i += MR) {
                        double *A = atilde + i * std::min(KC, k - P);
                        double *B = btilde + j * std::min(KC, k - P);
                        double *C = c + J * ldC + I + j * ldC + i;

                        kernel_packed(std::min(KC, k - P), A, B, C, ldC);
                    }
                }
            }
        }
    }

    delete[] atilde;
    delete[] btilde;
}
