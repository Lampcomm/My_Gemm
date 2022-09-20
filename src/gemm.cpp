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

void MT_packA(int m, int k, double *A, int ldA, double* atilde) {
    #pragma omp parallel for
    for (int i = 0; i < m; i += MR) {
        pack_panelA(std::min(MR, m - i), k, &alpha(i, 0), ldA, &atilde[i * k]);
    }
}

void MT_packB(int k, int n, double *B, int ldB, double* btilde) {
    #pragma omp parallel for
    for (int j = 0; j < n; j += NR) {
        pack_panelB(k, std::min(NR, n - j), &beta(0, j), ldB, &btilde[j * k]);
    }
}

//*****************************************************************************************************************
// MT loop 2

void JPI_JI_Loop_Packed_MT(int m, int n, int k, double* a, int ldA, double* b, int ldB, double* c, int ldC) {
    auto *atilde = new double[MC * KC];
    auto *btilde = new double[NC * KC];

    for (int J = 0; J < n; J += NC) {
        for (int P = 0; P < k; P += KC) {
            MT_packB(std::min(KC, k - P), std::min(NC, n - J), b + J * ldB + P, ldB, btilde);

            for (int I = 0; I < m; I += MC) {
                MT_packA(std::min(MC, m - I), std::min(KC, k - P), a + P * ldA + I, ldA, atilde);

                #pragma omp parallel for
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
//*****************************************************************************************************************

//*****************************************************************************************************************
// MT loop3

// void Loop2 (int m, int n, int k, double* atilde, double* btilde, double* c, int ldC) {
//     for (int j = 0; j < n; j += NR) {
//         for (int i = 0; i < m; i += MR) {
//             double *A = atilde + i * k;
//             double *B = btilde + j * k;
//             double *C = c + j * ldC + i;

//             kernel_packed(k, A, B, C, ldC);
//         }
//     }
// }

// void JPI_JI_Loop_Packed_MT(int m, int n, int k, double* a, int ldA, double* b, int ldB, double* c, int ldC) {
//     auto *atilde = new double[MC * KC * omp_get_max_threads()];
//     auto *btilde = new double[NC * KC];

//     for (int J = 0; J < n; J += NC) {
//         for (int P = 0; P < k; P += KC) {
//             MT_packB(std::min(KC, k - P), std::min(NC, n - J), b + J * ldB + P, ldB, btilde);

//             int MC_per_thread = ((MC / omp_get_max_threads()) / MR) * MR;

//             int whole_part = (m / (MC_per_thread * omp_get_max_threads())) * MC_per_thread * omp_get_max_threads();

//             int remainder_per_thread = (((m - whole_part) / omp_get_max_threads()) / MR) * MR;

//             if (!remainder_per_thread) {
//                 remainder_per_thread = MR;
//             }

//             #pragma omp parallel for
//             for (int I = 0; I < whole_part; I += MC_per_thread) {
//                 packA(std::min(MC_per_thread, m - I), std::min(KC, k - P), a + P * ldA + I, ldA, atilde + MC * KC * omp_get_thread_num());
//                 Loop2(std::min(MC_per_thread, m - I), std::min(NC, n - J), std::min(KC, k - P), atilde + MC * KC * omp_get_thread_num(), btilde, c + J * ldC + I, ldC);
//             }

//             #pragma omp parallel for
//             for (int I = whole_part; I < m; I += remainder_per_thread) {
//                 packA(std::min(remainder_per_thread, m - I), std::min(KC, k - P), a + P * ldA + I, ldA, atilde + MC * KC * omp_get_thread_num());
//                 Loop2(std::min(remainder_per_thread, m - I), std::min(NC, n - J), std::min(KC, k - P), atilde + MC * KC * omp_get_thread_num(), btilde, c + J * ldC + I, ldC);
//             }
//         }
//     }

//     delete[] atilde;
//     delete[] btilde;
// }
//*****************************************************************************************************************

//*****************************************************************************************************************
// MT loop 5

// void Loop4(int m, int n, int k, double* a, int ldA, double* b, int ldB, double* c, int ldC, double* atilde, double* btilde) {
//     for (int P = 0; P < k; P += KC) {
//             packB(std::min(KC, k - P), n, b + P, ldB, btilde + NC * KC * omp_get_thread_num());

//             for (int I = 0; I < m; I += MC) {
//                 packA(std::min(MC, m - I), std::min(KC, k - P), a + P * ldA + I, ldA, atilde + MC * KC * omp_get_thread_num());

//                 for (int j = 0; j < n; j += NR) {
//                     for (int i = 0; i < std::min(MC, m - I); i += MR) {
//                         double *A = atilde + MC * KC * omp_get_thread_num() + i * std::min(KC, k - P);
//                         double *B = btilde + NC * KC * omp_get_thread_num() + j * std::min(KC, k - P);
//                         double *C = c + I + j * ldC + i;

//                         kernel_packed(std::min(KC, k - P), A, B, C, ldC);
//                     }
//                 }
//             }
//         }
// }

// void JPI_JI_Loop_Packed_MT(int m, int n, int k, double* a, int ldA, double* b, int ldB, double* c, int ldC) {
//     auto *atilde = new double[MC * KC * omp_get_max_threads()];
//     auto *btilde = new double[NC * KC * omp_get_max_threads()];

//     int NC_per_thread = ((NC / omp_get_max_threads()) / NR) * NR;
//     int whole_part = (n / (NC_per_thread * omp_get_max_threads())) * NC_per_thread * omp_get_max_threads();

//     int remainder_per_thread = (((n - whole_part) / omp_get_max_threads()) / NR) * NR;

//     if (!remainder_per_thread) {
//         remainder_per_thread = NR;
//     }

//     #pragma omp parallel for
//     for (int J = 0; J < whole_part; J += NC_per_thread) {
//         Loop4(m, std::min(NC_per_thread, n - J), k, a, ldA, b + J * ldB, ldB, c + J * ldC, ldC, atilde, btilde);
//     }

//     #pragma omp parallel for
//     for (int J = whole_part; J < n; J += remainder_per_thread) {
//         Loop4(m, std::min(remainder_per_thread, n - J), k, a, ldA, b + J * ldB, ldB, c + J * ldC, ldC, atilde, btilde);
//     }

//     delete[] atilde;
//     delete[] btilde;
// }
//*****************************************************************************************************************