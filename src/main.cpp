#include <iostream>
#include <chrono>
#include <blis/blis.h>
#include "../include/stuff_for_matrix .h"
#include "../include/gemm.h"

constexpr int begin_size = 50,
              end_size = 1000,
              step = 5,
              number_of_tests = 10;

int main() {
    double one = 1.0;

    freopen("output.txt", "w", stdout);

    for (int size = begin_size; size <= end_size; size += step) {
        auto a = new double[size * size];
        auto b = new double[size * size];
        auto c_old = new double[size * size];
        auto c = new double[size * size];
        auto c_ref = new double[size * size];
        double gflops = 2.0 * size * size * size * 1e-09;

        generate_random_matrix(size * size, a);
        generate_random_matrix(size * size, b);
        generate_random_matrix(size * size, c_old);
        
        std::chrono::nanoseconds best_time;

        for (int test = 0; test < number_of_tests; ++test) {
            mempcpy(c_ref, c_old, size * size * sizeof(double));

            auto begin = std::chrono::steady_clock::now();
            bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, size, size, size, &one, a, 1, size, 
                      b, 1, size, &one, c_ref, 1, size);
            auto end = std::chrono::steady_clock::now();

            if (!test) {
                best_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            }
            else {
                best_time = std::min(best_time, 
                                     std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin));
            }
        }

        std::cout << "BLIS Size:          " << size << "x" << size << "\tTime: " << best_time.count() * 1e-09
                  << " sec. \t GFLOPS: " << gflops / (best_time.count() * 1e-09) << std::endl;

        // for (int test = 0; test < number_of_tests; ++test) {
        //     mempcpy(c, c_old, size * size * sizeof(double));

        //     auto begin = std::chrono::steady_clock::now();
        //     JPI_Loop(size, size, size, a, size, b, size, c, size);
        //     auto end = std::chrono::steady_clock::now();

        //     check(size, size, c_ref, size, c, size);

        //     if (!test) {
        //         best_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        //     }
        //     else {
        //         best_time = std::min(best_time, 
        //                              std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin));
        //     }
        // }
        // std::cout << "IJP_LOOP Size:      " << size << "x" << size << "\tTime: " << best_time.count() * 1e-09
        //           << " sec. \t GFLOPS: " << gflops / (best_time.count() * 1e-09) << std::endl;

        // for (int test = 0; test < number_of_tests; ++test) {
        //     mempcpy(c, c_old, size * size * sizeof(double));
            
        //     auto begin = std::chrono::steady_clock::now();
        //     JPI_JI_Loop(size, size, size, a, size, b, size, c, size);
        //     auto end = std::chrono::steady_clock::now();

        //     check(size, size, c_ref, size, c, size);

        //     if (!test) {
        //         best_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        //     }
        //     else {
        //         best_time = std::min(best_time, 
        //                              std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin));
        //     }
        //  }

        // std::cout << "JPI_JI_LOOP Size:   " << size << "x" << size << "\tTime: " << best_time.count() * 1e-09
        //           << " sec. \t GFLOPS: " << gflops / (best_time.count() * 1e-09) << std::endl;

        for (int test = 0; test < number_of_tests; ++test) {
            mempcpy(c, c_old, size * size * sizeof(double));
            
            auto begin = std::chrono::steady_clock::now();
            JPI_JI_Loop_Packed(size, size, size, a, size, b, size, c, size);
            auto end = std::chrono::steady_clock::now();

            check(size, size, c_ref, size, c, size);

            if (!test) {
                best_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            }
            else {
                best_time = std::min(best_time, 
                                     std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin));
            }
         }

        std::cout << "JPI_JI_LOOP_P Size: " << size << "x" << size << "\tTime: " << best_time.count() * 1e-09
                  << " sec. \t GFLOPS: " << gflops / (best_time.count() * 1e-09) << std::endl;

        
        for (int test = 0; test < number_of_tests; ++test) {
            mempcpy(c, c_old, size * size * sizeof(double));
            
            auto begin = std::chrono::steady_clock::now();
            MyGemm(size, size, size, a, size, b, size, c, size);
            auto end = std::chrono::steady_clock::now();

            check(size, size, c_ref, size, c, size);

            if (!test) {
                best_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            }
            else {
                best_time = std::min(best_time, 
                                     std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin));
            }
         }

        std::cout << "MyGemm Size:        " << size << "x" << size << "\tTime: " << best_time.count() * 1e-09
                  << " sec. \t GFLOPS: " << gflops / (best_time.count() * 1e-09) << std::endl;

        std::cout << std::endl;

        delete[] a;
        delete[] b;
        delete[] c_old;
        delete[] c;
        delete[] c_ref;
    }

    return 0;
}