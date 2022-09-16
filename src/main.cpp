#include <iostream>
#include <chrono>
#include <blis/blis.h>
#include <omp.h>
#include <fstream>
#include "../include/stuff_for_matrix .h"
#include "../include/gemm.h"

constexpr int begin_size = 100,
              end_size = 1000,
              step = 100,
              number_of_tests = 10;

int main() {
    double one = 1.0;
    double maxDiff = 0.0;

    std::ofstream out_blis("blis.csv"), out_jpi_ji_packed("jpi_ji_packed.csv"), out_jpi_ji_packed_mt("jpi_ji_packed_mt.csv");
    out_blis << "N,GFLOPS\n";
    out_jpi_ji_packed << "N,GFLOPS\n";
    out_jpi_ji_packed_mt << "N,GFLOPS,GFLOPS per thread\n";

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

        std::cout << "BLIS Size:                  " << size << "x" << size << "\tTime: " << best_time.count() * 1e-09
                  << " sec. \t GFLOPS:            " << gflops / (best_time.count() * 1e-09) << std::endl;
        
        out_blis << size << ',' << gflops / (best_time.count() * 1e-09) << '\n';

        // for (int test = 0; test < number_of_tests; ++test) {
        //     mempcpy(c, c_old, size * size * sizeof(double));

        //     auto begin = std::chrono::steady_clock::now();
        //     JPI_Loop(size, size, size, a, size, b, size, c, size);
        //     auto end = std::chrono::steady_clock::now();

        //     if (!test) {
        //         best_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        //     }
        //     else {
        //         best_time = std::min(best_time, 
        //                              std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin));
        //     }
        // }
        // std::cout << "IJP_LOOP Size:              " << size << "x" << size << "\tTime: " << best_time.count() * 1e-09
        //           << " sec. \t GFLOPS:            " << gflops / (best_time.count() * 1e-09) << std::endl;

        // for (int test = 0; test < number_of_tests; ++test) {
        //     mempcpy(c, c_old, size * size * sizeof(double));
            
        //     auto begin = std::chrono::steady_clock::now();
        //     JPI_JI_Loop(size, size, size, a, size, b, size, c, size);
        //     auto end = std::chrono::steady_clock::now();

        //     if (!test) {
        //         best_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        //     }
        //     else {
        //         best_time = std::min(best_time, 
        //                              std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin));
        //     }
        //  }

        // std::cout << "JPI_JI_LOOP Size:       " << size << "x" << size << "\tTime: " << best_time.count() * 1e-09
        //           << " sec. \t GFLOPS:            " << gflops / (best_time.count() * 1e-09) << std::endl;

        // for (int test = 0; test < number_of_tests; ++test) {
        //     mempcpy(c, c_old, size * size * sizeof(double));
            
        //     auto begin = std::chrono::steady_clock::now();
        //     JPI_JI_Loop_Packed(size, size, size, a, size, b, size, c, size);
        //     auto end = std::chrono::steady_clock::now();

        //     if (!test) {
        //         best_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        //     }
        //     else {
        //         best_time = std::min(best_time, 
        //                              std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin));
        //     }
        //  }

        // std::cout << "JPI_JI_LOOP_P Size:         " << size << "x" << size << "\tTime: " << best_time.count() * 1e-09
        //           << " sec. \t GFLOPS:            " << gflops / (best_time.count() * 1e-09) << std::endl;


        // out_jpi_ji_packed << size << ',' << gflops / (best_time.count() * 1e-09) << '\n';

        for (int test = 0; test < number_of_tests; ++test) {
            mempcpy(c, c_old, size * size * sizeof(double));
            
            auto begin = std::chrono::steady_clock::now();
            JPI_JI_Loop_Packed_MT(size, size, size, a, size, b, size, c, size);
            auto end = std::chrono::steady_clock::now();

            if (!test) {
                best_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            }
            else {
                best_time = std::min(best_time, 
                                     std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin));
            }
        }

        std::cout << "JPI_JI_LOOP_P Size:         " << size << "x" << size << "\tTime: " << best_time.count() * 1e-09
                  << " sec. \t GFLOPS:            " << gflops / (best_time.count() * 1e-09) 
                  << "\t GFLOPS per thread: " << (gflops / (best_time.count() * 1e-09)) / omp_get_max_threads() << std::endl;

        out_jpi_ji_packed_mt << size << ',' << gflops / (best_time.count() * 1e-09) << ',' << (gflops / (best_time.count() * 1e-09)) / omp_get_max_threads() << '\n';
        
        std::cout << std::endl;

        maxDiff = std::max(max_diff(size, size, c_ref, c), maxDiff);

        delete[] a;
        delete[] b;
        delete[] c_old;
        delete[] c;
        delete[] c_ref;
    }

    std::cout << "Max Diff: " << maxDiff << std::endl;

    return 0;
}