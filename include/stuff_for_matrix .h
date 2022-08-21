#ifndef STUFF_FOR_MATRIX_H
#define STUFF_FOR_MATRIX_H

void generate_random_matrix(int size, double* a);
void print_matrix(int n, int m, int ld, double* a);
void check(int n, int m, double* c_ref, int ldCRef, double* c, int ldC);

#endif