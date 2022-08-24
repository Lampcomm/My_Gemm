#ifndef GEMM_H
#define GEMM_H

void JPI_Loop(int m, int n, int k, double* a, int ldA, double* b, int ldB, double* c, int ldC);
void JPI_JI_Loop(int m, int n, int k, double* a, int ldA, double* b, int ldB, double* c, int ldC);
void JPI_JI_Loop_Packed(int m, int n, int k, double* a, int ldA, double* b, int ldB, double* c, int ldC);
void JPI_JI_Loop_Packed_MP(int m, int n, int k, double* a, int ldA, double* b, int ldB, double* c, int ldC);

#endif