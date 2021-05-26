
/**
 * blis_hwcn.cpp
 * 
 * Implements the gemm-based convolution algorithm in forward propagation mode.
 * Uses the BLIS library to avoid the im2col step. Executes it sequentially in 
 * the CPU, then compares the result with the direct approach.
 */

#include "../utils.hpp"

int 
  KC = 512,  //(C*R*S)/2, //368,
  NC = 6144, //(P*Q)/2,   //3072,
  MC = 96,   //K/2,       //560,
  NR = 12,   //NC/2,
  MR = 8;    //MC/2;

int CHW=C*H*W, HW=H*W, RS=R*S, QN=Q*N, HWN=H*W*N, WN=W*N, PQ=P*Q;

/**
 * Performs a simple matrix multiplication.
 */
void matmul(float *C, float *A, float *B, int M, int N, int K, int ldb, int ldc) {

  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
      for (int n = 0; n < N; n++) {
        C[m*ldc+n] += A[m*K+k] * B[k*ldb+n];
      }
    }
  }
}

/** 
 * Packs a block of matrix A into the buffer A_pack.
 */
void pack_A(float *A_pack, float *A, int lda, int M, int K) {
  
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
      A_pack[m*K+k] = A[m*lda+k];
    }
  }
}

/**
 * Packs a block of matrix B into the buffer B_pack 
 * doing the im2col and the format transformation.
 */
void pack_B(float *B_pack, float *B, int pc, int jc, int kc, int nc) {

  for (int ps = 0; ps < kc; ps++) {
    int tmp1 = (pc+ps)%RS;
    int c = (pc+ps)/RS;
    int r = tmp1/R;
    int s = tmp1%R;

    for (int js = 0; js < nc; js++) {
      int tmp2 = (jc+js)%PQ;
      int n = (jc+js)/PQ;
      int p = tmp2/P;
      int q = tmp2%P;

      B_pack[ps*nc + js] = B[n*CHW + c*HW + (p+r)*W + (q+s)];
    }
  }
}

/**
 * Matrix multiplication with implicit im2col.
 */
void blis(float *C, float *A, float *B, int m, int n, int k) {

  float *A_pack = new float[MC*KC];
  float *B_pack = new float[KC*NC];

  int lda = k;
  int ldc = n;

  for (int jc = 0; jc < n; jc += NC) {
    int nc = fmin(NC, n-jc);

    for (int pc = 0; pc < k; pc += KC) {
      int kc = fmin(KC, k-pc);

      pack_B(B_pack, B, pc, jc, kc, nc); // PACK B

      for (int ic = 0; ic < m; ic += MC) {
        int mc = fmin(MC, m-ic);

        pack_A(A_pack, &A[ic*lda + pc], lda, mc, kc); // PACK A
        float *C_pack = &C[ic*ldc + jc];

        for (int jr = 0; jr < nc; jr += NR) {
          int nr = fmin(NR, nc-jr);

          for (int ir = 0; ir < mc; ir += MR) {
            int mr = fmin(MR, mc-ir);
              
            float *Ar = &A_pack[ir*kc];
            float *Br = &B_pack[jr];
            float *Cr = &C_pack[ir*ldc + jr];

            matmul(Cr, Ar, Br, mr, nr, kc, nc, ldc);
          }
        }
      }
    }
  }

  delete [] A_pack;
  delete [] B_pack;
}

/**
 * im2col transformation + matrix multiplication
 */
void convolution() {

  std::vector<float> x_vec(N*C*H*W);
  std::vector<float> f_vec(K*C*R*S);
  std::vector<float> y_vec(N*K*P*Q);

  init_data(x_vec, f_vec, y_vec);

  CHW=C*H*W; HW=H*W; RS=R*S; QN=Q*N; HWN=H*W*N; WN=W*N; PQ=P*Q;
  blis(y_vec.data(), f_vec.data(), x_vec.data(), K, P*Q*N, C*R*S);
  
  #ifdef DEBUG // only run the sequential convolution if debugging
  compare(cpu_convolution(), y_vec);
  #endif
}

int main(int argc, char **argv) {
  return handle_errors(parse_arguments(argc,argv), convolution);
}

//    Copyright 2021 Sara Aguado Couselo
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
