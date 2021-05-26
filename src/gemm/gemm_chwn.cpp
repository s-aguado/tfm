
/**
 * gemm_chwn.cpp
 * 
 * Implements the gemm-based convolution algorithm in forward propagation mode.
 * Uses CHWN data format. Executes it sequentially in the CPU, then compares 
 * the result with the direct approach.
 */

#include "../utils.hpp"

/**
 * CHWN to NCHW formatter
 */
std::vector<float> format_NCHW(float *src, int C, int H, int W, int N) {
  
  std::vector<float> result(N*C*H*W);
  int HWN = H*W*N, WN = W*N, CHW = C*H*W, HW = H*W;

  for (int n = 0; n < N; n++) 
    for (int c = 0; c < C; c++)
      for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) 
          result[n*CHW + c*HW + h*W + w] = src[c*HWN + h*WN + w*N + n];

  return result;
}

void im2col(float *y, float *x) {

  int n, c, h, w, r, s, p, q, row, col;
  int chw=C*H*W, hw=H*W, pqn=P*Q*N, rspqn=R*S*P*Q*N;

  for (c = 0; c < C; c++) {
    int x_off = c * hw;
    int y_off = c * rspqn;

    for (r = 0; r < R; r++) {
      for (s = 0; s < S; s++) {
        for (p = 0; p < P; p++) {
          for (q = 0; q < Q; q++) {

            h = p + r; row = r*S + s;
            w = q + s; col = p*Q + q;

            for (n = 0; n < N; n++) {
              y[y_off + row*pqn + col*N + n] = x[n*chw + x_off + h*W + w];
            }
          }
        }
      }
    }
  }
}

void matmul(float *C, float *A, float *B, int M, int N, int K) {

  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
      for (int n = 0; n < N; n++) {
        C[m*N+n] += A[m*K+k] * B[k*N+n];
      }
    }
  }
}

/**
 * im2col transformation + matrix multiplication
 */
void convolution() {

  std::vector<float> x_vec(N*C*H*W);
  std::vector<float> f_vec(K*C*R*S);
  std::vector<float> y_vec(N*K*P*Q);

  init_data(x_vec, f_vec, y_vec);

  float *w = new float[C*R*S*P*Q*N];
  im2col(w, x_vec.data());
  matmul(y_vec.data(), f_vec.data(), w, K, P*Q*N, C*R*S);
  y_vec = format_NCHW(y_vec.data(), K, P, Q, N);

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
