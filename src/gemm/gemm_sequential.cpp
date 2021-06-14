
/**
 * gemm_sequential.cpp
 * 
 * Implements the gemm-based convolution algorithm in forward propagation mode.
 */

#include "../utils.hpp"

/**
 * Transforms a 3D input tensor into a 2D matrix.
 */
void im2col(float *y, float *x) {

  int c, h, w, r, s, p, q, row, col;
  int hw=H*W, pq=P*Q, rspq=R*S*P*Q;

  for (c = 0; c < C; c++) {
    int x_off = c * hw;
    int y_off = c * rspq;

    for (r = 0; r < R; r++) {
      for (s = 0; s < S; s++) {
        for (p = 0; p < P; p++) {
          for (q = 0; q < Q; q++) {

            h = p + r; row = r*S + s;
            w = q + s; col = p*Q + q;

            y[y_off + row*pq+col] = x[x_off + h*W+w];
          }
        }
      }
    }
  }
}

/**
 * Performs a simple matrix multiplication.
 */
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

  float *workspace = new float[C*R*S*P*Q];
  for (int n = 0; n < N; n++) {
    im2col(workspace, &x_vec[n*C*H*W]);
    matmul(&y_vec[n*K*P*Q], f_vec.data(), workspace, K, P*Q, C*R*S);
  }

  #ifdef DEBUG // only run the sequential convolution if debugging
  compare(cpu_convolution(), y_vec); 
  #endif

  delete[] workspace;
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
