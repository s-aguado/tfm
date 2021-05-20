
/**
 * matmul.cpp
 * 
 * Executes the matrix multiplication in standalone mode.
 */

#include <CL/sycl.hpp>
#include <iostream>

#include "dpc_common.hpp"
#include "../utils.hpp"

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

void standalone() {

  std::vector<float> x_vec(C*R*S*P*Q);
  std::vector<float> f_vec(K*C*R*S);
  std::vector<float> y_vec(N*K*P*Q);

  init_data(x_vec, f_vec, y_vec);

  for (int n = 0; n < N; n++) {
    matmul(&y_vec[n*K*P*Q], f_vec.data(), x_vec.data(), K, P*Q, C*R*S);
  }
}

int main(int argc, char **argv) {
  return handle_errors(parse_arguments(argc,argv), standalone);
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
