
/**
 * im2col.cpp
 * 
 * Executes the im2col function in standalone mode.
 */

#include <CL/sycl.hpp>
#include <iostream>

#include "dpc_common.hpp"
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

void standalone() {

  std::vector<float> x_vec(N*C*H*W);
  std::vector<float> f_vec(1);
  std::vector<float> y_vec(C*R*S*P*Q);

  init_data(x_vec, f_vec, y_vec);

  for (int n = 0; n < N; n++) {
    im2col(y_vec.data(), &x_vec[n*C*H*W]);
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
