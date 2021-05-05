
/**
 * convolution.cpp
 * 
 * Implements the direct convolution algorithm in forward propagation mode. 
 * Executes it in both, the CPU and the offload device, then compares 
 * the result. If the code executes on both CPU and the offload device, 
 * the name of the offload device and a success message are displayed.
 *
 * For comprehensive instructions regarding DPC++ Programming, go to
 * https://software.intel.com/en-us/oneapi-programming-guide and search based 
 * on relevant terms noted in the comments.
 */

#include <CL/sycl.hpp>
#include <iostream>
#include <limits>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPIHOME/dev-utilities/latest/include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace std;
using namespace sycl;

// Tensor dimensions.
constexpr int 
  m_size = 1 << 6,       
  SH = 1,                // height-wise stride
  SW = 1,                // width-wise stride
  N = 4,                 // batch size
  C = 4,                 // input channels
  K = 4,                 // output channels / number of filters
  H = m_size / 2,        // image height
  W = m_size / 2,        // image width
  R = m_size / 4,        // filter height
  S = m_size / 4,        // filter width
  P = (H - R) / SH + 1,  // output height
  Q = (W - S) / SW + 1;  // output width

// Precomputed variables.
constexpr int 
  hw  = H*W,
  rs  = R*S,
  pq  = P*Q,
  chw = C*H*W,
  crs = C*R*S,
  kpq = K*P*Q;

/**
 * Same initialization for device and host convolution.
 */
void InitializeTensors(float(*x)[C][H][W], float(*f)[C][R][S], float(*y)[K][P][Q]) {
  int n, c, k, h, w, r, s, p, q;

  // Initialize each element of the input tensor to 1
  for (n = 0; n < N; n++)
    for (c = 0; c < C; c++)
      for (h = 0; h < H; h++)
        for (w = 0; w < W; w++) x[n][c][h][w] = 1.0f;

  // Initialize each column of the filter to the sequence 1,2,...,S
  for (k = 0; k < K; k++)
    for (c = 0; c < C; c++)
      for (r = 0; r < R; r++)
        for (s = 0; s < S; s++) f[k][c][r][s] = s + 1.0f;

  // Initialize each element of the output tensor to 0
  for (n = 0; n < N; n++)
    for (k = 0; k < K; k++)
      for (p = 0; p < P; p++)
        for (q = 0; q < Q; q++) y[n][k][p][q] = 0.0f;
}


/**
 * Return true if both params have the same value.
 */
bool ValueSame(float a, float b) {
  return fabs(a - b) < numeric_limits<float>::epsilon();
}

/**
 * Compare host side results with the result buffer from device side: print
 * mismatched data 4 times only. 
 */
bool Compare(float(*y_host)[K][P][Q], float(*y_back)[K][P][Q]) {
  
  int printed_errors = 0;

  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k++) {
      for (int p = 0; p < P; p++) {
        for (int q = 0; q < Q; q++) {
          if (!ValueSame(y_back[n][k][p][q], y_host[n][k][p][q])) {
            cout << "\nFail - The result is incorrect for element: [" 
                 << n << ", " << k << ", " << p << ", " << q << "], expected: " 
                 << y_host[n][k][p][q] << ", but found: " << y_back[n][k][p][q];
            if (++printed_errors == 4) return true;
          }
        }
      }
    }
  }

  return false;
}

/**
 * Perform convolution on host to verify results from device.
 */
int VerifyResult(float(*y_back)[K][P][Q]) {  
  int n, c, k, h, w, r, s, p, q;

  // Tensors on host side
  float(*x_host)[C][H][W] = new float[N][C][H][W];
  float(*f_host)[C][R][S] = new float[K][C][R][S];
  float(*y_host)[K][P][Q] = new float[N][K][P][Q];

  InitializeTensors(x_host, f_host, y_host);

  // Do the convolution
  for (n = 0; n < N; n++)
    for (k = 0; k < K; k++)
      for (c = 0; c < C; c++)
        for (p = 0; p < P; p++)
          for (q = 0; q < Q; q++)
            for (r = 0; r < R; r++)
              for (s = 0; s < S; s++) {

                h = p + r;
                w = q + s;

                y_host[n][k][p][q] += x_host[n][c][h][w] * f_host[k][c][r][s];
              }

  bool error = Compare(y_host, y_back);

  delete[] x_host;
  delete[] f_host;
  delete[] y_host;

  cout << "\nResult of convolution using DPC++: ";
  if (error) {
    cout << "Fail - The results mismatch!\n";
    return -1;
  } else {
    cout << "Success - The results are correct!\n";
    return 0;
  }
}

/**
 * Perform convolution on device.
 */
int main() {

  // Tensors on device side
  float(*x_back)[C][H][W] = new float[N][C][H][W];
  float(*f_back)[C][R][S] = new float[K][C][R][S];
  float(*y_back)[K][P][Q] = new float[N][K][P][Q];

  InitializeTensors(x_back, f_back, y_back);

  try {

    // Initialize the device queue with the default selector. The device queue is
    // used to enqueue kernels. It encapsulates all states needed for execution.
    queue device_queue(default_selector{}, dpc_common::exception_handler);
    cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << "\n";

    // Create 4D buffers for tensors, buffer c is bound with host memory y_back
    // Allocate DPC++ buffers for input and output memory objects
    buffer x_buf(reinterpret_cast<float*>(x_back), range<1> { N*C*H*W });
    buffer f_buf(reinterpret_cast<float*>(f_back), range<1> { K*C*R*S });
    buffer y_buf(reinterpret_cast<float*>(y_back), range<1> { N*K*P*Q });

    cout << "Multiplication size: y(" << N << "," << K << "," << P << "," << Q 
         << ") = x(" << N << "," << C << "," << H << "," << W 
         << ") * f(" << K << "," << C << "," << R << "," << S << ")";

    // Submit command group to queue to perform convolution: y = x * f
    device_queue.submit([&](auto &context) {

      // Read from x and f, write to y
      accessor x(x_buf, context, read_only);
      accessor f(f_buf, context, read_only);
      accessor y(y_buf, context, read_write);

      // Execute kernel
      context.parallel_for(range(N,K), [=](auto index) {

        int n = index[0];
        int k = index[1];

        auto y_ = &y[n * kpq + k * pq];

        for (int c = 0; c < C; c++) {

          auto x_ = &x[n * chw + c * hw];
          auto f_ = &f[k * crs + c * rs];

          for (int p = 0; p < P; p++)
            for (int q = 0; q < Q; q++)
              for (int r = 0; r < R; r++)
                for (int s = 0; s < S; s++) {

                  int h = p + r;
                  int w = q + s;

                  y_[p*Q+q] += x_[h*W+w] * f_[r*S+s];
                }
        }
      });
    });
  } catch (sycl::exception const &e) {
    cout << "An exception was caught during the convolution.\n";
    terminate();
  } // y_back is updated when y_buf is destroyed upon exiting scope

  int result = VerifyResult(y_back);
  
  delete[] x_back;
  delete[] f_back;
  delete[] y_back;

  return result;
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
