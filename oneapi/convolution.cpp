//==============================================================
// Sara Aguado Couselo © 2020
//
// SPDX-License-Identifier: MIT
// =============================================================

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

// Matrix size constants.
constexpr int 
  m_size = 1 << 6,            // Must be a power of 2
  stride_h = 1,
  stride_w = 1,
  N = 4,                      // batch size *
  C = 4,                      // input channels *
  K = 4,                      // output channels/number of filters *
  H = m_size / 2,             // image height *
  W = m_size / 2,             // image width *
  R = m_size / 4,             // filter height *
  S = m_size / 4,             // filter width *
  P = (H - R) / stride_h + 1, // output height
  Q = (W - S) / stride_w + 1; // output width

constexpr int // helper
  hw = H*W,
  rs = R*S,
  pq = P*Q,
  chw = C*H*W,
  crs = C*R*S,
  kpq = K*P*Q;

/**
 * Same initialization for device and host convolution
 */
void InitializeTensors(float(*x)[C][H][W], float(*f)[C][R][S], float(*y)[K][P][Q]) {
  int n, c, k, h, w, r, s, p, q;

  // Initialize each element of matrix a to 1
  for (n = 0; n < N; n++)
    for (c = 0; c < C; c++)
      for (h = 0; h < H; h++)
        for (w = 0; w < W; w++) x[n][c][h][w] = 1.0f;

  // Initialize each column of b_host to the sequence 1,2,...,S
  for (k = 0; k < K; k++)
    for (c = 0; c < C; c++)
      for (r = 0; r < R; r++)
        for (s = 0; s < S; s++) f[k][c][r][s] = s + 1.0f;

  // Initialize c_host to zero
  for (n = 0; n < N; n++)
    for (k = 0; k < K; k++)
      for (p = 0; p < P; p++)
        for (q = 0; q < Q; q++) y[n][k][p][q] = 0.0f;
}

/**
 * Perform convolution on host to verify results from device.
 */
int VerifyResult(float(*c_back)[K][P][Q]);

/**
 * Perform convolution on device.
 */
int main() {

  // Tensors on device side
  float(*a_back)[C][H][W] = new float[N][C][H][W];
  float(*b_back)[C][R][S] = new float[K][C][R][S];
  float(*c_back)[K][P][Q] = new float[N][K][P][Q];

  InitializeTensors(a_back, b_back, c_back);

  try {

    // Initialize the device queue with the default selector. The device queue is
    // used to enqueue kernels. It encapsulates all states needed for execution.
    queue device_queue(default_selector{}, dpc_common::exception_handler);
    cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << "\n";

    // Create 4D buffers for tensors, buffer c is bound with host memory c_back
    // Allocate DPC++ buffers for input and output memory objects
    buffer a_buf(reinterpret_cast<float*>(a_back), range<1> { N*C*H*W });
    buffer b_buf(reinterpret_cast<float*>(b_back), range<1> { K*C*R*S });
    buffer c_buf(reinterpret_cast<float*>(c_back), range<1> { N*K*P*Q });

    cout << "Multiplication size: c(" << N << "," << K << "," << P << "," << Q 
         << ") = a(" << N << "," << C << "," << H << "," << W 
         << ") * b(" << K << "," << C << "," << R << "," << S << ")\n";

    // Submit command group to queue to perform convolution: y = x * f
    device_queue.submit([&](auto &context) {

      // Read from a and b, write to c
      accessor x(a_buf, context, read_only);
      accessor f(b_buf, context, read_only);
      accessor y(c_buf, context, read_write);

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
  } // c_back is updated when c_buf is destroyed upon exiting scope

  int result;
  cout << "Result of convolution using DPC++: ";
  result = VerifyResult(c_back);
  
  delete[] c_back;
  return result;
}

bool ValueSame(float a, float b) {
  return fabs(a - b) < numeric_limits<float>::epsilon();
}

bool Compare(float(*c_host)[K][P][Q], float(*c_back)[K][P][Q]) {

  // Compare host side results with the result buffer from device side: print
  // mismatched data 4 times only.
  
  int printed_errors = 0;

  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k++) {
      for (int p = 0; p < P; p++) {
        for (int q = 0; q < Q; q++) {
          if (!ValueSame(c_back[n][k][p][q], c_host[n][k][p][q])) {
            cout << "\nFail - The result is incorrect for element: [" 
                 << n << ", " << k << ", " << p << ", " << q << "], expected: " 
                 << c_host[n][k][p][q] << ", but found: " << c_back[n][k][p][q];
            if (++printed_errors == 4) return printed_errors;
          }
        }
      }
    }
  }

  return printed_errors;
}

int VerifyResult(float(*c_back)[K][P][Q]) {  
  int n, c, k, h, w, r, s, p, q;

  // Tensors on host side
  float(*a_host)[C][H][W] = new float[N][C][H][W];
  float(*b_host)[C][R][S] = new float[K][C][R][S];
  float(*c_host)[K][P][Q] = new float[N][K][P][Q];

  InitializeTensors(a_host, b_host, c_host);

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

                c_host[n][k][p][q] += a_host[n][c][h][w] * b_host[k][c][r][s];
              }

  bool printed_errors = Compare(c_host, c_back);

  delete[] a_host;
  delete[] b_host;
  delete[] c_host;

  if (printed_errors) {
    cout << "\nFail - The results mismatch!\n";
    return -1;
  } else {
    cout << "Success - The results are correct!\n";
    return 0;
  }
}
