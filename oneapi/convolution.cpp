//==============================================================
// Sara Aguado Couselo © 2020
//
// SPDX-License-Identifier: MIT
// =============================================================

/**
 * convolution.cpp
 * 
 * Implements a (basic 2D) convolution algorithm (TODO: in forward propagation mode). 
 * Executes it in both, the CPU and the offload device, then compares 
 * the result. If the code executes on both CPU and the offload device, 
 * the name of the offload device and a success message are displayed.
 *
 * For comprehensive instructions regarding DPC++ Programming, go to
 * https://software.intel.com/en-us/oneapi-programming-guide and search based on
 * relevant terms noted in the comments.
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
constexpr int stride_h = 1;
constexpr int stride_w = 1;
constexpr int m_size = 1 << 6; // Must be a power of 2
constexpr int H = m_size / 2;
constexpr int W = m_size / 2;
constexpr int R = m_size / 4;
constexpr int S = m_size / 4;
constexpr int P = (H - R) / stride_h + 1;
constexpr int Q = (W - S) / stride_w + 1;

/**
 * Perform convolution on host to verify results from device.
 */
int VerifyResult(float (*c_back)[Q]);

int main() {

  // Host memory buffer that device will write data back before destruction.
  float(*c_back)[Q] = new float[P][Q];

  // Initialize c_back to zero
  for (int i = 0; i < P; i++)
    for (int j = 0; j < Q; j++) c_back[i][j] = 0.0f;

  try {

    // Initialize the device queue with the default selector. The device queue is
    // used to enqueue kernels. It encapsulates all states needed for execution.
    queue device_queue(default_selector{}, dpc_common::exception_handler);
    cout << "Device: " << device_queue.get_device().get_info<info::device::name>() << "\n";

    // Create 2D buffers for matrices, buffer c is bound with host memory c_back
    buffer<float, 2> a_buf(range(H, W));
    buffer<float, 2> b_buf(range(R, S));
    buffer c_buf(reinterpret_cast<float *>(c_back), range(P, Q));

    cout << "Multiplication size: c(" << P << "," << Q << ") = a(" << H << "," << W
         << ") * b(" << R << "," << S << ")\n";

    // Using three command groups to illustrate execution order. The use of
    // first two command groups for initializing matrices is not the most
    // efficient way. It just demonstrates the implicit multiple command group
    // execution ordering.

    // Submit command group to queue to initialize matrix a
    device_queue.submit([&](auto &h) {
      accessor a(a_buf, h, write_only); // Get write only access to the buffer on the device
      h.parallel_for(range(H, W), [=](auto index) {
        a[index] = 1.0f; // Initialize each element of matrix a to 1
      });
    });

    // Submit command group to queue to initialize matrix b
    device_queue.submit([&](auto &h) {
      accessor b(b_buf, h, write_only); // Get write only access to the buffer on the device
      h.parallel_for(range(R, S), [=](auto index) {
        b[index] = index[0] + 1.0f; // Initialize each column of b_host to the sequence 1,2,...,R
      });
    });

    // Submit command group to queue to multiply matrices: c = a * b
    device_queue.submit([&](auto &h) {

      // Read from a and b, write to c
      accessor a(a_buf, h, read_only);
      accessor b(b_buf, h, read_only);
      accessor c(c_buf, h, write_only);

      // Execute kernel.
      h.parallel_for(range(P, Q), [=](auto index) {

        int p = index[0]; // Get global position in Y direction.
        int q = index[1]; // Get global position in X direction.

        // Compute the result of one element of c
        for (int r = 0; r < R; r++) {
          for (int s = 0; s < S; s++) {

            int h = p + r;
            int w = q + s;

            c[p][q] += a[h][w] * b[r][s];
          }
        }
      });
    });
  } catch (sycl::exception const &e) {
    cout << "An exception is caught while multiplying matrices.\n";
    terminate();
  }

  int result;
  cout << "Result of convolution using DPC++: ";
  result = VerifyResult(c_back);
  delete[] c_back;

  return result;
}

bool ValueSame(float a, float b) {
  return fabs(a - b) < numeric_limits<float>::epsilon();
}

int VerifyResult(float (*c_back)[Q]) {  
  int i, j, k;

  // 2D arrays on host side.
  float(*a_host)[W] = new float[H][W];
  float(*b_host)[S] = new float[R][S];
  float(*c_host)[Q] = new float[P][Q];

  // Initialize each element of matrix a to 1
  for (i = 0; i < H; i++)
    for (j = 0; j < W; j++) a_host[i][j] = 1.0f;

  // Initialize each column of b_host to the sequence 1,2,...,R
  for (i = 0; i < R; i++)
    for (j = 0; j < S; j++) b_host[i][j] = i + 1.0f;

  // Initialize c_host to zero
  for (i = 0; i < P; i++)
    for (j = 0; j < Q; j++) c_host[i][j] = 0.0f;

  // Do the convolution
  for (int p = 0; p < P; p++) { // output height
    for (int q = 0; q < Q; q++) { // output width
      for (int r = 0; r < R; r++) {
        for (int s = 0; s < S; s++) {

          int h = p + r;
          int w = q + s;

          c_host[p][q] += a_host[h][w] * b_host[r][s];
        }
      }
    }
  }

  // Compare host side results with the result buffer from device side: print
  // mismatched data 5 times only.
  int printed_errors = 0;

  for (i = 0; i < P; i++) {
    for (j = 0; j < Q; j++) {
      if (!ValueSame(c_back[i][j], c_host[i][j])) {
        cout << "Fail - The result is incorrect for element: [" << i << ", "
             << j << "], expected: " << c_host[i][j]
             << ", but found: " << c_back[i][j] << "\n";
        if (++printed_errors == 5) break;
      }
    }
    if (printed_errors == 5) break;
  }

  delete[] a_host;
  delete[] b_host;
  delete[] c_host;

  if (printed_errors == 0) {
    cout << "Success - The results are correct!\n";
    return 0;
  } else {
    cout << "Fail - The results mismatch!\n";
    return -1;
  }
}
