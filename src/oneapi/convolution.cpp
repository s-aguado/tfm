
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
#include "../utils.hpp"

struct constants_t {
  int N,C,K,H,W,R,S,P,Q; // tensor constants
  int hw,rs,pq,chw,crs,kpq; // precomputed variables
};

/**
 * Return true if both params have the same value.
 */
bool equals(float a, float b) {
  return fabs(a - b) < std::numeric_limits<float>::epsilon();
}

/**
 * Compare host side results with the result buffer from device side: print
 * mismatched data 4 times only. 
 */
bool compare(std::vector<float> &expected, std::vector<float> &result) {
  
  int printed_errors = 0;

  for (int i = 0; i < expected.size(); i++) {
    if (!equals(expected[i], result[i])) {
      std::cout << "\nFail - The result is incorrect for element: y(" 
           << i/(K*P*Q) << "·" << i/(P*Q) << "·" << i/(Q) << "·" << i%Q 
           << "), expected: " << expected[i] << ", but found: " << result[i];
      if (++printed_errors == 4) return true;
    }
  }

  return false;
}

/**
 * Perform convolution on host to verify results from device.
 */
void verify_result(std::vector<float> &y_back, constants_t args) {
  int n, c, k, h, w, r, s, p, q;

  std::vector<float> x_host(N*C*H*W);
  std::vector<float> f_host(K*C*R*S);
  std::vector<float> y_host(N*K*P*Q);

  init_data(x_host, f_host, y_host);

  for (n = 0; n < N; n++) {
    for (k = 0; k < K; k++) {
      auto y_ = &y_host[n * args.kpq + k * args.pq];
      
      for (c = 0; c < C; c++) {

        auto x_ = &x_host[n * args.chw + c * args.hw];
        auto f_ = &f_host[k * args.crs + c * args.rs];

        for (p = 0; p < P; p++) {
          for (q = 0; q < Q; q++) {
            for (r = 0; r < R; r++) {
              for (s = 0; s < S; s++) {

                h = p + r;
                w = q + s;

                y_[p*Q+q] += x_[h*W+w] * f_[r*S+s];
              }
            }
          }
        }
      }
    }
  }

  if (compare(y_host, y_back)) {
    std::cout << "\nFail - The results mismatch!\n";
  } else {
    std::cout << ": Success - The results are correct!\n";
  }
}

/**
 * Perform convolution on device. Uses the dnnl engine_kind only to parse the 
 * dpc++ device selector.
 */
void convolution(dnnl::engine::kind engine_kind) {

  constants_t constants = { N,C,K,H,W,R,S,P,Q,H*W,R*S,P*Q,C*H*W,C*R*S,K*P*Q };

  std::vector<float> x_back(N*C*H*W);
  std::vector<float> f_back(K*C*R*S);
  std::vector<float> y_back(N*K*P*Q);

  init_data(x_back, f_back, y_back);

  try {
    
    // Initialize the device queue with the custom selector. The device queue is
    // used to enqueue kernels. It encapsulates all states needed for execution.
    sycl::queue device_queue(
      select_device(engine_kind), dpc_common::exception_handler
    );
    
    #ifdef DEBUG
    std::cout << device_queue.get_device().get_info<sycl::info::device::name>();
    #endif

    // Create 4D buffers for tensors, buffer c is bound with host memory y_back
    // Allocate DPC++ buffers for input and output memory objects
    sycl::buffer x_buf(x_back.data(), sycl::range(N*C*H*W));
    sycl::buffer f_buf(f_back.data(), sycl::range(K*C*R*S));
    sycl::buffer y_buf(y_back.data(), sycl::range(N*K*P*Q));
    sycl::buffer args_buf(&constants, sycl::range(1));

    // Submit command group to queue to perform convolution: y = x * f
    device_queue.submit([&](sycl::handler &context) {

      // Read from x and f, write to y
      sycl::accessor x(x_buf, context, sycl::read_only);
      sycl::accessor f(f_buf, context, sycl::read_only);
      sycl::accessor y(y_buf, context, sycl::write_only);
      sycl::accessor args = args_buf.get_access(context);

      // Execute kernel
      context.parallel_for(sycl::range(N,K), [=](auto index) {

        auto a = args[0];
        int n = index[0];
        int k = index[1];
        int y_off = n*a.kpq + k*a.pq;

        for (int c = 0; c < a.C; c++) {

          int x_off = n*a.chw + c*a.hw;
          int f_off = k*a.crs + c*a.rs;

          for (int p = 0; p < a.P; p++) {
            for (int q = 0; q < a.Q; q++) {
              for (int r = 0; r < a.R; r++) {
                for (int s = 0; s < a.S; s++) {

                  int h = p + r;
                  int w = q + s;

                  y[y_off + p*a.Q+q] += x[x_off + h*a.W+w] * f[f_off + r*a.S+s];
                }
              }
            }
          }
        }
      });
    });

    device_queue.wait_and_throw();
  } catch (sycl::exception const &e) {
    std::cout << "An exception was caught during the convolution.\n";
    std::terminate();
  } // y_back is updated when y_buf is destroyed upon exiting scope

  #ifdef DEBUG
  verify_result(y_back, constants); // only run the sequential convolution if debugging
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
