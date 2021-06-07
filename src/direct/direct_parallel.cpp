
/**
 * direct_parallel.cpp
 * 
 * Implements the direct convolution algorithm in forward propagation mode.
 */

#include <CL/sycl.hpp>
#include <iostream>

#include "dpc_common.hpp"
#include "../utils.hpp"

/**
 * Struct to pass the dimensions to the kernel
 */
struct constants_t {
  int N,C,K,H,W,R,S,P,Q; // tensor constants
  int hw,rs,pq,chw,crs,kpq; // precomputed variables
};

/**
 * Perform convolution on device. Uses the dnnl engine_kind only to parse the 
 * dpc++ device selector.
 */
void convolution(dnnl::engine::kind engine_kind) {

  constants_t constants = { N,C,K,H,W,R,S,P,Q,H*W,R*S,P*Q,C*H*W,C*R*S,K*P*Q };

  std::vector<float> x_vec(N*C*H*W);
  std::vector<float> f_vec(K*C*R*S);
  std::vector<float> y_vec(N*K*P*Q);

  init_data(x_vec, f_vec, y_vec);

  {
    
    // Initialize the device queue with the custom selector. The device queue is
    // used to enqueue kernels. It encapsulates all states needed for execution.
    sycl::queue device_queue(
      select_device(engine_kind), dpc_common::exception_handler
    );
    
    #ifdef DEBUG
    std::cout << device_queue.get_device().get_info<sycl::info::device::name>();
    #endif

    // Create buffers for tensors, buffer c is bound with host memory y_vec
    // Allocate DPC++ buffers for input and output memory objects
    sycl::buffer x_buf(x_vec.data(), sycl::range(N*C*H*W));
    sycl::buffer f_buf(f_vec.data(), sycl::range(K*C*R*S));
    sycl::buffer y_buf(y_vec.data(), sycl::range(N*K*P*Q));
    sycl::buffer args_buf(&constants, sycl::range(1));

    // Submit command group to queue to perform convolution: y = x * f
    device_queue.submit([&](sycl::handler &context) {

      // Read from x and f, write to y
      sycl::accessor x(x_buf, context, sycl::read_only);
      sycl::accessor f(f_buf, context, sycl::read_only);
      sycl::accessor y(y_buf, context, sycl::write_only);
      sycl::accessor args(args_buf, context, sycl::read_only);

      // Execute kernel
      context.parallel_for(sycl::range(N,K,P*Q), [=](auto index) {
        
        auto arg = args[0];
        int n = index[0];
        int k = index[1];
        int p = index[2] / arg.Q;
        int q = index[2] % arg.Q;
        int y_off = n*arg.kpq + k*arg.pq;

        for (int c = 0; c < arg.C; c++) {

          int x_off = n*arg.chw + c*arg.hw;
          int f_off = k*arg.crs + c*arg.rs;

          for (int r = 0; r < arg.R; r++) {
            for (int s = 0; s < arg.S; s++) {

              int h = p + r;
              int w = q + s;

              y[y_off + p*arg.Q+q] += x[x_off + h*arg.W+w] * f[f_off + r*arg.S+s];
            }
          }
        }
      });
    });

  } // y_vec is updated when y_buf is destroyed upon exiting scope

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
