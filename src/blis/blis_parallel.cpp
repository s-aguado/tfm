
/**
 * blis_parallel.cpp
 * 
 * Implements the gemm-based convolution algorithm in forward propagation mode.
 * Reduces the memory consumption avoiding the im2col step.
 */

#include <CL/sycl.hpp>
#include <iostream>

#include "dpc_common.hpp"
#include "../utils.hpp"

#define SIZE 3072
#define MIN(a,b) ((a)<(b)?(a):(b))

/**
 * Struct to pass the dimensions to the kernel
 */
struct constants_t {
  int N,C,K,H,W,R,S,P,Q; // tensor constants
  int CHW,HW,RS,QN,HWN,WN,PQ,CRS,PQN; // precomputed variables
} constants;

/**
 * Performs a simple matrix multiplication.
 */
void matmul(
  sycl::accessor<float, 1, cl::sycl::access::mode::write> C, 
  sycl::accessor<float, 1, cl::sycl::access::mode::read>  A, 
  float *B, int M, int N, int K, int ldb, int ldc, int a_off, int c_off) {
  
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
      for (int n = 0; n < N; n++) {
        C[c_off + m*ldc+n] += A[a_off + m*K+k] * B[k*ldb+n];
      }
    }
  }
}

/**
 * Packs a column of matrix B into the buffer B_pack 
 * doing the im2col and the format transformation.
 */
void pack_B(float *B_pack, sycl::accessor<float, 1, cl::sycl::access::mode::read> B,
  int pc, int jc, int kc, constants_t arg) {

  for (int ps = 0; ps < kc; ps++) {

    int tmp1 = (pc+ps)%arg.RS;
    int c = (pc+ps)/arg.RS;
    int r = tmp1/arg.R;
    int s = tmp1%arg.R;
    
    int tmp2 = jc%arg.PQ;
    int n = jc/arg.PQ;
    int p = tmp2/arg.P;
    int q = tmp2%arg.P;

    B_pack[ps] = B[n*arg.CHW + c*arg.HW + (p+r)*arg.W + (q+s)];
  }
}

/**
 * im2col transformation + matrix multiplication
 */
void convolution(dnnl::engine::kind engine_kind) {

  constants = { N,C,K,H,W,R,S,P,Q,C*H*W,H*W,R*S,Q*N,H*W*N,W*N,P*Q,C*R*S,P*Q*N };

  std::vector<float> x_vec(N*C*H*W);
  std::vector<float> f_vec(K*C*R*S);
  std::vector<float> y_vec(N*K*P*Q);

  init_data(x_vec, f_vec, y_vec);

  try {

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

    // Submit command group to queue to perform matmul
    device_queue.submit([&](sycl::handler &context) {

      sycl::accessor x = x_buf.get_access<cl::sycl::access::mode::read>(context);
      sycl::accessor f = f_buf.get_access<cl::sycl::access::mode::read>(context);
      sycl::accessor y = y_buf.get_access<cl::sycl::access::mode::write>(context);
      sycl::accessor args(args_buf, context, sycl::read_only);

      context.parallel_for(sycl::range(P,Q,N), [=](auto index) {
        
        auto arg = args[0];
        int p = index[0];
        int q = index[1];
        int n = index[2];
        int jc = p*arg.QN + q*arg.N + n;

        float B_pack[SIZE];
        for (int pc = 0; pc < arg.CRS; pc += SIZE) {
          int kc = MIN(SIZE, arg.CRS-pc);

          // Pack an entire column of matrix B into B_pack, sequential memory
          pack_B(B_pack, x, pc, jc, kc, arg);

          // Perform matrix multiplication over the packed memory
          matmul(y, f, B_pack, arg.K, 1, kc, 1, arg.PQN, pc, jc);
        }
      });
    });

  } catch (sycl::exception const &e) {
    std::cout << "An exception was caught during the convolution.\n";
    std::terminate();
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
