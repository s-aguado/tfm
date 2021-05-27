
/**
 * utils.hpp
 * 
 * Common functions to use with the convolution codes.
 */

#ifndef UTILS_HPP
#define UTILS_HPP

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <initializer_list>

#include "dnnl.hpp"
#include "dnnl_debug.h"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
  #include "dnnl_ocl.hpp"
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
  #include "dnnl_sycl.hpp"
#endif

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP 
  #ifdef _MSC_VER
    #define PRAGMA_MACRo(x) __pragma(x)
    #define PRAGMA_MACRO(x) PRAGMA_MACRo(x)
  #else
    #define PRAGMA_MACRo(x) _Pragma(#x)
    #define PRAGMA_MACRO(x) PRAGMA_MACRo(x)
  #endif

  // MSVC doesn't support collapse clause in omp parallel
  #if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
    #define collapse(x)
  #endif

  #define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(n) PRAGMA_MACRO(omp parallel for collapse(n))

#else // DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_OMP
  #define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(n)
#endif

// Tensor constants.
int
  N = 16,                             // batch size
  C = 4,                              // input channels
  K = 4,                              // output channels / number of filters
  H = 32,                             // image height
  W = 32,                             // image width
  R = 3,                              // filter height
  S = 3,                              // filter width
  PH_L = 0,                           // height padding: left
  PH_R = 0,                           // height padding: right
  PW_L = 0,                           // width padding: left
  PW_R = 0,                           // width padding: right
  SH = 1,                             // height-wise stride
  SW = 1,                             // width-wise stride
  P = (H - R + PH_L + PH_R) / SH + 1, // output height
  Q = (W - S + PW_L + PW_R) / SW + 1; // output width

// Tensor dimensions.
dnnl::memory::dims 
  x_dims = {N, C, H, W},
  f_dims = {K, C, R, S},
  y_dims = {N, K, P, Q},
  bias_dims = {K},
  strides_dims = {SH, SW},
  padding_dims_l = {PH_L, PW_L},
  padding_dims_r = {PH_R, PW_R};

// Returns the string representation of the engine kind.
inline const std::string engine_to_string(dnnl::engine::kind engine_kind) {
  if (engine_kind == dnnl::engine::kind::cpu) return "CPU";
  if (engine_kind == dnnl::engine::kind::gpu) return "GPU";
  assert(!"not expected");
  return "<Unknown engine>";
}

// Runs example function with signature void() and catches errors.
// Returns `0` on success, `1` or oneDNN error, and `2` on program error.
inline int handle_errors(dnnl::engine::kind engine_kind, 
                         std::function<void()> example) {
  int exit_code = 0;

  try {
    example();
  } catch (dnnl::error &e) {
    std::cout << "oneDNN error caught: " << std::endl
      << "\tStatus: " << dnnl_status2str(e.status) << std::endl
      << "\tMessage: " << e.what() << std::endl;
    exit_code = 1;
  } catch (std::exception &e) {
    std::cout << "Error in the program: " << e.what() << "." << std::endl;
    exit_code = 2;
  }

  #ifdef DEBUG
    std::cout << "Convolution y(" << N << "·" << K << "·" << P << "·" << Q
                    << ") = x(" << N << "·" << C << "·" << H << "·" << W
                    << ") * f(" << K << "·" << C << "·" << R << "·" << S
                    << ") on " << engine_to_string(engine_kind) << ": "
                    << (exit_code ? "failed" : "passed") << std::endl;
  #endif

  return exit_code;
}

// Same as above, but for functions with signature
// void(dnnl::engine::kind engine_kind, int argc, char **argv).
inline int handle_errors(
  dnnl::engine::kind engine_kind, int argc, char **argv,
  std::function<void(dnnl::engine::kind, int, char **)> example) {
  return handle_errors(
    engine_kind, [&]() { example(engine_kind, argc, argv); });
}

// Same as above, but for functions with signature void(dnnl::engine::kind).
inline int handle_errors(
  dnnl::engine::kind engine_kind,
  std::function<void(dnnl::engine::kind)> example) {
  return handle_errors(
    engine_kind, [&]() { example(engine_kind); });
}

// Validates that at least one device of that kind exists on the machine.
inline dnnl::engine::kind validate_engine_kind(dnnl::engine::kind engine_kind) {
  if (dnnl::engine::get_count(engine_kind) == 0) {
    std::cout << "Application couldn't find any device for the selected engine."
              << " Try with other engine kind instead.\n";
    exit(0);
  }
  return engine_kind;
}

// Parses the program arguments and returns the engine kind.
inline dnnl::engine::kind parse_arguments(int argc, char **argv) {

  if (argc == 1)
    return validate_engine_kind(dnnl::engine::kind::cpu);

  if (argc == 9) {
    N = atoi(argv[2]);
    C = atoi(argv[3]);
    K = atoi(argv[4]);
    H = atoi(argv[5]);
    W = atoi(argv[6]);
    R = atoi(argv[7]);
    S = atoi(argv[8]);
    P = (H - R + PH_L + PH_R) / SH + 1; // output height
    Q = (W - S + PW_L + PW_R) / SW + 1; // output width

    x_dims = {N, C, H, W};
    f_dims = {K, C, R, S};
    y_dims = {N, K, P, Q};
    bias_dims = {K};
  }

  if (argc == 2 || argc == 9) {
    std::string engine_kind = argv[1];

    if (engine_kind == "cpu")
      return validate_engine_kind(dnnl::engine::kind::cpu);
    if (engine_kind == "gpu")
      return validate_engine_kind(dnnl::engine::kind::gpu);
  }

  std::cout << "Usage: " << argv[0] << " [cpu|gpu] [N C K H W R S]\n";
  exit(1);
}

// Returns a device selector depending on the device type.
const sycl::device_selector &select_device(dnnl::engine::kind engine_kind) {
  
  static const sycl::gpu_selector gpu;
  static const sycl::cpu_selector cpu;
  
  if (engine_kind == dnnl::engine::kind::gpu) {
    return gpu;
  }
  return cpu;
}

// Multiplies thedimensions to get the total size of the memory object.
inline dnnl::memory::dim product(const dnnl::memory::dims &dims) {
  return std::accumulate(dims.begin(), dims.end(), (dnnl::memory::dim)1,
    std::multiplies<dnnl::memory::dim>());
}

// Read from memory, write to handle.
inline void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {

  dnnl::engine eng = mem.get_engine();
  size_t size = mem.get_desc().get_size();
  if (!handle) throw std::runtime_error("handle is nullptr.");

  #ifdef DNNL_WITH_SYCL
    bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
      && eng.get_kind() == dnnl::engine::kind::cpu);
    bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
      && eng.get_kind() == dnnl::engine::kind::gpu);
    if (is_cpu_sycl || is_gpu_sycl) {
      auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
      if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
        auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
        auto src = buffer.get_access<cl::sycl::access::mode::read>();
        uint8_t *src_ptr = src.get_pointer();
        if (!src_ptr)
          throw std::runtime_error("get_pointer returned nullptr.");
        for (size_t i = 0; i < size; ++i)
          ((uint8_t *)handle)[i] = src_ptr[i];
      } else {
        assert(mkind == dnnl::sycl_interop::memory_kind::usm);
        uint8_t *src_ptr = (uint8_t *)mem.get_data_handle();
        if (!src_ptr)
          throw std::runtime_error("get_data_handle returned nullptr.");
        if (is_cpu_sycl) {
          for (size_t i = 0; i < size; ++i)
            ((uint8_t *)handle)[i] = src_ptr[i];
        } else {
          auto sycl_queue = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
          sycl_queue.memcpy(handle, src_ptr, size).wait();
        }
      }
      return;
    }
  #endif
  
  #if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (eng.get_kind() == dnnl::engine::kind::gpu) {
      dnnl::stream s(eng);
      cl_int ret = clEnqueueReadBuffer(
        dnnl::ocl_interop::get_command_queue(s), 
        dnnl::ocl_interop::get_mem_object(mem), 
        CL_TRUE, 0, size, handle, 0, NULL, NULL
      );
      if (ret != CL_SUCCESS)
        throw std::runtime_error("clEnqueueReadBuffer failed.");
      return;
    }
  #endif

  if (eng.get_kind() == dnnl::engine::kind::cpu) {
    uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
    if (!src) throw std::runtime_error("get_data_handle returned nullptr.");
      for (size_t i = 0; i < size; ++i)
        ((uint8_t *)handle)[i] = src[i];
    return;
  }

  assert(!"not expected");
}

// Read from handle, write to memory.
inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {

  dnnl::engine eng = mem.get_engine();
  size_t size = mem.get_desc().get_size();
  if (!handle) throw std::runtime_error("handle is nullptr.");

  #ifdef DNNL_WITH_SYCL
    bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
      && eng.get_kind() == dnnl::engine::kind::cpu);
    bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
      && eng.get_kind() == dnnl::engine::kind::gpu);
    if (is_cpu_sycl || is_gpu_sycl) {
      auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
      if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
        auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
        auto dst = buffer.get_access<cl::sycl::access::mode::write>();
        uint8_t *dst_ptr = dst.get_pointer();
        if (!dst_ptr)
          throw std::runtime_error("get_pointer returned nullptr.");
        for (size_t i = 0; i < size; ++i)
          dst_ptr[i] = ((uint8_t *)handle)[i];
      } else {
        assert(mkind == dnnl::sycl_interop::memory_kind::usm);
        uint8_t *dst_ptr = (uint8_t *)mem.get_data_handle();
        if (!dst_ptr)
          throw std::runtime_error("get_data_handle returned nullptr.");
        if (is_cpu_sycl) {
          for (size_t i = 0; i < size; ++i)
            dst_ptr[i] = ((uint8_t *)handle)[i];
        } else {
          auto sycl_queue = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
          sycl_queue.memcpy(dst_ptr, handle, size).wait();
        }
      }
      return;
    }
  #endif

  #if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (eng.get_kind() == dnnl::engine::kind::gpu) {
      dnnl::stream s(eng);
      cl_int ret = clEnqueueWriteBuffer(
        dnnl::ocl_interop::get_command_queue(s), 
        dnnl::ocl_interop::get_mem_object(mem), 
        CL_TRUE, 0, size, handle, 0, NULL, NULL
      );
      if (ret != CL_SUCCESS)
        throw std::runtime_error("clEnqueueWriteBuffer failed.");
      return;
    }
  #endif

  if (eng.get_kind() == dnnl::engine::kind::cpu) {
    uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
    if (!dst) throw std::runtime_error("get_data_handle returned nullptr.");
    for (size_t i = 0; i < size; ++i)
      dst[i] = ((uint8_t *)handle)[i];
    return;
  }

  assert(!"not expected");
}

// Initializes three vectors with synthetic values. Note: avoids the use 
// of floating point values due to precision errors between devices.
inline void init_data(std::vector<float> &a, 
                      std::vector<float> &b, 
                      std::vector<float> &c) {
  
  for (int i = 0; i < a.size(); i++) a[i] = i % H;
  for (int i = 0; i < b.size(); i++) b[i] = i % S;
  for (int i = 0; i < c.size(); i++) c[i] = 0;
}

// Perform convolution on host.
std::vector<float> cpu_convolution() {
  int n, c, k, h, w, r, s, p, q;
  int hw=H*W, rs=R*S, pq=P*Q, chw=C*H*W, crs=C*R*S, kpq=K*P*Q;
  
  std::vector<float> x(N*C*H*W);
  std::vector<float> f(K*C*R*S);
  std::vector<float> y(N*K*P*Q);

  init_data(x, f, y);

  for (n = 0; n < N; n++) {
    int n_chw = n * chw;
    int n_kpq = n * kpq;

    for (k = 0; k < K; k++) {
      int k_crs = k * crs;
      int y_off = n_kpq + k * pq;
      
      for (c = 0; c < C; c++) {
        int x_off = n_chw + c * hw;
        int f_off = k_crs + c * rs;

        for (p = 0; p < P; p++) {
          for (q = 0; q < Q; q++) {
            for (r = 0; r < R; r++) {
              for (s = 0; s < S; s++) {

                h = p + r;
                w = q + s;

                y[y_off + p*Q+q] += x[x_off + h*W+w] * f[f_off + r*S+s];
              }
            }
          }
        }
      }
    }
  }

  return y;
}

// Return true if both params have the same value.
bool equals(float a, float b) {
  return fabs(a - b) < std::numeric_limits<float>::epsilon();
}

// Compare host side results with the result buffer from device side: print
// mismatched data 4 times only. 
void compare(std::vector<float> expected, std::vector<float> result) {
  
  int printed_errors = 0;

  for (int i = 0; i < expected.size(); i++) {
    if (!equals(expected[i], result[i])) {
      std::cout << "\nFail - The result is incorrect for element: y(" 
           << i/(K*P*Q) << "·" << i/(P*Q) << "·" << i/Q << "·" << i%Q 
           << "), expected: " << expected[i] << ", but found: " << result[i];
      if (++printed_errors == 4) break;
    }
  }

  if (printed_errors) {
    std::cout << "\nFail - The results mismatch!\n";
  } else {
    std::cout << ": Success - The results are correct!\n";
  }
}

#endif

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
