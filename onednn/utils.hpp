
/**
 * utils.cpp
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

// Exception class to indicate that the example uses a feature that is not
// available on the current systems. It is not treated as an error then, but
// just notifies a user.
struct unimplemented : public std::exception {
  unimplemented(const char *msg) noexcept : message(msg) {}
  const char *what() const noexcept override { return message; }
  const char *message;
};

// Returns the string representation of the engine kind
inline const char *engine_kind2str_upper(dnnl::engine::kind kind) {
  if (kind == dnnl::engine::kind::cpu) return "CPU";
  if (kind == dnnl::engine::kind::gpu) return "GPU";
  assert(!"not expected");
  return "<Unknown engine>";
}

// Runs example function with signature void() and catches errors.
// Returns `0` on success, `1` or oneDNN error, and `2` on program error.
inline int handle_errors(
  dnnl::engine::kind engine_kind,
  std::function<void()> example) {
  int exit_code = 0;

  try {
    example();
  } catch (unimplemented &e) {
    std::cout << e.message << std::endl;
    exit_code = 0;
  } catch (dnnl::error &e) {
    std::cout << "oneDNN error caught: " << std::endl
      << "\tStatus: " << dnnl_status2str(e.status) << std::endl
      << "\tMessage: " << e.what() << std::endl;
    exit_code = 1;
  } catch (std::exception &e) {
    std::cout << "Error in the program: " << e.what() << "." << std::endl;
    exit_code = 2;
  }

  std::string engine_kind_str = engine_kind2str_upper(engine_kind);
  std::cout << "Example " << (exit_code ? "failed" : "passed") << " on "
    << engine_kind_str << "." << std::endl;
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

// Validates that at least one device of that kind exists on the machine
dnnl::engine::kind validate_engine_kind(dnnl::engine::kind akind) {
  if (dnnl::engine::get_count(akind) == 0) {
    std::cout << "Application couldn't find any device for the selected engine."
    << " Try with other engine kind instead.\n";
    exit(0);
  }
  return akind;
}

// Selects the device on which the operation will be performed, CPU by default.
inline dnnl::engine::kind parse_engine_kind(int argc, char **argv) {

  if (argc <= 1)
    return validate_engine_kind(dnnl::engine::kind::cpu);
  
  std::string engine_kind = argv[1];
  
  if (engine_kind == "cpu")
    return validate_engine_kind(dnnl::engine::kind::cpu);
  if (engine_kind == "gpu")
    return validate_engine_kind(dnnl::engine::kind::gpu);

  std::cout << "Inappropriate engine kind.\n"
    << "Please indicate the engine kind in the first argument: " 
    << argv[0] << " [cpu|gpu] [arguments..]\n";
  
  exit(1);
}

inline dnnl::memory::dim product(const dnnl::memory::dims &dims) {
  return std::accumulate(dims.begin(), dims.end(), (dnnl::memory::dim)1,
    std::multiplies<dnnl::memory::dim>());
}

// Read from memory, write to handle
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

// Read from handle, write to memory
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
