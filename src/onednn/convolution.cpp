
/**
 * convolution.cpp
 * 
 * This C++ API example demonstrates how to create and execute a
 * Convolution primitive in forward propagation mode.
 * 
 * Key optimizations included in this example:
 *  - Creation of optimized memory format from the primitive descriptor;
 *  - Primitive attributes with fused post-ops.
 *
 * More information about oneDNN convolution algorithms:
 * https://oneapi-src.github.io/oneDNN/dev_guide_convolution.html
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"
#include "dnnl_debug.h"
#include "../utils.hpp"

using namespace std;
using namespace dnnl;

using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;

void convolution(dnnl::engine::kind engine_kind) {

  // Create execution engine and stream
  dnnl::engine engine(engine_kind, 0);
  dnnl::stream stream(engine);

  // Create memory descriptors with format_tag::any for the primitive. This
  // enables the convolution primitive to choose memory layouts for an
  // optimized primitive implementation, and these layouts may differ from the
  // ones provided by the user.
  auto x_md = memory::desc(x_dims, dt::f32, tag::any);
  auto f_md = memory::desc(f_dims, dt::f32, tag::any);
  auto y_md = memory::desc(y_dims, dt::f32, tag::any);
  auto bias_md = memory::desc(bias_dims, dt::f32, tag::a);

  // Forces the fallback to the gemm algorithm indicating the format_tag.
  #ifdef GEMM
    x_md = memory::desc(x_dims, dt::f32, tag::nchw);
    f_md = memory::desc(f_dims, dt::f32, tag::oihw);
    y_md = memory::desc(y_dims, dt::f32, tag::nchw);
  #endif

  // Create memory objects = memory descriptors + data. In this example, 
  // NCHW layout is assumed for src and dst, and OIHW for weights.
  auto x_mem = memory({x_dims, dt::f32, tag::nchw}, engine);
  auto f_mem = memory({f_dims, dt::f32, tag::oihw}, engine);
  auto y_mem = memory({y_dims, dt::f32, tag::nchw}, engine);
  auto bias_mem = memory(bias_md, engine);

  // Allocate buffers.
  std::vector<float> x_vec(product(x_dims));
  std::vector<float> f_vec(product(f_dims));
  std::vector<float> y_vec(product(y_dims), 0);
  std::vector<float> bias_vec(product(bias_dims));

  // Initialize tensors.
  init_data(x_vec, f_vec, bias_vec);

  // Fill the memory object's handle with the data.
  write_to_dnnl_memory(x_vec.data(), x_mem);
  write_to_dnnl_memory(f_vec.data(), f_mem);
  write_to_dnnl_memory(bias_vec.data(), bias_mem);

  // Create operation descriptor.
  #ifdef WINOGRAD
    auto convolution_algorithm = algorithm::convolution_winograd;
  #else
    auto convolution_algorithm = algorithm::convolution_direct;
  #endif

  auto conv_desc = convolution_forward::desc(
    prop_kind::forward_inference,
    convolution_algorithm, // convolution_winograd | convolution_direct | gemm (fallback)
    x_md, f_md, bias_md, y_md, 
    strides_dims, padding_dims_l, padding_dims_r
  );
  
  // We could indicate additional operations to apply to the result.
  // For example ReLU: y[:] = ReLU(y[:] + convolution(x[:],f[:]))
  primitive_attr conv_attr;
  //post_ops conv_ops;
  //const float scale = 1.f, alpha = 0.f, beta = 0.f;
  //conv_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
  //conv_attr.set_post_ops(conv_ops);

  // Create primitive descriptor.
  auto conv_pd
    = convolution_forward::primitive_desc(conv_desc, conv_attr, engine);

  // For now, assume that the src, weights, and dst memory layouts generated
  // by the primitive and the ones provided by the user are identical.
  auto conv_x_mem = x_mem;
  auto conv_f_mem = f_mem;
  auto conv_y_mem = y_mem;

  // Reorder the data in case the src and weights memory layouts generated by
  // the primitive and the ones provided by the user are different. In this
  // case, we create additional memory objects with internal buffers that will
  // contain the reordered data. The data in dst will be reordered after the
  // convolution computation has finalized.
  if (conv_pd.src_desc() != x_mem.get_desc()) {
    conv_x_mem = memory(conv_pd.src_desc(), engine);
    reorder(x_mem, conv_x_mem)
      .execute(stream, x_mem, conv_x_mem);
  }

  if (conv_pd.weights_desc() != f_mem.get_desc()) {
    conv_f_mem = memory(conv_pd.weights_desc(), engine);
    reorder(f_mem, conv_f_mem)
      .execute(stream, f_mem, conv_f_mem);
  }

  if (conv_pd.dst_desc() != y_mem.get_desc()) {
    conv_y_mem = memory(conv_pd.dst_desc(), engine);
  }

  // Create the primitive.
  auto conv_prim = convolution_forward(conv_pd);

  // Primitive arguments.
  std::unordered_map<int, memory> conv_args;
  conv_args.insert({DNNL_ARG_SRC, conv_x_mem});
  conv_args.insert({DNNL_ARG_WEIGHTS, conv_f_mem});
  conv_args.insert({DNNL_ARG_BIAS, bias_mem});
  conv_args.insert({DNNL_ARG_DST, conv_y_mem});

  // Primitive execution: forward convolution direct algorithm.
  conv_prim.execute(stream, conv_args); // <----------------------------------    time measurement ?

  // Reorder the data in case the dst memory descriptor generated by the
  // primitive and the one provided by the user are different.
  if (conv_pd.dst_desc() != y_mem.get_desc()) {
    reorder(conv_y_mem, y_mem)
      .execute(stream, conv_y_mem, y_mem);
  } else {
    y_mem = conv_y_mem;
  }

  // Wait for the computation to finalize.
  stream.wait();

  // Read data from memory object's handle.
  read_from_dnnl_memory(y_vec.data(), y_mem);
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
