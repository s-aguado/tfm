
/**
 * onednn.cpp
 * 
 * This C++ API example demonstrates how to create and execute a
 * Convolution primitive in forward propagation mode using all 
 * the algorithms supported by oneDNN.
 *
 * https://oneapi-src.github.io/oneDNN/dev_guide_convolution.html
 */

#include "../utils.hpp"
using namespace dnnl;

using format = dnnl::memory::format_tag;
using type = dnnl::memory::data_type;

void convolution(dnnl::engine::kind engine_kind) {

  // Define memory dims
  dnnl::memory::dims 
    x_dims = {N,C,H,W},
    f_dims = {K,C,R,S},
    y_dims = {N,K,P,Q},
    b_dims = {K};

  // Create execution engine and stream
  engine engine(engine_kind, 0);
  stream stream(engine);

  // Create memory descriptors with format_tag::any for the primitive. This
  // enables the convolution primitive to choose memory layouts for an
  // optimized primitive implementation, and these layouts may differ from the
  // ones provided by the user.
  memory::desc x_desc(x_dims, type::f32, format::any);
  memory::desc f_desc(f_dims, type::f32, format::any);
  memory::desc y_desc(y_dims, type::f32, format::any);
  memory::desc b_desc(b_dims, type::f32, format::a);

  // Forces the fallback to the gemm algorithm indicating the format_tag.
  #ifdef GEMM
    x_desc = memory::desc(x_dims, type::f32, format::nchw);
    f_desc = memory::desc(f_dims, type::f32, format::oihw);
    y_desc = memory::desc(y_dims, type::f32, format::nchw);
  #endif

  // Create memory objects = memory descriptors + data. In this example, 
  // NCHW layout is assumed for src and dst, and OIHW for weights.
  memory x_mem({x_dims, type::f32, format::nchw}, engine);
  memory f_mem({f_dims, type::f32, format::oihw}, engine);
  memory y_mem({y_dims, type::f32, format::nchw}, engine);
  memory b_mem(b_desc, engine);

  // Allocate buffers.
  std::vector<float> x_vec(product(x_dims));
  std::vector<float> f_vec(product(f_dims));
  std::vector<float> y_vec(product(y_dims), 0);
  std::vector<float> bias_vec(product(b_dims));

  // Initialize tensors.
  init_data(x_vec, f_vec, bias_vec);

  // Fill the memory object's handle with the data.
  write_to_dnnl_memory(x_vec.data(), x_mem);
  write_to_dnnl_memory(f_vec.data(), f_mem);
  write_to_dnnl_memory(bias_vec.data(), b_mem);

  // Create operation descriptor.
  #ifdef WINOGRAD
    auto convolution_algorithm = algorithm::convolution_winograd;
  #else
    auto convolution_algorithm = algorithm::convolution_direct;
  #endif

  convolution_forward::desc conv_desc(
    prop_kind::forward_inference,
    convolution_algorithm,
    x_desc, f_desc, b_desc, y_desc,
    {SH,SW}, {PH_L,PW_L}, {PH_R,PW_R}
  );
  
  // We could indicate additional operations to apply to the result.
  // For example ReLU: y[:] = ReLU(y[:] + convolution(x[:],f[:]))
  primitive_attr conv_attr;
  //post_ops conv_ops;
  //const float scale = 1.f, alpha = 0.f, beta = 0.f;
  //conv_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
  //conv_attr.set_post_ops(conv_ops);

  // Create primitive descriptor.
  convolution_forward::primitive_desc conv_pd(conv_desc, conv_attr, engine);

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
  convolution_forward conv_prim(conv_pd);

  // Execute the primitive.
  conv_prim.execute(stream, {
    {DNNL_ARG_SRC, conv_x_mem},
    {DNNL_ARG_WEIGHTS, conv_f_mem},
    {DNNL_ARG_BIAS, b_mem},
    {DNNL_ARG_DST, conv_y_mem}
  });

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
