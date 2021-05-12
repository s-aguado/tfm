
/**
 * convolution.cpp
 * 
 * Implements the direct convolution algorithm in forward propagation mode. 
 * Executes it sequentially in the CPU.
 */

#include <iostream>
#include <limits>
#include "../utils.hpp"

void convolution() {
  std::vector<float> result = cpu_convolution();
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
