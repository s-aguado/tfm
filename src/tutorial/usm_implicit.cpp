#include <CL/sycl.hpp>
#include <iostream>
#define N 8

using namespace sycl;

int main() {

  // Create a QUEUE passing a device_selector
  queue q(gpu_selector{});

  int *shared_array = malloc_shared<int>(N,q);
  int *host_array = malloc_host<int>(N,q);
  for (int i = 0; i < N; i++) host_array[i] = i;

  // Execute the KERNEL in the device
  q.submit([&](handler &h) {
    h.parallel_for(N, [=](auto &i) {
      shared_array[i] = host_array[i] + 1;
    });
  }).wait();
  
  for (int i = 0; i < N; i++)
    std::cout << "array[" << i << "] = " << shared_array[i] << "\n";

  return 0;
}
