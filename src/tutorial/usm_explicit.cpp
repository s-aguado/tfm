#include <CL/sycl.hpp>
#include <iostream>
#define N 8

using namespace sycl;

int main() {

  // Create a QUEUE passing a device_selector
  queue q(gpu_selector{});

  int *device_array = malloc_device<int>(N,q);
  int *host_array = new int[N];
  for (int i = 0; i < N; i++) host_array[i] = i;

  // Copy the initialized data from host to device
  q.submit([&](handler &h) {
    h.memcpy(device_array, host_array, N*sizeof(int));
  }).wait();

  // Execute the KERNEL in the device
  q.submit([&](handler &h) {
    h.parallel_for(N, [=](auto &i) {
      device_array[i] += 1;
    });
  }).wait();

  // Copy the result data from device to host
  q.submit([&](handler &h) {
    h.memcpy(host_array, device_array, N*sizeof(int)); 
  }).wait();
  
  for (int i = 0; i < N; i++)
    std::cout << "array[" << i << "] = " << host_array[i] << "\n";

  return 0;
}
