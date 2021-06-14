
/**
 * hello.cpp
 * 
 * Simple usage example of the SYCL/DPC++ API using
 * buffers to access to the host/device memory.
 */

#include <CL/sycl.hpp>
using namespace sycl;

#define N 8

int main() {

  int *array = new int[N];
  for (int i = 0; i < N; i++) array[i] = i;

  {
    // Create a QUEUE passing a device_selector
    queue q(gpu_selector{});
  
    // Create a one-dimensional BUFFER using host allocated array
    buffer b(array, range(N));
  
    q.submit([&](handler &h) {
      
      // Get an ACCESSOR to the host data
      accessor a(b, h, write_only);

      // Execute the KERNEL in the device
      h.parallel_for(N, [=](auto &i) {
        a[i] += 1;
      });
    });
  }
  
  // The array will be updated upon exiting the scope
  for (int i = 0; i < N; i++)
    std::cout << "array[" << i << "] = " << array[i] << "\n";

  return 0;
}
