Dependencies:

- Intel oneAPI Base Toolkit
  - Intel oneAPI Deep Neural Network Library (oneDNN)
- Intel oneAPI HPC Toolkit
  - Intel C++ Compiler Classic (`icc`)

Compile and execute the examples:

```bash
source /opt/intel/oneapi/setvars.sh # replace with custom installation path 

cd tfm/oneapi # for the oneAPI DPC++ (Data Parallel C++) example
cd tfm/onednn # for the oneDNN example

make
./convolution
```
