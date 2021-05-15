### Batched forward convolution with Intel OneAPI

Code repository for the Master Thesis *Implementación do algoritmo da convolución por lotes usando Intel oneAPI*.

#### Dependencies

- [Intel oneAPI Base Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html?operatingsystem=linux&distributions=webdownload&options=online)
  - Intel oneAPI Deep Neural Network Library (oneDNN)
  - Intel oneAPI DPC++/C++ Compiler
- [Intel oneAPI HPC Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/hpc-toolkit/download.html?operatingsystem=linux&distributions=webdownload&options=online)
  - Intel C++ Compiler Classic (`icc`)

#### Local

Clone this repository:

```bash
git clone https://git.fic.udc.es/s.aguado/tfm.git && cd tfm
```

Set the environment:

```bash
source /opt/intel/inteloneapi/setvars.sh # replace with custom installation path 
```

Compile the codes:

```bash
./build debug # to print some feedback while running the codes
./build       # to run the codes in quiet mode
```

After running these commands, the executables should be in the `bin/` folder. All of them share the same interface:

```bash
./executable [ (cpu|gpu) [N C K H W R S] ]
```

Examples:

```bash
./bin/convolution # Run convolution with default parameters in the CPU
./bin/gemm gpu    # Run convolution with default parameters in the GPU
./bin/winograd gpu 4 3 3 64 64 3 3 
```

#### Cloud

1. [Sign up for Intel DevCloud for oneAPI](https://www.intel.com/content/www/us/en/forms/idz/devcloud-enrollment/oneapi-request.html)
2. [Connect via SSH from Linux/macOS](https://devcloud.intel.com/oneapi/documentation/connect-with-ssh-linux-macos/)
3. [Submit a job to the queue](https://devcloud.intel.com/oneapi/documentation/job-submission/)
4. [Advanced queue management](https://devcloud.intel.com/oneapi/documentation/advanced-queue/)

This repo includes a [script](./job.sh) ready to be launched as a job in DevCloud.