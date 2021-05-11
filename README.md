Dependencies:

- Intel oneAPI Base Toolkit
  - Intel oneAPI Deep Neural Network Library (oneDNN)
- Intel oneAPI HPC Toolkit
  - Intel C++ Compiler Classic (`icc`)

Set the environment:

```bash
source /opt/intel/oneapi/setvars.sh # replace with custom installation path 
```

Clone this repository:

```bash
git clone https://git.fic.udc.es/s.aguado/tfm.git
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