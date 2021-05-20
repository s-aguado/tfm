#!/bin/bash
#PBS -N convolution
#PBS -l walltime=00:30:00
#PBS -l nodes=1:gpu:ppn=2
#PBS -d .

# Set the environment
source /opt/intel/inteloneapi/setvars.sh &> /dev/null

# Build the project
./build > /dev/null && cd bin/

# Run the tests
TIMEFORMAT='%4R';
echo "executable,device,parameters,time1,time2,time3,time4";

device="cpu";

for executable in "direct_sequential" "gemm_sequential" "im2col" "matmul"; do
  for params in\
    "4  2 2 1024 1024 3 3"\
    "8  2 2 1024 1024 3 3"\
    "16 2 2 1024 1024 3 3"\
    "4  4 4 1024 1024 3 3"\
    "8  4 4 1024 1024 3 3"\
    "16 4 4 1024 1024 3 3"\
    "4  2 2 2048 2048 3 3"\
    "8  2 2 2048 2048 3 3"\
    "16 2 2 2048 2048 3 3"\
    "4  4 4 2048 2048 3 3"\
    "8  4 4 2048 2048 3 3"\
    "16 4 4 2048 2048 3 3"
  do
    printf "${executable},${device},${params}"
    for i in {1..4}; do
      timei=$( { time ./${executable} ${device} ${params}; } 2>&1 )
      printf ",${timei}"
    done; echo
  done;
done;

device="gpu";
executable="winograd_onednn";

for params in\
  "4  2 2 1024 1024 3 3"\
  "8  2 2 1024 1024 3 3"\
  "16 2 2 1024 1024 3 3"\
  "4  4 4 1024 1024 3 3"\
  "8  4 4 1024 1024 3 3"\
  "16 4 4 1024 1024 3 3"\
  "4  2 2 2048 2048 3 3"\
  "8  2 2 2048 2048 3 3"\
  "16 2 2 2048 2048 3 3"\
  "4  4 4 2048 2048 3 3"\
  "8  4 4 2048 2048 3 3"\
  "16 4 4 2048 2048 3 3"
do
  printf "${executable},${device},${params}"
  for i in {1..4}; do
    timei=$( { time ./${executable} ${device} ${params}; } 2>&1 )
    printf ",${timei}"
  done; echo
done;

for executable in "direct_parallel" "direct_onednn" "gemm_parallel" "gemm_onednn"; do
  for device in "cpu" "gpu"; do
    for params in\
      "4  2 2 1024 1024 3 3"\
      "8  2 2 1024 1024 3 3"\
      "16 2 2 1024 1024 3 3"\
      "4  4 4 1024 1024 3 3"\
      "8  4 4 1024 1024 3 3"\
      "16 4 4 1024 1024 3 3"\
      "4  2 2 2048 2048 3 3"\
      "8  2 2 2048 2048 3 3"\
      "16 2 2 2048 2048 3 3"\
      "4  4 4 2048 2048 3 3"\
      "8  4 4 2048 2048 3 3"\
      "16 4 4 2048 2048 3 3"
    do
      printf "${executable},${device},${params}"
      for i in {1..4}; do
        timei=$( { time ./${executable} ${device} ${params}; } 2>&1 )
        printf ",${timei}"
      done; echo
    done;
  done;
done;

# CPU - Intel(R) Xeon(R) E-2176G CPU @ 3.70GHz
# GPU - Intel(R) UHD Graphics P630 [0x3e96]
