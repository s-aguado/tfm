#!/bin/bash
#PBS -N size_p630
#PBS -l nodes=1:gpu:ppn=2
#PBS -d .

# Set the environment
source /opt/intel/inteloneapi/setvars.sh &> /dev/null

# Build the project
cd .. && ./build > /dev/null && cd bin/

# Run the tests
TIMEFORMAT='%4R';
echo "executable,device,parameters,time1,time2,time3,time4";

device="cpu";

for executable in "direct_sequential" "gemm_sequential" "blis_sequential" "im2col" "matmul"; do
  for params in\
    "8 4 4 64 64 3 3"\
    "8 4 4 128 128 3 3"\
    "8 4 4 256 256 3 3"\
    "8 4 4 512 512 3 3"\
    "8 4 4 1024 1024 3 3"\
    "8 4 4 2048 2048 3 3"\
    "8 4 4 4096 4096 3 3"
  do
    printf "${executable},${device},${params}"
    for i in {1..4}; do
      timei=$( { time ./${executable} ${device} ${params}; } 2>&1 )
      printf ",${timei}"
    done; echo
  done;
done;

device="gpu";

for executable in "winograd_onednn"; do
  for params in\
    "8 4 4 64 64 3 3"\
    "8 4 4 128 128 3 3"\
    "8 4 4 256 256 3 3"\
    "8 4 4 512 512 3 3"\
    "8 4 4 1024 1024 3 3"\
    "8 4 4 2048 2048 3 3"\
    "8 4 4 4096 4096 3 3"   
  do
    printf "${executable},${device},${params}"
    for i in {1..4}; do
      timei=$( { time ./${executable} ${device} ${params}; } 2>&1 )
      printf ",${timei}"
    done; echo
  done;
done;

for executable in "direct_onednn" "gemm_onednn" "direct_parallel" "gemm_parallel" "blis_parallel"; do
  for device in "cpu" "gpu"; do
    for params in\
      "8 4 4 64 64 3 3"\
      "8 4 4 128 128 3 3"\
      "8 4 4 256 256 3 3"\
      "8 4 4 512 512 3 3"\
      "8 4 4 1024 1024 3 3"\
      "8 4 4 2048 2048 3 3"\
      "8 4 4 4096 4096 3 3"
    do
      printf "${executable},${device},${params}"
      for i in {1..4}; do
        timei=$( { time ./${executable} ${device} ${params}; } 2>&1 )
        printf ",${timei}"
      done; echo
    done;
  done;
done;

# xeon,cfl,e-2176g,ram64gb,net1gbe,gpu,gen9