#!/bin/bash
#PBS -N batch
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

for executable in "direct_sequential" "gemm_sequential" "gemm_chwn" "blis_sequential" "blis_chwn" "im2col" "matmul"; do
  for params in\
    "8  2 2 1024 1024 3 3"\
    "16 2 2 1024 1024 3 3"\
    "24 2 2 1024 1024 3 3"\
    "32 2 2 1024 1024 3 3"\
    "40 2 2 1024 1024 3 3"\
    "48 2 2 1024 1024 3 3"\
    "56 2 2 1024 1024 3 3"\
    "64 2 2 1024 1024 3 3"\
    "72 2 2 1024 1024 3 3"\
    "80 2 2 1024 1024 3 3"
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
    "8  2 2 1024 1024 3 3"\
    "16 2 2 1024 1024 3 3"\
    "24 2 2 1024 1024 3 3"\
    "32 2 2 1024 1024 3 3"\
    "40 2 2 1024 1024 3 3"\
    "48 2 2 1024 1024 3 3"\
    "56 2 2 1024 1024 3 3"\
    "64 2 2 1024 1024 3 3"\
    "72 2 2 1024 1024 3 3"\
    "80 2 2 1024 1024 3 3"      
do
  printf "${executable},${device},${params}"
  for i in {1..4}; do
    timei=$( { time ./${executable} ${device} ${params}; } 2>&1 )
    printf ",${timei}"
  done; echo
done;

for executable in "direct_parallel" "direct_onednn" "gemm_parallel" "gemm_onednn" "blis_parallel"; do
  for device in "cpu" "gpu"; do
    for params in\
      "8  2 2 1024 1024 3 3"\
      "16 2 2 1024 1024 3 3"\
      "24 2 2 1024 1024 3 3"\
      "32 2 2 1024 1024 3 3"\
      "40 2 2 1024 1024 3 3"\
      "48 2 2 1024 1024 3 3"\
      "56 2 2 1024 1024 3 3"\
      "64 2 2 1024 1024 3 3"\
      "72 2 2 1024 1024 3 3"\
      "80 2 2 1024 1024 3 3"
    do
      printf "${executable},${device},${params}"
      for i in {1..4}; do
        timei=$( { time ./${executable} ${device} ${params}; } 2>&1 )
        printf ",${timei}"
      done; echo
    done;
  done;
done;

