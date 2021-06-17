#!/bin/bash
#PBS -N batch_iris
#PBS -l nodes=1:iris_xe_max:dual_gpu:ppn=2
#PBS -d .

# Set the environment
source /opt/intel/inteloneapi/setvars.sh &> /dev/null

# Build the project
cd .. && ./build > /dev/null && cd bin/

# Run the tests
TIMEFORMAT='%4R';
echo "executable,device,parameters,time1,time2,time3,time4";

device="gpu";

for executable in "direct_parallel" "gemm_parallel" "blis_parallel"\
                  "direct_onednn" "gemm_onednn" "winograd_onednn"; do
  for params in\
    "8  4 4 1024 1024 3 3"\
    "16 4 4 1024 1024 3 3"\
    "24 4 4 1024 1024 3 3"\
    "32 4 4 1024 1024 3 3"\
    "40 4 4 1024 1024 3 3"\
    "48 4 4 1024 1024 3 3"\
    "56 4 4 1024 1024 3 3"\
    "64 4 4 1024 1024 3 3"\
    "72 4 4 1024 1024 3 3"\
    "80 4 4 1024 1024 3 3"
  do
    printf "${executable},${device},${params}"
    for i in {1..4}; do
      timei=$( { time ./${executable} ${device} ${params}; } 2>&1 )
      printf ",${timei}"
    done; echo
  done;
done;

# Intel® Iris® Xe MAX