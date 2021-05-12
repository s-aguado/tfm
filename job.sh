#!/bin/bash
#PBS -N convolution
#PBS -l walltime=00:30:00
#PBS -l nodes=1:gpu:ppn=2
#PBS -d .

# Build the project
./build &> /dev/null && cd bin/

# Run the tests
TIMEFORMAT='%4R';
echo "executable,device,parameters,time1,time2,time3,time4";

executable="sequential"; 
device="cpu";

for params in "4 2 2 1024 1024 3 3" "4 4 4 1024 1024 3 3"; do
  printf "${executable},${device},${params}"
  for i in {1..4}; do
    timei=$( { time ./${executable} ${device} ${params}; } 2>&1 )
    printf ",${timei}"
  done; echo
done;

executable="winograd"; 
device="gpu";

for params in "4 2 2 1024 1024 3 3" "4 4 4 1024 1024 3 3"; do
  printf "${executable},${device},${params}"
  for i in {1..4}; do
    timei=$( { time ./${executable} ${device} ${params}; } 2>&1 )
    printf ",${timei}"
  done; echo
done;

executable="direct"; 

for device in "cpu" "gpu"; do
  for params in "4 2 2 1024 1024 3 3" "4 4 4 1024 1024 3 3"; do
    printf "${executable},${device},${params}"
    for i in {1..4}; do
      timei=$( { time ./${executable} ${device} ${params}; } 2>&1 )
      printf ",${timei}"
    done; echo
  done;
done;

for executable in "gemm" "convolution"; do
  for device in "cpu" "gpu"; do
    for params in "4 2 2 1024 1024 3 3" "4 4 4 1024 1024 3 3"; do
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
