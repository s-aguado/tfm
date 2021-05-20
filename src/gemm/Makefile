
# ~$ source ${ONEAPIHOME}/setvars.sh
# ~$ make

TARGET=gemm_sequential gemm_parallel im2col matmul

CXX=dpcpp
CXXFLAGS=-std=c++17
LDFLAGS=-I${DNNLROOT}/include -L${DNNLROOT}/lib 
LDLIBS=-ldnnl

all: ${TARGET}

debug: CXXFLAGS += -g -O0 -fsycl -Wall -DDEBUG
debug: all;

clean:
	rm ${TARGET}