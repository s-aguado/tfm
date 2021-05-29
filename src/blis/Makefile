
# ~$ source ${ONEAPIHOME}/setvars.sh
# ~$ make

TARGET=blis_sequential blis_parallel blis_parallel_gpu

CXX=dpcpp
CXXFLAGS=-std=c++17
LDFLAGS=-I${DNNLROOT}/include -L${DNNLROOT}/lib
LDLIBS=-ldnnl

all: ${TARGET}

blis_parallel_gpu: CXXFLAGS += -DGPU
blis_parallel_gpu: 
	$(CXX) $(CXXFLAGS) $(LDFLAGS) blis_parallel.cpp $(LDLIBS) -o blis_parallel_gpu

debug: CXXFLAGS += -g -O0 -fsycl -Wall -DDEBUG
debug: all;

clean:
	rm ${TARGET}