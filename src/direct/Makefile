
# ~$ source ${ONEAPIHOME}/setvars.sh
# ~$ make

TARGET=direct_sequential direct_parallel

CXX=dpcpp
CXXFLAGS=-std=c++17
LDFLAGS=-I${DNNLROOT}/include -L${DNNLROOT}/lib 
LDLIBS=-ldnnl

all: ${TARGET}

debug: CXXFLAGS += -g -O0 -fsycl -Wall -DDEBUG
debug: all;

clean:
	rm ${TARGET}