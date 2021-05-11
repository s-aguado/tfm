
# ~$ source ${ONEAPIHOME}/setvars.sh
# ~$ make

TARGET=convolution

CXX=dpcpp
CXXFLAGS=-std=c++17
LDFLAGS=-I${DNNLROOT}/include -L${DNNLROOT}/lib 
LDLIBS=-ldnnl

all: ${TARGET}

debug: CXXFLAGS += -g -O0 -fsycl -DDEBUG
debug: all;

clean:
	rm ${TARGET}