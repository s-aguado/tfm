
# ~$ source ${ONEAPIHOME}/setvars.sh
# ~$ make

TARGET=convolution

CXX=dpcpp
CXXFLAGS=-std=c++17 -g
LDFLAGS=-I${DNNLROOT}/include -L${DNNLROOT}/lib 
LDLIBS=-ldnnl

all: ${TARGET}

clean:
	rm ${TARGET}
