
# ~$ source ${ONEAPIHOME}/setvars.sh
# ~$ make

TARGET=convolution
ONEAPIHOME=/opt/intel/oneapi
SYCLROOT=${ONEAPIHOME}/compiler/2021.1.1/linux

CXX=icc
CXXFLAGS=-I${DNNLROOT}/include -I${SYCLROOT}/include
LDFLAGS=-L${DNNLROOT}/lib 
LDLIBS=-ldnnl

all: ${TARGET}

clean:
	rm ${TARGET}
