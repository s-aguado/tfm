
# ~$ source ${ONEAPIHOME}/setvars.sh
# ~$ make

TARGET=convolution
#ONEAPIHOME=/opt/intel/oneapi
#SYCLROOT=${ONEAPIHOME}/compiler/2021.1.1/linux

CXX=dpcpp
CXXFLAGS=-std=c++17 -g
LDFLAGS=#-I${SYCLROOT}/include #-I${ONEAPIHOME}/dev-utilities/latest/include
LDLIBS=

all: ${TARGET}

clean:
	rm ${TARGET}
