
# ~$ source ${ONEAPIHOME}/setvars.sh
# ~$ make

TARGET=convolution

CXX=dpcpp
CXXFLAGS=-std=c++17
LDFLAGS=-I${DNNLROOT}/include -L${DNNLROOT}/lib 
LDLIBS=-ldnnl

all:
	$(CXX) $(CXXFLAGS) $(LDFLAGS) ${TARGET}.cpp $(LDLIBS) -o sequential

debug: CXXFLAGS += -g -O0 -fsycl -DDEBUG
debug: all;

clean:
	rm sequential