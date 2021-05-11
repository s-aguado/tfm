
# ~$ source ${ONEAPIHOME}/setvars.sh
# ~$ make

TARGET=convolution

CXX=dpcpp
CXXFLAGS=-std=c++17
LDFLAGS=-I${DNNLROOT}/include -L${DNNLROOT}/lib 
LDLIBS=-ldnnl

all: direct winograd gemm

debug: CXXFLAGS += -g -O0 -fsycl -DDEBUG 
debug: all;

direct: CXXFLAGS += -DDIRECT
direct: 
	$(CXX) $(CXXFLAGS) $(LDFLAGS) ${TARGET}.cpp $(LDLIBS) -o direct

winograd: CXXFLAGS += -DWINOGRAD
winograd: 
	$(CXX) $(CXXFLAGS) $(LDFLAGS) ${TARGET}.cpp $(LDLIBS) -o winograd

gemm: CXXFLAGS += -DGEMM
gemm:
	$(CXX) $(CXXFLAGS) $(LDFLAGS) ${TARGET}.cpp $(LDLIBS) -o gemm

clean:
	rm direct winograd gemm