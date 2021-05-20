
# ~$ source ${ONEAPIHOME}/setvars.sh
# ~$ make

TARGET=onednn

CXX=dpcpp
CXXFLAGS=-std=c++17
LDFLAGS=-I${DNNLROOT}/include -L${DNNLROOT}/lib 
LDLIBS=-ldnnl

all: direct winograd gemm

debug: CXXFLAGS += -g -O0 -fsycl -Wall -DDEBUG 
debug: all;

direct: CXXFLAGS += -DDIRECT
direct: 
	$(CXX) $(CXXFLAGS) $(LDFLAGS) ${TARGET}.cpp $(LDLIBS) -o direct_onednn

winograd: CXXFLAGS += -DWINOGRAD
winograd: 
	$(CXX) $(CXXFLAGS) $(LDFLAGS) ${TARGET}.cpp $(LDLIBS) -o winograd_onednn

gemm: CXXFLAGS += -DGEMM
gemm:
	$(CXX) $(CXXFLAGS) $(LDFLAGS) ${TARGET}.cpp $(LDLIBS) -o gemm_onednn

clean:
	rm direct_onednn winograd_onednn gemm_onednn