
# ~$ source ${ONEAPIHOME}/setvars.sh
# ~$ make

TARGET=convolution

CXX=dpcpp
CXXFLAGS=-std=c++17
LDFLAGS=-I${DNNLROOT}/include -L${DNNLROOT}/lib 
LDLIBS=-ldnnl

all: ${TARGET}

debug:
	$(CXX) -DDEBUG -g -O0 -fsycl $(CXXFLAGS) $(LDFLAGS) $(LDLIBS) ${TARGET}.cpp $(LDLIBS) -o ${TARGET}

clean:
	rm ${TARGET}
