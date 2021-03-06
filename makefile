CXX = g++-7
CXXFLAGS = -std=c++11 -fopenmp -O3 -flto -I src/utility/ -I src/model/ -I src/opt/ -I src/param/ -I Random123/include/ -I Random123/examples/

all: runlogit runmixedlogit functionevaltime testlr_SGD testlr_HGD testlr_AGD testmxl 

runlogit: run/runlogit.cpp src/param/matvec.o src/param/param.o src/model/logistic_regression.o src/utility/readinput.o src/utility/common.o
	$(CXX) $(CXXFLAGS) -o runlogit run/runlogit.cpp src/param/matvec.o src/param/param.o src/model/logistic_regression.o src/utility/readinput.o src/utility/common.o

runmixedlogit: run/runmixedlogit.cpp src/param/matvec.o src/param/param.o src/model/mxl_gaussblockdiag.o src/utility/readinput.o src/utility/common.o
	$(CXX) $(CXXFLAGS) -o runmixedlogit run/runmixedlogit.cpp src/param/matvec.o src/param/param.o src/model/mxl_gaussblockdiag.o src/utility/readinput.o src/utility/common.o

functionevaltime: run/functionevaltime.cpp src/param/matvec.o src/param/param.o src/model/mxl_gaussblockdiag.o src/utility/readinput.o src/utility/common.o
	$(CXX) $(CXXFLAGS) -o functionevaltime run/functionevaltime.cpp src/param/matvec.o src/param/param.o src/model/mxl_gaussblockdiag.o src/utility/readinput.o src/utility/common.o

testlr_SGD: testing/test_logistic_regression_SGD.cpp src/param/matvec.o src/param/param.o src/model/logistic_regression.o src/utility/readinput.o src/utility/common.o
	$(CXX) $(CXXFLAGS) -o testLR_StochasticGD testing/test_logistic_regression_SGD.cpp src/param/matvec.o src/param/param.o src/model/logistic_regression.o src/utility/readinput.o src/utility/common.o

testlr_HGD: testing/test_logistic_regression_HGD.cpp src/param/matvec.o src/param/param.o src/model/logistic_regression.o src/utility/readinput.o src/utility/common.o
	$(CXX) $(CXXFLAGS) -o testLR_HybridGD testing/test_logistic_regression_HGD.cpp src/param/matvec.o src/param/param.o src/model/logistic_regression.o src/utility/readinput.o src/utility/common.o

testlr_AGD: testing/test_logistic_regression_AGD.cpp src/param/matvec.o src/param/param.o src/model/logistic_regression.o src/utility/readinput.o src/utility/common.o
	$(CXX) $(CXXFLAGS) -o testLR_FullGD testing/test_logistic_regression_AGD.cpp src/param/matvec.o src/param/param.o src/model/logistic_regression.o src/utility/readinput.o src/utility/common.o

testmxl: testing/test_mxl.cpp src/param/matvec.o src/param/param.o src/model/mxl_gaussblockdiag.o src/utility/readinput.o src/utility/common.o
	$(CXX) $(CXXFLAGS) -o testmxl testing/test_mxl.cpp src/param/matvec.o src/param/param.o src/model/mxl_gaussblockdiag.o src/utility/readinput.o src/utility/common.o

matvec.o: src/param/matvec.cpp 
	$(CXX) $(CXXFLAGS) -c src/param/matvec.cpp 

param.o: src/param/param.cpp
	$(CXX) $(CXXFLAGS) -c src/param/param.cpp

mxl_gaussblockdiag.o: src/model/mxl_gaussblockdiag.cpp
	$(CXX) $(CXXFLAGS) -c src/model/mxl_gaussblockdiag.cpp

logistic_regression.o: src/model/logistic_regression.cpp
	$(CXX) $(CXXFLAGS) -c src/model/logistic_regression.cpp

readinput.o: src/utility/readinput.cpp 
	$(CXX) $(CXXFLAGS) -c src/utility/readinput.cpp 

common.o: src/utility/common.cpp
	$(CXX) $(CXXFLAGS) -c src/utility/common.cpp




