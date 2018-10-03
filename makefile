CXX = g++
CXXFLAGS = -std=c++11 -fopenmp -O3 -flto -I src/utility/ -I src/model/ -I src/param/ -I Random123/include/ -I Random123/examples/

main: testing/testgmxl.cpp src/param/matvec.o src/param/param.o src/model/mxl_gaussblockdiag.o src/model/logistic_regression.o src/utility/readinput.o src/utility/common.o
	$(CXX) $(CXXFLAGS) -o main testing/testgmxl.cpp src/param/matvec.o src/param/param.o src/model/mxl_gaussblockdiag.o src/model/logistic_regression.o src/utility/readinput.o src/utility/common.o

testgmxl.o: testing/testgmxl.cpp 
	$(CXX) $(CXXFLAGS) -c testing/testgmxl.cpp

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




