CXX = g++
CXXFLAGS = -std=c++11 -O3 -flto -I src/utility/ -I src/model/ -I src/param/ -I ../../../work/boost_1_63_0/

main: testing/testgmxl.cpp src/param/matvec.o src/param/param.o src/model/mxl_gaussblockdiag.o src/utility/readinput.o src/utility/rnggenerator.o
	$(CXX) $(CXXFLAGS) -o main testing/testgmxl.cpp src/param/matvec.o src/param/param.o src/model/mxl_gaussblockdiag.o src/utility/readinput.o src/utility/rnggenerator.o

testgmxl.o: testing/testgmxl.cpp 
	$(CXX) $(CXXFLAGS) -c testing/testgmxl.cpp

matvec.o: src/param/matvec.cpp 
	$(CXX) $(CXXFLAGS) -c src/param/matvec.cpp 

param.o: src/param/param.cpp
	$(CXX) $(CXXFLAGS) -c src/param/param.cpp

mxl_gaussblockdiag.o: src/model/mxl_gaussblockdiag.cpp
	$(CXX) $(CXXFLAGS) -c src/model/mxl_gaussblockdiag.cpp

readinput.o: src/utility/readinput.cpp 
	$(CXX) $(CXXFLAGS) -c src/utility/readinput.cpp 

rnggenerator.o: src/utility/rnggenerator.cpp
	$(CXX) $(CXXFLAGS) -c src/utility/rnggenerator.cpp




