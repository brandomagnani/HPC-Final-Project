# makefile 

# mainExec : main.o 
# 	g++ -std=c++11 -o mainExec main.o 

CXX = g++
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp


TARGETS = $(basename $(wildcard *.cpp))

all : $(TARGETS)

%:%.cpp *.h
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

main:main.cpp #MMult.h sgd.h
	g++ -std=c++11 -fopenmp -O3 main.cpp -o main

main_gd.o:main_gd.cpp
	g++ -std=c++11 -fopenmp -O3 main_gd.cpp -o main_gd

clean:
	rm -f main *.out output/*.csv
