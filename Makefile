HEADERS = utils.h
CXX = g++
CXXFLAGS = -std=c++11 -O3

TARGETS = $(basename $(wildcard *.cpp))

all : $(TARGETS)

main:main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o main.out

# %:%.cpp *.h
# 	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	