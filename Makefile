# HEADERS = utils.h
CXX = g++
CXXFLAGS = -std=c++11 -O3

main: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o main

# %:%.cpp *.h
# 	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	rm main