CC = clang++-17
LFLAGS = -std=c++14 -Wall -Ofast -march=native

all: gmix test prep

debug: LFLAGS = -std=c++14 -Wall -ggdb
debug: gmix test prep

SRC_FILES := $(filter-out $(wildcard src/runner/*.cpp), $(wildcard src/*.cpp) $(wildcard src/*/*.cpp)) src/runner/runner-utils.cpp
HDR_FILES := $(wildcard src/*.h) $(wildcard src/*/*.cpp) src/runner/runner-utils.h

gmix: $(SRC_FILES) $(HDR_FILES) src/runner/runner.cpp
	$(CC) $(LFLAGS) $(SRC_FILES) src/runner/runner.cpp -o gmix

test: $(SRC_FILES) $(HDR_FILES) src/runner/tester.cpp
	$(CC) $(LFLAGS) $(SRC_FILES) src/runner/tester.cpp -o test

prep: src/preprocess/dictionary.cpp src/preprocess/dictionary.h src/runner/prep.cpp
	$(CC) $(LFLAGS) src/preprocess/dictionary.cpp src/runner/prep.cpp -o prep

clean:
	rm -f gmix test prep
