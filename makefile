CC = clang++-17
LFLAGS = -std=c++14 -Wall -Ofast -march=native

all: gmix test prep enwik9-preproc

debug: LFLAGS = -std=c++14 -Wall -ggdb
debug: gmix test prep

SRC_FILES := $(filter-out $(wildcard src/runner/*.cpp) $(wildcard src/preprocess/*.cpp), $(wildcard src/*.cpp) $(wildcard src/*/*.cpp) $(wildcard src/*/*/*.cpp)) src/runner/runner-utils.cpp
HDR_FILES := $(filter-out $(wildcard src/preprocess/*.h) $(wildcard src/preprocess/*/*.h), $(wildcard src/*.h) $(wildcard src/*/*.h) $(wildcard src/*/*/*.h))

gmix: $(SRC_FILES) $(HDR_FILES) src/runner/runner.cpp
	$(CC) $(LFLAGS) $(SRC_FILES) src/runner/runner.cpp -o gmix

test: $(SRC_FILES) $(HDR_FILES) src/runner/tester.cpp
	$(CC) $(LFLAGS) $(SRC_FILES) src/runner/tester.cpp -o test

prep: src/preprocess/dictionary.cpp src/preprocess/dictionary.h src/runner/prep.cpp
	$(CC) $(LFLAGS) src/preprocess/dictionary.cpp src/runner/prep.cpp -o prep

enwik9-preproc: src/preprocess/enwik9/article_reorder.h src/runner/enwik9-preproc.cpp src/preprocess/enwik9/misc.h src/preprocess/enwik9/phda9_preprocess.h
	$(CC) $(LFLAGS) src/runner/enwik9-preproc.cpp -o enwik9-preproc

clean:
	rm -f gmix test prep enwik9-preproc
