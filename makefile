CC = clang++-17
LFLAGS = -std=c++14 -Wall -Ofast -march=native

all: gmix test dictionary-prep enwik9-prep

debug: LFLAGS = -std=c++14 -Wall -ggdb
debug: gmix test dictionary-prep enwik9-prep

SRC_FILES := $(filter-out $(wildcard src/runner/*.cpp) $(wildcard src/preprocess/*.cpp), $(wildcard src/*.cpp) $(wildcard src/*/*.cpp) $(wildcard src/*/*/*.cpp)) src/runner/runner-utils.cpp
HDR_FILES := $(filter-out $(wildcard src/preprocess/*.h) $(wildcard src/preprocess/*/*.h), $(wildcard src/*.h) $(wildcard src/*/*.h) $(wildcard src/*/*/*.h))

gmix: $(SRC_FILES) $(HDR_FILES) src/runner/runner.cpp
	$(CC) $(LFLAGS) $(SRC_FILES) src/runner/runner.cpp -o gmix

test: $(SRC_FILES) $(HDR_FILES) src/runner/tester.cpp
	$(CC) $(LFLAGS) $(SRC_FILES) src/runner/tester.cpp -o test

dictionary-prep: src/preprocess/dictionary.cpp src/preprocess/dictionary.h src/runner/dictionary-prep.cpp
	$(CC) $(LFLAGS) src/preprocess/dictionary.cpp src/runner/dictionary-prep.cpp -o dictionary-prep

enwik9-prep: src/preprocess/enwik9/article_reorder.h src/runner/enwik9-prep.cpp src/preprocess/enwik9/misc.h src/preprocess/enwik9/phda9_preprocess.h
	$(CC) $(LFLAGS) src/runner/enwik9-prep.cpp -o enwik9-prep

clean:
	rm -f gmix test dictionary-prep enwik9-prep
