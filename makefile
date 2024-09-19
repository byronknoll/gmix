CC = clang++-17
LFLAGS = -std=c++14 -Wall

all: LFLAGS += -Ofast -march=native
all: gmix

debug: LFLAGS += -ggdb
debug: gmix test

test: LFLAGS += -Ofast -march=native
test: tester

GMIX_SRC_FILES := $(filter-out src/tester.cpp, $(wildcard src/*.cpp) $(wildcard src/*/*.cpp))
TEST_SRC_FILES := $(filter-out src/runner.cpp, $(wildcard src/*.cpp) $(wildcard src/*/*.cpp))
HDR_FILES := $(wildcard src/*.h) $(wildcard src/*/*.cpp)

gmix: $(GMIX_SRC_FILES) $(HDR_FILES)
	$(CC) $(LFLAGS) $(GMIX_SRC_FILES) -o gmix

tester: $(TEST_SRC_FILES) $(HDR_FILES)
	$(CC) $(LFLAGS) $(TEST_SRC_FILES) -o tester

clean:
	rm -f gmix tester
