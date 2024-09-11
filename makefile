CC = clang++-17
LFLAGS = -std=c++14 -Wall

all: LFLAGS += -Ofast -march=native
all: gmix

debug: LFLAGS += -ggdb
debug: gmix

SRC_FILES := $(wildcard src/*.cpp) $(wildcard src/*/*.cpp)
HDR_FILES := $(wildcard src/*.h) $(wildcard src/*/*.cpp)

gmix: $(SRC_FILES) $(HDR_FILES)
	$(CC) $(LFLAGS) $(SRC_FILES) -o gmix

clean:
	rm -f gmix
