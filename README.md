# machine_learning_library

A small, from-scratch C++ matrix and ML-ops sandbox. work-in-progress.

## Build

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pedantic -Iinclude -c src/utils.cpp src/matrix.cpp
ar rcs libml.a utils.o matrix.o
```

## Notes

Experimental and Evolving.