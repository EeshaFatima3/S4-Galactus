# ============================================================
# Makefile  —  S4D Galaxy Classifier  |  Milestone 2
# ============================================================
# Builds the demo, benchmark, and test programs at various
# optimization levels for Task 3.3 Performance Benchmarking.
#
# Usage:
#   make demo           # Build demo program (-O2)
#   make benchmark      # Build benchmark program (-O2)
#   make all_opts       # Build benchmark at -O0, -O1, -O2, -O3, -Ofast
#   make assembly       # Generate assembly dumps (.s files)
#   make clean          # Remove all build artifacts
# ============================================================

CC      = gcc
CFLAGS  = -Wall -Wextra -std=c11
LDFLAGS = -lm

SRCS_NN    = nn.c
SRCS_MAIN  = main.c
SRCS_BENCH = benchmark.c

# --- Default target ---
all: demo benchmark

# --- 3.1.8 Demo Program ---
demo: $(SRCS_NN) $(SRCS_MAIN) nn.h
	$(CC) $(CFLAGS) -O2 -o demo $(SRCS_NN) $(SRCS_MAIN) $(LDFLAGS)

# --- 3.3 Benchmark (default -O2) ---
benchmark: $(SRCS_NN) $(SRCS_BENCH) nn.h
	$(CC) $(CFLAGS) -O2 -o benchmark $(SRCS_NN) $(SRCS_BENCH) $(LDFLAGS)

# --- 3.3 Compiler Optimization Comparison ---
all_opts: bench_O0 bench_O1 bench_O2 bench_O3 bench_Ofast

bench_O0: $(SRCS_NN) $(SRCS_BENCH) nn.h
	$(CC) $(CFLAGS) -O0 -o bench_O0 $(SRCS_NN) $(SRCS_BENCH) $(LDFLAGS)

bench_O1: $(SRCS_NN) $(SRCS_BENCH) nn.h
	$(CC) $(CFLAGS) -O1 -o bench_O1 $(SRCS_NN) $(SRCS_BENCH) $(LDFLAGS)

bench_O2: $(SRCS_NN) $(SRCS_BENCH) nn.h
	$(CC) $(CFLAGS) -O2 -o bench_O2 $(SRCS_NN) $(SRCS_BENCH) $(LDFLAGS)

bench_O3: $(SRCS_NN) $(SRCS_BENCH) nn.h
	$(CC) $(CFLAGS) -O3 -o bench_O3 $(SRCS_NN) $(SRCS_BENCH) $(LDFLAGS)

bench_Ofast: $(SRCS_NN) $(SRCS_BENCH) nn.h
	$(CC) $(CFLAGS) -Ofast -o bench_Ofast $(SRCS_NN) $(SRCS_BENCH) $(LDFLAGS)

# --- 3.3 Assembly Analysis ---
assembly: nn_O0.s nn_O2.s nn_O3.s

nn_O0.s: $(SRCS_NN) nn.h
	$(CC) $(CFLAGS) -S -O0 $(SRCS_NN) -o nn_O0.s

nn_O2.s: $(SRCS_NN) nn.h
	$(CC) $(CFLAGS) -S -O2 $(SRCS_NN) -o nn_O2.s

nn_O3.s: $(SRCS_NN) nn.h
	$(CC) $(CFLAGS) -S -O3 $(SRCS_NN) -o nn_O3.s

# --- Clean ---
clean:
	rm -f demo benchmark bench_O0 bench_O1 bench_O2 bench_O3 bench_Ofast
	rm -f *.s *.o *.exe
	rm -f benchmark_results*.csv

.PHONY: all all_opts assembly clean
