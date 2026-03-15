# =============================================================
#  S4D Galaxy Classifier — Makefile
#  Milestone 2: C/C++ Implementation & Numerical Validation
# =============================================================

CC      = gcc
CFLAGS  = -Wall -std=c11 -lm
SRC     = nn.c

# Default: build all targets with -O2
all: main validate benchmark

main: main.c $(SRC)
	$(CC) $(CFLAGS) -O2 -o main main.c $(SRC)

validate: validate.c $(SRC)
	$(CC) $(CFLAGS) -O2 -o validate validate.c $(SRC)

benchmark: benchmark.c $(SRC)
	$(CC) $(CFLAGS) -O2 -o benchmark benchmark.c $(SRC)

# ── Optimization level targets ─────────────────────────────────
O0: main.c $(SRC)
	$(CC) $(CFLAGS) -O0 -o main_O0 main.c $(SRC)

O1: main.c $(SRC)
	$(CC) $(CFLAGS) -O1 -o main_O1 main.c $(SRC)

O3: main.c $(SRC)
	$(CC) $(CFLAGS) -O3 -o main_O3 main.c $(SRC)

Ofast: main.c $(SRC)
	$(CC) $(CFLAGS) -Ofast -o main_Ofast main.c $(SRC)

benchmark_O0: benchmark.c $(SRC)
	$(CC) $(CFLAGS) -O0 -o benchmark_O0 benchmark.c $(SRC)

benchmark_O1: benchmark.c $(SRC)
	$(CC) $(CFLAGS) -O1 -o benchmark_O1 benchmark.c $(SRC)

benchmark_O3: benchmark.c $(SRC)
	$(CC) $(CFLAGS) -O3 -o benchmark_O3 benchmark.c $(SRC)

benchmark_Ofast: benchmark.c $(SRC)
	$(CC) $(CFLAGS) -Ofast -o benchmark_Ofast benchmark.c $(SRC)

# ── Assembly dumps ──────────────────────────────────────────────
asm:
	$(CC) -S -O0 $(SRC) -o nn_O0.s
	$(CC) -S -O2 $(SRC) -o nn_O2.s
	$(CC) -S -O3 $(SRC) -o nn_O3.s

# ── Memory / address sanitizer build ───────────────────────────
sanitize: main.c $(SRC)
	$(CC) $(CFLAGS) -O1 -fsanitize=address -o main_asan main.c $(SRC)

# ── Clean ───────────────────────────────────────────────────────
clean:
	rm -f main validate benchmark \
	      main_O0 main_O1 main_O3 main_Ofast \
	      benchmark_O0 benchmark_O1 benchmark_O3 benchmark_Ofast \
	      main_asan \
	      nn_O0.s nn_O2.s nn_O3.s \
	      test_suite

.PHONY: all clean asm sanitize O0 O1 O3 Ofast \
        benchmark_O0 benchmark_O1 benchmark_O3 benchmark_Ofast
