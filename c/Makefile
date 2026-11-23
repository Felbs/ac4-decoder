# SPDX-License-Identifier: Apache-2.0
# libac4dec + ac4dec-c. C99, libm only.
#
#   make            # libac4dec.a, libac4dec.so, ac4dec-c
#   make test       # unit + robustness tests (no captures needed)
#   make clean
#
# Use at most `make -j2` on small machines.
#
# IEEE-pinned like ac4/build_kernels.py: no -ffast-math, no FMA contraction,
# so the arithmetic is what the source says.

CC      ?= cc
CFLAGS  ?= -O2 -g
CFLAGS  += -std=c99 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Wno-unused-parameter \
           -ffp-contract=off -fPIC -Iinclude -Isrc
LDLIBS  += -lm

SRC = src/ac4_tables.c src/toc.c src/asf.c src/spectra.c src/filterbank.c \
      src/resample.c src/rates.c src/loudness.c src/aspx.c src/aspx_render.c src/acpl.c src/decoder.c src/mp4.c src/input.c
OBJ = $(SRC:.c=.o)
HDR = include/ac4dec.h include/ac4dec_mp4.h include/ac4dec_input.h src/ac4_internal.h src/ac4_tables.h

all: libac4dec.a libac4dec.so ac4dec-c

$(OBJ): $(HDR)

libac4dec.a: $(OBJ)
	$(AR) rcs $@ $(OBJ)

libac4dec.so: $(OBJ)
	$(CC) -shared -o $@ $(OBJ) $(LDLIBS)

ac4dec-c: cli/ac4dec_main.c libac4dec.a $(HDR)
	$(CC) $(CFLAGS) -o $@ cli/ac4dec_main.c libac4dec.a $(LDLIBS)

tests/unit_test: tests/unit_test.c libac4dec.a $(HDR)
	$(CC) $(CFLAGS) -o $@ tests/unit_test.c libac4dec.a $(LDLIBS)

tests/fuzz_test: tests/fuzz_test.c libac4dec.a $(HDR)
	$(CC) $(CFLAGS) -o $@ tests/fuzz_test.c libac4dec.a $(LDLIBS)

test: tests/unit_test tests/fuzz_test
	./tests/unit_test
	./tests/fuzz_test

# the damaged-input gate under AddressSanitizer + UBSan (slow)
SAN = -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all
tests/fuzz_test_asan: tests/fuzz_test.c $(SRC) $(HDR)
	$(CC) $(filter-out -O2,$(CFLAGS)) $(SAN) -o $@ tests/fuzz_test.c $(SRC) $(LDLIBS)

asan: tests/fuzz_test_asan
	./tests/fuzz_test_asan

# regenerate the tables from ac4/tables/*.json (needs python3)
tables:
	python3 tools/gen_tables.py > src/ac4_tables.c

clean:
	rm -f $(OBJ) libac4dec.a libac4dec.so ac4dec-c tests/unit_test tests/fuzz_test \
	      tests/fuzz_test_asan

.PHONY: all test asan clean tables
