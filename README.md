# dir-benchmark-dgemm-C
Benchmark K20 dgemm.c

Copy the appropriate Makefile.(bolt|cyence) to Makefile

make [options] BLAS=(atlas|openBLAS|mkl)

options:

all: default

  Compiler optimization level: -O3
  Builds benchmark with no pinned host memory, with no niave cpu dgemm calculation, with no verification of results.

debug:

  Compiler optimization level: not set
  Builds the default with verbose printing.

cpu-dgemm:

  Compiler optimization level: -O3
  A niave cpu dgemm implementation is performed.

verify:

  Compiler optimization level: -O3
  Accelerator blas calculated DGEMM product and CPU blas calcualated DGEMM product is compared elememt by element a niave cpu calculated DGEMM product.  The accumulated unsigned error is printed out.
  
debug-verify:

  Compiler optimization level: not set
  Adds verbose printing on top of a verify build.

pinned:

  Compiler optimization level: -O3
  Builds benchmark with pinned host memory.
  
debug-pinned:

  Compiler optimization level: not set
  Adds verbose printing on top of a pinned build.
  
cpu-dgemm-pinned:

  Compiler optimization level: -O3
  Builds benchmark with pinned host memory and a niave cpu dgemm implementation is performed.
  
verify-pinned:

  Compiler optimization level: -O3
  Builds benchmark with pinned host memory.  Accelerator blas calculated DGEMM product and CPU blas calcualated DGEMM product is compared elememt by element a niave cpu calculated DGEMM product.  The accumulated unsigned error is printed out.

debug-verify-pinned:

  Compiler optimization level: not set
  Adds verbose printing on top of a verify-pinned build.
  
