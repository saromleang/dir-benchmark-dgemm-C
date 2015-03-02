# dir-benchmark-dgemm-C
Benchmark K20 dgemm.c

Copy the appropriate Makefile.(bolt|cyence) to Makefile

**make [options] BLAS=(atlas|openBLAS|mkl)**

options:

**all: default**<br>
  Compiler optimization level: -O3<br>
  Builds benchmark with no pinned host memory, with no niave cpu dgemm calculation, with no verification of results.

**debug:**<br>
  Compiler optimization level: not set<br>
  Builds the default with verbose printing.

**cpu-dgemm:**<br>
  Compiler optimization level: -O3<br>
  A niave cpu dgemm implementation is performed.

**verify:**<br>
  Compiler optimization level: -O3<br>
  Accelerator blas calculated DGEMM product and CPU blas calcualated DGEMM product is compared elememt by element a niave cpu calculated DGEMM product.  The accumulated unsigned error is printed out.
  
**debug-verify:**<br>
  Compiler optimization level: not set<br>
  Adds verbose printing on top of a verify build.

**pinned:**<br>
  Compiler optimization level: -O3<br>
  Builds benchmark with pinned host memory.
  
**debug-pinned:**<br>
  Compiler optimization level: not set<br>
  Adds verbose printing on top of a pinned build.
  
**cpu-dgemm-pinned:**<br>
  Compiler optimization level: -O3<br>
  Builds benchmark with pinned host memory and a niave cpu dgemm implementation is performed.
  
**verify-pinned:**<br>
  Compiler optimization level: -O3<br>
  Builds benchmark with pinned host memory.  Accelerator blas calculated DGEMM product and CPU blas calcualated DGEMM product is compared elememt by element a niave cpu calculated DGEMM product.  The accumulated unsigned error is printed out.

**debug-verify-pinned:**<br>
  Compiler optimization level: not set<br>
  Adds verbose printing on top of a verify-pinned build.
  
