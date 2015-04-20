# dir-benchmark-dgemm-C
Benchmark K20 dgemm.c

Copy the appropriate Makefile.(bolt|cyence) to Makefile

bolt:
make [options] BLAS=(atlas-sandybridge-gnu|atlas-haswell-gnu|atlas-dev-sandybridge-gnu|atlas-dev-haswell-gnu|openBLAS-sandybridge-gnu|openBLAS-sandybridge-intel|openBLAS-haswell-gnu|openBLAS-haswell-intel|mkl|mkl-with-cuda)

cyence:
make [options] BLAS=(atlas-sandybridge-gnu|atlas-dev-sandybridge-gnu|openBLAS-sandybridge-gnu|openBLAS-sandybridge-intel|mkl|mkl-with-cuda)

LOCAL(sarom):
make [options] BLAS=(atlas|openBLAS|mkl)

options:

all: default<br>
<br>Compiler optimization level: -O3
<br>Builds benchmark with no pinned host memory, with no niave cpu dgemm calculation, with no verification of results.

debug:<br>
<br>Compiler optimization level: not set
<br>Builds the default with verbose printing.

cpu-dgemm:<br>
<br>Compiler optimization level: -O3
<br>A niave cpu dgemm implementation is performed.

verify:<br>
<br>Compiler optimization level: -O3
<br>Accelerator blas calculated DGEMM product and CPU blas calcualated DGEMM product is compared elememt by element a niave cpu calculated DGEMM product.  The accumulated unsigned error is printed out.
  
debug-verify:<br>
<br>Compiler optimization level: not set
<br>Adds verbose printing on top of a verify build.

pinned:<br>
<br>Compiler optimization level: -O3
<br>Builds benchmark with pinned host memory.
  
debug-pinned:<br>
<br>Compiler optimization level: not set
<br>Adds verbose printing on top of a pinned build.
  
cpu-dgemm-pinned:<br>
<br>Compiler optimization level: -O3
<br>Builds benchmark with pinned host memory and a niave cpu dgemm implementation is performed.
  
verify-pinned:<br>
<br>Compiler optimization level: -O3
<br>Builds benchmark with pinned host memory.  Accelerator blas calculated DGEMM product and CPU blas calcualated DGEMM product is compared elememt by element a niave cpu calculated DGEMM product.  The accumulated unsigned error is printed out.

debug-verify-pinned:<br>
<br>Compiler optimization level: not set
<br>Adds verbose printing on top of a verify-pinned build.

no-gpu:<br>
<br>Compiler optimization level: -O3
<br>Pure CPU code. Random-number generation and DGEMM operations performed on the CPU.
