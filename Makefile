libs_for_gpu=-L/share/apps/cuda/lib64 -lcublas -lcurand
includes_for_gpu=-I/share/apps/cuda/include
opt= -O3

ifndef BLAS
$(error BLAS is not set. Define BLAS=(atlas|openBLAS|mkl))
endif

ifeq ($(BLAS),atlas)
  opt= -O3
	libs_for_blas=-L/share/apps/share/blas/$(BLAS)/lib -lptcblas -latlas -lpthread
	includes_for_blas=-I/share/apps/share/blas/$(BLAS)/include
endif

ifeq ($(BLAS),openBLAS)
  opt= -O3
	libs_for_blas=-L/share/apps/share/blas/$(BLAS)/lib -lopenblas -lpthread
	includes_for_blas=-I/share/apps/share/blas/$(BLAS)/include
endif

ifeq ($(BLAS),mkl)
  opt= -O3 -DMKL
	libs_for_blas=${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a -ldl -lpthread -lm
	includes_for_blas= -DMKL_ILP64 -Xcompiler -fopenmp -m64 -I${MKLROOT}/include
endif

all:
	nvcc $(opt) $(includes_for_gpu) $(includes_for_blas) dgemm.c $(libs_for_gpu) $(libs_for_blas) -o dgemm.x

debug:
	nvcc -DDEBUG $(includes_for_gpu) $(includes_for_blas) dgemm.c $(libs_for_gpu) $(libs_for_blas) -o dgemm.x

cpu-dgemm:
	nvcc $(opt) -DCPU_DGEMM $(includes_for_gpu) $(includes_for_blas) dgemm.c $(libs_for_gpu) $(libs_for_blas) -o dgemm.x

verify:
	nvcc $(opt) -DCPU_DGEMM $(includes_for_gpu) $(includes_for_blas) dgemm.c $(libs_for_gpu) $(libs_for_blas) -o dgemm.x

debug-verify:
	nvcc -DDEBUG -DCPU_DGEMM -DVERIFY $(includes_for_gpu) $(includes_for_blas) dgemm.c $(libs_for_gpu) $(libs_for_blas) -o dgemm.x

pinned:
	nvcc $(opt) -DPINNED $(includes_for_gpu) $(includes_for_blas) dgemm.c $(libs_for_gpu) $(libs_for_blas) -o dgemm.x

debug-pinned:
	nvcc -DPINNED -DDEBUG $(includes_for_gpu) $(includes_for_blas) dgemm.c $(libs_for_gpu) $(libs_for_blas) -o dgemm.x

cpu-dgemm-pinned:
	nvcc $(opt) -DPINNED -DCPU_DGEMM $(includes_for_gpu) $(includes_for_blas) dgemm.c $(libs_for_gpu) $(libs_for_blas) -o dgemm.x

verify-pinned:
	nvcc $(opt) -DPINNED -DCPU_DGEMM -DVERIFY $(includes_for_gpu) $(includes_for_blas) dgemm.c $(libs_for_gpu) $(libs_for_blas) -o dgemm.x

debug-verify-pinned:
	nvcc -DPINNED -DDEBUG -DCPU_DGEMM -DVERIFY $(includes_for_gpu) $(includes_for_blas) dgemm.c $(libs_for_gpu) $(libs_for_blas) -o dgemm.x

