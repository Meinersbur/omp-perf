
add_executable(atax-host-unroll atax-host-unroll.cpp)
target_link_libraries(atax-host-unroll PRIVATE benchmark::benchmark_main)
target_compile_options(atax-host-unroll PUBLIC -fno-exceptions)

# Assume everything else is based on clang (e.g. Cray)
set(OFFLOAD_OPTIONS ${OpenMP_CXX_FLAGS} --cuda-path=/opt/cuda -fopenmp=libomp -fopenmp-version=51 -fopenmp-targets=nvptx64 "--libomptarget-nvptx-bc-path=/home/meinersbur/build/llvm-project/release/projects/openmp/libomptarget" -Wall -save-temps
    -fno-unroll-loops -Rpass=loop-unroll -Rpass-missed=loop-unroll -Rpass-analysis=loop-unroll
    -fno-vectorize )# -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize)

target_compile_options(atax-host-unroll PRIVATE ${OFFLOAD_OPTIONS})
target_link_options(atax-host-unroll PRIVATE ${OFFLOAD_OPTIONS} "-Wl,-rpath,/home/meinersbur/build/llvm-project/release/projects/openmp/libomptarget" "-Wl,-rpath,/home/meinersbur/build/llvm-project/release/lib"  "-Wl,-rpath,/opt/cuda/lib64")


