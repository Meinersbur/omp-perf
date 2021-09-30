
add_executable(heat-3d heat-3d.cpp)
target_link_libraries(heat-3d PRIVATE benchmark::benchmark_main)
target_compile_options(heat-3d PUBLIC -fno-exceptions)

set(OFFLOAD_OPTIONS ${OpenMP_CXX_FLAGS} --cuda-path=/opt/cuda -fopenmp -fopenmp-version=51 -fopenmp-targets=nvptx64 "--libomptarget-nvptx-bc-path=/home/meinersbur/build/llvm-project/release/projects/openmp/libomptarget" -Wall -save-temps -march=native
   # -fno-unroll-loops -Rpass=loop-unroll    -Rpass-missed=loop-unroll    -Rpass-analysis=loop-unroll
   # -fno-vectorize    -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
    )

target_compile_options(heat-3d PRIVATE ${OFFLOAD_OPTIONS})
target_link_options(heat-3d PRIVATE ${OFFLOAD_OPTIONS} "-Wl,-rpath,/home/meinersbur/build/llvm-project/release/projects/openmp/libomptarget" "-Wl,-rpath,/home/meinersbur/build/llvm-project/release/lib"  "-Wl,-rpath,/opt/cuda/lib64")

