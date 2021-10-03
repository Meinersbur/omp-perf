cmake_minimum_required(VERSION 3.14)
project(gamessminimini CXX)

## Configuration ###############################################################

set(GAMESS_REPETITIONS "20" CACHE STRING "Number of test repetitions")


## Dependencies ################################################################

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)


include(FetchContent)
set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_Declare(googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
)
FetchContent_MakeAvailable(googlebenchmark)


#############################################################################

message("CUDA_ROOT: ${CUDA_ROOT}")
message("CUDAToolkit_LIBRARY_DIR: ${CUDAToolkit_LIBRARY_DIR}")
message("CUDAToolkit_LIBRARY_ROOT: ${CUDAToolkit_LIBRARY_ROOT}")

set(OFFLOAD_OPTIONS ${OpenMP_CXX_FLAGS} -fno-exceptions -march=native --cuda-path=${CUDA_ROOT} -fopenmp=libomp -fopenmp-version=51  -fopenmp-targets=nvptx64
"--libomptarget-nvptx-bc-path=/home/meinersbur/build/llvm-project/release/projects/openmp/libomptarget" -Wall
"-Wl,-rpath,${CUDAToolkit_LIBRARY_DIR}"
)

set(OFFLOAD_CC_OPTIONS ${OFFLOAD_OPTIONS}
)

set(OFFLOAD_LD_OPTIONS ${OFFLOAD_OPTIONS}
"-Wl,-rpath,/home/meinersbur/build/llvm-project/release/lib" 
"-Wl,-rpath,/home/meinersbur/build/llvm-project/release/projects/openmp/libomptarget" "-Wl,-rpath,/home/meinersbur/build/llvm-project/release/lib"  
)


## Build #######################################################################

add_subdirectory(atax-host-unroll)
add_subdirectory(heat-3d)
add_subdirectory(su3)
add_subdirectory(chunksize)

## Run #########################################################################

set(GBENCH_ARGS)
list(APPEND GBENCH_ARGS --benchmark_min_time=10)
#if (GAMESS_REPETITIONS GREATER 1)
#    list(APPEND GBENCH_ARGS --benchmark_repetitions=${GAMESS_REPETITIONS} --benchmark_display_aggregates_only=true --benchmark_min_time=3)
#endif ()



add_custom_target(run
    DEPENDS heat-3d atax-host-unroll
    COMMAND $<TARGET_FILE:atax-host-unroll> ${GBENCH_ARGS}
    #COMMAND $<TARGET_FILE:heat-3d> ${GBENCH_ARGS}
    #COMMAND $<TARGET_FILE:su3> ${GBENCH_ARGS}
    #COMMAND $<TARGET_FILE:chunksize> ${GBENCH_ARGS}
    USES_TERMINAL
    COMMENT "Run benchmark ..."
    VERBATIM
)

