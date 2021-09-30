cmake_minimum_required(VERSION 3.14)
project(gamessminimini CXX)

## Configuration ###############################################################

set(GAMESS_REPETITIONS "20" CACHE STRING "Number of test repetitions")


## Dependencies ################################################################

find_package(Threads REQUIRED)
if (OMP_VERSION)
    find_package(OpenMP REQUIRED)
endif ()

include(FetchContent)
set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_Declare(googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
)
FetchContent_MakeAvailable(googlebenchmark)


## Build #######################################################################

add_subdirectory(atax-host-unroll)
add_subdirectory(heat-3d)


## Run #########################################################################

set(GBENCH_ARGS)
#if (GAMESS_REPETITIONS GREATER 1)
#    list(APPEND GBENCH_ARGS --benchmark_repetitions=${GAMESS_REPETITIONS} --benchmark_display_aggregates_only=true)
#endif ()



add_custom_target(run
    DEPENDS heat-3d atax-host-unroll
    #COMMAND $<TARGET_FILE:atax-host-unroll> ${GBENCH_ARGS}
    COMMAND $<TARGET_FILE:heat-3d> ${GBENCH_ARGS}
    USES_TERMINAL
    COMMENT "Run benchmark ..."
    VERBATIM
)
