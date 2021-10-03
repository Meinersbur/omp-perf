add_executable(atax-host-unroll atax-host-unroll.cpp)
target_link_libraries(atax-host-unroll PRIVATE benchmark::benchmark_main)


target_compile_options(atax-host-unroll PRIVATE ${OFFLOAD_CC_OPTIONS} 
    -save-temps
    -fno-unroll-loops -Rpass=loop-unroll -Rpass-missed=loop-unroll -Rpass-analysis=loop-unroll
    -fno-vectorize  # -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize)
)
target_link_options(atax-host-unroll PRIVATE ${OFFLOAD_LD_OPTIONS})


