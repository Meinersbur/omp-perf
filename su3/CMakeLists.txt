add_executable(su3 su3.cpp)
target_link_libraries(su3 PRIVATE benchmark::benchmark_main)


target_compile_options(su3 PRIVATE ${OFFLOAD_CC_OPTIONS}
    -fno-unroll-loops -Rpass=loop-unroll    -Rpass-missed=loop-unroll    -Rpass-analysis=loop-unroll
    -fno-vectorize
   #   -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
   "SHELL:-mllvm -interleave-loops=0"
     #"SHELL:-mllvm -debug-only=loop-unroll"
)
target_link_options(su3 PRIVATE ${OFFLOAD_LD_OPTIONS})


