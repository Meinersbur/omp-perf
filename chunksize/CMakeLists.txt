add_executable(chunksize chunksize.cpp)
target_link_libraries(chunksize PRIVATE benchmark::benchmark_main)


target_compile_options(chunksize PRIVATE ${OFFLOAD_CC_OPTIONS}
-Rpass=loop-unroll    -Rpass-missed=loop-unroll    -Rpass-analysis=loop-unroll
   # -fno-unroll-loops 
   # -fno-vectorize
   #   -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
  # "SHELL:-mllvm -interleave-loops=0"
     #"SHELL:-mllvm -debug-only=loop-unroll"
)
target_link_options(chunksize PRIVATE ${OFFLOAD_LD_OPTIONS})


