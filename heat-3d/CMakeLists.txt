add_executable(heat-3d heat-3d.cpp)
target_link_libraries(heat-3d PRIVATE benchmark::benchmark_main)


target_compile_options(heat-3d PRIVATE  PRIVATE ${OFFLOAD_CC_OPTIONS} 
   # -fno-unroll-loops -Rpass=loop-unroll    -Rpass-missed=loop-unroll    -Rpass-analysis=loop-unroll
   # -fno-vectorize    -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
)
target_link_options(heat-3d PRIVATE ${OFFLOAD_LD_OPTIONS})


