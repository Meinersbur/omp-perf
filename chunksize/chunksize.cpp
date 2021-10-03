#include <benchmark/benchmark.h>
#include <string.h>

#define N 128*1024



/* Inter-array padding, for use with . By default, none is used. */
# ifndef POLYBENCH_INTER_ARRAY_PADDING_FACTOR
/* default: */
#  define POLYBENCH_INTER_ARRAY_PADDING_FACTOR 0
#  undef POLYBENCH_ENABLE_INTARRAY_PAD
# else
#  define POLYBENCH_ENABLE_INTARRAY_PAD
# endif


static size_t polybench_inter_array_padding_sz = 0;



static
void*
xmalloc(size_t alloc_sz)
{
  void* ret = NULL;
  /* By default, post-pad the arrays. Safe behavior, but likely useless. */
  polybench_inter_array_padding_sz += POLYBENCH_INTER_ARRAY_PADDING_FACTOR;
  size_t padded_sz = alloc_sz + polybench_inter_array_padding_sz;
  int err = posix_memalign (&ret, 4096, padded_sz);
  if (! ret || err)
    {
      fprintf (stderr, "[PolyBench] posix_memalign: cannot allocate memory");
      exit (1);
    }
  /* Safeguard: this is invoked only if polybench.c has been compiled
     with inter-array padding support from polybench.h. If so, move
     the starting address of the allocation and return it to the
     user. The original pointer is registered in an allocation table
     internal to polybench.c. Data must then be freed using
     polybench_free_data, which will inspect the allocation table to
     free the original pointer.*/
#ifdef POLYBENCH_ENABLE_INTARRAY_PAD
  /* This moves the 'ret' pointer by (padded_sz - alloc_sz) positions, and
  registers it in the lookup table for future free using
  polybench_free_data. */
  ret = register_padded_pointer(ret, alloc_sz, padded_sz);
#endif

  return ret;
}






struct su3_vector {
  float v[3];
};
struct su3_matrix {
  su3_vector v[3];
};







static
void kernel_chunksize_threads_nochunk(int n,
    su3_vector C[N],
    su3_vector A[N],
    su3_matrix B[N])
{

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k)
           C[i].v[j] += A[i].v[k] * B[i].v[k].v[j];
  }
}


static void benchmark_chunksize_threads_nochunk(benchmark::State& state) {
  int n = N;

  su3_vector *C = (su3_vector*)xmalloc(n * sizeof(*C));
  su3_vector *A = (su3_vector*)xmalloc(n * sizeof(*A));
  su3_matrix *B = (su3_matrix*)xmalloc(n * sizeof(*B));

  memset(C, '\0', n * sizeof(*C));
  memset(A, '\0', n * sizeof(*A));
  memset(B, '\0', n * sizeof(*B));

  for (auto _ : state) {
    kernel_chunksize_threads_nochunk(n, C, A, B);
    benchmark::ClobberMemory();
  }

  free(C);
  free(A);
  free(B);
}
BENCHMARK(benchmark_chunksize_threads_nochunk)->Unit(benchmark::kMicrosecond)->MeasureProcessCPUTime()->UseRealTime();


static
void kernel_chunksize_threads(int n,
    su3_vector C[N],
    su3_vector A[N],
    su3_matrix B[N],int chunksize)
{

  #pragma omp parallel for schedule(static,chunksize)
  for (int i = 0; i < n; ++i) {
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k)
           C[i].v[j] += A[i].v[k] * B[i].v[k].v[j];
  }
}


static void benchmark_chunksize_threads(benchmark::State& state) {
  int n = N;

  su3_vector *C = (su3_vector*)xmalloc(n * sizeof(*C));
  su3_vector *A = (su3_vector*)xmalloc(n * sizeof(*A));
  su3_matrix *B = (su3_matrix*)xmalloc(n * sizeof(*B));

  memset(C, '\0', n * sizeof(*C));
  memset(A, '\0', n * sizeof(*A));
  memset(B, '\0', n * sizeof(*B));

  for (auto _ : state) {
    kernel_chunksize_threads(n, C, A, B, state.range(0));
    benchmark::ClobberMemory();
  }

  free(C);
  free(A);
  free(B);
}


BENCHMARK(benchmark_chunksize_threads)->Unit(benchmark::kMicrosecond)->MeasureProcessCPUTime()->UseRealTime()
  ->Arg(1)
  ->Arg(2)
  ->Arg(3)
  ->Arg(4)
  ->Arg(8)
  ->Arg(16)
  ->Arg(32)
  ->Arg(64)
  ->Arg(128)
  ->Arg(256)
  ->Arg(512)
  ;





#if 0
static
void kernel_su3_target(int n,
    su3_vector C[N],
    su3_vector A[N],
    su3_matrix B[N])
{
  #pragma omp target teams distribute parallel for
  for (int i = 0; i < n; ++i) {
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k)
           C[i].v[j] += A[i].v[k] * B[i].v[k].v[j];
  }
  #pragma omp taskwait
}
static void bench_su3_target(benchmark::State& state) {
  int n = N;

  su3_vector *C = (su3_vector*)xmalloc(n * sizeof(*C));
  su3_vector *A = (su3_vector*)xmalloc(n * sizeof(*A));
  su3_matrix *B = (su3_matrix*)xmalloc(n * sizeof(*B));

  memset(C, '\0', n * sizeof(*C));
  memset(A, '\0', n * sizeof(*A));
  memset(B, '\0', n * sizeof(*B));


  #pragma omp target data map(to:A[0:N]) map(to:B[0:N]) map(from:C[0:N])
  for (auto _ : state) {
    kernel_su3_target(n, C, A, B);
    benchmark::ClobberMemory();
  }

  free(C);
  free(A);
  free(B);
}
BENCHMARK(bench_su3_target)->Unit(benchmark::kMicrosecond)->UseRealTime();
#endif

