
#include <cstdlib>
#include <iostream>

#define STATIC_SIZE 512
#define STATIC_SIZESQ (STATIC_SIZE * STATIC_SIZE)

template <typename R, typename W> void matmult(R &A, R &B, W &C, size_t N) {
#pragma hls inline off
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      int acc = 0;
      for (int k = 0; k < N; k++) {
        acc += A[i * N + k] * B[k * N + j];
        if (k == (N - 1))
          C[i * N + j] = acc;
      }
    }
  }
}

void matmult_hw(int *a, int *b, int *c, size_t N) {
#pragma omp target map(to                                                      \
                       : a [0:STATIC_SIZESQ], b [0:STATIC_SIZESQ], N)          \
    map(tofrom                                                                 \
        : c [0:STATIC_SIZESQ])                                                 \
        orkaTranslate(a_cacheConfig                                            \
                      : "dataflow:true:false:512:1:8:0:0")                     \
            orkaTranslate(b_cacheConfig                                        \
                          : "dataflow:true:false:512:1:8:0:0")
  { matmult(a, b, c, N); }
}

int main(int argc, char **argv) {
  size_t N = -1;
  if (argc > 1) {
    N = atoi(argv[1]);
  } else {
    std::cerr << "usage: " << argv[0] << " <size>" << std::endl;
    abort();
  }
  srand(N);
  int *a = new int[STATIC_SIZESQ];
  int *b = new int[STATIC_SIZESQ];
  int *c = new int[STATIC_SIZESQ];
  int *c_ref = new int[N * N];
  for (int i = 0; i < N * N; i++) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
  }
  std::cout << "computing ref" << std::endl;
  matmult(a, b, c_ref, N);
  std::cout << "computing fpga" << std::endl;
  matmult_hw(a, b, c, N);
  std::cout << "comparing" << std::endl;
  bool err = false;
  for (int i = 0; i < N * N; i++) {
    if (c[i] != c_ref[i])
      err = true;
  }
  if (err) {
    std::cout << "mismatch" << std::endl;
    for (int i = 0; i < N * N; i++) {
      std::cout << c[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < N * N; i++) {
      std::cout << c_ref[i] << " ";
    }
    std::cout << std::endl;
  } else {
    std::cout << "computation successful!" << std::endl;
    std::cout << "size: " << N << std::endl;
  }
}
