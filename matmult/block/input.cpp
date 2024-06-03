
#include <cstdlib>
#include <iostream>

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
                       : a [0:N * N], b [0:N * N], N) map(tofrom               \
                                                          : c [0:N * N])
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
  int *a = new int[N * N];
  int *b = new int[N * N];
  int *c = new int[N * N];
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
