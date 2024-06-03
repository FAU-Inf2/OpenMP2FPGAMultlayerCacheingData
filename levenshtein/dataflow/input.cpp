#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#define STATIC_SIZE 512
#define STATIC_SIZE_SQ (524288)

char *gen_rand_string(int len) {
  char *res = new char[STATIC_SIZE];
  for (int i = 0; i < len; i++) {
    res[i] = 'A' + (rand()) % 6;
  }
  res[len] = '\0';
  return res;
}

int lin(int x, int y, int n) {
#pragma hls inline
  return x * (n + 1) + y;
}

template <typename A, typename B, typename C>
int leven_dst(A &str_a, B &str_b, C &grd, int n) {
#pragma hls inline off
  for (int i = 0; i < n + 1; i++) {
    grd[lin(i, 0, n)] = i;
  }
  for (int i = 0; i < n + 1; i++) {
    grd[lin(0, i, n)] = i;
  }

  for (int i = 1; i < n + 1; i++) {
    for (int j = 1; j < n + 1; j++) {
      int opt_t =
          grd[lin(i - 1, j - 1, n)] + ((str_a[i - 1] == str_b[j - 1]) ? 0 : 1);
      int opt_i = grd[lin(i, j - 1, n)] + 1;
      int opt_d = grd[lin(i - 1, j, n)] + 1;
      int in = (opt_i < opt_d) ? opt_i : opt_d;
      grd[lin(i, j, n)] = in < opt_t ? in : opt_t;
    }
  }
  return grd[lin(n, n, n)];
}

int main(int argc, char **argv) {
  int n = 500;
  if (argc > 1)
    n = atoi(argv[1]);
  else
    abort();
  srand(n);
  char *str_a = gen_rand_string(n);
  char *str_b = gen_rand_string(n);
  int *grd = new int[STATIC_SIZE_SQ];
  int dst_fpga, dst_cpu;

  dst_cpu = leven_dst(str_a, str_b, grd, n);

#pragma omp target map(to                                                      \
                       : str_a [0:STATIC_SIZE], str_b [0:STATIC_SIZE], n)      \
    map(tofrom                                                                 \
        : grd [0:STATIC_SIZE_SQ], dst_fpga, grd) \ 
	orkaTranslate(str_a_cacheConfig                                        \
                      : "dataflow:true:false:64:1:8:0:0")                      \
        orkaTranslate(str_b_cacheConfig                                        \
                      : "dataflow:true:false:64:1:8:0:0")                      \
            orkaTranslate(grd_cacheConfig                                      \
                          : "dataflow:true:true:512:1:8:0:0")
  { dst_fpga = leven_dst(str_a, str_b, grd, n); }

  if (dst_fpga != dst_cpu) {
    fprintf(stderr, "Mismatch!\b");
    printf("str_a: %s\n", str_a);
    printf("str_b: %s\n", str_b);
    printf("dst_cpu: %d dst_fpga: %d\n", dst_cpu, dst_fpga);
    exit(EXIT_FAILURE);
  }
  printf("dst: %d\n", dst_cpu);
}
