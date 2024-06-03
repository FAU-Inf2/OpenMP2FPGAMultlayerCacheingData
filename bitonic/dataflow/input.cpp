#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const size_t STATIC_SIZE = 65536;

void mov_dat(int *from, int *to, int n) {
  for (int i = 0; i < n; i++) {
    to[i] = from[i];
  }
}

// https://en.wikipedia.org/wiki/Bitonic_sorter
template <typename T> void sort(T &arr, int n) {
  // given an array arr of length n, this code sorts it in place
  // all indices run from 0 to n-1
  size_t arr_offs = 0;
  size_t buff_offs = n;
  for (int k = 2; k <= n; k *= 2) {      // k is doubled every iteration
    for (int j = k / 2; j > 0; j /= 2) { // j is halved at every iteration, with
                                         // truncation of fractional parts
      for (int i = 0; i < n; i++) {
#pragma HLS unroll factor = 32
        int l = i ^ j; // in C-like languages this is "i ^ j"
        if (l > i) {
          if (((i & k) == 0) && (arr[arr_offs + i] > arr[arr_offs + l]) ||
              ((i & k) != 0) && (arr[arr_offs + i] < arr[arr_offs + l])) {
            // swap the elements arr[i] and arr[l]
            arr[buff_offs + i] = arr[arr_offs + l];
            arr[buff_offs + l] = arr[arr_offs + i];
          } else {
            arr[buff_offs + l] = arr[arr_offs + l];
            arr[buff_offs + i] = arr[arr_offs + i];
          }
        }

        if (i == (n - 1)) {
          size_t tmp = arr_offs;
          arr_offs = buff_offs;
          buff_offs = tmp;
        }
      }
    }
  }
  if (arr_offs != 0) {
    for (int i = 0; i < n; i++) {
      arr[i] = arr[i + arr_offs];
    }
  }
}

int ref_cmp(const void *a, const void *b) {
  int ai = *((const int *)a);
  int bi = *((const int *)b);
  return ai > bi;
}

int main(int argc, char **argv) {
  srand(0);
  int n = 16384;
  if (argc > 1)
    n = atoi(argv[1]);
  int *array = (int *)malloc(2 * sizeof(int) * STATIC_SIZE);
  for (int i = 0; i < n; i++) {
    array[i] = rand();
  }
  int *ref = (int *)malloc(sizeof(int) * n);
  memcpy(ref, array, n * sizeof(int));
  qsort(ref, n, sizeof(int), ref_cmp);

  int *buf = (int *)malloc(sizeof(int) * n);
#pragma omp target map(to                                                      \
                       : n) map(tofrom                                         \
                                : array [0:2 * STATIC_SIZE])                   \
    orkaTranslate(array_cacheConfig                                            \
                  : "dataflow:true:true:512:1:8:0:0")
  { sort(array, n); }
  char wrong = 0;
  for (int i = 0; i < n; i++) {
    if (array[i] != ref[i])
      wrong = 1;
  }
  if (wrong) {
    for (int i = 0; i < n; i++) {
      printf("%d,%d\n", array[i], ref[i]);
    }
    printf("OUTPUT MISMATCH!\n");
    exit(EXIT_FAILURE);
  } else {
    printf("COMPUTATION SUCCESFULL!\n");
  }
  free(array);
}
