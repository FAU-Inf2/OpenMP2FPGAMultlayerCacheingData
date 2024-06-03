#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define KERNEL_SIZE 32
#define KERNEL_SIZESQ (KERNEL_SIZE * KERNEL_SIZE)
#define DATA_SIZE 512
#define DATA_SIZESQ (DATA_SIZE * DATA_SIZE)

int linearize(int x, int y, int bndx, int bndy) {
  if (x >= bndx || y >= bndy)
    return -1;
  return x + bndx * y;
}

void fill_rand(double *data, int bndx, int bndy) {
  for (int i = 0; i < bndx; i++) {
    for (int j = 0; j < bndy; j++) {
      data[linearize(i, j, bndx, bndy)] = ((double)(rand() % 10000)) / 10000;
    }
  }
}

template <typename A, typename B, typename C>
void apply_filter(A &kernel, B &data, C &result, const int N, const int M) {
  for (int x = 0; x < N; x++) {
    for (int y = 0; y < N; y++) {
      double weightsum = 0;
      double sum = 0;
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
          double weight = kernel[linearize(i, j, M, M)];
          int target = linearize(x + i, y + j, N, N);
          if (target < 0)
            continue;
          weightsum += weight;
          sum += weight * data[target];
        }
      }
      result[linearize(x, y, N, N)] = sum / weightsum;
    }
  }
}

int main(int argc, char **argv) {
  int N = 100;
  int M = 20;
  if (argc == 3) {
    N = atoi(argv[1]);
    M = atoi(argv[2]);
  }
  srand(0);
  double *kernel = (double *)malloc(sizeof(double) * KERNEL_SIZESQ);
  double *data = (double *)malloc(sizeof(double) * DATA_SIZESQ);
  double *resulta = (double *)malloc(sizeof(double) * DATA_SIZESQ);
  double *resultb = (double *)malloc(sizeof(double) * DATA_SIZESQ);
  fill_rand(kernel, M, M);
  fill_rand(data, N, N);

  struct timeval a, b, c, d, sub_one, sub_two;

  gettimeofday(&a, NULL);
  apply_filter(kernel, data, resulta, N, M);
  gettimeofday(&b, NULL);
  timersub(&b, &a, &sub_one);
  printf("CPU filtering complete, Time: %ld.%06ld s\n",
         (long int)sub_one.tv_sec, (long int)sub_one.tv_usec);
  gettimeofday(&c, NULL);
#pragma omp target map(tofrom                                                  \
                       : data [0:DATA_SIZESQ], kernel [0:KERNEL_SIZESQ],       \
                         resultb [0:DATA_SIZESQ])                              \
    orkaTranslate(data_cacheConfig                                             \
                  : "dataflow:true:false:512:1:8:0:0")                         \
        orkaTranslate(kernel_cacheConfig                                       \
                      : "dataflow:true:false:256:1:8:0:0")
  { apply_filter(kernel, data, resultb, N, M); }
  gettimeofday(&d, NULL);

  timersub(&d, &c, &sub_two);
  printf("FPGA filtering complete, Time: %ld.%06ld s\n",
         (long int)sub_two.tv_sec, (long int)sub_two.tv_usec);

  FILE *output = fopen("output.txt", "w");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      int target = linearize(i, j, N, N);
      if (fabs(resulta[target] - resultb[target]) > 0.01) {
        printf("OUTPUT MISMATCH!!!!\nABORTING\n");
        exit(1);
      } else {
        fprintf(output, "%f ", resultb[target]);
      }
    }
    fprintf(output, "\n");
  }
  fclose(output);
  printf("Computation successful!\n");
  free(kernel);
  free(data);
  free(resulta);
  free(resultb);
}
