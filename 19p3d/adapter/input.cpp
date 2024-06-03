#include <cstdlib>
#include <iostream>

float *float_stencil19(unsigned int size) {

  int i, j, k, iter;
  int n = size - 2;
  float fac = 1.0 / 18;

  /* Work buffers, with halos */
  float *a0 = (float *)malloc(sizeof(float) * size * size * size);
  float *a1 = (float *)malloc(sizeof(float) * size * size * size);

  if (a0 == NULL || a1 == NULL) {
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf(
        "19-point Single Precision Stencil Error: Unable to allocate memory\n");
  }

  /* zero all of array (including halos) */
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k < size; k++) {
        a0[i * size * size + j * size + k] = 0.0;
        a1[i * size * size + j * size + k] = 0.0;
      }
    }
  }

  /* use random numbers to fill interior */
  for (i = 1; i < n + 1; i++) {
    for (j = 1; j < n + 1; j++) {
      for (k = 1; k < n + 1; k++) {
        a0[i * size * size + j * size + k] =
            (float)rand() / (float)(1.0 + RAND_MAX);
      }
    }
  }

  /* run main computation on fpga */
#pragma omp target map(to                                                      \
                       : size, n, a0 [0:size * size * size])                   \
    map(tofrom                                                                 \
        : a1 [0:size * size * size]) \ 
orkaTranslate(a0_cacheConfig                                                   \
              : "vitis:512:8")
  {
    for (int i = 1; i < n + 1; i++) {
      for (int j = 1; j < n + 1; j++) {
        for (int k = 1; k < n + 1; k++) {
          a1[i * size * size + j * size + k] =
              (a0[i * size * size + (j - 1) * size + k] +
               a0[i * size * size + (j + 1) * size + k] +
               a0[(i - 1) * size * size + j * size + k] +
               a0[(i + 1) * size * size + j * size + k] +
               a0[(i - 1) * size * size + (j - 1) * size + k] +
               a0[(i - 1) * size * size + (j + 1) * size + k] +
               a0[(i + 1) * size * size + (j - 1) * size + k] +
               a0[(i + 1) * size * size + (j + 1) * size + k] +

               a0[i * size * size + (j - 1) * size + (k - 1)] +
               a0[i * size * size + (j + 1) * size + (k - 1)] +
               a0[(i - 1) * size * size + j * size + (k - 1)] +
               a0[(i + 1) * size * size + j * size + (k - 1)] +

               a0[i * size * size + (j - 1) * size + (k + 1)] +
               a0[i * size * size + (j + 1) * size + (k + 1)] +
               a0[(i - 1) * size * size + j * size + (k + 1)] +
               a0[(i + 1) * size * size + j * size + (k + 1)] +

               a0[i * size * size + j * size + (k - 1)] +
               a0[i * size * size + j * size + (k + 1)]) *
              fac;
        }
      }
    }
  }

  /* Free malloc'd memory to prevent leaks */
  free(a0);
  return a1;
}

float *float_stencil19_ref(unsigned int size) {

  int i, j, k, iter;
  int n = size - 2;
  float fac = 1.0 / 18;

  /* Work buffers, with halos */
  float *a0 = (float *)malloc(sizeof(float) * size * size * size);
  float *a1 = (float *)malloc(sizeof(float) * size * size * size);

  if (a0 == NULL || a1 == NULL) {
    /* Something went wrong in the memory allocation here, fail gracefully */
    printf(
        "19-point Single Precision Stencil Error: Unable to allocate memory\n");
  }

  /* zero all of array (including halos) */
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k < size; k++) {
        a0[i * size * size + j * size + k] = 0.0;
        a1[i * size * size + j * size + k] = 0.0;
      }
    }
  }

  /* use random numbers to fill interior */
  for (i = 1; i < n + 1; i++) {
    for (j = 1; j < n + 1; j++) {
      for (k = 1; k < n + 1; k++) {
        a0[i * size * size + j * size + k] =
            (float)rand() / (float)(1.0 + RAND_MAX);
      }
    }
  }

  /* run main computation on host */

  for (i = 1; i < n + 1; i++) {
    for (j = 1; j < n + 1; j++) {
      for (k = 1; k < n + 1; k++) {
        a1[i * size * size + j * size + k] =
            (a0[i * size * size + (j - 1) * size + k] +
             a0[i * size * size + (j + 1) * size + k] +
             a0[(i - 1) * size * size + j * size + k] +
             a0[(i + 1) * size * size + j * size + k] +
             a0[(i - 1) * size * size + (j - 1) * size + k] +
             a0[(i - 1) * size * size + (j + 1) * size + k] +
             a0[(i + 1) * size * size + (j - 1) * size + k] +
             a0[(i + 1) * size * size + (j + 1) * size + k] +

             a0[i * size * size + (j - 1) * size + (k - 1)] +
             a0[i * size * size + (j + 1) * size + (k - 1)] +
             a0[(i - 1) * size * size + j * size + (k - 1)] +
             a0[(i + 1) * size * size + j * size + (k - 1)] +

             a0[i * size * size + (j - 1) * size + (k + 1)] +
             a0[i * size * size + (j + 1) * size + (k + 1)] +
             a0[(i - 1) * size * size + j * size + (k + 1)] +
             a0[(i + 1) * size * size + j * size + (k + 1)] +

             a0[i * size * size + j * size + (k - 1)] +
             a0[i * size * size + j * size + (k + 1)]) *
            fac;
      }
    }
  }

  /* Free malloc'd memory to prevent leaks */
  free(a0);
  return a1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <size>" << std::endl;
    abort();
  }
  size_t size = atoi(argv[1]);
  srand(size);
  float *res_fpga = float_stencil19(size);
  srand(size);
  float *res_cpu = float_stencil19_ref(size);

  bool match = true;
  for (size_t i = 0; i < size * size * size; i++) {
    if (res_fpga[i] != res_cpu[i]) {
      match = false;
      std::cout << "mismatch! : " << std::endl;
    }

    std::cout << res_fpga[i] << " " << res_cpu[i] << std::endl;
  }

  if (!match) {
    std::cout << "output mismatch!" << std::endl;
    exit(-1);
  } else {
    std::cout << "computation successful!" << std::endl;
  }
  free(res_cpu);
  free(res_fpga);
}
