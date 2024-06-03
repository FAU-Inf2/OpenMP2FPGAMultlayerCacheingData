#include <stdio.h>
#include <stdlib.h>

#define NUM_ITEMS 512
#define SPACE 512
#define BUF_LEN 524288

void gen_input(int *values, int *weights, int num_items) {
  for (int i = 0; i < num_items; i++) {
    values[i] = (rand() % 1000) + 1;
    weights[i] = (rand() % 1000) + 1;
  }
}

size_t lin(int x, int y, int dim_x) {
#pragma hls inline
  return x + y * dim_x;
}

template <typename A, typename B, typename C>
int solve(int num_items, int space, A &sizes, B &values, C &res) {
#pragma hls inline off
  int x = num_items + 1;
  for (int j = 0; j < space + 1; j++) {
    res[lin(0, j, x)] = 0;
  }
  for (int i = 1; i < num_items + 1; i++) {
    for (int j = 0; j < space + 1; j++) {
      int size = sizes[i - 1];
      int value = values[i - 1];
      int base = res[lin(i - 1, j, x)];
      if (size <= j) {
        int opt = res[lin(i - 1, j - size, x)] + value;
        if (opt > base)
          res[lin(i, j, x)] = opt;
        else
          res[lin(i, j, x)] = base;
      } else {
        res[lin(i, j, x)] = base;
      }
    }
  }
  return res[lin(num_items, space, x)];
}

void print_problem(int num_items, int space, int *sizes, int *values) {
  printf("num items %d\nspace %d\n", num_items, space);
  printf("values \n");
  for (int i = 0; i < num_items; i++) {
    printf("%d ", values[i]);
  }
  printf("\n");
  printf("sizes \n");
  for (int i = 0; i < num_items; i++) {
    printf("%d ", sizes[i]);
  }
  printf("\n");
}

int main(int argc, char **argv) {
  if (argc == 2) {
    fprintf(stderr, "usage: %s [num_items space\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  int num_items = 100;
  int space = 300;
  if (argc >= 3) {
    num_items = atoi(argv[1]);
    space = atoi(argv[2]);
  }
  srand(num_items + space * 17);
  int *values = new int[NUM_ITEMS];
  int *sizes = new int[NUM_ITEMS];
  int buf_len = (num_items + 1) * (space + 1);
  int *buf = new int[BUF_LEN];
  gen_input(values, sizes, num_items);
  int res_cpu = solve(num_items, space, sizes, values, buf);
  int res_fpga;
  int *res_fpgap = &res_fpga;

#pragma omp target map(to                                                      \
                       : sizes [0:NUM_ITEMS], values [0:NUM_ITEMS],            \
                         buf [0:BUF_LEN], num_items, space)                    \
    map(tofrom                                                                 \
        : res_fpgap [0:1])
  { *res_fpgap = solve(num_items, space, sizes, values, buf); }

  // print_problem(num_items,space,sizes,values);
  if (res_cpu != res_fpga) {
    printf("result CPU: %d\n", res_cpu);
    printf("result FPGA: %d\n", res_fpga);
    fprintf(stderr, "MISMATCH!\n");
    exit(EXIT_FAILURE);
  } else {
    print_problem(num_items, space, sizes, values);
    printf("Computation Successfull!\nresult: %d\n", res_fpga);
  }

  delete[] values;
  delete[] sizes;
  delete[] buf;
}
