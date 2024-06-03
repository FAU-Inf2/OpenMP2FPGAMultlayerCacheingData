#include <cstdlib>
#include <iostream>

#define FLOW_SCALING_FACTOR (0.25f) //(1.0f/4.0f)

int PO(int x, int y, int width) { return x + y * width; }

int Px(int x, int width) {
#pragma HLS INLINE
  if (x >= width)
    x = width - 1;
  else if (x < 0)
    x = 0;
  return x;
}
int Py(int y, int height) {
#pragma HLS INLINE
  if (y >= height)
    y = height - 1;
  else if (y < 0)
    y = 0;
  return y;
}
int P(int x, int y, int width, int height) {
#pragma HLS INLINE
  return Py(y, height) * width + Px(x, width);
}

int get_matrix_inv(float *G, float *G_inv) {
  float detG = (float)G[0] * G[3] - (float)G[1] * G[2];
  if (detG <= 1.0f) {
    return 0;
  }
  float detG_inv = 1.0f / detG;
  G_inv[0] = G[3] * detG_inv;
  G_inv[1] = -G[1] * detG_inv;
  G_inv[2] = -G[2] * detG_inv;
  G_inv[3] = G[0] * detG_inv;
  return 1;
}

extern "C" void knp(unsigned char *im1, unsigned char *im2, float *out,
                    int width, int height, int window_size) {
loop_main:
  for (int j = 0; j < height; j++)
    for (int i = 0; i < width; i++) {
      float G_inv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma HLS ARRAY_PARTITION variable = G_inv complete dim = 1
      float G[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma HLS ARRAY_PARTITION variable = G complete dim = 1
      float b_k[2] = {0.0f, 0.0f};
#pragma HLS ARRAY_PARTITION variable = b_k complete dim = 1

    loop_column:
      for (int wj = -1 * window_size; wj <= window_size; wj++) {
      loop_row:
        for (int wi = -1 * window_size; wi <= window_size; wi++) {
#pragma HLS PIPELINE
          int px = Px(i + wi, width), py = Py(j + wj, height);
          int im2_val = im2[px + py * width];

          int deltaIk = im1[px + py * width] - im2_val;
          int cx = Px(i + wi + 1, width) + py * width,
              dx = Px(i + wi - 1, width) + py * width;
          int cIx = im1[cx];

          cIx -= im1[dx];
          cIx >>= 1;
          int cy = px + Py(j + wj + 1, height) * width,
              dy = px + Py(j + wj - 1, height) * width;

          int cIy = im1[cy];
          cIy -= im1[dy];
          cIy >>= 1;

          G[0] += cIx * cIx;
          G[1] += cIx * cIy;
          G[2] += cIx * cIy;
          G[3] += cIy * cIy;
          b_k[0] += deltaIk * cIx;
          b_k[1] += deltaIk * cIy;
        }
      }

      get_matrix_inv(G, G_inv);

      float fx = 0.0f, fy = 0.0f;
      fx = G_inv[0] * b_k[0] + G_inv[1] * b_k[1];
      fy = G_inv[2] * b_k[0] + G_inv[3] * b_k[1];

      out[2 * (j * width + i)] = fx;
      out[2 * (j * width + i) + 1] = fy;
    }
}

std::pair<unsigned char *, unsigned char *>
gen_shifted_img_pair(size_t width, size_t height, int shiftx, int shifty) {
  unsigned char *imga = new unsigned char[width * height];
  unsigned char *imgb = new unsigned char[width * height];
  for (size_t i = 0; i < width * height; i++) {
    imga[i] = rand();
  }
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      int si = i + shiftx;
      int sj = j + shifty;
      if (si >= 0 && si < width && sj >= 0 && sj < height) {
        imgb[i + j * width] = imga[si + sj * width];
      } else {
        imgb[i + j * width] = rand();
      }
    }
  }
  return std::make_pair(imga, imgb);
}

void print_img(unsigned char *img, int width, int height) {
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      std::cout << (int)img[i + j * width] << " ";
    }
    std::cout << std::endl;
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " <size> <window size>" << std::endl;
    abort();
  }
  int size = atoi(argv[1]);
  int window_size = atoi(argv[2]);

  srand((size << 5) ^ window_size);
  int shiftx = (rand() % (window_size)) - window_size / 2;
  int shifty = (rand() % (window_size)) - window_size / 2;
  auto inp = gen_shifted_img_pair(size, size, shiftx, shifty);
  unsigned char *imga = inp.first;
  unsigned char *imgb = inp.second;

  float *result_fpga = new float[size * size * 2];
#pragma omp target map(to                                                      \
                       : imga [0:size * size], imgb [0:size * size], size,     \
                         window_size) map(from                                 \
                                          : result_fpga [0:size * size * 2])   \
    orkaTranslate(imga_cacheConfig                                             \
                  : "vitis:512:8") orkaTranslate(imgb_cacheConfig              \
                                                 : "vitis:512:8")
  { knp(imga, imgb, result_fpga, size, size, window_size); }

  float *result_cpu = new float[size * size * 2];
  knp(imga, imgb, result_cpu, size, size, window_size);

  bool match = true;
  for (int i = 0; i < 2 * size * size; i++) {
    if (result_cpu[i] != result_fpga[i]) {
      match = false;
    }
  }
  if (!match) {
    std::cout << "output mismatch!" << std::endl;
    abort();
  } else {
    std::cout << "computation successful!" << std::endl;
  }

  delete[] inp.first;
  delete[] inp.second;
  delete[] result_fpga;
  delete[] result_cpu;
}
