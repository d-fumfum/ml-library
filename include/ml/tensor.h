#pragma once

#include <cstdint>
#include <vector>

struct matrix;

struct shape {
    int row = 0;
    int col = 0;
    int layer = 0;
};

struct tensor {
    std::vector<float> data;
    shape s;
};

tensor tsr_create(int r, int c, int l);
void tsr_copy(tensor* dst, const tensor* src);
void tsr_clear(tensor* tsr);
void tsr_fill(tensor* tsr, float x);
int tsr_add(tensor* out, const tensor* a, const tensor* b);
int tsr_sub(tensor* out, const tensor* a, const tensor* b);
int tsr_mul(tensor* out, const tensor* a, const tensor* b,
            int zero_out, int transpose_a, int transpose_b);
int tsr_softmax(tensor* out, const tensor* in);
int tsr_cross_entropy(tensor* out, const tensor* p, const tensor* q);
matrix tsr_sum_depth(const tensor& t);
tensor& tsr_zero(tensor& t);
tensor& tsr_arrange(tensor& t);
tensor tsr_random_uniform(tensor& out, float lo, float hi);
void tsr_set_seed(int seed);
tensor tsr_random_normal(tensor& out, float mean, float stddev);
int tsr_get(const tensor& t, shape idx, float* out);
tensor tsr_clone(const tensor& t);
tensor tsr_like(const tensor& t);
tensor tsr_zeros_like(const tensor& t);
