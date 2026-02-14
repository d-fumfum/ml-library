#pragma once

#include <vector>

struct matrix {
    std::vector<float> data;
    int row = 0;
    int col = 0;
};

matrix mat_create(int r, int c);
void mat_copy(matrix* dst, matrix* src);
void mat_clear(matrix* mat);
void mat_fill(matrix* mat, float x);
int mat_add(matrix* out, const matrix* a, const matrix* b);
int mat_sub(matrix* out, const matrix* a, const matrix* b);
int mat_mul(matrix* out, const matrix* a, const matrix* b,
            int zero_out, int transpose_a, int transpose_b);
int mat_softmax(matrix* out, const matrix* in);
int mat_cross_entropy(matrix* out, const matrix* p, const matrix* q);
