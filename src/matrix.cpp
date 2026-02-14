#include "ml/matrix.h"
#include "ml/utils.h"
using namespace std;



matrix mat_create(int r, int c){
    matrix m;
    m.row = r;
    m.col = c;
    m.data.assign(r*c,0.0f);
    return m;
}


void mat_copy(matrix *dst, matrix *src){
    if((dst->row != src->row) || (dst->col != src->col))
        error_occured("Matrices aren't the same size");

    dst->data = src->data;
}


void mat_clear(matrix *mat){
    mat->data.clear();
    mat->row=0;
    mat->col=0;
}


void mat_fill(matrix *mat, float x){
    mat->data.assign(mat->row * mat->col, x);
}


int mat_add(matrix *out, const matrix *a, const matrix *b){
    if(a->row != b->row || a->col != b->col)
        return error_occured("a and b aren't the same size");

    if(out->row != a->row || out->col != a->col)
        return error_occured("out and matrices aren't the same size");

    const size_t n = a->data.size();
    out->data.resize(n);
    for(size_t i=0; i<n; ++i){
        out->data[i] = a->data[i] + b->data[i];
    }

    return 0;
}


int mat_sub(matrix *out, const matrix *a, const matrix *b){
    if(a->row != b->row || a->col != b->col)
        return error_occured("a and b aren't the same size");

    if(out->row != a->row || out->col != a->col)
        return error_occured("out and matrices aren't the same size");

    const size_t n = a->data.size();
    out->data.resize(n);
    for(size_t i=0; i<n; ++i){
        out->data[i] = a->data[i] - b->data[i];
    }

    return 0;
}


int mat_mul(matrix *out, const matrix *a, const matrix *b,
            int zero_out, int transpose_a, int transpose_b){

    if(!out || !a || !b) return error_occured("null matrix pointer");

    const int a_rows = transpose_a ? a->col : a->row;
    const int a_cols = transpose_a ? a->row : a->col;
    const int b_rows = transpose_b ? b->col : b->row;
    const int b_cols = transpose_b ? b->row : b->col;
    
    if (a_cols != b_rows)
        return error_occured("a cols != b rows");

    if (out->row != a_rows || out->col != b_cols)
        return error_occured("out has wrong shape");


    const size_t out_size = static_cast<size_t>(out->row) * out->col;

    if(zero_out)
        out->data.assign(out_size, 0.0f);
        
    else if (out->data.size() != out_size)
        return error_occured("out data size mismatxh");

    for(int idx = 0; idx < a_rows; ++idx){
        for(int k = 0; k < a_cols; ++k){
            const float a_val = transpose_a
            ? a->data[k*a->col + idx]
            : a->data[idx*a->col + k];
        

            for(int jdx =0; jdx < b_cols; ++jdx){
                const float b_val = transpose_b
                ? b->data[jdx*b->col + k]
                : b->data[k*b->col + jdx];

                out->data[idx*out->col + jdx] += a_val * b_val;
            }
        }
    }
    return 0;
}


int mat_softmax(matrix *out, const matrix *in){
    if (!out || !in)
        return error_occured("null matrix pointer");

    if(out->row != in->row || out->col != in->col)
        return error_occured("out and in aren't the same shape");

    const int rows = in->row;
    const int cols = in->col;

    out->data.resize(static_cast<size_t>(rows)*cols);

    for(int r=0; r<rows; ++r){
        float max_val = in->data[r*cols];
        for(int c = 1; c < cols; ++c){
            float v = in->data[r*cols + c];
            if(v > max_val) max_val = v;
        }

        float sum = 0.0f;
        for (int c = 0; c < cols; ++c){
            float e = my_exp(in->data[r * cols + c] - max_val);
            out->data[r * cols + c] = e;
            sum += e; 
        }

        if(sum == 0.0f)
            return error_occured("softmax sum is zero");
 
        for(int c=0; c<cols; ++c)
            out->data[r*cols+c] /= sum;
    }
    return 0;
}


int mat_cross_entropy(matrix *out, const matrix *p, const matrix *q){
    if(!out || !p || !q)
        return error_occured("null matrix pointer");

    if(p->row != q->row || p->col != q->col)
        return error_occured("p and q aren't the same size");

    const int rows = p->row;
    const int cols = p->col;

    const bool out_per_row = (out->row == rows && out->col == 1);
    const bool out_scalar = (out->row == 1 && out->col ==1);
    if(!out_per_row && !out_scalar) 
        return error_occured("out must be (rows x 1) or (1 x 1)");
        
    const float small = 1e-7;
        
    if (out_per_row) {
        out->data.assign(static_cast<size_t>(rows), 0.0f);
        for (int r = 0; r < rows; ++r) {
            float loss = 0.0f;
            for (int c = 0; c < cols; ++c) {
                float pc = my_max(p->data[r * cols + c], small);
                float qc = q->data[r * cols + c];
                loss -= qc * my_log(pc);
            }
            out->data[r] = loss;
        }
    }
    else{
        float total = 0.0f;
        for (int r = 0; r < rows; ++r) {
            float loss = 0.0f;
            for (int c = 0; c < cols; ++c) {
                float pc = my_max(p->data[r * cols + c], small);
                float qc = q->data[r * cols + c];
                loss -= qc * my_log(pc);
            }
            total += loss;
        }
            out->data.assign(1, total / rows);
        }
    return 0;
}


int mat_relu_add_grad(matrix *out, const matrix *in){
    if(!out || !in)
        return error_occured("null matrix pointer");

    if(out->row != in->row || out->col != in->col)
        return error_occured("out and in aren't the same size");

    const size_t n = static_cast<size_t>(in->row) * in->col;
    if(out->data.size() != n || in->data.size() != n)
        return error_occured("data size mismatch");
 
    for(size_t i = 0; i < n; ++i){
        if(in->data[i] <= 0.0f)
            out->data[i] = 0.0f;
    }

    return 0;
}


int mat_softmax_add_grad(matrix *out, const matrix *softmax_out){
    if(!out || !softmax_out)
        return error_occured("null matrix pointer");

    if(out->row != softmax_out->row || out->col != softmax_out->col)
        return error_occured("out and softmax_out aren't the same size");

    const int rows = softmax_out->row;
    const int cols = softmax_out->col;
    const size_t n = static_cast<size_t>(rows) * cols;
    if(out->data.size() != n || softmax_out->data.size() != n)
        return error_occured("data size mismatch");

    for(int r = 0; r < rows; ++r){
        const int base = r * cols;
        float dot = 0.0f;
        for(int c = 0; c < cols; ++c){
            dot += out->data[base + c] * softmax_out->data[base + c];
        }
        for(int c = 0; c < cols; ++c){
            const float y = softmax_out->data[base + c];
            out->data[base + c] = y * (out->data[base + c] - dot);
        }
    }

    return 0;
}


int mat_cross_entropy_add_grad(matrix *out, const matrix *p, const matrix *q){
    if(!out || !p || !q)
        return error_occured("null matrix pointer");

    if(p->row != q->row || p->col != q->col)
        return error_occured("p and q aren't the same size");

    if(out->row != p->row || out->col != p->col)
        return error_occured("out and p aren't the same size");

    const size_t n = static_cast<size_t>(p->row) * p->col;
    if(out->data.size() != n)
        out->data.assign(n, 0.0f);

    if(p->data.size() != n || q->data.size() != n)
        return error_occured("data size mismatch");

    const float small = 1e-7f;
    const float scale = (p->row > 0) ? (1.0f / p->row) : 1.0f;

    for(size_t i = 0; i < n; ++i){
        const float pc = my_max(p->data[i], small);
        out->data[i] += (-q->data[i] / pc) * scale;
    }

    return 0;
}
