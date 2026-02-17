#include "ml/matrix.h"
#include "ml/tensor.h"
#include "ml/utils.h"
using namespace std;

static tensor matrix_as_tensor(const matrix* mat){
    tensor t;
    t.data = mat->data;
    t.s.row = mat->row;
    t.s.col = mat->col;
    t.s.layer = 1;
    return t;
}

static void tensor_to_matrix(matrix* mat, const tensor& t){
    mat->row = t.s.row;
    mat->col = t.s.col;
    mat->data = t.data;
}


matrix mat_create(int r, int c){
    const tensor t = tsr_create(r, c, 1);
    matrix m;
    tensor_to_matrix(&m, t);
    return m;
}


void mat_copy(matrix *dst, matrix *src){
    if(!dst || !src){
        error_occured("null matrix pointer");
        return;
    }

    tensor t_dst = matrix_as_tensor(dst);
    const tensor t_src = matrix_as_tensor(src);
    tsr_copy(&t_dst, &t_src);
    tensor_to_matrix(dst, t_dst);
}


void mat_clear(matrix *mat){
    if(!mat){
        error_occured("null matrix pointer");
        return;
    }

    tensor t = matrix_as_tensor(mat);
    tsr_clear(&t);
    tensor_to_matrix(mat, t);
}


void mat_fill(matrix *mat, float x){
    if(!mat){
        error_occured("null matrix pointer");
        return;
    }

    tensor t = matrix_as_tensor(mat);
    tsr_fill(&t, x);
    tensor_to_matrix(mat, t);
}


int mat_add(matrix *out, const matrix *a, const matrix *b){
    if(!out || !a || !b)
        return error_occured("null matrix pointer");

    tensor t_out = matrix_as_tensor(out);
    const tensor t_a = matrix_as_tensor(a);
    const tensor t_b = matrix_as_tensor(b);

    const int rc = tsr_add(&t_out, &t_a, &t_b);
    if(rc != 0)
        return rc;

    tensor_to_matrix(out, t_out);
    return 0;
}


int mat_sub(matrix *out, const matrix *a, const matrix *b){
    if(!out || !a || !b)
        return error_occured("null matrix pointer");

    tensor t_out = matrix_as_tensor(out);
    const tensor t_a = matrix_as_tensor(a);
    const tensor t_b = matrix_as_tensor(b);

    const int rc = tsr_sub(&t_out, &t_a, &t_b);
    if(rc != 0)
        return rc;

    tensor_to_matrix(out, t_out);
    return 0;
}


int mat_mul(matrix *out, const matrix *a, const matrix *b,
            int zero_out, int transpose_a, int transpose_b){
    if(!out || !a || !b) return error_occured("null matrix pointer");

    tensor t_out = matrix_as_tensor(out);
    const tensor t_a = matrix_as_tensor(a);
    const tensor t_b = matrix_as_tensor(b);

    const int rc = tsr_mul(&t_out, &t_a, &t_b, zero_out, transpose_a, transpose_b);
    if(rc != 0)
        return rc;

    tensor_to_matrix(out, t_out);
    return rc;
}


int mat_softmax(matrix *out, const matrix *in){
    if (!out || !in)
        return error_occured("null matrix pointer");

    tensor t_out = matrix_as_tensor(out);
    const tensor t_in = matrix_as_tensor(in);

    const int rc = tsr_softmax(&t_out, &t_in);
    if(rc != 0)
        return rc;

    tensor_to_matrix(out, t_out);
    return 0;
}


int mat_cross_entropy(matrix *out, const matrix *p, const matrix *q){
    if(!out || !p || !q)
        return error_occured("null matrix pointer");

    tensor t_out = matrix_as_tensor(out);
    const tensor t_p = matrix_as_tensor(p);
    const tensor t_q = matrix_as_tensor(q);

    const int rc = tsr_cross_entropy(&t_out, &t_p, &t_q);
    if(rc != 0)
        return rc;

    tensor_to_matrix(out, t_out);
    return rc;
}
