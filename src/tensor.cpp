#include "ml/tensor.h"
#include "ml/matrix.h"
#include "ml/utils.h"

#include <random>

static std::mt19937& global_rng() {
    static std::mt19937 rng(std::random_device{}());
    return rng;
}

tensor tsr_create(int r, int c, int l){
    tensor t;
    t.s.row = r;
    t.s.col = c;
    t.s.layer = l;
    t.data.assign(r*c*l, 0.0f);
    return t;
};


void tsr_copy(tensor* dst, const tensor* src){
    if((dst->s.row != src->s.row) || (dst->s.col != src->s.col) || (dst->s.layer != src->s.layer))
        error_occured("Tensors aren't the same size");

    dst->data = src->data;
}


void tsr_clear(tensor* tsr){
    tsr -> data.clear();
    tsr->s.row=0;
    tsr->s.col=0;
    tsr->s.layer=0;
}


void tsr_fill(tensor* tsr, float x){
    if(!tsr){
        error_occured("null tensor pointer");
        return;
    }

    if(tsr->s.row <= 0 || tsr->s.col <= 0 || tsr->s.layer <= 0){
        tsr->data.clear();
        return;
    }

    const size_t total =
        static_cast<size_t>(tsr->s.row) *
        static_cast<size_t>(tsr->s.col) *
        static_cast<size_t>(tsr->s.layer);
    tsr->data.assign(total, x);
}


int tsr_add(tensor* out, const tensor* a, const tensor* b){
    if(a->s.row != b->s.row || a->s.col != b->s.col || a->s.layer != b-> s.layer)
        return error_occured("tensors aren't the same size");

    if(out->s.row != a->s.row || out->s.col != a->s.col || out->s.layer != a->s.layer)
        return error_occured("out and tensors aren't the same size");

    const size_t n = a->data.size();
    out->data.resize(n);
    for(size_t i =0; i<n; ++i)
        out->data[i] = a->data[i] + b->data[i];


    return 0;
}


int tsr_sub(tensor* out, const tensor* a, const tensor* b){
    if(a->s.row != b->s.row || a->s.col != b->s.col || a->s.layer != b-> s.layer)
        return error_occured("tensors aren't the same size");

    if(out->s.row != a->s.row || out->s.col != a->s.col || out->s.layer != a->s.layer)
        return error_occured("out and tensors aren't the same size");

    const size_t n = a->data.size();
    out->data.resize(n);
    for(size_t i =0; i<n; ++i)
        out->data[i] = a->data[i] - b->data[i];


    return 0;
}


int tsr_mul(tensor* out, const tensor* a, const tensor* b,
            int zero_out, int transpose_a, int transpose_b){


    if(!out || !a || !b) return error_occured("null tensor pointer");

    const int a_rows = transpose_a ? a->s.col : a->s.row;
    const int a_cols = transpose_a ? a->s.row : a->s.col;
    const int b_rows = transpose_b ? b->s.col : b->s.row;
    const int b_cols = transpose_b ? b->s.row : b->s.col;

    if(a_cols != b_rows)
        return error_occured("a cols != b rows");

    if(a->s.layer != b->s.layer)
        return error_occured("a and b s.layer counts differ");

    if(out->s.row != a_rows || out->s.col != b_cols || out->s.layer != a->s.layer)
        return error_occured("out has wrong shape");

    const size_t out_size =
        static_cast<size_t>(out->s.row) * out->s.col * out->s.layer;
    if(zero_out)
        out->data.assign(out_size, 0.0f);
    else if(out->data.size() != out_size)
        return error_occured("out data size mismatch");

    const int a_layer_stride = a->s.row * a->s.col;
    const int b_layer_stride = b->s.row * b->s.col;
    const int out_layer_stride = out->s.row * out->s.col;

    for(int l = 0; l < out->s.layer; ++l){
        const int a_layer_base = l * a_layer_stride;
        const int b_layer_base = l * b_layer_stride;
        const int out_layer_base = l * out_layer_stride;

        for(int i = 0; i < a_rows; ++i){
            for(int k = 0; k < a_cols; ++k){
                const float a_val = transpose_a
                    ? a->data[a_layer_base + k * a->s.col + i]
                    : a->data[a_layer_base + i * a->s.col + k];

                for(int j = 0; j < b_cols; ++j){
                    const float b_val = transpose_b
                        ? b->data[b_layer_base + j * b->s.col + k]
                        : b->data[b_layer_base + k * b->s.col + j];

                    out->data[out_layer_base + i * out->s.col + j] += a_val * b_val;
                }
            }
        }
    }
    return 0;
}


int tsr_softmax(tensor* out, const tensor* in){
    if(!out || !in)
        return error_occured("null tensor pointer");

    if(out->s.row != in->s.row || out->s.col != in->s.col || out->s.layer != in->s.layer)
        return error_occured("out and in aren't the same shape");

    const int rows = in->s.row;
    const int cols = in->s.col;
    const int layers = in->s.layer;
    if(rows <= 0 || cols <= 0 || layers <= 0)
        return error_occured("softmax expects positive tensor shape");

    const size_t total =
        static_cast<size_t>(rows) *
        static_cast<size_t>(cols) *
        static_cast<size_t>(layers);
    if(in->data.size() != total)
        return error_occured("in data size mismatch");

    out->data.resize(total);

    const int layer_stride = rows * cols;
    for(int l = 0; l < layers; ++l){
        const int layer_base = l * layer_stride;
        for(int r = 0; r < rows; ++r){
            const int row_base = layer_base + r * cols;

            float max_val = in->data[row_base];
            for(int c = 1; c < cols; ++c){
                const float v = in->data[row_base + c];
                if(v > max_val) max_val = v;
            }

            float sum = 0.0f;
            for(int c = 0; c < cols; ++c){
                const float e = my_exp(in->data[row_base + c] - max_val);
                out->data[row_base + c] = e;
                sum += e;
            }

            if(!(sum > 0.0f))
                return error_occured("softmax sum is not positive");

            for(int c = 0; c < cols; ++c)
                out->data[row_base + c] /= sum;
        }
    }

    return 0;
}


int tsr_cross_entropy(tensor* out, const tensor* p, const tensor* q){
    if(!out || !p || !q)
        return error_occured("null tensor pointer");

    if(p->s.row != q->s.row || p->s.col != q->s.col || p->s.layer != q->s.layer)
        return error_occured("p and q aren't the same shape");

    const int rows = p->s.row;
    const int cols = p->s.col;
    const int layers = p->s.layer;
    if(rows <= 0 || cols <= 0 || layers <= 0)
        return error_occured("cross entropy expects positive tensor shape");

    const size_t total =
        static_cast<size_t>(rows) *
        static_cast<size_t>(cols) *
        static_cast<size_t>(layers);
    if(p->data.size() != total || q->data.size() != total)
        return error_occured("tensor data size mismatch");

    const bool out_per_row = (out->s.row == rows && out->s.col == 1 && out->s.layer == layers);
    const bool out_per_layer = (out->s.row == 1 && out->s.col == 1 && out->s.layer == layers);
    const bool out_scalar = (out->s.row == 1 && out->s.col == 1 && out->s.layer == 1);
    if(!out_per_row && !out_per_layer && !out_scalar)
        return error_occured("out must be (rows x 1 x layers), (1 x 1 x layers), or (1 x 1 x 1)");

    const float small = 1e-7f;
    const int layer_stride = rows * cols;

    if(out_per_row){
        out->data.assign(static_cast<size_t>(rows) * static_cast<size_t>(layers), 0.0f);
        for(int l = 0; l < layers; ++l){
            const int layer_base = l * layer_stride;
            const int out_layer_base = l * rows;
            for(int r = 0; r < rows; ++r){
                float loss = 0.0f;
                const int row_base = layer_base + r * cols;
                for(int c = 0; c < cols; ++c){
                    const float pc = my_max(p->data[row_base + c], small);
                    const float qc = q->data[row_base + c];
                    loss -= qc * my_log(pc);
                }
                out->data[out_layer_base + r] = loss;
            }
        }
        return 0;
    }

    if(out_per_layer){
        out->data.assign(static_cast<size_t>(layers), 0.0f);
        for(int l = 0; l < layers; ++l){
            const int layer_base = l * layer_stride;
            float total_loss = 0.0f;
            for(int r = 0; r < rows; ++r){
                float loss = 0.0f;
                const int row_base = layer_base + r * cols;
                for(int c = 0; c < cols; ++c){
                    const float pc = my_max(p->data[row_base + c], small);
                    const float qc = q->data[row_base + c];
                    loss -= qc * my_log(pc);
                }
                total_loss += loss;
            }
            out->data[l] = total_loss / static_cast<float>(rows);
        }
        return 0;
    }

    float total_loss = 0.0f;
    for(int l = 0; l < layers; ++l){
        const int layer_base = l * layer_stride;
        for(int r = 0; r < rows; ++r){
            float loss = 0.0f;
            const int row_base = layer_base + r * cols;
            for(int c = 0; c < cols; ++c){
                const float pc = my_max(p->data[row_base + c], small);
                const float qc = q->data[row_base + c];
                loss -= qc * my_log(pc);
            }
            total_loss += loss;
        }
    }

    const float denom = static_cast<float>(rows) * static_cast<float>(layers);
    out->data.assign(1, total_loss / denom);
    return 0;
}


matrix tsr_sum_depth(const tensor& t){
    if(t.s.row < 0 || t.s.col < 0 || t.s.layer < 0){
        error_occured("negative tensor shape");
        return mat_create(0, 0);
    }

    matrix out = mat_create(t.s.row, t.s.col);
    if(t.s.row == 0 || t.s.col == 0 || t.s.layer == 0)
        return out;

    const size_t expected =
        static_cast<size_t>(t.s.row) *
        static_cast<size_t>(t.s.col) *
        static_cast<size_t>(t.s.layer);
    if(t.data.size() != expected){
        error_occured("tensor data size mismatch");
        return out;
    }

    const size_t plane = static_cast<size_t>(t.s.row) * static_cast<size_t>(t.s.col);
    for(int l = 0; l < t.s.layer; ++l){
        const size_t base = static_cast<size_t>(l) * plane;
        for(size_t i = 0; i < plane; ++i){
            out.data[i] += t.data[base + i];
        }
    }

    return out;
}


tensor& tsr_zero(tensor& t){
    tsr_fill(&t, 0.0f);
    return t;
}


tensor& tsr_arrange(tensor& t){
    size_t total = t.data.size();
    if(t.s.row > 0 && t.s.col > 0 && t.s.layer > 0){
        total = static_cast<size_t>(t.s.row) *
                static_cast<size_t>(t.s.col) *
                static_cast<size_t>(t.s.layer);
        if(t.data.size() != total)
            t.data.resize(total, 0.0f);
    }

    t.s.row = 1;
    t.s.col = static_cast<int>(total);
    t.s.layer = 1;
    return t;
}


tensor tsr_random_uniform(tensor& out, float lo, float hi){
    std::uniform_real_distribution<float> dist(lo,hi);
    auto& rng = global_rng();

    for(float &v : out.data)
        v = dist(rng);

    return out;
}


void tsr_set_seed(int seed) {
    global_rng().seed(seed);
}


tensor tsr_random_normal(tensor& out, float mean, float stddev){
    
    std::normal_distribution<float> dist(mean,stddev);
    auto& rng = global_rng();

    for(float &v : out.data)
        v = dist(rng);

    return out;
}

int tsr_get(const tensor& t, shape idx, float* out){
    if(!out) return error_occured("null out pointer");
    if(idx.row < 0 || idx.col < 0 || idx.layer < 0) return error_occured("negative index");
    if(idx.row >= t.s.row || idx.col >= t.s.col || idx.layer >= t.s.layer)
        return error_occured("index out of range");

    const size_t total = static_cast<size_t>(t.s.row) * t.s.col * t.s.layer;
    if(t.data.size() != total) return error_occured("tensor data size mismatch");

    const size_t flat =
        static_cast<size_t>(idx.layer) * static_cast<size_t>(t.s.row) * static_cast<size_t>(t.s.col) +
        static_cast<size_t>(idx.row) * static_cast<size_t>(t.s.col) +
        static_cast<size_t>(idx.col);

    *out = t.data[flat];
    return 0;
}


tensor tsr_clone(const tensor& t){
    return t;
}


tensor tsr_like(const tensor& t){
    return tsr_create(t.s.row, t.s.col, t.s.layer);
}

tensor tsr_zeros_like(const tensor& t){
    tensor out = tsr_like(t);
    tsr_fill(&out, 0.0f);
    return out;
}