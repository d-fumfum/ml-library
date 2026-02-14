#include "ml/matrix.h"
#include "ml/utils.h"

#include <random>
#include <vector>

using namespace std;


inline float sigmoid(float x) {
    return 1.0f / (1.0f + my_exp(-x));
}



struct parameter {
    matrix data;
    matrix grad;
};



static matrix zeros(int rows, int cols) {
    return mat_create(rows, cols);
}



// stdev - standard deviation
static matrix random_normal(int rows, int cols, float mean, float stddev){
    matrix out = mat_create(rows, cols);
    static mt19937 rng(random_device{}()); // generate random number
    normal_distribution<float> dist(mean, stddev);
    for (float& v : out.data) {
        v = dist(rng);
    }
    return out; // matrix full of random values
}



inline void zero_grad(parameter& p) {
    if (p.grad.row != p.data.row || p.grad.col != p.data.col) 
        p.grad = mat_create(p.data.row, p.data.col);

    else
        mat_fill(&p.grad, 0.0f);
}


struct layer {
    virtual ~layer() = default;
    virtual matrix forward(const matrix& input) = 0;
    virtual matrix backward(const matrix& grad_output) = 0;
    virtual vector<parameter*> parameters() = 0;
};



struct dense : layer {
    parameter W;
    parameter b;
    matrix cached_input;

    dense(int in_features, int out_features);
    matrix forward(const matrix& input) override;
    matrix backward(const matrix& grad_output) override;
    vector<parameter*> parameters() override;
};



// He/Kaiming's ReLU standard deviation = sqrt(2/input) 
dense::dense(int in_features, int out_features) {
    W.data = random_normal(in_features, out_features, 0.0f, my_sqrt(2.0f / in_features));
    W.grad = zeros(in_features, out_features);
    b.data = zeros(1, out_features);
    b.grad = zeros(1, out_features);
}



matrix dense::forward(const matrix& input) {
    cached_input = input;

    matrix out = mat_create(input.row, W.data.col);
    if (mat_mul(&out, &input, &W.data, 1, 0, 0) != 0)
        return out;
    
    if (b.data.row == 1 && b.data.col == out.col) {
        for (int r = 0; r < out.row; ++r) {
            for (int c = 0; c < out.col; ++c)
                out.data[r * out.col + c] += b.data.data[c];
        }
    }

    return out;
}



matrix dense::backward(const matrix& grad_output) {
    matrix grad_input = mat_create(grad_output.row, W.data.row);
    mat_mul(&grad_input, &grad_output, &W.data, 1, 0, 1);

    if (W.grad.row != W.data.row || W.grad.col != W.data.col) 
        W.grad = mat_create(W.data.row, W.data.col);
    
    mat_mul(&W.grad, &cached_input, &grad_output, 1, 1, 0);

    if (b.grad.row != 1 || b.grad.col != grad_output.col)
        b.grad = mat_create(1, grad_output.col);
    else
        mat_fill(&b.grad, 0.0f);
    

    for (int r = 0; r < grad_output.row; ++r)
        for (int c = 0; c < grad_output.col; ++c) 
            b.grad.data[c] += grad_output.data[r * grad_output.col + c];


    return grad_input;
}



vector<parameter*> dense::parameters() {
    return {&W, &b};
}