#include "ml/abstraction.h"
#include "ml/utils.h"

#include <cstddef>
#include <utility>

namespace {
bool valid_tensor(const tensor& t) {
    if(t.s.row <= 0 || t.s.col <= 0 || t.s.layer <= 0)
        return false;

    const std::size_t expected =
        static_cast<std::size_t>(t.s.row) *
        static_cast<std::size_t>(t.s.col) *
        static_cast<std::size_t>(t.s.layer);
    return t.data.size() == expected;
}
}

dense::dense(int in_features, int out_features, bool use_bias)
    : in_features_(in_features),
      out_features_(out_features),
      use_bias_(use_bias) {
    if(in_features_ <= 0 || out_features_ <= 0){
        error_occured("dense: in_features and out_features must be positive");
        return;
    }

    tensor w = tsr_create(in_features_, out_features_, 1);
    const float stddev = my_sqrt(2.0f / static_cast<float>(in_features_));
    tsr_random_normal(w, 0.0f, stddev);
    weight_ = create_parameter(std::move(w), true);

    if(use_bias_){
        tensor b = tsr_create(1, out_features_, 1);
        tsr_zero(b);
        bias_ = create_parameter(std::move(b), true);
    }
}

tensor_list dense::forward_many(const tensor_list& inputs) {
    if(inputs.size() != 1){
        error_occured("dense::forward_many expects exactly one input tensor");
        return {};
    }
    if(!weight_){
        error_occured("dense: weight parameter is not initialized");
        return {};
    }

    const tensor& input = inputs[0];
    const tensor& w = weight_->data;
    if(!valid_tensor(input) || !valid_tensor(w)){
        error_occured("dense::forward_many invalid tensor layout");
        return {};
    }
    if(input.s.col != w.s.row){
        error_occured("dense::forward_many input feature mismatch");
        return {};
    }
    if(w.s.layer != 1 && w.s.layer != input.s.layer){
        error_occured("dense::forward_many weight layer mismatch");
        return {};
    }

    tensor out = tsr_create(input.s.row, w.s.col, input.s.layer);
    tsr_zero(out);

    const int in_rows = input.s.row;
    const int in_cols = input.s.col;
    const int out_cols = w.s.col;

    const int input_layer_stride = in_rows * in_cols;
    const int weight_layer_stride = w.s.row * w.s.col;
    const int out_layer_stride = out.s.row * out.s.col;

    for(int l = 0; l < input.s.layer; ++l){
        const int input_layer_base = l * input_layer_stride;
        const int weight_layer = (w.s.layer == 1) ? 0 : l;
        const int weight_layer_base = weight_layer * weight_layer_stride;
        const int out_layer_base = l * out_layer_stride;

        for(int r = 0; r < in_rows; ++r){
            const int input_row_base = input_layer_base + r * in_cols;
            const int out_row_base = out_layer_base + r * out_cols;
            for(int k = 0; k < in_cols; ++k){
                const float x = input.data[input_row_base + k];
                const int weight_row_base = weight_layer_base + k * out_cols;
                for(int c = 0; c < out_cols; ++c)
                    out.data[out_row_base + c] += x * w.data[weight_row_base + c];
            }
        }
    }

    if(use_bias_){
        if(!bias_){
            error_occured("dense: bias parameter is not initialized");
            return {};
        }
        const tensor& b = bias_->data;
        if(!valid_tensor(b)){
            error_occured("dense::forward_many invalid bias layout");
            return {};
        }
        const bool bias_shape_ok =
            b.s.row == 1 &&
            b.s.col == out_cols &&
            (b.s.layer == 1 || b.s.layer == out.s.layer);
        if(!bias_shape_ok){
            error_occured("dense::forward_many bias shape mismatch");
            return {};
        }

        const int bias_layer_stride = b.s.row * b.s.col;
        for(int l = 0; l < out.s.layer; ++l){
            const int bias_layer = (b.s.layer == 1) ? 0 : l;
            const int bias_layer_base = bias_layer * bias_layer_stride;
            const int out_layer_base = l * out_layer_stride;
            for(int r = 0; r < out.s.row; ++r){
                const int out_row_base = out_layer_base + r * out_cols;
                for(int c = 0; c < out_cols; ++c)
                    out.data[out_row_base + c] += b.data[bias_layer_base + c];
            }
        }
    }

    cached_input_ = tsr_clone(input);
    has_cache_ = true;
    return tensor_list{out};
}

tensor_list dense::backward_many(const tensor_list& grad_outputs) {
    if(grad_outputs.size() != 1){
        error_occured("dense::backward_many expects exactly one gradient tensor");
        return {};
    }
    if(!has_cache_){
        error_occured("dense::backward_many called before forward_many");
        return {};
    }
    if(!weight_){
        error_occured("dense: weight parameter is not initialized");
        return {};
    }

    const tensor& grad_output = grad_outputs[0];
    const tensor& input = cached_input_;
    const tensor& w = weight_->data;

    if(!valid_tensor(grad_output) || !valid_tensor(input) || !valid_tensor(w)){
        error_occured("dense::backward_many invalid tensor layout");
        return {};
    }
    if(grad_output.s.row != input.s.row ||
       grad_output.s.col != out_features_ ||
       grad_output.s.layer != input.s.layer){
        error_occured("dense::backward_many grad_output shape mismatch");
        return {};
    }
    if(w.s.row != in_features_ || w.s.col != out_features_ || (w.s.layer != 1 && w.s.layer != input.s.layer)){
        error_occured("dense::backward_many weight shape mismatch");
        return {};
    }

    tensor grad_input = tsr_create(input.s.row, in_features_, input.s.layer);
    tsr_zero(grad_input);

    const int batch = input.s.row;
    const int in_cols = in_features_;
    const int out_cols = out_features_;
    const int layers = input.s.layer;

    const int input_layer_stride = batch * in_cols;
    const int grad_output_layer_stride = batch * out_cols;
    const int grad_input_layer_stride = batch * in_cols;
    const int weight_layer_stride = w.s.row * w.s.col;

    for(int l = 0; l < layers; ++l){
        const int grad_layer_base = l * grad_output_layer_stride;
        const int grad_input_layer_base = l * grad_input_layer_stride;
        const int weight_layer = (w.s.layer == 1) ? 0 : l;
        const int weight_layer_base = weight_layer * weight_layer_stride;

        for(int r = 0; r < batch; ++r){
            const int grad_row_base = grad_layer_base + r * out_cols;
            const int grad_input_row_base = grad_input_layer_base + r * in_cols;
            for(int c = 0; c < out_cols; ++c){
                const float g = grad_output.data[grad_row_base + c];
                for(int k = 0; k < in_cols; ++k){
                    const int weight_idx = weight_layer_base + k * out_cols + c;
                    grad_input.data[grad_input_row_base + k] += g * w.data[weight_idx];
                }
            }
        }
    }

    if(weight_->grad.s.row != w.s.row || weight_->grad.s.col != w.s.col || weight_->grad.s.layer != w.s.layer)
        weight_->grad = tsr_zeros_like(w);
    else
        tsr_zero(weight_->grad);

    for(int l = 0; l < layers; ++l){
        const int input_layer_base = l * input_layer_stride;
        const int grad_layer_base = l * grad_output_layer_stride;
        const int weight_layer = (weight_->grad.s.layer == 1) ? 0 : l;
        const int weight_grad_layer_base = weight_layer * weight_layer_stride;

        for(int r = 0; r < batch; ++r){
            const int input_row_base = input_layer_base + r * in_cols;
            const int grad_row_base = grad_layer_base + r * out_cols;
            for(int k = 0; k < in_cols; ++k){
                const float x = input.data[input_row_base + k];
                const int weight_grad_row_base = weight_grad_layer_base + k * out_cols;
                for(int c = 0; c < out_cols; ++c)
                    weight_->grad.data[weight_grad_row_base + c] += x * grad_output.data[grad_row_base + c];
            }
        }
    }

    if(use_bias_){
        if(!bias_){
            error_occured("dense: bias parameter is not initialized");
            return {};
        }

        if(bias_->grad.s.row != bias_->data.s.row ||
           bias_->grad.s.col != bias_->data.s.col ||
           bias_->grad.s.layer != bias_->data.s.layer) {
            bias_->grad = tsr_zeros_like(bias_->data);
        } else {
            tsr_zero(bias_->grad);
        }

        const bool bias_grad_ok =
            bias_->grad.s.row == 1 &&
            bias_->grad.s.col == out_cols &&
            (bias_->grad.s.layer == 1 || bias_->grad.s.layer == layers);
        if(!bias_grad_ok){
            error_occured("dense::backward_many bias gradient shape mismatch");
            return {};
        }

        const int bias_grad_layer_stride = bias_->grad.s.row * bias_->grad.s.col;
        for(int l = 0; l < layers; ++l){
            const int grad_layer_base = l * grad_output_layer_stride;
            const int bias_layer = (bias_->grad.s.layer == 1) ? 0 : l;
            const int bias_layer_base = bias_layer * bias_grad_layer_stride;
            for(int r = 0; r < batch; ++r){
                const int grad_row_base = grad_layer_base + r * out_cols;
                for(int c = 0; c < out_cols; ++c)
                    bias_->grad.data[bias_layer_base + c] += grad_output.data[grad_row_base + c];
            }
        }
    }

    return tensor_list{grad_input};
}

parameter* dense::weight() noexcept {
    return weight_;
}

const parameter* dense::weight() const noexcept {
    return weight_;
}

parameter* dense::bias() noexcept {
    return bias_;
}

const parameter* dense::bias() const noexcept {
    return bias_;
}
