#include "ml/loss.h"
#include "ml/utils.h"

#include <cstddef>

namespace {
bool valid_tensor_layout(const tensor& t) {
    if(t.s.row < 0 || t.s.col < 0 || t.s.layer < 0)
        return false;

    const std::size_t expected =
        static_cast<std::size_t>(t.s.row) *
        static_cast<std::size_t>(t.s.col) *
        static_cast<std::size_t>(t.s.layer);
    return t.data.size() == expected;
}

bool same_shape(const tensor& a, const tensor& b) {
    return a.s.row == b.s.row && a.s.col == b.s.col && a.s.layer == b.s.layer;
}

std::size_t element_count(const tensor& t) {
    return static_cast<std::size_t>(t.s.row) *
           static_cast<std::size_t>(t.s.col) *
           static_cast<std::size_t>(t.s.layer);
}

tensor make_scalar(float value) {
    tensor out = tsr_create(1, 1, 1);
    out.data[0] = value;
    return out;
}
}

mse_loss::mse_loss(bool mean_reduction) noexcept
    : mean_reduction_(mean_reduction) {}

tensor mse_loss::forward(const tensor& prediction, const tensor& target) {
    has_cache_ = false;

    if(!valid_tensor_layout(prediction) || !valid_tensor_layout(target)){
        error_occured("mse_loss: invalid tensor layout");
        return tsr_create(0, 0, 0);
    }
    if(!same_shape(prediction, target)){
        error_occured("mse_loss: prediction and target shape mismatch");
        return tsr_create(0, 0, 0);
    }

    const std::size_t n = element_count(prediction);
    if(n == 0){
        error_occured("mse_loss: empty tensor");
        return tsr_create(0, 0, 0);
    }

    if(!same_shape(grad_cache_, prediction))
        grad_cache_ = tsr_create(prediction.s.row, prediction.s.col, prediction.s.layer);

    float* g = grad_cache_.data.data();
    const float* p = prediction.data.data();
    const float* t = target.data.data();

    float total = 0.0f;
    const float grad_scale = mean_reduction_ ? (2.0f / static_cast<float>(n)) : 2.0f;

    for(std::size_t i = 0; i < n; ++i){
        const float diff = p[i] - t[i];
        total += diff * diff;
        g[i] = grad_scale * diff;
    }

    has_cache_ = true;
    if(mean_reduction_)
        total /= static_cast<float>(n);

    return make_scalar(total);
}

tensor mse_loss::backward() const {
    if(!has_cache_){
        error_occured("mse_loss: backward called before forward");
        return tsr_create(0, 0, 0);
    }
    return tsr_clone(grad_cache_);
}

cross_entropy_loss::cross_entropy_loss(bool mean_reduction, float epsilon) noexcept
    : mean_reduction_(mean_reduction), epsilon_(epsilon > 0.0f ? epsilon : 1e-7f) {}

tensor cross_entropy_loss::forward(const tensor& prediction, const tensor& target) {
    has_cache_ = false;

    if(!valid_tensor_layout(prediction) || !valid_tensor_layout(target)){
        error_occured("cross_entropy_loss: invalid tensor layout");
        return tsr_create(0, 0, 0);
    }
    if(!same_shape(prediction, target)){
        error_occured("cross_entropy_loss: prediction and target shape mismatch");
        return tsr_create(0, 0, 0);
    }
    if(prediction.s.row <= 0 || prediction.s.layer <= 0){
        error_occured("cross_entropy_loss: invalid shape");
        return tsr_create(0, 0, 0);
    }

    const std::size_t n = element_count(prediction);
    if(n == 0){
        error_occured("cross_entropy_loss: empty tensor");
        return tsr_create(0, 0, 0);
    }

    if(!same_shape(grad_cache_, prediction))
        grad_cache_ = tsr_create(prediction.s.row, prediction.s.col, prediction.s.layer);

    const std::size_t batch_count =
        static_cast<std::size_t>(prediction.s.row) * static_cast<std::size_t>(prediction.s.layer);
    const float scale = mean_reduction_ ? (1.0f / static_cast<float>(batch_count)) : 1.0f;

    float* g = grad_cache_.data.data();
    const float* p = prediction.data.data();
    const float* t = target.data.data();

    float total = 0.0f;
    for(std::size_t i = 0; i < n; ++i){
        const float pc = my_max(p[i], epsilon_);
        const float tc = t[i];
        total -= tc * my_log(pc);
        g[i] = (-tc / pc) * scale;
    }

    has_cache_ = true;
    if(mean_reduction_)
        total *= scale;

    return make_scalar(total);
}

tensor cross_entropy_loss::backward() const {
    if(!has_cache_){
        error_occured("cross_entropy_loss: backward called before forward");
        return tsr_create(0, 0, 0);
    }
    return tsr_clone(grad_cache_);
}
