#pragma once

#include "ml/tensor.h"

class loss {
public:
    virtual ~loss() = default;

    virtual tensor forward(const tensor& prediction, const tensor& target) = 0;
    virtual tensor backward() const = 0;

    tensor operator()(const tensor& prediction, const tensor& target) {
        return forward(prediction, target);
    }
};

class mse_loss final : public loss {
public:
    explicit mse_loss(bool mean_reduction = true) noexcept;

    tensor forward(const tensor& prediction, const tensor& target) override;
    tensor backward() const override;

private:
    bool mean_reduction_ = true;
    tensor grad_cache_;
    bool has_cache_ = false;
};

class cross_entropy_loss final : public loss {
public:
    explicit cross_entropy_loss(bool mean_reduction = true, float epsilon = 1e-7f) noexcept;

    tensor forward(const tensor& prediction, const tensor& target) override;
    tensor backward() const override;

private:
    bool mean_reduction_ = true;
    float epsilon_ = 1e-7f;
    tensor grad_cache_;
    bool has_cache_ = false;
};
