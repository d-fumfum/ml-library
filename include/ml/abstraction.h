#pragma once

#include "ml/module.h"

class dense final : public module {
public:
    dense(int in_features, int out_features, bool use_bias = true);

    tensor_list forward_many(const tensor_list& inputs) override;
    tensor_list backward_many(const tensor_list& grad_outputs) override;

    parameter* weight() noexcept;
    const parameter* weight() const noexcept;
    parameter* bias() noexcept;
    const parameter* bias() const noexcept;

private:
    int in_features_ = 0;
    int out_features_ = 0;
    bool use_bias_ = true;

    parameter* weight_ = nullptr;
    parameter* bias_ = nullptr;

    tensor cached_input_;
    bool has_cache_ = false;
};
