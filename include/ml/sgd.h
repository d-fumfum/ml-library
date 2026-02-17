#pragma once

#include "ml/optimizer.h"

#include <unordered_map>
#include <vector>

struct sgd_options {
    float lr = 1e-3f;
    float momentum = 0.0f;
    float dampening = 0.0f;
    float weight_decay = 0.0f;
    bool nesterov = false;
    bool maximize = false;
};

class sgd final : public optimizer {
public:
    explicit sgd(std::vector<parameter*> parameters, sgd_options options = {});
    explicit sgd(module& model, sgd_options options = {});

    void step() override;

    const sgd_options& options() const noexcept;
    void set_options(sgd_options options) noexcept;
    void clear_state() noexcept;

private:
    tensor* ensure_velocity(parameter* param);

    sgd_options opts_;
    std::unordered_map<parameter*, tensor> velocity_;
};
