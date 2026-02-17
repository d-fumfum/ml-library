#pragma once

#include "ml/module.h"

#include <cstddef>
#include <vector>

struct parameter_group {
    std::vector<parameter*> params;
    float lr_scale = 1.0f;
    float weight_decay = 0.0f;
};

class optimizer {
public:
    optimizer() = default;
    explicit optimizer(std::vector<parameter*> parameters);
    explicit optimizer(module& model);
    virtual ~optimizer() = default;

    optimizer(const optimizer&) = delete;
    optimizer& operator=(const optimizer&) = delete;
    optimizer(optimizer&&) = default;
    optimizer& operator=(optimizer&&) = default;

    virtual void step() = 0;
    virtual void zero_grad() noexcept;

    void add_param_group(parameter_group group);

    const std::vector<parameter_group>& param_groups() const noexcept;
    std::size_t parameter_count() const noexcept;

protected:
    std::vector<parameter_group>& mutable_param_groups() noexcept;

private:
    static void normalize_group(parameter_group* group) noexcept;
    void recompute_parameter_count() noexcept;

    std::vector<parameter_group> groups_;
    std::size_t parameter_count_ = 0;
};
