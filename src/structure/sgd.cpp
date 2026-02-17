#include "ml/sgd.h"
#include "ml/utils.h"

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace {
sgd_options sanitize_options(sgd_options options) noexcept {
    if(!(options.lr > 0.0f))
        options.lr = 1e-3f;
    if(options.momentum < 0.0f)
        options.momentum = 0.0f;
    if(options.dampening < 0.0f)
        options.dampening = 0.0f;
    if(options.weight_decay < 0.0f)
        options.weight_decay = 0.0f;

    if(options.nesterov && options.momentum <= 0.0f)
        options.nesterov = false;
    if(options.nesterov && options.dampening != 0.0f)
        options.dampening = 0.0f;

    return options;
}

bool same_shape(const tensor& a, const tensor& b) {
    return a.s.row == b.s.row && a.s.col == b.s.col && a.s.layer == b.s.layer;
}
}

sgd::sgd(std::vector<parameter*> parameters, sgd_options options)
    : optimizer(std::move(parameters)), opts_(sanitize_options(options)) {}

sgd::sgd(module& model, sgd_options options)
    : optimizer(model), opts_(sanitize_options(options)) {}

void sgd::step() {
    auto& groups = mutable_param_groups();
    if(groups.empty())
        return;

    for(parameter_group& group : groups){
        const float group_lr = opts_.lr * group.lr_scale;
        if(!(group_lr > 0.0f))
            continue;

        const float weight_decay = opts_.weight_decay + group.weight_decay;

        for(parameter* p : group.params){
            if(!p || !p->requires_grad)
                continue;

            if(!same_shape(p->data, p->grad)){
                error_occured("sgd: data/grad shape mismatch");
                continue;
            }

            const std::size_t n = p->data.data.size();
            if(p->grad.data.size() != n){
                error_occured("sgd: data/grad size mismatch");
                continue;
            }
            if(n == 0)
                continue;

            float* data_ptr = p->data.data.data();
            const float* grad_ptr = p->grad.data.data();

            if(opts_.momentum > 0.0f){
                tensor* velocity = ensure_velocity(p);
                if(!velocity){
                    error_occured("sgd: unable to allocate momentum buffer");
                    continue;
                }

                float* v = velocity->data.data();
                const float momentum = opts_.momentum;
                const float damp = opts_.dampening;
                const float grad_factor = 1.0f - damp;

                if(opts_.nesterov){
                    for(std::size_t i = 0; i < n; ++i){
                        float g = grad_ptr[i];
                        if(weight_decay != 0.0f)
                            g += weight_decay * data_ptr[i];

                        v[i] = momentum * v[i] + grad_factor * g;
                        const float update = g + momentum * v[i];
                        if(opts_.maximize)
                            data_ptr[i] += group_lr * update;
                        else
                            data_ptr[i] -= group_lr * update;
                    }
                    continue;
                }

                for(std::size_t i = 0; i < n; ++i){
                    float g = grad_ptr[i];
                    if(weight_decay != 0.0f)
                        g += weight_decay * data_ptr[i];

                    v[i] = momentum * v[i] + grad_factor * g;
                    if(opts_.maximize)
                        data_ptr[i] += group_lr * v[i];
                    else
                        data_ptr[i] -= group_lr * v[i];
                }
                continue;
            }

            for(std::size_t i = 0; i < n; ++i){
                float g = grad_ptr[i];
                if(weight_decay != 0.0f)
                    g += weight_decay * data_ptr[i];

                if(opts_.maximize)
                    data_ptr[i] += group_lr * g;
                else
                    data_ptr[i] -= group_lr * g;
            }
        }
    }
}

const sgd_options& sgd::options() const noexcept {
    return opts_;
}

void sgd::set_options(sgd_options options) noexcept {
    opts_ = sanitize_options(options);
}

void sgd::clear_state() noexcept {
    velocity_.clear();
}

tensor* sgd::ensure_velocity(parameter* param) {
    if(!param)
        return nullptr;

    auto it = velocity_.find(param);
    if(it == velocity_.end())
        it = velocity_.emplace(param, tsr_zeros_like(param->data)).first;

    if(!same_shape(it->second, param->data))
        it->second = tsr_zeros_like(param->data);

    return &it->second;
}
