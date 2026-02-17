#include "ml/optimizer.h"

#include <unordered_set>
#include <vector>

optimizer::optimizer(std::vector<parameter*> parameters) {
    parameter_group group;
    group.params = std::move(parameters);
    add_param_group(std::move(group));
}

optimizer::optimizer(module& model)
    : optimizer(std::vector<parameter*>(model.parameters().begin(), model.parameters().end())) {}

void optimizer::zero_grad() noexcept {
    for(parameter_group& group : groups_){
        for(parameter* p : group.params){
            if(!p || !p->requires_grad)
                continue;
            p->zero_grad();
        }
    }
}

void optimizer::add_param_group(parameter_group group) {
    normalize_group(&group);
    if(group.params.empty())
        return;

    groups_.push_back(std::move(group));
    recompute_parameter_count();
}

const std::vector<parameter_group>& optimizer::param_groups() const noexcept {
    return groups_;
}

std::size_t optimizer::parameter_count() const noexcept {
    return parameter_count_;
}

std::vector<parameter_group>& optimizer::mutable_param_groups() noexcept {
    return groups_;
}

void optimizer::normalize_group(parameter_group* group) noexcept {
    if(!group)
        return;

    if(!(group->lr_scale > 0.0f))
        group->lr_scale = 1.0f;
    if(group->weight_decay < 0.0f)
        group->weight_decay = 0.0f;

    std::unordered_set<parameter*> seen;
    seen.reserve(group->params.size() * 2 + 1);

    std::vector<parameter*> normalized;
    normalized.reserve(group->params.size());

    for(parameter* p : group->params){
        if(!p)
            continue;
        if(!seen.insert(p).second)
            continue;
        normalized.push_back(p);
    }

    group->params = std::move(normalized);
}

void optimizer::recompute_parameter_count() noexcept {
    parameter_count_ = 0;
    for(const parameter_group& group : groups_)
        parameter_count_ += group.params.size();
}
