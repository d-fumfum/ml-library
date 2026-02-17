#include "ml/module.h"
#include "ml/utils.h"

#include <algorithm>
#include <unordered_set>
#include <vector>

namespace {
thread_local std::vector<graph_step>* g_active_execution_tape = nullptr;

tensor make_empty_tensor() {
    return tsr_create(0, 0, 0);
}

tensor_list make_single_list(const tensor& value) {
    tensor_list out;
    out.push_back(value);
    return out;
}
}

void parameter::zero_grad() {
    if(!requires_grad)
        return;

    if(grad.s.row != data.s.row || grad.s.col != data.s.col || grad.s.layer != data.s.layer)
        grad = tsr_zeros_like(data);
    else
        tsr_zero(grad);
}

tensor module::forward(const tensor& input) {
    tensor_list outputs = forward_many(make_single_list(input));
    if(outputs.size() != 1){
        error_occured("forward(tensor) expected exactly 1 output tensor");
        return make_empty_tensor();
    }
    return outputs[0];
}

tensor module::backward(const tensor& grad_output) {
    tensor_list grad_inputs = backward_many(make_single_list(grad_output));
    if(grad_inputs.size() != 1){
        error_occured("backward(tensor) expected exactly 1 input gradient tensor");
        return make_empty_tensor();
    }
    return grad_inputs[0];
}

bool module::record_on_tape() const noexcept {
    return own_children_.empty();
}

tensor_list module::operator()(const tensor_list& inputs) {
    if(!training_){
        execution_tape_.clear();
        return forward_many(inputs);
    }

    const bool top_level_call = (g_active_execution_tape == nullptr);
    if(top_level_call){
        execution_tape_.clear();
        g_active_execution_tape = &execution_tape_;
    }

    tensor_list outputs = forward_many(inputs);

    if(g_active_execution_tape && record_on_tape()){
        g_active_execution_tape->push_back(graph_step{
            this,
            inputs.size(),
            outputs.size()
        });
    }

    if(top_level_call)
        g_active_execution_tape = nullptr;

    return outputs;
}

tensor module::operator()(const tensor& input) {
    tensor_list outputs = (*this)(make_single_list(input));
    if(outputs.size() != 1){
        error_occured("operator()(tensor) expected exactly 1 output tensor");
        return make_empty_tensor();
    }
    return outputs[0];
}

tensor_list module::backward_graph(const tensor_list& grad_outputs, bool retain_graph) {
    if(execution_tape_.empty())
        return backward_many(grad_outputs);

    tensor_list grad = grad_outputs;
    for(auto it = execution_tape_.rbegin(); it != execution_tape_.rend(); ++it){
        const graph_step& step = *it;
        if(!step.node)
            continue;

        if(step.output_count != 0 && grad.size() != step.output_count){
            error_occured("backward_graph gradient count mismatch for module output");
            return {};
        }

        grad = step.node->backward_many(grad);

        if(step.input_count != 0 && grad.size() != step.input_count){
            error_occured("backward_graph gradient count mismatch for module input");
            return {};
        }
    }

    if(!retain_graph)
        execution_tape_.clear();

    return grad;
}

tensor module::backward_graph(const tensor& grad_output, bool retain_graph) {
    tensor_list grad_inputs = backward_graph(make_single_list(grad_output), retain_graph);
    if(grad_inputs.size() != 1){
        error_occured("backward_graph(tensor) expected exactly 1 input gradient tensor");
        return make_empty_tensor();
    }
    return grad_inputs[0];
}

tensor_list module::train_step(const tensor_list& inputs,
                               const tensor_list& grad_outputs,
                               float learning_rate,
                               bool retain_graph) {
    if(!(learning_rate > 0.0f)){
        error_occured("learning_rate must be positive");
        return {};
    }

    zero_grad();
    tensor_list outputs = (*this)(inputs);
    tensor_list grad_inputs = backward_graph(grad_outputs, retain_graph);
    if(grad_inputs.size() != inputs.size()){
        error_occured("train_step aborted: backward/input gradient arity mismatch");
        return {};
    }

    sgd_step(learning_rate);
    return outputs;
}

tensor module::train_step(const tensor& input,
                          const tensor& grad_output,
                          float learning_rate,
                          bool retain_graph) {
    tensor_list outputs = train_step(make_single_list(input),
                                     make_single_list(grad_output),
                                     learning_rate,
                                     retain_graph);
    if(outputs.size() != 1){
        error_occured("train_step(tensor) expected exactly 1 output tensor");
        return make_empty_tensor();
    }
    return outputs[0];
}

void module::sgd_step(float learning_rate) {
    if(!(learning_rate > 0.0f)){
        error_occured("learning_rate must be positive");
        return;
    }

    const auto& params = parameters();
    for(parameter* p : params){
        if(!p || !p->requires_grad)
            continue;

        if(p->grad.s.row != p->data.s.row || p->grad.s.col != p->data.s.col || p->grad.s.layer != p->data.s.layer){
            error_occured("parameter gradient shape mismatch");
            continue;
        }

        const size_t n = p->data.data.size();
        if(p->grad.data.size() != n){
            error_occured("parameter gradient data size mismatch");
            continue;
        }

        for(size_t i = 0; i < n; ++i)
            p->data.data[i] -= learning_rate * p->grad.data[i];
    }
}

void module::train(bool mode) {
    std::vector<module*> stack;
    stack.push_back(this);

    std::unordered_set<module*> visited;
    visited.reserve(32);

    while(!stack.empty()){
        module* current = stack.back();
        stack.pop_back();
        if(!visited.insert(current).second)
            continue;

        current->training_ = mode;
        const auto& children = current->own_children_;
        for(auto it = children.rbegin(); it != children.rend(); ++it){
            if(*it)
                stack.push_back(*it);
        }
    }
}

void module::eval() {
    train(false);
}

bool module::is_training() const noexcept {
    return training_;
}

void module::zero_grad() {
    const auto& params = parameters();
    for(parameter* p : params){
        if(p)
            p->zero_grad();
    }
}

const std::vector<parameter*>& module::parameters() const {
    if(cache_dirty_)
        rebuild_parameter_cache();
    return flat_parameter_cache_;
}

void module::reserve_parameters(std::size_t count) {
    owned_parameters_storage_.reserve(count);
    own_parameters_.reserve(count);
    mark_graph_dirty();
}

void module::reserve_modules(std::size_t count) {
    owned_children_storage_.reserve(count);
    own_children_.reserve(count);
    mark_graph_dirty();
}

bool module::register_parameter(parameter* param) {
    if(!param){
        error_occured("null parameter pointer");
        return false;
    }

    if(std::find(own_parameters_.begin(), own_parameters_.end(), param) != own_parameters_.end())
        return false;

    own_parameters_.push_back(param);
    mark_graph_dirty();
    return true;
}

parameter* module::add_parameter(std::unique_ptr<parameter> param) {
    if(!param){
        error_occured("null parameter pointer");
        return nullptr;
    }

    parameter* raw = param.get();
    if(!register_parameter(raw))
        return nullptr;

    owned_parameters_storage_.push_back(std::move(param));
    return raw;
}

parameter* module::create_parameter(tensor data, bool requires_grad) {
    auto param = std::make_unique<parameter>();
    param->requires_grad = requires_grad;
    param->data = std::move(data);
    param->grad = tsr_zeros_like(param->data);
    return add_parameter(std::move(param));
}

bool module::register_module(module* child) {
    if(!child){
        error_occured("null child module pointer");
        return false;
    }
    if(child == this){
        error_occured("cannot register module as its own child");
        return false;
    }
    if(std::find(own_children_.begin(), own_children_.end(), child) != own_children_.end())
        return false;

    std::vector<const module*> stack;
    stack.push_back(child);

    std::unordered_set<const module*> visited;
    visited.reserve(32);

    while(!stack.empty()){
        const module* current = stack.back();
        stack.pop_back();
        if(!visited.insert(current).second)
            continue;
        if(current == this){
            error_occured("module graph cycle detected");
            return false;
        }

        for(const module* next : current->own_children_){
            if(next)
                stack.push_back(next);
        }
    }

    own_children_.push_back(child);
    child->link_parent(this);
    mark_graph_dirty();
    return true;
}

module* module::add_module(std::unique_ptr<module> child) {
    if(!child){
        error_occured("null child module pointer");
        return nullptr;
    }

    module* raw = child.get();
    if(!register_module(raw))
        return nullptr;

    owned_children_storage_.push_back(std::move(child));
    return raw;
}

void module::mark_graph_dirty() noexcept {
    std::vector<module*> stack;
    stack.push_back(this);

    std::unordered_set<module*> visited;
    visited.reserve(32);

    while(!stack.empty()){
        module* current = stack.back();
        stack.pop_back();
        if(!visited.insert(current).second)
            continue;

        current->cache_dirty_ = true;
        for(module* parent : current->parents_){
            if(parent)
                stack.push_back(parent);
        }
    }
}

void module::link_parent(module* parent) {
    if(!parent)
        return;

    if(std::find(parents_.begin(), parents_.end(), parent) == parents_.end())
        parents_.push_back(parent);
}

void module::rebuild_parameter_cache() const {
    flat_parameter_cache_.clear();

    std::vector<const module*> stack;
    stack.push_back(this);

    std::unordered_set<const module*> visited_modules;
    visited_modules.reserve(32);

    std::unordered_set<const parameter*> seen_parameters;
    seen_parameters.reserve(64);

    while(!stack.empty()){
        const module* current = stack.back();
        stack.pop_back();
        if(!visited_modules.insert(current).second)
            continue;

        for(parameter* p : current->own_parameters_){
            if(!p)
                continue;
            if(seen_parameters.insert(p).second)
                flat_parameter_cache_.push_back(p);
        }

        const auto& children = current->own_children_;
        for(auto it = children.rbegin(); it != children.rend(); ++it){
            if(*it)
                stack.push_back(*it);
        }
    }

    cache_dirty_ = false;
}
