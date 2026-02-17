#pragma once

#include "ml/tensor.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

class module;
using tensor_list = std::vector<tensor>;

struct graph_step {
    module* node = nullptr;
    std::size_t input_count = 0;
    std::size_t output_count = 0;
};

struct parameter {
    tensor data;
    tensor grad;
    bool requires_grad = true;

    void zero_grad();
};

class module {
public:
    virtual ~module() = default;
    module() = default;
    module(const module&) = delete;
    module& operator=(const module&) = delete;
    module(module&&) = delete;
    module& operator=(module&&) = delete;

    virtual tensor_list forward_many(const tensor_list& inputs) = 0;
    virtual tensor_list backward_many(const tensor_list& grad_outputs) = 0;
    virtual bool record_on_tape() const noexcept;

    tensor forward(const tensor& input);
    tensor backward(const tensor& grad_output);
    tensor_list operator()(const tensor_list& inputs);
    tensor operator()(const tensor& input);
    tensor_list backward_graph(const tensor_list& grad_outputs, bool retain_graph = false);
    tensor backward_graph(const tensor& grad_output, bool retain_graph = false);
    tensor_list train_step(const tensor_list& inputs,
                           const tensor_list& grad_outputs,
                           float learning_rate,
                           bool retain_graph = false);
    tensor train_step(const tensor& input,
                      const tensor& grad_output,
                      float learning_rate,
                      bool retain_graph = false);
    void sgd_step(float learning_rate);

    void train(bool mode = true);
    void eval();
    bool is_training() const noexcept;

    void zero_grad();

    const std::vector<parameter*>& parameters() const;

    void reserve_parameters(std::size_t count);
    void reserve_modules(std::size_t count);

protected:
    bool register_parameter(parameter* param);
    bool register_module(module* child);

    parameter* add_parameter(std::unique_ptr<parameter> param);
    parameter* create_parameter(tensor data, bool requires_grad = true);
    module* add_module(std::unique_ptr<module> child);

    template<typename ModuleT, typename... Args>
    ModuleT* emplace_module(Args&&... args) {
        auto child = std::make_unique<ModuleT>(std::forward<Args>(args)...);
        ModuleT* raw = child.get();
        module* added = add_module(std::move(child));
        if(added != raw)
            return nullptr;
        return raw;
    }

private:
    void mark_graph_dirty() noexcept;
    void link_parent(module* parent);
    void rebuild_parameter_cache() const;

    bool training_ = true;
    std::vector<std::unique_ptr<parameter>> owned_parameters_storage_;
    std::vector<std::unique_ptr<module>> owned_children_storage_;
    std::vector<parameter*> own_parameters_;
    std::vector<module*> own_children_;
    std::vector<module*> parents_;

    mutable std::vector<graph_step> execution_tape_;
    mutable bool cache_dirty_ = true;
    mutable std::vector<parameter*> flat_parameter_cache_;
};
