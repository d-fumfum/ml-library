#pragma once

#include "ml/module.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class dense;

class model final : public module {
public:
    using named_module = std::pair<std::string, module*>;
    using named_module_const = std::pair<std::string, const module*>;
    using named_parameter = std::pair<std::string, parameter*>;
    using named_parameter_const = std::pair<std::string, const parameter*>;

    model() = default;

    void reserve(std::size_t count);
    bool add(std::string name, std::unique_ptr<module> child);

    template<typename ModuleT, typename... Args>
    ModuleT* emplace(std::string name, Args&&... args) {
        auto child = std::make_unique<ModuleT>(std::forward<Args>(args)...);
        ModuleT* raw = child.get();
        if(!add(std::move(name), std::move(child)))
            return nullptr;
        return raw;
    }

    dense* add_dense(std::string name, int in_features, int out_features, bool use_bias = true);

    std::size_t size() const noexcept;
    bool empty() const noexcept;
    module* at(std::size_t index) noexcept;
    const module* at(std::size_t index) const noexcept;
    const std::string* name_at(std::size_t index) const noexcept;

    module* find_module(const std::string& name) noexcept;
    const module* find_module(const std::string& name) const noexcept;

    std::vector<named_module> named_modules() noexcept;
    std::vector<named_module_const> named_modules() const noexcept;
    std::vector<named_parameter> named_parameters() noexcept;
    std::vector<named_parameter_const> named_parameters() const noexcept;

    bool save_state_dict(const std::string& path) const;
    bool load_state_dict(const std::string& path, bool strict = true);

    tensor_list forward_many(const tensor_list& inputs) override;
    tensor_list backward_many(const tensor_list& grad_outputs) override;
    bool record_on_tape() const noexcept override;

private:
    struct module_entry {
        std::string name;
        module* ptr = nullptr;
    };

    static bool valid_module_name(const std::string& name) noexcept;
    bool has_name_conflict(const std::string& name) const noexcept;

    std::vector<module_entry> modules_;
};
