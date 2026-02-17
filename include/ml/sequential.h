#pragma once

#include "ml/module.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

class sequential final : public module {
public:
    sequential() = default;

    void reserve(std::size_t count);
    bool add(std::unique_ptr<module> layer);

    template<typename ModuleT, typename... Args>
    ModuleT* emplace(Args&&... args) {
        auto layer = std::make_unique<ModuleT>(std::forward<Args>(args)...);
        ModuleT* raw = layer.get();
        if(!add(std::move(layer)))
            return nullptr;
        return raw;
    }

    std::size_t size() const noexcept;
    bool empty() const noexcept;

    module* at(std::size_t index) noexcept;
    const module* at(std::size_t index) const noexcept;

    tensor_list forward_many(const tensor_list& inputs) override;
    tensor_list backward_many(const tensor_list& grad_outputs) override;
    bool record_on_tape() const noexcept override;

private:
    std::vector<module*> layers_;
};
