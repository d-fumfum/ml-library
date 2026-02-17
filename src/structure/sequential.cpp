#include "ml/sequential.h"
#include "ml/utils.h"

void sequential::reserve(std::size_t count) {
    layers_.reserve(count);
    reserve_modules(count);
}

bool sequential::add(std::unique_ptr<module> layer) {
    if(!layer){
        error_occured("sequential::add received null layer");
        return false;
    }

    module* raw = add_module(std::move(layer));
    if(!raw)
        return false;

    layers_.push_back(raw);
    return true;
}

std::size_t sequential::size() const noexcept {
    return layers_.size();
}

bool sequential::empty() const noexcept {
    return layers_.empty();
}

module* sequential::at(std::size_t index) noexcept {
    if(index >= layers_.size())
        return nullptr;
    return layers_[index];
}

const module* sequential::at(std::size_t index) const noexcept {
    if(index >= layers_.size())
        return nullptr;
    return layers_[index];
}

tensor_list sequential::forward_many(const tensor_list& inputs) {
    if(layers_.empty())
        return inputs;

    tensor_list current = inputs;
    for(module* layer : layers_){
        if(!layer){
            error_occured("sequential contains null layer");
            return {};
        }
        current = (*layer)(current);
        if(current.empty()){
            error_occured("sequential layer forward returned no outputs");
            return {};
        }
    }

    return current;
}

tensor_list sequential::backward_many(const tensor_list& grad_outputs) {
    if(layers_.empty())
        return grad_outputs;

    tensor_list current = grad_outputs;
    for(auto it = layers_.rbegin(); it != layers_.rend(); ++it){
        module* layer = *it;
        if(!layer){
            error_occured("sequential contains null layer");
            return {};
        }
        current = layer->backward_many(current);
        if(current.empty()){
            error_occured("sequential layer backward returned no gradients");
            return {};
        }
    }

    return current;
}

bool sequential::record_on_tape() const noexcept {
    return false;
}
