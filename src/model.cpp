#include "ml/model.h"

#include "ml/abstraction.h"
#include "ml/utils.h"

#include <cctype>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
constexpr const char* k_state_dict_magic = "ML_STATE_DICT_V1";

std::string make_parameter_name(const std::string& module_name, std::size_t index) {
    std::string name = module_name;
    name += ".param_";
    name += std::to_string(index);
    return name;
}
}

void model::reserve(std::size_t count) {
    modules_.reserve(count);
    reserve_modules(count);
}

bool model::add(std::string name, std::unique_ptr<module> child) {
    if(!child){
        error_occured("model::add received null child module");
        return false;
    }
    if(!valid_module_name(name)){
        error_occured("model::add received invalid module name");
        return false;
    }
    if(has_name_conflict(name)){
        error_occured("model::add module name already exists");
        return false;
    }

    module* raw = add_module(std::move(child));
    if(!raw)
        return false;

    modules_.push_back(module_entry{std::move(name), raw});
    return true;
}

dense* model::add_dense(std::string name, int in_features, int out_features, bool use_bias) {
    return emplace<dense>(std::move(name), in_features, out_features, use_bias);
}

std::size_t model::size() const noexcept {
    return modules_.size();
}

bool model::empty() const noexcept {
    return modules_.empty();
}

module* model::at(std::size_t index) noexcept {
    if(index >= modules_.size())
        return nullptr;
    return modules_[index].ptr;
}

const module* model::at(std::size_t index) const noexcept {
    if(index >= modules_.size())
        return nullptr;
    return modules_[index].ptr;
}

const std::string* model::name_at(std::size_t index) const noexcept {
    if(index >= modules_.size())
        return nullptr;
    return &modules_[index].name;
}

module* model::find_module(const std::string& name) noexcept {
    for(module_entry& entry : modules_){
        if(entry.name == name)
            return entry.ptr;
    }
    return nullptr;
}

const module* model::find_module(const std::string& name) const noexcept {
    for(const module_entry& entry : modules_){
        if(entry.name == name)
            return entry.ptr;
    }
    return nullptr;
}

std::vector<model::named_module> model::named_modules() noexcept {
    std::vector<named_module> out;
    out.reserve(modules_.size());
    for(module_entry& entry : modules_)
        out.push_back(named_module{entry.name, entry.ptr});
    return out;
}

std::vector<model::named_module_const> model::named_modules() const noexcept {
    std::vector<named_module_const> out;
    out.reserve(modules_.size());
    for(const module_entry& entry : modules_)
        out.push_back(named_module_const{entry.name, entry.ptr});
    return out;
}

std::vector<model::named_parameter> model::named_parameters() noexcept {
    std::vector<named_parameter> out;
    std::unordered_set<parameter*> seen;
    seen.reserve(parameters().size() * 2 + 1);

    for(module_entry& entry : modules_){
        if(!entry.ptr)
            continue;

        const auto& params = entry.ptr->parameters();
        for(std::size_t i = 0; i < params.size(); ++i){
            parameter* p = params[i];
            if(!p)
                continue;
            if(!seen.insert(p).second)
                continue;
            out.push_back(named_parameter{make_parameter_name(entry.name, i), p});
        }
    }

    return out;
}

std::vector<model::named_parameter_const> model::named_parameters() const noexcept {
    std::vector<named_parameter_const> out;
    std::unordered_set<const parameter*> seen;
    seen.reserve(parameters().size() * 2 + 1);

    for(const module_entry& entry : modules_){
        if(!entry.ptr)
            continue;

        const auto& params = entry.ptr->parameters();
        for(std::size_t i = 0; i < params.size(); ++i){
            const parameter* p = params[i];
            if(!p)
                continue;
            if(!seen.insert(p).second)
                continue;
            out.push_back(named_parameter_const{make_parameter_name(entry.name, i), p});
        }
    }

    return out;
}

bool model::save_state_dict(const std::string& path) const {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if(!out){
        error_occured("model::save_state_dict unable to open output file");
        return false;
    }

    const auto params = named_parameters();
    out << k_state_dict_magic << '\n';
    out << params.size() << '\n';
    out << std::setprecision(9);

    for(const named_parameter_const& named : params){
        const parameter* p = named.second;
        if(!p){
            error_occured("model::save_state_dict encountered null parameter");
            return false;
        }

        const tensor& t = p->data;
        out << named.first << '\n';
        out << (p->requires_grad ? 1 : 0) << ' '
            << t.s.row << ' ' << t.s.col << ' ' << t.s.layer << ' '
            << t.data.size() << '\n';

        for(std::size_t i = 0; i < t.data.size(); ++i){
            if(i != 0)
                out << ' ';
            out << t.data[i];
        }
        out << '\n';
    }

    if(!out){
        error_occured("model::save_state_dict failed while writing");
        return false;
    }
    return true;
}

bool model::load_state_dict(const std::string& path, bool strict) {
    std::ifstream in(path);
    if(!in){
        error_occured("model::load_state_dict unable to open input file");
        return false;
    }

    std::string magic;
    if(!(in >> magic) || magic != k_state_dict_magic){
        error_occured("model::load_state_dict invalid checkpoint format");
        return false;
    }

    std::size_t record_count = 0;
    if(!(in >> record_count)){
        error_occured("model::load_state_dict failed to read checkpoint count");
        return false;
    }

    const auto named = named_parameters();
    std::unordered_map<std::string, parameter*> table;
    table.reserve(named.size() * 2 + 1);
    for(const named_parameter& item : named){
        if(!item.second){
            error_occured("model::load_state_dict encountered null parameter slot");
            return false;
        }
        if(!table.emplace(item.first, item.second).second){
            error_occured("model::load_state_dict duplicate parameter name");
            return false;
        }
    }

    std::unordered_set<std::string> loaded;
    loaded.reserve(record_count * 2 + 1);

    for(std::size_t rec = 0; rec < record_count; ++rec){
        std::string name;
        int requires_grad = 0;
        int rows = 0;
        int cols = 0;
        int layers = 0;
        std::size_t data_count = 0;

        if(!(in >> name >> requires_grad >> rows >> cols >> layers >> data_count)){
            error_occured("model::load_state_dict failed to read checkpoint record header");
            return false;
        }

        std::vector<float> values(data_count);
        for(std::size_t i = 0; i < data_count; ++i){
            if(!(in >> values[i])){
                error_occured("model::load_state_dict failed to read checkpoint parameter payload");
                return false;
            }
        }

        auto it = table.find(name);
        if(it == table.end()){
            if(strict){
                error_occured("model::load_state_dict found unknown parameter name");
                return false;
            }
            continue;
        }

        if(!loaded.insert(name).second){
            error_occured("model::load_state_dict duplicate parameter record");
            return false;
        }

        parameter* p = it->second;
        if(!p){
            error_occured("model::load_state_dict null destination parameter");
            return false;
        }

        if(rows < 0 || cols < 0 || layers < 0){
            error_occured("model::load_state_dict negative tensor shape in checkpoint");
            return false;
        }

        const std::size_t expected_count =
            static_cast<std::size_t>(rows) *
            static_cast<std::size_t>(cols) *
            static_cast<std::size_t>(layers);
        if(expected_count != data_count){
            error_occured("model::load_state_dict invalid tensor payload size");
            return false;
        }

        if(p->data.s.row != rows || p->data.s.col != cols || p->data.s.layer != layers){
            error_occured("model::load_state_dict parameter shape mismatch");
            return false;
        }

        p->requires_grad = (requires_grad != 0);
        p->data.data = std::move(values);
        p->grad = tsr_zeros_like(p->data);
    }

    if(strict && loaded.size() != table.size()){
        error_occured("model::load_state_dict missing parameter records");
        return false;
    }

    return true;
}

tensor_list model::forward_many(const tensor_list& inputs) {
    if(modules_.empty())
        return inputs;

    tensor_list current = inputs;
    for(module_entry& entry : modules_){
        if(!entry.ptr){
            error_occured("model contains null module");
            return {};
        }
        current = (*entry.ptr)(current);
        if(current.empty()){
            error_occured("model forward returned empty tensor list");
            return {};
        }
    }

    return current;
}

tensor_list model::backward_many(const tensor_list& grad_outputs) {
    if(modules_.empty())
        return grad_outputs;

    tensor_list current = grad_outputs;
    for(auto it = modules_.rbegin(); it != modules_.rend(); ++it){
        if(!it->ptr){
            error_occured("model contains null module");
            return {};
        }
        current = it->ptr->backward_many(current);
        if(current.empty()){
            error_occured("model backward returned empty gradient list");
            return {};
        }
    }

    return current;
}

bool model::record_on_tape() const noexcept {
    return false;
}

bool model::valid_module_name(const std::string& name) noexcept {
    if(name.empty())
        return false;

    for(char ch : name){
        if(std::isspace(static_cast<unsigned char>(ch)))
            return false;
    }
    return true;
}

bool model::has_name_conflict(const std::string& name) const noexcept {
    for(const module_entry& entry : modules_){
        if(entry.name == name)
            return true;
    }
    return false;
}