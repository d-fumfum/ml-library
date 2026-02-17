#include "ml/trainer.h"
#include "ml/utils.h"

#include <cstddef>
#include <vector>

namespace {
bool is_scalar_tensor(const tensor& t) {
    return t.s.row == 1 && t.s.col == 1 && t.s.layer == 1 && t.data.size() == 1;
}
}

trainer::trainer(module& model, optimizer& opt, loss& loss_fn) noexcept
    : model_(model), optimizer_(opt), loss_fn_(loss_fn) {}

trainer_step_result trainer::train_batch(const tensor_list& inputs,
                                         const tensor_list& targets,
                                         bool retain_graph) {
    trainer_step_result result;

    if(inputs.empty()){
        error_occured("trainer: inputs cannot be empty");
        result.loss = tsr_create(0, 0, 0);
        return result;
    }
    if(targets.empty()){
        error_occured("trainer: targets cannot be empty");
        result.loss = tsr_create(0, 0, 0);
        return result;
    }

    model_.train(true);
    optimizer_.zero_grad();

    result.outputs = model_(inputs);
    if(result.outputs.size() != targets.size()){
        error_occured("trainer: output/target count mismatch");
        result.loss = tsr_create(0, 0, 0);
        return result;
    }

    tensor_list output_grads;
    output_grads.reserve(result.outputs.size());

    float total_loss = 0.0f;
    for(std::size_t i = 0; i < result.outputs.size(); ++i){
        const tensor l = loss_fn_.forward(result.outputs[i], targets[i]);
        if(!is_scalar_tensor(l)){
            error_occured("trainer: loss must return scalar tensor");
            result.loss = tsr_create(0, 0, 0);
            return result;
        }

        total_loss += l.data[0];
        output_grads.push_back(loss_fn_.backward());
    }

    const tensor_list input_grads = model_.backward_graph(output_grads, retain_graph);
    if(input_grads.size() != inputs.size()){
        error_occured("trainer: backward input gradient count mismatch");
        result.loss = tsr_create(0, 0, 0);
        return result;
    }

    optimizer_.step();
    result.loss = make_scalar_loss(total_loss);
    return result;
}

tensor trainer::train_batch(const tensor& input,
                            const tensor& target,
                            bool retain_graph) {
    trainer_step_result result = train_batch(tensor_list{input}, tensor_list{target}, retain_graph);
    return result.loss;
}

tensor trainer::make_scalar_loss(float value) {
    tensor out = tsr_create(1, 1, 1);
    out.data[0] = value;
    return out;
}
