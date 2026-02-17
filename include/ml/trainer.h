#pragma once

#include "ml/loss.h"
#include "ml/module.h"
#include "ml/optimizer.h"

#include <vector>

struct trainer_step_result {
    tensor loss;
    tensor_list outputs;
};

class trainer {
public:
    trainer(module& model, optimizer& opt, loss& loss_fn) noexcept;

    trainer_step_result train_batch(const tensor_list& inputs,
                                    const tensor_list& targets,
                                    bool retain_graph = false);

    tensor train_batch(const tensor& input,
                       const tensor& target,
                       bool retain_graph = false);

private:
    static tensor make_scalar_loss(float value);

    module& model_;
    optimizer& optimizer_;
    loss& loss_fn_;
};
