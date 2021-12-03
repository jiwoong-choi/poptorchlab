# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import poptorch


def get_optimizer(args, model):
    regularized_params = []
    non_regularized_params = []

    # Filter biases and norm parameters.
    for param in model.parameters():
        if param.requires_grad:
            if len(param.shape) == 1:
                non_regularized_params.append(param)
            else:
                regularized_params.append(param)

    params = [
        {'params': regularized_params, 'weight_decay': args.weight_decay},
        {'params': non_regularized_params, 'weight_decay': 0}
    ]

    optimizer_class = getattr(poptorch.optim, args.optimizer)
    optimizer = optimizer_class(params, lr=args.lr, **args.optimizer_kwargs)

    return optimizer
