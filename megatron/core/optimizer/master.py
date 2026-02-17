def step(param, config):
    grad = param.grad

    if project_gradient:
        grad = project(grad, param)

    direction = update_direction()

    if orthogonalize_direction:
        direction = orthogonalize(direction)

    if weight_decay > 0 and decouple_weight_decay:
        direction = lr*direction + weight_decay*param
    elif weight_decay > 0:
        direction = lr*(direction + param)
    else:
        direction = lr*direction

    param = param - direction

    if normalize_weights:
        param = normalize(param)
