def poly_learning_rate(
        optimizer, base_lr, curr_iter, max_iter,
        power=0.9, index_split=-1, scale_lr=10.,
        warmup=False, warmup_step=500):

    if warmup and curr_iter < warmup_step:
        lr = base_lr * (0.1 + 0.9 * (curr_iter/warmup_step))
    else:
        lr = base_lr * (1 - float(curr_iter) / max_iter) ** power

    if curr_iter % 50 == 0:
        print(
            'Base LR: {:.4f}, Curr LR: {:.4f}, Warmup: {}.'.format(
                base_lr, lr, (warmup and curr_iter < warmup_step)
            )
        )

    for index, param_group in enumerate(optimizer.param_groups):
        if index <= index_split:
            param_group['lr'] = lr
        else:
            param_group['lr'] = lr * scale_lr
