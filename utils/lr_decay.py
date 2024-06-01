def adjust_lr(optimizer, init_lr, epoch, num_epochs, power=2):
    lr = init_lr * (1 - epoch / num_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
