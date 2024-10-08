def grad_update(old_grad, new_grad):
    assert old_grad is not None
    # print(f"grad_update \n    {old_grad=}\n    {new_grad=}")
    old_grad.add_(new_grad)
    return old_grad