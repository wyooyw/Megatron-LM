from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
import torch
if __name__=="__main__":
    # with FakeTensorMode():
    #     x = torch.Tensor([1, 2, 3])
    # print(x)
    # print(type(x))
    # y = x.clone()
    # print(y)
    # transpose_3 = torch.randn((2,2,2,2), dtype=torch.float32, device="cuda")
    # transpose_3 = transpose_3.permute(3,2,1,0)

    # cur_shape = [16]
    # # stride = [1] * len(cur_shape)
    # # for i in range(len(cur_shape) - 2, -1, -1):
    # #     stride[i] = stride[i+1] * cur_shape[i+1]
    # # print(stride)
    # transpose_3 = torch.ops.aten.as_strided(transpose_3, cur_shape, (1,),0)
    # print(transpose_3.is_contiguous())
    # view_96 = torch.ops.aten.view(transpose_3, torch.bfloat16)
    # view_96 = torch.ops.aten.view(view_96, (4,8))
    # # resize__10 = torch.ops.aten.resize_(view_96, (0,))
    # # resize__10 = view_96.reshape(4,8)
    # print(view_96.shape)
    # print(view_96.numel())

    a = torch.zeros((4,4), device="cuda")
    b = torch.ops.aten.unsqueeze(a, 2)
    b[0,0] = 1
    print(a)