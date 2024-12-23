import torch

def compare(tensor1, tensor2):
    if tensor1 is None or tensor2 is None:
        return tensor1 is None and tensor2 is None
    if type(tensor1)==tuple:
        assert type(tensor2)==tuple
        assert len(tensor1)==len(tensor2)
        for t1,t2 in zip(tensor1, tensor2):
            return compare(t1, t2)
    are_close = torch.allclose(tensor1, tensor2, atol=1e-3, rtol=1e-3)
    print(f"    ({tensor1.shape}, {tensor1.dtype}){tensor1}")
    print(f"    ({tensor2.shape}, {tensor2.dtype}){tensor2}")
    return are_close


def main():
    for name in [
        # "input",
        "input_layernorm_output",
        "sa_hidden_states",
        "sa_qkv_weight",
        "sa_qkv_ln_weight",
        # "mixed_qkv_2",
        # "split_query",
        # "split_key",
        # "split_value",
        # "query",
        # "key",
        # "value",
        # "attention_output_with_bias"
    ]:
        path1 = f"save_hiddens/megatron/rank0/layer1/{name}.pth"
        path2 = f"save_hiddens/wyo/rank0/layer1/{name}.pth"
        tensor1 = torch.load(path1, map_location="cuda")
        tensor2 = torch.load(path2, map_location="cuda")
        print(f"\nCompare {name}: \n  {path1=},\n  {path2=}")
        result = compare(tensor1, tensor2)
        print(f"  result: {result}")
        # print(f"    {tensor1}")
        # print(f"    {tensor2}")

def show_stats(big_dir, verbose=False):
    for layer in range(1,40):
        path = f"save_hiddens/{big_dir}/rank0/layer{layer}/input_layernorm_output.pth"
        tensor = torch.load(path, map_location="cuda")
        max_val = tensor.max().item()
        min_val = tensor.min().item()
        mean_val = tensor.mean().item()
        if verbose:
            print(f"{path=}")
            print(f"    {max_val=}")
            print(f"    {min_val=}")
            print(f"    {mean_val=}")
        else:
            print(f"{layer} {max_val=}")

if __name__=="__main__":
    # show_stats("megatron")
    # print("---------------------")
    show_stats("wyo")