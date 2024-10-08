import torch
import torch.utils._pytree as pytree


class ForwardInputs:
    def __init__(self, parmeters, inputs):
        self.parmeters = parmeters
        self.inputs = inputs

    def dump(self, first_none=True):
        returns = [*self.parmeters, *self.inputs]
        if first_none:
            returns = [None, *returns]
        return returns

    def show(self):
        for idx,tensor in enumerate(self.dump()):
            if type(tensor)==torch.Tensor or type(tensor)==torch.nn.parameter.Parameter:
                print(f"{idx} {tensor.dtype=}, {tensor.shape=}")
            else:
                print(f"{idx} {type(tensor)=}")
                

class ForwardOutputs:
    def __init__(self, model_outputs, saved_for_backwards):
        self.model_outputs = model_outputs
        self.saved_for_backwards = saved_for_backwards

    def get_model_outputs(self):
        return self.model_outputs

    def get_saved_for_backwards(self):
        return self.saved_for_backwards

    def dump(self, first_none=True):
        returns = [*self.model_outputs, *self.saved_for_backwards]
        if first_none:
            returns = [None, *returns]
        return returns


class BackwardInputs:
    def __init__(self, save_for_backwards, output_grads, parameters_grad):
        self.save_for_backwards = save_for_backwards
        self.output_grads = output_grads
        self.parameters_grad = parameters_grad

    # @staticmethod
    # def make_from_forward_outputs(forward_outputs:ForwardOutputs, parameters_grad):
    #     origin_outputs = forward_outputs.get_model_outputs()
    #     save_for_backwards = forward_outputs.get_saved_for_backwards()

    #     output_grads = []
    #     for ori_out in origin_outputs:
    #         out_grad = torch.ones(ori_out.shape, dtype=ori_out.dtype, device=ori_out.device)
    #         output_grads.append(out_grad)

    #     return BackwardInputs(save_for_backwards, output_grads, parameters_grad)

    def dump(self, first_none=True):
        returns = [*self.save_for_backwards, *self.output_grads, *self.parameters_grad]
        if first_none:
            returns = [None, *returns]
        return returns


class BackwardOutputs:
    def __init__(self, parameters_grad, inputs_grad):
        self.parameters_grad = parameters_grad
        self.inputs_grad = inputs_grad


class BackwardForwardInputs:
    def __init__(self, backward_inputs: BackwardInputs, forward_inputs: ForwardInputs):
        assert type(backward_inputs) == BackwardInputs, f"{type(backward_inputs)=}"
        assert type(forward_inputs) == ForwardInputs, f"{type(forward_inputs)=}"
        self.backward_inputs = backward_inputs
        self.forward_inputs = forward_inputs

    # @staticmethod
    # def make_from_fwd_outs_ins(last_step_forward_outputs:ForwardOutputs, this_step_forward_inputs:ForwardInputs):
    #     backward_inputs = BackwardInputs.make_from_forward_outputs(last_step_forward_outputs)
    #     return BackwardForwardInputs(backward_inputs, this_step_forward_inputs)

    def dump(self, first_none=True):
        backward_inputs = self.backward_inputs.dump(first_none=False)
        forward_inputs = self.forward_inputs.dump(first_none=False)

        returns = [*backward_inputs, *forward_inputs]
        if first_none:
            returns = [None, *returns]
        return returns


class BackwardForwardOutputs:
    def __init__(
        self, backward_outputs: BackwardOutputs, forward_outputs: ForwardOutputs
    ):
        self.backward_outputs = backward_outputs
        self.forward_outputs = forward_outputs


class ForwardForwardInputs:
    def __init__(self, forward_inputs_0, forward_inputs_1):
        self.forward_inputs_0 = forward_inputs_0
        self.forward_inputs_1 = forward_inputs_1

    def dump(self, first_none=True):
        forward_inputs_0 = self.forward_inputs_0.dump(first_none=False)
        forward_inputs_1 = self.forward_inputs_1.dump(first_none=False)

        returns = [*forward_inputs_0, *forward_inputs_1]
        if first_none:
            returns = [None, *returns]
        return returns


class ForwardForwardOutputs:
    def __init__(self, forward_outputs_0, forward_outputs_1):
        self.forward_outputs_0 = forward_outputs_0
        self.forward_outputs_1 = forward_outputs_1


class BackwardBackwardInputs:
    def __init__(self, backward_inputs_0, backward_inputs_1):
        self.backward_inputs_0 = backward_inputs_0
        self.backward_inputs_1 = backward_inputs_1

    def dump(self, first_none=True):
        backward_inputs_0 = self.backward_inputs_0.dump(first_none=False)
        backward_inputs_1 = self.backward_inputs_1.dump(first_none=False)

        returns = [*backward_inputs_0, *backward_inputs_1]
        if first_none:
            returns = [None, *returns]
        return returns


class BakwardBackwardOutputs:
    def __init__(self, backward_outputs_0, backward_outputs_1):
        self.backward_outputs_0 = backward_outputs_0
        self.backward_outputs_1 = backward_outputs_1


class ModelArgsAndReturnsManager:
    def __init__(self, model, params_flat, n_model_outputs, n_model_inputs):
        self.model = model
        self.params_flat = params_flat
        self.n_params = len(self.params_flat)
        self.n_model_outputs = n_model_outputs
        self.n_model_inputs = n_model_inputs
        # self._make_params()

    def _make_params(self):
        params = {
            **dict(self.model.named_parameters(remove_duplicate=False)),
            **dict(self.model.named_buffers(remove_duplicate=False)),
        }
        print("ModelArgsAndReturnsManager parameters:")
        for idx,(key,value) in enumerate(params.items()):
            print(idx, key)
        params_flat, params_spec = pytree.tree_flatten(params)
        params_flat = list(params_flat)
        self.params_flat = params_flat
        self.n_params = len(self.params_flat)

    def _make_grads(self):
        grads = []
        for param in self.params_flat:
            grads.append(param.grad)
        return grads

    def make_forward_inputs(self, inputs):
        """
        arg:
            inputs
        return:
            forward_inputs: ForwardInputs
        """
        return ForwardInputs(self.params_flat, inputs)

    def make_forward_outputs(self, forward_outputs_list):
        model_outputs = forward_outputs_list[: self.n_model_outputs]
        saved_for_backwards = forward_outputs_list[self.n_model_outputs :]
        forward_outputs = ForwardOutputs(model_outputs, saved_for_backwards)
        return forward_outputs

    def make_backward_inputs(self, forward_outputs: ForwardOutputs):
        origin_outputs = forward_outputs.get_model_outputs()
        save_for_backwards = forward_outputs.get_saved_for_backwards()

        output_grads = []
        for ori_out in origin_outputs:
            out_grad = torch.ones(
                ori_out.shape, dtype=ori_out.dtype, device=ori_out.device
            )
            output_grads.append(out_grad)

        parameters_grad = self._make_grads()

        return BackwardInputs(save_for_backwards, output_grads, parameters_grad)

    def make_backward_outputs(self, backward_outputs_list):
        parameters_grad = backward_outputs_list[: self.n_params]
        inputs_grad = backward_outputs_list[self.n_params :]
        backward_outputs = BackwardOutputs(parameters_grad, inputs_grad)
        return backward_outputs

    def make_backward_forward_inputs(
        self,
        last_step_forward_outputs: ForwardOutputs,
        this_step_forward_inputs: ForwardInputs,
    ):
        backward_inputs = self.make_backward_inputs(last_step_forward_outputs)
        return BackwardForwardInputs(backward_inputs, this_step_forward_inputs)

    def make_backward_forward_outputs(self, outputs):
        forward_input_len = self.n_model_inputs + self.n_params

        backward_outputs_list = outputs[:forward_input_len]
        backward_outputs = self.make_backward_outputs(backward_outputs_list)

        forward_outputs_list = outputs[forward_input_len:]
        forward_outputs = self.make_forward_outputs(forward_outputs_list)

        return BackwardForwardOutputs(backward_outputs, forward_outputs)

    def make_forward_forward_inputs(self, inputs_0, inputs_1):
        return ForwardForwardInputs(
            self.make_forward_inputs(inputs_0), self.make_forward_inputs(inputs_1)
        )

    def make_forward_forward_outputs(self, forward_outputs_list):
        l = len(forward_outputs_list)
        assert l % 2 == 0
        forward_outputs_0 = forward_outputs_list[: l // 2]
        forward_outputs_1 = forward_outputs_list[l // 2 :]
        return ForwardForwardOutputs(
            self.make_forward_outputs(forward_outputs_0),
            self.make_forward_outputs(forward_outputs_1),
        )

    def make_backward_backward_inputs(
        self, forward_forward_outputs: ForwardForwardOutputs
    ):
        return BackwardBackwardInputs(
            self.make_backward_inputs(forward_forward_outputs.forward_outputs_0),
            self.make_backward_inputs(forward_forward_outputs.forward_outputs_1),
        )
