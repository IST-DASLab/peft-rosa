import torch
from .layer import RosaLayer

class GradCollectorHook:
    def __init__(self, name: str, module: RosaLayer, grad_acc_mode: str) -> None:
        assert grad_acc_mode in ['mean', 'mean_squared']
        self._name = name
        self._module = module
        self._grad_acc_mode = grad_acc_mode

    def __call__(self, param):
        print('hook called for', self._name)

        if not hasattr(self._module, 'collected_grad'):
            self._module.register_buffer('collected_grad', torch.zeros_like(param.grad, device='cpu'))
            setattr(self._module, 'collected_grad_cnt', 0)

        with torch.no_grad():
            prev_cnt = getattr(self._module, 'collected_grad_cnt')
            new_cnt = prev_cnt + 1

            prev_grad = self._module.collected_grad
            new_grad = param.grad.detach().cpu()
            
            if self._grad_acc_mode == 'mean_squared':
                new_grad = new_grad ** 2
            
            self._module.collected_grad = (prev_grad * prev_cnt + new_grad) / new_cnt
            self._module.collected_grad_cnt = new_cnt

        # remove the gradient to save memory
        param.grad = None

class SaveInputHook:
    def __init__(self, name: str, module: RosaLayer) -> None:
        self._name = name
        self._module = module

    def __call__(self, model, module_in, module_out):
        if not isinstance(module_in, torch.Tensor):
            if len(module_in) > 1:
                print(f'found {len(module_in)} inputs, keeping only the first one.')
            module_in = module_in[0]
        
        if hasattr(self._module, 'saved_input'):
            self._module.saved_input = module_in
        else:
            self._module.register_buffer('saved_input', module_in)
        
        print(f'saved input for {self._name}')

class ManualGradCollectorHook:
    def __init__(self, name: str, module: RosaLayer, grad_acc_mode: str) -> None:
        assert grad_acc_mode in ['mean', 'mean_squared']
        self._name = name
        self._module = module
        self._grad_acc_mode = grad_acc_mode

    def __call__(self, model, grad_in, grad_out):
        print('hook called for', self._name)
        if not isinstance(grad_out, torch.Tensor):
            if len(grad_out) > 1:
                print(f'found {len(grad_out)} grad_outs, keeping only the first one.')
            grad_out = grad_out[0]

        with torch.no_grad():
            saved_input = self._module.saved_input
            new_grad = torch.mm(
                grad_out.reshape(-1, grad_out.shape[-1]).T,
                saved_input.reshape(-1, saved_input.shape[-1]),
            )
            if isinstance(self._module.get_base_layer(), torch.nn.Embedding):
                new_grad = new_grad.T
            if self._grad_acc_mode == 'mean_squared':
                new_grad = new_grad ** 2
            new_grad = new_grad.cpu()

            if not hasattr(self._module, 'collected_grad'):
                self._module.register_buffer('collected_grad', torch.zeros_like(new_grad))
                setattr(self._module, 'collected_grad_cnt', 0)

            prev_grad = self._module.collected_grad
            prev_cnt = getattr(self._module, 'collected_grad_cnt')

            new_cnt = prev_cnt + 1
            self._module.collected_grad = (prev_grad * prev_cnt + new_grad) / new_cnt
            self._module.collected_grad_cnt = new_cnt

            self._module.saved_input.zero_()