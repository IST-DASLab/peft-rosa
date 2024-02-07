import torch
from torch.autograd.function import once_differentiable
import bitsandbytes as bnb
from spops import sddmm, csr_add

class RoSALinearFunction(torch.autograd.Function):
    """
    
    Inputs: 
        X, W_module (with weight W and bias b), LA, LB, S_val, and S related indices as input.
        impl: a string indicating which implementation to use. options are 'sp_add', and 'spmm'

    Forward: 
        Output is O = X . W^T + X . S^T + X . (LB . LA) ^ T + b,
        which is equivalent to:
        O = X . (W + S)^T + (X . LA^T) . LB^T + b

    Backward:
        Returns gradients for LA, LB, S_val, and the input X.
        dLA = (LB^T . dO^T) . X
        dLB = dO^T . (X . LA^T)
        dS = dO^T . X -> SDDMM
        dX = dO . (W + S) + (dO . LB) . LA

    Shapes:
        X: (*, m)
        W, S: (n, m) -> S_val: (nnz, ), S_row_offs: (n + 1, ), S_row_idx: (n, ), S_col_idx: (nnz, )
        LA: (r, m), LB: (n, r)
        b: (n, )
        O: (*, n)
    
    Notes:
        - This function handles all computations of a RoSA layer (even dropout) to avoid storing different versions
          of the inputs for backward (this happens in the spmm implementations where the base module, 
          the low rank adapter, and the sparse adapter each save a different version of the input
          for backward).
        - If W is not a floating point, it means that some quantization is at work. In this case
          We simply dequantize it first, and then proceed. 
    """
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, X, W_module, LA, LB, S_val, S_row_offs, S_row_idx, S_col_idx, lora_scaling, lora_dropout_rate, training):
        # assert S_val is not None, 'sp_add implementation of RoSA is suboptimal if there is no sparse adapter, please switch to the spmm implementation.'

        input_shape = X.shape
        X = X.reshape(-1, X.shape[-1])

        needs_4bit_deq = False
        orig_W = W_module.weight if hasattr(W_module, 'weight') else W_module.qweight
        b = W_module.bias if hasattr(W_module, 'bias') else None
        if orig_W.dtype in [torch.bfloat16, torch.float16, torch.float32]:
            W = orig_W.to(X.dtype)
        else:
            assert isinstance(W_module, bnb.nn.Linear4bit), 'only [bf16, fp16, fp32] and 4bit quantization are supported in the sp_add implementation of RoSA. Change the implementation to spmm.'
            needs_4bit_deq = True
            W = bnb.functional.dequantize_4bit(orig_W.data, orig_W.quant_state).to(X.dtype)
        
        if S_val is None:
            O = torch.mm(X, W.T)
        else:
            O = torch.mm(X, csr_add(S_val, S_row_offs, S_row_idx, S_col_idx, W).T)

        if b is not None:
            O += b.to(X.dtype).unsqueeze(0)
        
        keep_prob = None
        D = None # the dropout mask
        if LA is not None:
            if training:
                keep_prob = 1 - lora_dropout_rate
                D = torch.rand_like(X) < keep_prob
                O += lora_scaling * torch.mm(torch.mm((X * D) / keep_prob, LA.T), LB.T)
            else:
                O += lora_scaling * torch.mm(torch.mm(X, LA.T), LB.T)

        ctx.save_for_backward(X, orig_W, LA, LB, S_val, S_row_offs, S_row_idx, S_col_idx, D)
        ctx.needs_4bit_deq = needs_4bit_deq
        ctx.input_shape = input_shape
        ctx.lora_scaling = lora_scaling
        ctx.keep_prob = keep_prob
        
        return O.reshape(*input_shape[:-1], O.shape[-1])

    @staticmethod
    @once_differentiable
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dO):
        dO = dO.reshape(-1, dO.shape[-1])
        X, orig_W, LA, LB, S_val, S_row_offs, S_row_idx, S_col_idx, D = ctx.saved_tensors

        if ctx.needs_4bit_deq:
            W = bnb.functional.dequantize_4bit(orig_W.data, orig_W.quant_state).to(X.dtype)
        else:
            W = orig_W.to(X.dtype)
        
        # Backward:
        # Returns gradients for LA, LB, S_val, and the input X.
        # dLA = (LB^T . dO^T) . X
        # dLB = dO^T . (X . LA^T)
        # dS = dO^T . X -> SDDMM
        # dX = dO . (W + S) + (dO . LB) . LA
        
        if S_val is None:
            dS_val = None
            dX = torch.mm(dO, W)
        else:
            dS_val = sddmm(S_row_offs, S_row_idx, S_col_idx, dO.T.contiguous(), X.T.contiguous())
            dX = torch.mm(dO, csr_add(S_val, S_row_offs, S_row_idx, S_col_idx, W))

        if LA is not None:
            if D is None:
                dLA = ctx.lora_scaling * torch.mm(torch.mm(LB.T, dO.T), X)
                dLB = ctx.lora_scaling * torch.mm(dO.T, torch.mm(X, LA.T))
                dX += ctx.lora_scaling * torch.mm(torch.mm(dO, LB), LA)
            else:
                XD = X * D
                dLA = ctx.lora_scaling * torch.mm(torch.mm(LB.T, dO.T), XD) / ctx.keep_prob
                dLB = ctx.lora_scaling * torch.mm(dO.T, torch.mm(XD, LA.T)) / ctx.keep_prob
                dX += ctx.lora_scaling * torch.mm(torch.mm(dO, LB), LA) * D / ctx.keep_prob
        else:
            dLA = None
            dLB = None
        
        dX = dX.reshape(*ctx.input_shape)
        return dX, None, dLA, dLB, dS_val, None, None, None, None, None, None