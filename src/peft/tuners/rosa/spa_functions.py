import torch
from spops import spmm, sddmm, csr_transpose
from torch.autograd.function import once_differentiable

class SpMMFunction(torch.autograd.Function):
    """
    returns the grad with respect to A_val and B.
    """
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, A_val, A_row_offsets, A_row_indices, A_col_indices, B, M, AT_val, AT_row_offsets, AT_row_indices, AT_col_indices):
        ctx.save_for_backward(A_val, A_row_offsets, A_row_indices, A_col_indices, B, AT_val, AT_row_offsets, AT_row_indices, AT_col_indices)
        C = spmm(A_val, A_row_offsets, A_row_indices, A_col_indices, B, M)
        return C

    @staticmethod
    @once_differentiable
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dLdC):
        # dLdA = dLdC.B^T
        # dLdB = A^T.dLdC

        A_val, A_row_offsets, A_row_indices, A_col_indices, B, AT_val, AT_row_offsets, AT_row_indices, AT_col_indices = ctx.saved_tensors
        dLdC = dLdC.contiguous()
        
        dLdA_val = sddmm(A_row_offsets, A_row_indices, A_col_indices, dLdC, B)

        if AT_val is None:
            AT_val, AT_row_offsets, AT_col_indices = csr_transpose(A_val, A_row_offsets, A_col_indices, dLdC.shape[0], B.shape[0])
            AT_row_indices = torch.argsort(-1 * torch.diff(AT_row_offsets)).int()
            
        dLdB = spmm(AT_val, AT_row_offsets, AT_row_indices, AT_col_indices, dLdC, B.shape[0])
        return dLdA_val.to(A_val.dtype), None, None, None, dLdB.to(B.dtype), None, None, None, None, None
        
class SpMMTFunction(torch.autograd.Function):
    """
    returns the grad with respect to AT_val and B.
    """
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, AT_val, AT_row_offsets, AT_row_indices, AT_col_indices, B, M, A_val, A_row_offsets, A_row_indices, A_col_indices): # A: (M, K), AT: (K, M)
        if A_val is None:
            A_val, A_row_offsets, A_col_indices = csr_transpose(AT_val, AT_row_offsets, AT_col_indices, B.shape[0], M)
            A_row_indices = torch.argsort(-1 * torch.diff(A_row_offsets)).int()

        ctx.save_for_backward(AT_val, AT_row_offsets, AT_row_indices, AT_col_indices, B)
        C = spmm(A_val, A_row_offsets, A_row_indices, A_col_indices, B, M)
        return C

    @staticmethod
    @once_differentiable
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dLdC):
        # dLdAT = B.dLdCT
        # dLdB = AT.dLdC

        AT_val, AT_row_offsets, AT_row_indices, AT_col_indices, B = ctx.saved_tensors
        dLdC = dLdC.contiguous()
        
        dLdAT_val = sddmm(AT_row_offsets, AT_row_indices, AT_col_indices, B, dLdC)
        dLdB = spmm(AT_val, AT_row_offsets, AT_row_indices, AT_col_indices, dLdC, B.shape[0])
        return dLdAT_val.to(AT_val.dtype), None, None, None, dLdB.to(B.dtype), None, None, None, None, None