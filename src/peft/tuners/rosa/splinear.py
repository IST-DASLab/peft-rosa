from typing import Any, Mapping
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix
import spops
from .spa_functions import SpMMFunction, SpMMTFunction

class SparseLinear(nn.Module):
    def __init__(self, density, shape, store_transpose=False, dtype=torch.bfloat16):
        super(SparseLinear, self).__init__()
        
        self.shape = shape
        self.store_transpose = store_transpose

        nnz = int(density * np.prod(shape))
        self.values = nn.Parameter(torch.zeros((nnz, ), dtype=dtype))

        self.register_buffer('row_offs', torch.zeros((shape[0] + 1, ), dtype=torch.int32))
        self.register_buffer('row_idx', torch.zeros((shape[0], ), dtype=torch.int16))
        self.register_buffer('col_idx', torch.zeros((nnz, ), dtype=torch.int16))

        if self.store_transpose:
            self.register_buffer('tr_perm', torch.zeros((nnz, ), dtype=torch.int32))
            self.register_buffer('tr_row_offs', torch.zeros((shape[1] + 1, ), dtype=torch.int32))
            self.register_buffer('tr_row_idx', torch.zeros((shape[1], ), dtype=torch.int16))
            self.register_buffer('tr_col_idx', torch.zeros((nnz, ), dtype=torch.int16))

    @torch.no_grad()
    def set_mask(self, mask):
        nnz = mask.sum().int().item()
        assert self.values.numel() == nnz, f'mask.nnz does not match the numel of spa values. mask.nnz: {nnz}, spa.values.numel: {spa_module.values.numel()}'
        assert mask.shape[0] == self.shape[0] and mask.shape[1] == self.shape[1], f'mask.shape does not match spa.shape. mask.shape: {mask.shape}, spa.shape: {self.shape}'
        
        sparse_tensor = csr_matrix(mask.cpu())
        self.row_offs = torch.tensor(sparse_tensor.indptr, dtype=torch.int32, device=self.values.device)
        self.col_idx = torch.tensor(sparse_tensor.indices, dtype=torch.int16, device=self.values.device)
        self.row_idx = torch.argsort(-1 * torch.diff(self.row_offs)).to(torch.int16)

    @torch.no_grad()
    def to_dense(self):
        assert self.exists(), 'spa.to_dense() called before spa mask is set'
        return torch.sparse_csr_tensor(
            self.row_offs.to(torch.int64),
            self.col_idx.to(torch.int64),
            self.values.data,
            size=self.shape,
            dtype=self.values.dtype,
            device=self.values.device
        ).to_dense()
    
    @torch.no_grad()
    def tr(self, none_if_not_exist=False):
        if self.store_transpose:
            if self.tr_row_offs[-1] == 0:
                tr_perm_plus_one, tr_row_offs, tr_col_idx = spops.csr_transpose(
                    torch.arange(self.values.shape[0], dtype=torch.float32, device=self.values.device) + 1,
                    self.row_offs,
                    self.col_idx,
                    *self.shape
                )
                self.tr_perm = (tr_perm_plus_one - 1).int()
                self.tr_row_offs = tr_row_offs.int()
                self.tr_col_idx = tr_col_idx.to(torch.int16)
                self.tr_row_idx = torch.argsort(-1 * torch.diff(tr_row_offs)).to(torch.int16)
            return (
                self.values.data[self.tr_perm].contiguous(), 
                self.tr_row_offs, 
                self.tr_row_idx, 
                self.tr_col_idx
            )
        else:
            if none_if_not_exist:
                return [None] * 4   
            tr_values, tr_row_offs, tr_col_idx = spops.csr_transpose(
                self.values.data,
                self.row_offs,
                self.col_idx,
                *self.shape
            )
            tr_row_idx = torch.argsort(-1 * torch.diff(tr_row_offs)).int()
            return (tr_values, tr_row_offs, tr_row_idx, tr_col_idx)

    def exists(self):
        if None in [
            self.values,
            self.row_offs,
            self.row_idx,
            self.col_idx
        ]:
            return False
        return self.row_offs[-1] != 0

    def forward(self, x):
        assert self.exists(), 'spa.forward() called before spa mask is set'
        tr_values, tr_row_offs, tr_row_idx, tr_col_idx = self.tr(none_if_not_exist=True)

        return SpMMFunction.apply( # only calculates grad for spa_values and x
            self.values,
            self.row_offs,
            self.row_idx,
            self.col_idx,
            x.reshape(-1, x.shape[-1]).T.contiguous(),
            self.shape[0],
            tr_values.detach() if tr_values is not None else None,
            tr_row_offs,
            tr_row_idx,
            tr_col_idx
        ).T.reshape(*x.shape[:-1], self.shape[0])

    # def __repr__(self) -> str:
    #     rep = super().__repr__()
    #     return "rosa." + rep    

class SparseLinearT(SparseLinear):
    def forward(self, x):
        assert self.exists(), 'spa.forward() called before spa mask is set'
        tr_values, tr_row_offs, tr_row_idx, tr_col_idx = self.tr(none_if_not_exist=True)

        return SpMMTFunction.apply( # only calculates grad for spa_values and x_onehot
            self.values,
            self.row_offs,
            self.row_idx,
            self.col_idx,
            x.reshape(-1, x.shape[-1]).T.contiguous(),
            self.shape[1],
            tr_values.detach() if tr_values is not None else None,
            tr_row_offs,
            tr_row_idx,
            tr_col_idx
        ).T.reshape(*x.shape[:-1], self.shape[1])