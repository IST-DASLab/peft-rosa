import torch


def pack_to_i4(x: torch.Tensor):
    # X_i8 = two_compl(X.to(dtype=torch.int8), 4).to(torch.uint8)
    assert x.dtype == torch.uint8
    X_i4 = x[0::2] | (x[1::2] << 4)
    return X_i4


def unpack_to_i8(x: torch.Tensor):
    x_i8 = torch.zeros(x.numel() * 2, dtype=torch.uint8, device=x.device)
    x_i8[0::2] = x & 15
    x_i8[1::2] = x >> 4
    return x_i8

class QuantConfig:
    def __init__(self, bits: int, bucket_size: int = 128):
        assert bits in [4, 8]
        self.bits = bits
        self.bucket_size = bucket_size


class Quantizer:
    def __init__(self, config: QuantConfig):
        self._config = config
        self.is_enabled = self._config is not None

    def get_compressed_buffer(self, tensor: torch.Tensor):
        if self.is_enabled:
            if self._config.bits == 4:
                return torch.zeros(tensor.numel() // 2, dtype=torch.uint8, device=tensor.device)
            else:
                return torch.zeros(tensor.numel(), dtype=torch.uint8, device=tensor.device)
        else:
            return torch.zeros_like(tensor)

    def get_metadata_buffer(self, tensor: torch.Tensor):
        if not self.is_enabled:
            # empty tensor
            return torch.zeros(1, device='cpu')
        numel = tensor.numel()
        num_buckets = ((numel + self._config.bucket_size - 1) // self._config.bucket_size)
        return torch.zeros((2, num_buckets), dtype=tensor.dtype, device=tensor.device)

    @torch.no_grad()
    def quantize(self, t: torch.Tensor, output: torch.Tensor, meta: torch.Tensor):
        if not self.is_enabled:
            output.copy_(t)
        if t.numel() & 1 and self._config.bits == 4:
            raise ValueError("Uneven sizes of tensors with quantization to 4 bits are not supported")
        target_device = t.device
        bits = self._config.bits
        num_levels = 1 << bits
        bucket_size = self._config.bucket_size
        t = t.clone().view(-1)
        t = t.cuda()
        numel = t.numel()
        main_chunk_size = (numel // bucket_size) * bucket_size
        tail_chunk_size = numel - main_chunk_size
        if main_chunk_size > 0:
            chunk = t[:main_chunk_size].view((-1, bucket_size))
            fmin = torch.min(chunk, dim=1)[0]
            fmax = torch.max(chunk, dim=1)[0]
            unit = (fmax - fmin) / (num_levels - 1)
            unit = torch.max(unit, torch.Tensor([1e-11]).expand_as(unit).to(unit.device))
            meta[0, : numel // bucket_size] = fmin.to(target_device)
            meta[1, :numel // bucket_size] = unit.to(target_device)
            unit = unit[:, None]
            fmin = fmin[:, None]

            chunk -= fmin
            chunk /= unit
            # round to nearest
            chunk += 0.5
            chunk = chunk.floor_()
            chunk = chunk.view(-1).to(torch.uint8)
            if bits == 8:
                output[:main_chunk_size] = chunk
            else:
                output[:main_chunk_size//2] = pack_to_i4(chunk)
        if tail_chunk_size > 0:
            chunk = t[main_chunk_size:]
            fmin = torch.min(chunk)
            fmax = torch.max(chunk)
            unit = (fmax - fmin) / (num_levels - 1)
            meta[0, -1] = fmin.to(target_device)
            meta[1, -1] = unit.to(target_device)

            chunk -= fmin
            chunk /= unit
            # round to nearest
            chunk += 0.5
            chunk = chunk.floor_()
            chunk = chunk.view(-1).to(torch.uint8)
            if bits == 8:
                output[main_chunk_size:] = chunk
            else:
                output[main_chunk_size//2:] = pack_to_i4(chunk)

    @torch.no_grad()
    def dequantize(self, qtensor: torch.Tensor, meta: torch.Tensor):
        if not self.is_enabled:
            return qtensor
        numel = qtensor.numel()
        bits = self._config.bits
        bucket_size = self._config.bucket_size
        meta = meta.cuda()
        orig_dtype = meta.dtype
        main_chunk_size = (numel // bucket_size) * bucket_size
        tail_chunk_size = numel - main_chunk_size

        t = qtensor.cuda()
        if bits == 4:
            t = unpack_to_i8(t)
        t = t.to(orig_dtype)
        if main_chunk_size > 0:
            chunk = t[:main_chunk_size].view(-1, bucket_size)
            fmin = meta[0, :numel // bucket_size][:, None]
            unit = meta[1, :numel // bucket_size][:, None]
            chunk *= unit
            chunk += fmin
        if tail_chunk_size > 0:
            chunk = t[main_chunk_size:]
            fmin = meta[0, -1]
            unit = meta[1, -1]
            chunk *= unit
            chunk += fmin
        return t.cpu()


if __name__ == "__main__":
    # tensor = torch.randint(0, 15, (10,), dtype=torch.uint8)
    # print(tensor)
    # t_i4 = pack_to_i4(tensor)
    # t_i8 = unpack_to_i8(t_i4)
    # print(t_i8)
    tensor = torch.rand(36, dtype=torch.float16)
    print(tensor)
    config = QuantConfig(4, 8)
    quantizer = Quantizer(config)
    qmeta = quantizer.get_metadata_buffer(tensor)
    buffer = quantizer.get_compressed_buffer(tensor)
    quantizer.quantize(tensor, buffer, qmeta)
    result = quantizer.dequantize(buffer, qmeta)
    result = result.reshape(tensor.shape)
    print(result)

