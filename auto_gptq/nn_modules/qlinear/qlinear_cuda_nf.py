import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import transformers
import torch.functional as F
import time

logger = getLogger(__name__)
try:
    import autogptq_cuda_256
    import autogptq_cuda_64
    _autogptq_cuda_available = True
except ImportError:
    logger.warning('CUDA extension not installed.')
    _autogptq_cuda_available = False

def fptoint4(w, scale):
    dev = w.device
    q = w / scale
    q = torch.where(q >= 0.8614784181118011, torch.tensor(15).to(dev), q)
    q = torch.where((q < 0.8614784181118011) & (q >= 0.6427869200706482) , torch.tensor(14).to(dev), q)
    q = torch.where((q < 0.6427869200706482) & (q >= 0.5016634166240692) , torch.tensor(13).to(dev), q)
    q = torch.where((q < 0.5016634166240692) & (q >= 0.3893125355243683) , torch.tensor(12).to(dev), q)
    q = torch.where((q < 0.3893125355243683) & (q >= 0.2920137718319893) , torch.tensor(11).to(dev), q)
    q = torch.where((q < 0.2920137718319893) & (q >= 0.2035212516784668) , torch.tensor(10).to(dev), q)
    q = torch.where((q < 0.2035212516784668) & (q >= 0.1202552504837513) , torch.tensor(9).to(dev), q)
    q = torch.where((q < 0.1202552504837513) & (q >= 0.03979014977812767) , torch.tensor(8).to(dev), q)
    q = torch.where((q < 0.03979014977812767) & (q >= -0.045525018125772476) , torch.tensor(7).to(dev), q)
    q = torch.where((q < -0.045525018125772476) & (q >= -0.13791173323988914) , torch.tensor(6).to(dev), q)
    q = torch.where((q < -0.13791173323988914) & (q >= -0.23460740596055984) , torch.tensor(5).to(dev), q)
    q = torch.where((q < -0.23460740596055984) & (q >= -0.33967943489551544) , torch.tensor(4).to(dev), q)
    q = torch.where((q < -0.33967943489551544) & (q >= -0.4599952697753906) , torch.tensor(3).to(dev), q)
    q = torch.where((q < -0.4599952697753906) & (q >= -0.6106329262256622) , torch.tensor(2).to(dev), q)
    q = torch.where((q < -0.6106329262256622) & (q >= -0.8480964004993439) , torch.tensor(1).to(dev), q)
    q = torch.where(q < -0.8480964004993439 , torch.tensor(0).to(dev), q)
    q = q.to(torch.int)
    return q

def int4tofp(q, scale):
    dev = q.device
    w = torch.zeros_like(q).to(dev)
    w = torch.where(q == 15, torch.tensor(1.0).to(dev), w)
    w = torch.where(q == 14, torch.tensor(0.7229568362236023).to(dev), w)
    w = torch.where(q == 13, torch.tensor(0.5626170039176941).to(dev), w)
    w = torch.where(q == 12, torch.tensor(0.44070982933044434).to(dev), w)
    w = torch.where(q == 11, torch.tensor(0.33791524171829224).to(dev), w)
    w = torch.where(q == 10, torch.tensor(0.24611230194568634).to(dev), w)
    w = torch.where(q == 9, torch.tensor(0.16093020141124725).to(dev), w)
    w = torch.where(q == 8, torch.tensor(0.07958029955625534).to(dev), w)
    w = torch.where(q == 7, torch.tensor(0).to(dev), w)
    w = torch.where(q == 6, torch.tensor(-0.09105003625154495).to(dev), w)
    w = torch.where(q == 5, torch.tensor(-0.18477343022823334).to(dev), w)
    w = torch.where(q == 4, torch.tensor(-0.28444138169288635).to(dev), w)
    w = torch.where(q == 3, torch.tensor(-0.39491748809814453).to(dev), w)
    w = torch.where(q == 2, torch.tensor(-0.5250730514526367).to(dev), w)
    w = torch.where(q == 1, torch.tensor(-0.6961928009986877).to(dev), w)
    w = torch.where(q == 0, torch.tensor(-1.0).to(dev), w)
    w = (w * scale).to(torch.float16)
    return w

class QuantLinear(nn.Module):
    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        use_cuda_fp16=True,
        kernel_switch_threshold=128,
        trainable=False
    ):
        super().__init__()
        global _autogptq_cuda_available
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        if trainable:
            _autogptq_cuda_available = False
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2 ** self.bits - 1

        self.register_buffer(
            'qweight',
            torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32)
        )
        self.register_buffer(
            'scales',
            torch.zeros((math.ceil(infeatures / self.group_size), outfeatures), dtype=torch.float16)
        )
        self.register_buffer(
            'g_idx',
            torch.tensor([i // self.group_size for i in range(infeatures)], dtype=torch.int32)
        )

        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None
        self.half_indim = self.infeatures // 2

        self.use_cuda_fp16 = use_cuda_fp16 if bits != 8 else False
        
        # is performed by unpacking the weights and using torch.matmul
        if self.bits in [2, 4, 8]:
            self.wf = torch.tensor(list(range(0, 32, self.bits)), dtype=torch.int32).unsqueeze(0)
        elif self.bits == 3:
            self.wf = torch.tensor(
                [
                    [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                    [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                    [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
                ],
                dtype=torch.int32
            ).reshape(1, 3, 12)

        self.kernel_switch_threshold = kernel_switch_threshold
        self.autogptq_cuda_available = _autogptq_cuda_available
        if _autogptq_cuda_available:
            self.autogptq_cuda = autogptq_cuda_256
            if infeatures % 256 != 0 or outfeatures % 256 != 0:
                self.autogptq_cuda = autogptq_cuda_64
            if infeatures % 64 != 0 or outfeatures % 64 != 0:
                self.autogptq_cuda_available = False

        self.trainable = trainable

    def pack(self, linear, scales, g_idx):
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()

        scales = scales.t().contiguous()
        self.scales = scales.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(self.infeatures):
            g_idx = idx // self.group_size
            intweight.append(
                fptoint4(W[:, idx], self.scales[g_idx])[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros(
            (intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32
        )
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            elif self.bits == 3:
                raise NotImplementedError("3 bits unimplemented")
                # for j in range(i, i + 10):
                #     qweight[row] |= intweight[j] << (3 * (j - i))
                # i += 10
                # qweight[row] |= intweight[i] << 30
                # row += 1
                # qweight[row] |= (intweight[i] >> 2) & 1
                # i += 1
                # for j in range(i, i + 10):
                #     qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                # i += 10
                # qweight[row] |= intweight[i] << 31
                # row += 1
                # qweight[row] |= (intweight[i] >> 1) & 0x3
                # i += 1
                # for j in range(i, i + 10):
                #     qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                # i += 10
                # row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

    def forward(self, x):
        # print("QuantLinear forward")
        # print("x: ", x)
        # print("qweight: ", self.qweight)
        # print("scales: ", self.scales)
        # print("bias: ", self.bias)
        # print("g_idx: ", self.g_idx)
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        if self.autogptq_cuda_available is True and (
            self.kernel_switch_threshold is False or x.shape[0] < self.kernel_switch_threshold
        ):
            logger.info('cuda unimplemented')
            # out = torch.zeros(x.shape[0], out_shape[-1], dtype=torch.float, device=x.device)
            # if self.use_cuda_fp16:
            #     x = x.half()
            #     if self.bits == 2:
            #         self.autogptq_cuda.vecquant2matmul_faster_old(x, self.qweight, out, self.scales.float(), self.qzeros, self.group_size, self.half_indim)
            #     elif self.bits == 3:
            #         self.autogptq_cuda.vecquant3matmul_faster_old(x, self.qweight, out, self.scales.float(), self.qzeros, self.group_size, self.half_indim)
            #     elif self.bits == 4:
            #         self.autogptq_cuda.vecquant4matmul_faster_old(x, self.qweight, out, self.scales.float(), self.qzeros, self.group_size, self.half_indim)

            #     else:
            #         raise NotImplementedError("Only 2,3,4 bits are supported.")
            # else:
            #     x = x.float()
            #     if self.bits == 2:
            #         self.autogptq_cuda.vecquant2matmul_old(x, self.qweight, out, self.scales.float(), self.qzeros, self.group_size)
            #     elif self.bits == 3:
            #         self.autogptq_cuda.vecquant3matmul_old(x, self.qweight, out, self.scales.float(), self.qzeros, self.group_size)
            #     elif self.bits == 4:
            #         self.autogptq_cuda.vecquant4matmul_old(x, self.qweight, out, self.scales.float(), self.qzeros, self.group_size)
            #     elif self.bits == 8:
            #         self.autogptq_cuda.vecquant8matmul_old(x, self.qweight, out, self.scales.float(), self.qzeros, self.group_size)
            #     else:
            #         raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        else:
            if self.wf.device != self.scales.device:
               self.wf = self.wf.to(self.scales.device)
                
            if self.bits in [2,4,8]:
   
               scales = self.scales
               scales = scales.reshape(-1, 1, scales.shape[-1])
                
               weight = torch.bitwise_right_shift(torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1), self.wf.unsqueeze(-1)).to(torch.int16 if self.bits == 8 else torch.int8)
               torch.bitwise_and(weight,(2 ** self.bits) - 1, out=weight)
               weight = weight.reshape(-1, self.group_size, weight.shape[2])
            elif self.bits == 3:
                raise NotImplementedError("3 bits unimplemented")
                # zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1]//3, 3, 1).expand(-1, -1, -1, 12)
                # zeros = (zeros >> self.wf.unsqueeze(0))
                # zeros[:,:,0,10] = (zeros[:,:,0,10]&0x3) | ((zeros[:,:,1,0] << 2)&0x4)
                # zeros[:,:,1,11] = (zeros[:,:,1,11]&0x1) | ((zeros[:,:,2,0] << 1)&0x6)
                # zeros = zeros & 0x7
                # zeros = torch.cat([zeros[:,:,0,:11], zeros[:,:,1,1:12], zeros[:,:,2,1:11]], dim=2)
                
                # zeros = zeros + 1
                # zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2]) 
                
                # scales = self.scales
                # scales = scales.reshape(-1, 1, scales.shape[-1])
                
                # weight = self.qweight.reshape(self.qweight.shape[0]//3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
                # weight = (weight >> self.wf.unsqueeze(-1))&0x7
                # weight[:,0,10] = (weight[:,0,10]&0x3) | ((weight[:,1,0] << 2)&0x4)
                # weight[:,1,11] = (weight[:,1,11]&0x1) | ((weight[:,2,0] << 1)&0x6)
                # weight = weight & 0x7
                # weight = torch.cat([weight[:,0,:11], weight[:,1,1:12], weight[:,2,1:11]], dim=1)
                # weight = weight.reshape(-1, self.group_size, weight.shape[2])
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")
            weight = int4tofp(weight, scales)
            weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

            out = torch.matmul(x.half(), weight)
        out = out.half().reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        # print("out: ", out)
        return out


__all__ = ["QuantLinear"]
