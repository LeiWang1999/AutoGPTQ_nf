from typing import Any
import torch
import pycuda
import pycuda.autoprimaryctx
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from .workloads import (
    get_gemv_workloads,
    get_gemv_workloads_nf4,
    get_gemm_workloads,
    get_gemm_workloads_nf4,
    _apply_gemv_schedule_nf4,
    get_permutate_workloads
)
import welder
from welder.graph import IRNode, OutputNode
from welder.policy import *
from tvm import relay
import os.path as osp
from tvm.contrib.target.onnx import to_onnx
from tvm.relay.testing import run_infer_type
from tvm.contrib import graph_executor
import os
from tvm.script import tir as T
from tvm import te
from welder.te_utils import connect_tensor_graph
import pathlib
from string import Template
import numpy as np
import os
import nni
from nni.experiment import Experiment
from .nni_database import NNIDatabase
import time

cutlass_header = """
#include "cutlass/cutlass.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_sm70.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_sm70.h"

namespace cutlass
{
    namespace gemm
    {
        namespace warp
        {

            template <class MmaWarp, int KSize>
            class MMAWarpWrapper
            {
            public:
                typename MmaWarp::FragmentA frag_A[2];
                typename MmaWarp::FragmentB frag_B[2];
                typename MmaWarp::FragmentC accum;
                MmaWarp mma_op;
                typename MmaWarp::IteratorA iter_A;
                typename MmaWarp::IteratorB iter_B;
                const int warp_idx_m_, warp_idx_n_, lane_id_;

                using TensorRefA = typename MmaWarp::IteratorA::TensorRef;
                using TensorRefB = typename MmaWarp::IteratorB::TensorRef;
                static_assert(KSize % MmaWarp::Shape::kK == 0);
                static int constexpr kKgroups = KSize / MmaWarp::Shape::kK;

                CUTLASS_DEVICE
                MMAWarpWrapper(int warp_idx_m, int warp_idx_n, int lane_id)
                    : warp_idx_m_(warp_idx_m), warp_idx_n_(warp_idx_n), lane_id_(lane_id), iter_A({nullptr, 0}, 0), iter_B({nullptr, 0}, 0)
                {
                    accum.clear();
                }

                CUTLASS_DEVICE
                void prologue(const TensorRefA &ref_A, const TensorRefB &ref_B)
                {
                    iter_A = typename MmaWarp::IteratorA(ref_A, lane_id_);
                    iter_B = typename MmaWarp::IteratorB(ref_B, lane_id_);
                    iter_A.add_tile_offset({warp_idx_m_, 0});
                    iter_B.add_tile_offset({0, warp_idx_n_});
                    iter_A.load(frag_A[0]);
                    iter_B.load(frag_B[0]);
                    ++iter_A;
                    ++iter_B;
                }
                CUTLASS_DEVICE
                void body()
                {
                    CUTLASS_PRAGMA_UNROLL
                    for (int k = 0; k < kKgroups - 1; ++k)
                    {
                        iter_A.load(frag_A[(k + 1) % 2]);
                        iter_B.load(frag_B[(k + 1) % 2]);
                        ++iter_A;
                        ++iter_B;
                        mma_op(accum, frag_A[k % 2], frag_B[k % 2], accum);
                    }
                    __syncthreads();
                }
                CUTLASS_DEVICE
                void epilogue()
                {
                    mma_op(accum, frag_A[(kKgroups - 1) % 2], frag_B[(kKgroups - 1) % 2], accum);
                }
            };

            template <
                typename Shape,
                typename SMemLayoutA,
                typename SMemLayoutB>
            class GemmTensorOp
            {
            public:
                using InstructionShape = GemmShape<16, 8, 16>;
                using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
                    cutlass::arch::Mma<
                        InstructionShape,
                        32,
                        cutlass::half_t,
                        cutlass::layout::RowMajor,
                        cutlass::half_t,
                        cutlass::layout::ColumnMajor,
                        cutlass::half_t,
                        cutlass::layout::RowMajor,
                        cutlass::arch::OpMultiplyAdd>,
                    cutlass::MatrixShape<1, 1>>;

                using MmaWarp = typename cutlass::gemm::warp::MmaTensorOp<
                    GemmShape<Shape::kM, Shape::kN, InstructionShape::kK>,
                    cutlass::half_t,
                    SMemLayoutA,
                    cutlass::half_t,
                    SMemLayoutB,
                    cutlass::half_t,
                    cutlass::layout::RowMajor,
                    Policy>;
                using MMA = MMAWarpWrapper<MmaWarp, Shape::kK>;
                MMA mma;

                CUTLASS_DEVICE
                GemmTensorOp(int warp_idx_m, int warp_idx_n, int lane_id)
                    : mma(warp_idx_m, warp_idx_n, lane_id) {}
                CUTLASS_DEVICE
                half &operator[](size_t i) const
                {
                    return ((half *)mma.accum.data())[i];
                }
                CUTLASS_DEVICE
                half *operator+(size_t i) const
                {
                    return (half *)mma.accum.data() + i;
                }
            };

            template <
                typename Shape,
                typename SMemLayoutA,
                typename LayoutA,
                typename SMemLayoutB,
                typename LayoutB,
                typename LayoutC>
            class VoltaGemmTensorOp
            {
            public:
                using InstructionShape = GemmShape<16, 16, 4>;
                using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
                    cutlass::arch::Mma<
                        InstructionShape,
                        32,
                        cutlass::half_t,
                        LayoutA,
                        cutlass::half_t,
                        LayoutB,
                        cutlass::half_t,
                        LayoutC,
                        cutlass::arch::OpMultiplyAdd>,
                    cutlass::MatrixShape<1, 1>>;

                using MmaWarp = typename cutlass::gemm::warp::MmaVoltaTensorOp<
                    GemmShape<Shape::kM, Shape::kN, InstructionShape::kK>,
                    cutlass::half_t,
                    SMemLayoutA,
                    cutlass::half_t,
                    SMemLayoutB,
                    cutlass::half_t,
                    LayoutC,
                    Policy>;
                using MMA = MMAWarpWrapper<MmaWarp, Shape::kK>;
                MMA mma;

                CUTLASS_DEVICE
                VoltaGemmTensorOp(int warp_idx_m, int warp_idx_n, int lane_id)
                    : mma(warp_idx_m, warp_idx_n, lane_id) {}
                CUTLASS_DEVICE
                half &operator[](size_t i) const
                {
                    return ((half *)mma.accum.data())[i];
                }
                CUTLASS_DEVICE
                half *operator+(size_t i) const
                {
                    return (half *)mma.accum.data() + i;
                }
            };

            template <
                typename Shape,
                typename SMemLayoutA,
                typename SMemLayoutB>
            class GemmI8TensorOp
            {
            public:
                using InstructionShape = GemmShape<16, 8, 32>;
                using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
                    cutlass::arch::Mma<
                        InstructionShape,
                        32,
                        int8_t,
                        cutlass::layout::RowMajor,
                        int8_t,
                        cutlass::layout::ColumnMajor,
                        int,
                        cutlass::layout::RowMajor,
                        cutlass::arch::OpMultiplyAdd>,
                    cutlass::MatrixShape<1, 1>>;

                using MmaWarp = typename cutlass::gemm::warp::MmaTensorOp<
                    GemmShape<Shape::kM, Shape::kN, InstructionShape::kK>,
                    int8_t,
                    SMemLayoutA,
                    int8_t,
                    SMemLayoutB,
                    int,
                    cutlass::layout::RowMajor,
                    Policy>;
                using MMA = MMAWarpWrapper<MmaWarp, Shape::kK>;
                MMA mma;

                CUTLASS_DEVICE
                GemmI8TensorOp(int warp_idx_m, int warp_idx_n, int lane_id)
                    : mma(warp_idx_m, warp_idx_n, lane_id) {}
                CUTLASS_DEVICE
                int &operator[](size_t i) const
                {
                    return ((int *)mma.accum.data())[i];
                }
                CUTLASS_DEVICE
                int *operator+(size_t i) const
                {
                    return (int *)mma.accum.data() + i;
                }
            };

        }
    }
}

template <class TensorOp>
CUTLASS_DEVICE void call_cutlass_mma_body(TensorOp &op)
{
    op.mma.body();
}

template <class TensorOp>
CUTLASS_DEVICE void call_cutlass_mma_epilogue(TensorOp &op)
{
    op.mma.epilogue();
}

template <class TensorOp>
CUTLASS_DEVICE void call_cutlass_mma_prologue(TensorOp &op, void *pA, void *pB, int sA, int sB)
{
    using TensorRefA = typename TensorOp::MMA::TensorRefA;
    using TensorRefB = typename TensorOp::MMA::TensorRefB;
    TensorRefA refA{(typename TensorRefA::Element *)pA, sA};
    TensorRefB refB{(typename TensorRefB::Element *)pB, sB};
    op.mma.prologue(refA, refB);
}

#define ALLOCATE_CUTLASS_OBJECT(var, ...) auto var = __VA_ARGS__;

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

"""

cuda_fp16_header = """
#include <cuda_fp16.h>

namespace {

__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}

#define __int8_t_defined

#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
inline __device__ half HALF_MATH_NAME(half x, half y) {            \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
inline __device__ half HALF_MATH_NAME(half x) {                   \
  float tmp_x = __half2float(x);                                         \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2half(result);                                           \
}

CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY

// Pack two half values.
inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// There is no make_int8 in cuda, but TVM codegen seem to use it
inline __device__ longlong4 make_int8(int x0, int x1, int x2, int x3, int x4, int x5, int x6, int x7) {
  int2 i0 = make_int2(x0, x1);
  int2 i1 = make_int2(x2, x3);
  int2 i2 = make_int2(x4, x5);
  int2 i3 = make_int2(x6, x7);
  long long l0 = *(long long*)&i0;
  long long l1 = *(long long*)&i1;
  long long l2 = *(long long*)&i2;
  long long l3 = *(long long*)&i3;
  return make_longlong4(l0, l1, l2, l3);
}

template<int row_size, int col_size, int panel_width>
__device__ int rasterization2DRow(int idx) {
  const int block_size = row_size * col_size;
  const int panel_size = panel_width * col_size;
  const int block_offset = idx % block_size;
  const int block_idx = idx / block_size;
  const int panel_offset = block_offset % panel_size;
  const int panel_idx = block_offset / panel_size;
  const int total_panel = (block_size + panel_size - 1) / panel_size;
  const int stride = panel_idx + 1 < total_panel ? panel_width : (block_size - panel_idx * panel_size) / col_size;
  const int col_idx = (panel_idx & 1) ? col_size - 1 - panel_offset / stride : panel_offset / stride;
  const int row_idx = panel_offset % stride + panel_idx * panel_width;
  return block_idx * block_size + row_idx * col_size + col_idx;
}

template<int row_size, int col_size, int panel_width>
__device__ int rasterization2DColumn(int idx) {
  const int block_size = row_size * col_size;
  const int panel_size = panel_width * row_size;
  const int block_offset = idx % block_size;
  const int block_idx = idx / block_size;
  const int panel_offset = block_offset % panel_size;
  const int panel_idx = block_offset / panel_size;
  const int total_panel = (block_size + panel_size - 1) / panel_size;
  const int stride = panel_idx + 1 < total_panel ? panel_width : (block_size - panel_idx * panel_size) / row_size;
  const int row_idx = (panel_idx & 1) ? row_size - 1 - panel_offset / stride : panel_offset / stride;
  const int col_idx = panel_offset % stride + panel_idx * panel_width;
  return block_idx * block_size + row_idx * col_size + col_idx;
}

}
"""

nni_database_path = '.nnidatabase'

class TensorHolder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super(TensorHolder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_pointer(self):
        return self.t.data_ptr()


class TVMExecutable(object):
    def __init__(self, src, name):
        super(TVMExecutable, self).__init__()
        self.source_code: str = src
        self.func_name: str = name
        self.kernel_func = self._get_kernel(self.source_code, self.func_name)

    def __call__(self, input, qweight=None, output=None, scales=None, g_idx=None, lut=None, M=None, grid=None, block=None, format="nf4") -> Any:
        if qweight is not None and output is not None and scales is not None:
            self.kernel_func(TensorHolder(input), TensorHolder(qweight), TensorHolder(lut), TensorHolder(scales), TensorHolder(output), grid=tuple(grid), block=tuple(block))
        else:
            self.kernel_func(TensorHolder(input), TensorHolder(output), grid=tuple(grid), block=tuple(block))

    def _get_kernel(self, src_code, name):
        src = torch.cuda.ByteTensor(8)
        mod = SourceModule(src_code, no_extern_c=True)
        return mod.get_function(name)


class TVMHandler(object):
    def __init__(self, n: int, k: int, bits: int, group_size: int, load_from_cache: bool = False, format: str = "nf4"):
        super(TVMHandler, self).__init__()
        self.k = k
        self.n = n
        self.bits = bits
        self.group_size = group_size
        self.format = format
        self.m_candidates = [1, 16, 32, 64, 128, 256, 512]
        self.workspace = torch.zeros((1024, k), dtype=torch.float16, device="cuda")
        self.configurations = {
            f'm1n{n}k{k}g{group_size}':
                {
                    'num_warps': 4
                },
        }
        if not load_from_cache:
            for m in self.m_candidates:
                if m == 1:
                    setattr(self, f'm1n{n}k{k}g{group_size}', self._get_executable_m1(bits, n, k, group_size))
                else:
                    setattr(self, f'm{m}n{n}k{k}g{group_size}_prmt', self._get_permutate_mx(
                    bits, m, n, k, group_size))
                    setattr(self, f'm{m}n{n}k{k}g{group_size}', self._get_executable_mx(
                    bits, m, n, k, group_size))

    def __call__(self, input, qweight, output, scales, g_idx, lut) -> Any:
        assert len(output.shape) >= 2, "output should be larger than 2D"
        M = 1
        for i in range(len(input.shape) - 1):
            M *= input.shape[i]
        N = output.shape[-1]
        K = input.shape[-1]
        group_size = self.group_size
        args = (input, qweight, output, scales,
                    g_idx, lut, M)
        
        def ladder_call_mx(mx):
            prmt = f"{mx}_prmt"
            mx_config = self.configurations[prmt]
            block = mx_config['block_size']
            grid = mx_config['grid_size']
            _func = getattr(self, prmt)
            _func(input=args[0], output=self.workspace, block=block, grid=grid)
            new_args = (self.workspace, *args[1:])
            mx_config = self.configurations[mx]
            block = mx_config['block_size']
            grid = mx_config['grid_size']
            _func = getattr(self, mx)
            _func(*new_args, block=block, grid=grid)

        if M == 1:
            m1_config = self.configurations[f'm1n{N}k{K}g{group_size}']
            block = (32, m1_config['num_warps'], 1)
            grid = (N // m1_config['num_warps'], 1, 1)
            _func = getattr(self, f'm1n{N}k{K}g{group_size}')
            _func(*args, block=block, grid=grid)
        elif 1< M <= 16:
            mx = f'm16n{N}k{K}g{group_size}'
            ladder_call_mx(mx)
        elif 16 < M <= 32:
            mx = f'm32n{N}k{K}g{group_size}'
            ladder_call_mx(mx)
        elif 32 < M <= 64:
            mx = f'm64n{N}k{K}g{group_size}'
            ladder_call_mx(mx)
        elif 64 < M <= 128:
            mx = f'm128n{N}k{K}g{group_size}'
            ladder_call_mx(mx)
        elif 128 < M <= 256:
            mx = f'm256n{N}k{K}g{group_size}'
            ladder_call_mx(mx)
        elif 256 < M <= 512:
            mx = f'm512n{N}k{K}g{group_size}'
            ladder_call_mx(mx)
        elif 512 < M <= 1024:
            mx = f'm1024n{N}k{K}g{group_size}'
            ladder_call_mx(mx)

    def _get_executable_m1(self, bits: int, n: int, k: int, group_size: int = -1):
        # get src code
        if group_size == -1:
            group_size = k
        mx = f'm1n{n}k{k}g{group_size}'
        args = get_gemv_workloads_nf4(bits, n, k, group_size)
        m1_module = te.create_prim_func(args)
        num_warps = self.configurations[mx]['num_warps']
        m1_mod = _apply_gemv_schedule_nf4(m1_module, bits, k, num_warps)
        code = m1_mod.imported_modules[0].get_source()
        name = f"tir_halfxnf{bits}_simt_bn{num_warps}_n{n}_k{k}"
        code = code.replace(
            "main_kernel0", name)
        code = code.split("extern \"C\"")[1]
        code = "extern \"C\"" + code
        code = "#include <cuda_fp16.h>\n" + code
        return TVMExecutable(code, name)

    def _get_executable_mx(self, bits: int, m:int, n: int, k: int, group_size: int = -1):
        arch = "cuda"
        arch = welder.arch.__getattribute__(arch)()
        mx = f'm{m}n{n}k{k}g{group_size}'
        args = get_gemm_workloads_nf4(bits, m, n, k, group_size)
        input_args = args[:-1]
        output_args = [args[-1]]
        node = IRNode([None for _ in input_args], args, "ladder_gemm")
        node.add_tag("tensorCoreConfig", [2, 3])
        node.add_tag("ladder_config", (True, True))
        output_nodes = [OutputNode(node)]
        policy = LadderPolicy(output_nodes, arch)
        configs = policy.emit_config(20)

        compile_results = []
        cgen = welder.CodeGenerator()
        for config in configs:
            cpresult = cgen.compile(output_nodes, config, "cuda", kernel_name="_fused_kernel_")
            compile_results.append(cpresult)
        welder.utils.compile_and_load_parallel(compile_results, arch)
        best_latency = 10000
        best = None
        values = []
        for cpresult in compile_results:
            print(cpresult.config)
            code = cpresult.code
            if cpresult.lib is None:
                latency = 10000
            else:
                latency = cpresult.profile()
            values.append(latency)
            if latency < best_latency:
                best_latency = latency
                best = cpresult
            print(latency)
            
        grid_size = tuple(best.grid_size)
        block_size = tuple(best.block_size)
        
        grid_size = [int(x) for x in grid_size]
        block_size = [int(x) for x in block_size]
        
        # create mx
        self.configurations[mx] = {}
        self.configurations[mx]['block_size'] = block_size
        self.configurations[mx]['grid_size'] = grid_size
        _block = best.config[node].block
        if len(_block) == 4:
            BM = _block[0] * _block[2]
            BN = _block[1] * _block[3]
        else:
            BM, BN = _block[:2]
        BK = best.config[node].rstep[0]
        name = f"tir_halfx{self.format}_tensorop_{BM}x{BN}x{BK}_K{k}_align8"
        code = best.code
        code = code.replace(
            "_fused_kernel_", name)
        code = code.replace("__global__ void __launch_bounds__", f'extern "C" __global__ void __launch_bounds__')
        code = "#include <mma.h>\n" + code
        code = cuda_fp16_header + code
        return TVMExecutable(code, name)
    
    def _get_permutate_mx(self, bits: int, m:int, n: int, k: int, group_size: int = -1):
        arch = "cuda"
        arch = welder.arch.__getattribute__(arch)()
        mx = f'm{m}n{n}k{k}g{group_size}_prmt'
        args = get_permutate_workloads(bits, m, n, k, group_size)
        input_args = args[:-1]
        output_args = [args[-1]]
        node = IRNode([None for _ in input_args], args, "ladder_permutate")
        output_nodes = [OutputNode(node)]
        policy = DefaultPolicy(output_nodes, arch)
        configs = policy.emit_config(20)

        compile_results = []
        cgen = welder.CodeGenerator()
        for config in configs:
            cpresult = cgen.compile(output_nodes, config, "cuda", kernel_name="_fused_kernel_")
            compile_results.append(cpresult)
        welder.utils.compile_and_load_parallel(compile_results, arch)
        best_latency = 10000
        best = None
        values = []
        for cpresult in compile_results:
            print(cpresult.config)
            code = cpresult.code
            if cpresult.lib is None:
                latency = 10000
            else:
                latency = cpresult.profile()
            values.append(latency)
            if latency < best_latency:
                best_latency = latency
                best = cpresult
            print(latency)
            
        grid_size = tuple(best.grid_size)
        block_size = tuple(best.block_size)
        grid_size = [int(x) for x in grid_size]
        block_size = [int(x) for x in block_size]
        # create mx
        self.configurations[mx] = {}
        self.configurations[mx]['block_size'] = block_size
        self.configurations[mx]['grid_size'] = grid_size
        _block = best.config[node].block
        if len(_block) == 4:
            BM = _block[0] * _block[2]
            BN = _block[1] * _block[3]
        else:
            BM, BN = _block[:2]
        name = f"tir_halfx{self.format}_simtop_{BM}x{BN}_K{k}_prmt"
        code = best.code
        code = code.replace(
            "_fused_kernel_", name)
        code = code.replace("__global__ void __launch_bounds__", f'extern "C" __global__ void __launch_bounds__')
        code = "#include <mma.h>\n" + code
        code = cuda_fp16_header + code
        return TVMExecutable(code, name)