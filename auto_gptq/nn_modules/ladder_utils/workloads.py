import tvm
import numpy as np
import tvm.testing
from tvm import te
from tvm.script import tir as T
import os
from welder.te_utils import connect_tensor_graph


def _apply_gemv_schedule_nf4(ir_module, bits, K, num_warps=4):
    num_warps = num_warps
    warp_size = 32
    vec = 8
    sch = tvm.tir.Schedule(ir_module, debug_mask="all")
    block_b = sch.get_block("B")
    block_b_decompress = sch.get_block("B_decompress")
    block_b_rescale = sch.get_block("B_rescale")
    sch.compute_inline(block_b_decompress)
    sch.compute_inline(block_b_rescale)
    i, j, k = sch.get_loops(block_b)
    block_shared_local_A = sch.cache_read(block_b, 0, "local")
    block_shared_local_B = sch.cache_read(block_b, 2, "local")
    block_local_C = sch.cache_write(block_b, 0, "local")
    bx, j = sch.split(
        j, factors=[None, num_warps])
    k, tx, vk = sch.split(k, factors=[None, warp_size, vec])
    sch.reorder(bx, j, i, k, tx)

    sch.bind(i, "blockIdx.y")
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")
    sch.bind(j, "threadIdx.y")

    sch.compute_at(block_shared_local_A, tx, preserve_unit_loops=True)
    sch.compute_at(block_shared_local_B, tx, preserve_unit_loops=True)
    sch.reverse_compute_at(block_local_C, j, preserve_unit_loops=True)
    block_local_a_v = sch.get_loops(block_shared_local_A)[-1]
    sch.vectorize(block_local_a_v)
    
    block_local_b_v = sch.get_loops(block_shared_local_B)[-1]
    sch.vectorize(block_local_b_v)
    
    cuda_mod = tvm.build(sch.mod, target="cuda")
    return cuda_mod
        
def _apply_gemv_schedule(ir_module, bits, K, num_warps=4):
    num_warps = num_warps
    warp_size = 32
    vec = 8
    sch = tvm.tir.Schedule(ir_module, debug_mask="all")
    block_b = sch.get_block("B")
    block_b_decompress = sch.get_block("B_decompress")
    block_b_rescale = sch.get_block("B_rescale")
    sch.compute_inline(block_b_decompress)
    sch.compute_inline(block_b_rescale)
    i, j, k = sch.get_loops(block_b)
    block_shared_local_A = sch.cache_read(block_b, 0, "local")
    block_local_C = sch.cache_write(block_b, 0, "local")
    bx, j = sch.split(
        j, factors=[None, num_warps])
    k, tx, vk = sch.split(k, factors=[None, warp_size, vec])
    sch.reorder(bx, j, i, k, tx)

    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")
    sch.bind(j, "threadIdx.y")

    sch.compute_at(block_shared_local_A, tx, preserve_unit_loops=True)
    sch.reverse_compute_at(block_local_C, j, preserve_unit_loops=True)
    block_local_a_v = sch.get_loops(block_shared_local_A)[-1]
    sch.vectorize(block_local_a_v)
    ctx = tvm.cuda(0)
    cuda_mod = tvm.build(sch.mod, target="cuda")
    return cuda_mod

def _apply_gemm_schedule(ir_module, bits, K, config):
    from tvm.tir.tensor_intrin.cuda import (
        WMMA_FILL_16x16x16_F16_INTRIN,
        WMMA_LOAD_16x16x16_F16_A_INTRIN,
        WMMA_LOAD_16x16x16_F16_A_DYN_INTRIN,
        WMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN,
        WMMA_LOAD_16x16x16_F16_B_TRANS_DYN_INTRIN,
        WMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN,
        WMMA_STORE_16x16x16_F16_SHARED_INTRIN,
        WMMA_STORE_16x16x16_F16_SHARED_DYN_INTRIN,
        WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN,
    )

    wmma_m = 16
    wmma_n = 16
    wmma_k = 16
    warp_size = 32
    BM = config['BM']
    BN = config['BN']
    BK = config['BK']
    block_row_warps = config['block_row_warps']
    block_col_warps = config['block_col_warps']
    raster = config['raster']
    stage = config['stage']
    warp_row_tiles = BM // (wmma_m * block_row_warps)
    warp_col_tiles = BN // (wmma_n * block_col_warps)
    chunk = BK // (wmma_k)
    vec = 8
    shared_pad = 8
    sch = tvm.tir.Schedule(ir_module, debug_mask="all")
    block_b = sch.get_block("B")
    block_b_decompress = sch.get_block("B_decompress")
    block_b_rescale = sch.get_block("B_rescale")
    block_shared_A = sch.cache_read(block_b, 0, "shared")
    block_shared_local_A = sch.cache_read(block_b, 0, "wmma.matrix_a")
    block_shared_B = sch.cache_read(block_b, 1, "shared")
    block_shared_local_B = sch.cache_read(block_b, 1, "wmma.matrix_b")
    block_local_C = sch.cache_write(block_b, 0, "wmma.accumulator")

    sch.compute_inline(block_b_decompress)
    sch.compute_inline(block_b_rescale)

    (i, j, k) = sch.get_loops(block_b)
    i, kernel_i = sch.split(i, factors=[None, wmma_m])
    j, kernel_j = sch.split(j, factors=[None, wmma_n])
    k, kernel_k = sch.split(k, factors=[None, wmma_k])
    block_i, i, ii = sch.split(i, factors=[None, block_row_warps, warp_row_tiles])
    block_j, j, jj = sch.split(j, factors=[None, block_col_warps, warp_col_tiles])
    if raster > 0:
        block_j, block_k = sch.split(block_j, factors=[None, raster])
        ko, ki = sch.split(k, factors=[None, chunk])
        sch.reorder(block_k, block_i, block_j, i, j, ko, ki,
                    ii, jj, kernel_i, kernel_j, kernel_k)
        sch.bind(block_k, "blockIdx.z")
        sch.bind(block_i, "blockIdx.y")
        sch.bind(block_j, "blockIdx.x")
        sch.bind(i, "threadIdx.y")
        sch.bind(j, "threadIdx.z")
    else:
        ko, ki = sch.split(k, factors=[None, chunk])
        sch.reorder(block_i, block_j, i, j, ko, ki, ii,
                    jj, kernel_i, kernel_j, kernel_k)
        sch.bind(block_i, "blockIdx.y")
        sch.bind(block_j, "blockIdx.x")
        sch.bind(i, "threadIdx.y")
        sch.bind(j, "threadIdx.z")


    # cache read A from global memory to shared_memory
    sch.compute_at(block_shared_local_A, ki)
    sch.compute_at(block_shared_A, ko, preserve_unit_loops=True)
    sch.compute_at(block_shared_local_B, ki)
    sch.compute_at(block_shared_B, ko, preserve_unit_loops=True)
    sch.reverse_compute_at(block_local_C, j)


    A_shared_fused = sch.fuse(*sch.get_loops(block_shared_A)[-2:])
    A_shared_ty, A_shared_tz, A_shared_inner, A_shared_tx, A_shared_vi = sch.split(
        A_shared_fused, factors=[block_row_warps, block_col_warps, None, warp_size, vec])
    sch.vectorize(A_shared_vi)
    sch.bind(A_shared_tx, "threadIdx.x")
    sch.bind(A_shared_ty, "threadIdx.y")
    sch.bind(A_shared_tz, "threadIdx.z")
    sch.storage_align(block_shared_A, 0, axis=-2, factor=32, offset=shared_pad)

    B_shared_fused = sch.fuse(*sch.get_loops(block_shared_B)[-2:])
    B_shared_ty, B_shared_tz, B_shared_inner, B_shared_tx, B_shared_vi = sch.split(
        B_shared_fused, factors=[block_row_warps, block_col_warps, None, warp_size, 1])
    sch.vectorize(B_shared_vi)
    sch.bind(B_shared_tx, "threadIdx.x")
    sch.bind(B_shared_ty, "threadIdx.y")
    sch.bind(B_shared_tz, "threadIdx.z")
    sch.storage_align(block_shared_B, 0, axis=-2, factor=32, offset=shared_pad)



    A_local_i, A_local_j = sch.get_loops(block_shared_local_A)[-2:]
    A_local_i, A_local_kernel_i = sch.split(A_local_i, factors=[None, wmma_m])
    A_local_j, A_local_kernel_j = sch.split(A_local_j, factors=[None, wmma_k])
    sch.reorder(A_local_i, A_local_j, A_local_kernel_i, A_local_kernel_j)

    B_local_i, B_local_j = sch.get_loops(block_shared_local_B)[-2:]
    B_local_i, B_local_kernel_i = sch.split(B_local_i, factors=[None, wmma_n])
    B_local_j, B_local_kernel_j = sch.split(B_local_j, factors=[None, wmma_k])
    sch.reorder(B_local_i, B_local_j, B_local_kernel_i, B_local_kernel_j)

    C_local_i, C_local_j = sch.get_loops(block_local_C)[-2:]
    C_local_i, C_local_kernel_i = sch.split(C_local_i, factors=[None, wmma_m])
    C_local_j, C_local_kernel_j = sch.split(C_local_j, factors=[None, wmma_n])
    sch.reorder(C_local_i, C_local_j, C_local_kernel_i, C_local_kernel_j)

    # decompose reduction
    init_block_b = sch.decompose_reduction(block_b, ko)

    # transpose layout

    init_block_b_i, init_block_b_j = sch.get_loops(init_block_b)[-4:-2]
    sch.tensorize(sch.get_loops(init_block_b)[-2], WMMA_FILL_16x16x16_F16_INTRIN)

    block_shared_local_A_i, block_shared_local_A_j = sch.get_loops(
        block_shared_local_A)[-4:-2]
    sch.tensorize(sch.get_loops(block_shared_local_A)
                [-2], WMMA_LOAD_16x16x16_F16_A_INTRIN)

    block_shared_local_B_i, block_shared_local_B_j = sch.get_loops(
        block_shared_local_B)[-4:-2]
    sch.tensorize(sch.get_loops(block_shared_local_B)
                [-2], WMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN)
    sch.tensorize(kernel_i, WMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN)

    sch.tensorize(sch.get_loops(block_local_C)
                [-2], WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN)

    if stage > 1:

        sch.annotate(ki, ann_key="software_pipeline_stage", ann_val=[0, 0, 0])
        sch.annotate(ki, ann_key="software_pipeline_order", ann_val=[0, 1, 2])

        sch.annotate(ko, ann_key="software_pipeline_stage",
                    ann_val=[0, 0, 0, stage - 1, 0])
        sch.annotate(ko, ann_key="software_pipeline_order",
                    ann_val=[0, 1, 3, 2, 4])
        sch.annotate(ko, ann_key="software_pipeline_async_stages", ann_val=[0])


    ctx = tvm.cuda(0)
    with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
        cuda_mod = tvm.build(sch.mod, target="cuda")
    
    return cuda_mod


def get_gemm_workloads(bits, N, K, transposed=True):
    M = 512
    N = N
    K = K
    group_stride = 32 * bits // 8
    mask = (1 << bits) - 1
    if transposed:
        @tvm.script.ir_module
        class MyModule:
            @T.prim_func
            def main(a: T.handle, b: T.handle, c: T.handle, scales: T.handle, zeros: T.handle):
                T.func_attr({"global_symbol": "main", "tir.noalias": True})
                A = T.match_buffer(a, [M, K], dtype="float16")
                B = T.match_buffer(b, [N, K // 8 * bits], dtype="int8")
                C = T.match_buffer(c, [M, N], dtype="float16")
                Scales = T.match_buffer(scales, [N], dtype="float16")
                Zeros = T.match_buffer(zeros, [N], dtype="float16")

                B_decompress = T.alloc_buffer([N, K], dtype="float16")
                B_rescale = T.alloc_buffer([N, K], dtype="float16")

                for i, j in T.grid(N, K):
                    with T.block("B_decompress"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B_decompress[vi, vj] = T.Select(((vj % 32) * bits) % 8 <= 5, ((B[vi, (vj // 32) * group_stride + (vj % 32) * bits // 8] >> (((vj % 32) * bits) % 8) & mask)).astype("float16"), (((B[vi, (vj // 32) * group_stride + (vj % 32) * bits // 8] >> ((vj % 32) * bits) % 8) & (
                            1 << (8 - ((vj % 32) * bits) % 8)) - 1).astype("int8") | ((B[vi, (vj // 32) * group_stride + (vj % 32) * bits // 8 + 1] << (8 - ((vj % 32) * bits) % 8)) & (mask << (8 - ((vj % 32) * bits) % 8)) & mask).astype("int8")).astype("float16"))
                
                for i, j in T.grid(N, K):
                    with T.block("B_rescale"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B_rescale[vi, vj] = B_decompress[vi, vj] * \
                            Scales[vi].astype('float16') - Zeros[vi].astype('float16')
                            
                for i, j, k in T.grid(M, N, K):
                    with T.block("B"):
                        vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                        with T.init():
                            C[vi, vj] = T.float16(0)
                        C[vi, vj] = C[vi, vj] + \
                            A[vi, vk].astype("float16") * \
                            B_rescale[vj, vk].astype("float16")
    else:
        @tvm.script.ir_module
        class MyModule:
            @T.prim_func
            def main(a: T.handle, b: T.handle, c: T.handle, scales: T.handle, zeros: T.handle):
                T.func_attr({"global_symbol": "main", "tir.noalias": True})
                A = T.match_buffer(a, [M, K], dtype="float16")
                B = T.match_buffer(b, [K // 8 * bits, N], dtype="int8")
                C = T.match_buffer(c, [M, N], dtype="float16")
                Scales = T.match_buffer(scales, [N], dtype="float16")
                Zeros = T.match_buffer(zeros, [N], dtype="float16")

                B_decompress = T.alloc_buffer([K, N], dtype="float16")
                B_rescale = T.alloc_buffer([K, N], dtype="float16")

                for i, j in T.grid(K, N):
                    with T.block("B_decompress"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B_decompress[vi, vj] = T.Select(((vi % 32) * bits) % 8 <= 5, ((B[(vi // 32) * group_stride + (vi % 32) * bits // 8, vj] >> (((vi % 32) * bits) % 8) & mask)).astype("float16"), (((B[(vi // 32) * group_stride + (vi % 32) * bits // 8, vj] >> ((vi % 32) * bits) % 8) & (
                            1 << (8 - ((vi % 32) * bits) % 8)) - 1).astype("int8") | ((B[(vi // 32) * group_stride + (vi % 32) * bits // 8 + 1, vj] << (8 - ((vi % 32) * bits) % 8)) & (mask << (8 - ((vi % 32) * bits) % 8)) & mask).astype("int8")).astype("float16"))
                
                for i, j in T.grid(K, N):
                    with T.block("B_rescale"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B_rescale[vi, vj] = B_decompress[vi, vj] * \
                            Scales[vj].astype('float16') - Zeros[vj].astype('float16')
                            
                for i, j, k in T.grid(M, N, K):
                    with T.block("B"):
                        vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                        with T.init():
                            C[vi, vj] = T.float16(0)
                        C[vi, vj] = C[vi, vj] + \
                            A[vi, vk].astype("float16") * \
                            B_rescale[vk, vj].astype("float16")
    return MyModule

def get_gemv_workloads(bits, N, K, group_size=-1):
    group_stride = 32 * bits // 8
    M = 1
    N = N
    K = K
    group_size = K if group_size == -1 else group_size
    
    mask = (1 << bits) - 1
    vec = 8 if bits == 3 else 8
    num_warps = 4
    warp_size = 32

    @tvm.script.ir_module
    class MyModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle, scales: T.handle, zeros: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, [M, K], dtype="float16")
            B = T.match_buffer(b, [N, K // 8 * bits], dtype="int8")
            C = T.match_buffer(c, [M, N], dtype="float16")
            Scales = T.match_buffer(scales, [N, K // group_size], dtype="float16")
            Zeros = T.match_buffer(zeros, [N, K // group_size], dtype="float16")
            
            B_decompress = T.alloc_buffer([N, K], dtype="float16")
            B_rescale= T.alloc_buffer([N, K], dtype="float16")

            for i, j  in T.grid(N, K):
                with T.block("B_decompress"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B_decompress[vi, vj] = T.Select(((vj % 32) * bits) % 8 <= 5, ((B[vi, (vj // 32) * group_stride +  (vj % 32)* bits // 8] >> (((vj % 32) * bits) % 8) & mask)).astype("float16"), (((B[vi, (vj // 32) * group_stride +  (vj % 32)* bits // 8] >> ((vj % 32) * bits) % 8) & (1 << (8 - ((vj % 32) * bits) % 8)) - 1).astype("int8") | ((B[vi, (vj // 32) * group_stride +  (vj % 32)* bits // 8 + 1] << (8 - ((vj % 32) * bits) % 8)) & (mask << (8 - ((vj % 32) * bits) % 8)) & mask).astype("int8")).astype("float16")) 
            
            for i, j in T.grid(N, K):
                with T.block("B_rescale"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B_rescale[vi, vj] = B_decompress[vi, vj] * \
                        Scales[vi, vj // group_size].astype('float16') - Zeros[vi, vj // group_size].astype('float16')
            
            for i, j, k in T.grid(M, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = 0.0
                    C[vi, vj] = C[vi, vj] + \
                        A[vi, vk].astype("float16") * \
                        B_rescale[vj, vk].astype("float16")

    return MyModule

def get_gemv_workloads_nf4(bits, N, K, group_size=-1):

    M = 1
    bit = 4
    n_float_per_i8 = 8 // bit

    def _tir_u8_to_int(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return (val >> (pos * nbit).astype("int8")) & mask
        

    A = te.placeholder((M, K), name='A', dtype='float16')
    B = te.placeholder((N, K // 8 * bit), name='B', dtype='int8')
    LUT = te.placeholder((1 << bit, ), name='LUT', dtype='float16')
    Scales = te.placeholder((N, K // group_size), name='Scale', dtype='float16')
    
    def decode_func(n, k):
        w = _tir_u8_to_int(bit, B[n, k // n_float_per_i8], k % n_float_per_i8)
        return LUT[w]
    B_decode = te.compute(
        (N, K),
        decode_func,
        name='B_decompress'
    )
    B_rescale = te.compute(
      (N, K),
        lambda i, j: B_decode[i, j] * Scales[i, j // group_size],
        name='B_rescale'
    )  
    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name='k')
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B_rescale[j, k], axis=k),
        name='B'
    )
    return A, B, LUT, Scales, C

def get_gemm_workloads_nf4(bits, M, N, K, group_size=-1, transposed=True):
    bit = 4
    n_float_per_i8 = 8 // bit
    mask = (1 << bit) - 1
    if group_size == -1:
        group_size = K
    wmma_m = wmma_n = wmma_k = 16

    def _tir_u8_to_int(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return (val >> (pos * nbit).astype("int8")) & mask
        

    def ladder_gemm(M, N, K, wmma_m, wmma_n, wmma_k):
        A = te.placeholder((M // wmma_m, K // wmma_k, wmma_m ,wmma_k), name='A', dtype='float16')
        B = te.placeholder((N // wmma_n, K // wmma_k, wmma_n, wmma_k // 8 * bit), name='B', dtype='int8')
        LUT = te.placeholder((1 << bit, ), name='LUT', dtype='float16')
        Scales = te.placeholder((K // group_size, N), name='Scales', dtype='float16')
        def decode_func(n, k, nn, kk):
            w = _tir_u8_to_int(bit, B[n, k, nn, kk // n_float_per_i8], kk % n_float_per_i8)
            return LUT[w] * Scales[k // group_size, n]

        B_decode = te.compute(
            (N // wmma_n, K // wmma_k, wmma_n, wmma_k),
            decode_func,
            name='B_decode'
        )

        # Describe the matrix multiplication in TE
        k = te.reduce_axis((0, K // wmma_k), name='k')
        kk = te.reduce_axis((0, wmma_k), name='kk')

        C = te.compute(
            (M // wmma_m, N // wmma_n, wmma_m, wmma_n),
            lambda i, j, ii, jj: te.sum(A[i, k, ii, kk] * B_decode[j, k, jj, kk], axis=[k, kk]),
            name='C'
        )

        return A, B, LUT, Scales, C


    def reshape(M, N, wmma_m, wmma_n):
        C = te.placeholder((M // wmma_m, N // wmma_n, wmma_m, wmma_n), name='C', dtype='float16')
        C_reshape = te.compute(
            (M, N),
            lambda i, j: C[i // wmma_m, j // wmma_n, i % wmma_m, j % wmma_n],
            name='C_reshape'
        )
        return C, C_reshape
    arg1 = ladder_gemm(M, N, K, wmma_m, wmma_n, wmma_k)
    arg2 = reshape(M, N, wmma_m, wmma_n)
    args = arg1
    args = connect_tensor_graph(arg1, arg2, {arg2[0]:arg1[-1]})
    return args

def get_permutate_workloads(bits, M, N, K, group_size=-1, transposed=True):
    def layout_transform(_M, _N, wmma_m = 16, wmma_n = 16):
        A = te.placeholder((_M, _N), name='A', dtype='float16')
        B = te.compute(
            (_M // wmma_m, _N // wmma_n, wmma_m, wmma_n),
            lambda n, c, nn, cc: A[n * wmma_m + nn, c * wmma_n + cc],
            name='B'
        )
        return A, B

    def A_global_16x16_to_shared_load_16x16_layout(i, j):
        thread_id = i * 2 + j // 8
        row = thread_id % 16
        col = (j % 8) + (thread_id // 16) * 8
        return row, col

    def B_global_16x16_to_shared_load_16x16_layout(i, j):
        thread_id = i * 2 + j // 8
        row = (i // 8) * 8 + (thread_id % 8)
        col = (j % 8) + 8 * ((thread_id // 8) % 2)
        return row, col

    def layout_transform_with_func(_M, _N, wmma_m = 16, wmma_n = 16, func=None):
        def fcompute(*args):
            warp_i, warp_j = args[-2:]
            spatial_args = args[:-2]
            permutate_i, permutate_j = func(warp_i, warp_j)
            new_index = (*spatial_args, permutate_i, permutate_j)
            return A[new_index]
        A = te.placeholder((_M // wmma_m, _N // wmma_n, wmma_m, wmma_n), name='A', dtype='float16')
        B = te.compute(
            (_M // wmma_m, _N // wmma_n, wmma_m, wmma_n),
            fcompute,
            name='B'
        )
        return A, B

    arg1 = layout_transform(M, K)        
    arg2 = layout_transform_with_func(M, K, func=A_global_16x16_to_shared_load_16x16_layout)
    args = arg1
    args = tuple(connect_tensor_graph(args, arg2, {arg2[0]:args[-1]}))
    return args