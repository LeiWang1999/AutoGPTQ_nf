__global__ void __launch_bounds__(128) Fused(half* __restrict__ A, int8_t* __restrict__ B, half* __restrict__ LUT, half* __restrict__ C) {
  
  __shared__ half LUT_shared[16];
  half C_shared_warp[16];
  __shared__ half A_shared[2048];
  __shared__ half B_decode_shared[4096];
  signed char B_local[4];
  half B_decode_local[8];
  half A_shared_warp[16];
  half B_decode_shared_warp[8];
  if (((int)threadIdx.x) < 16) {
    LUT_shared[((int)threadIdx.x)] = LUT[((int)threadIdx.x)];
  }

  const int MAX_BLOCK_N = 9;
  const auto baseBlockIdx = blockIdx.x + gridDim.x *blockIdx.y;
  const auto totalPanel = (gridDim.x * gridDim.y +MAX_BLOCK_N * gridDim.x - 1) / (MAX_BLOCK_N * gridDim.x);
  const auto totalBlock = gridDim.x * gridDim.y;
  const auto panelIdx = baseBlockIdx / (MAX_BLOCK_N *gridDim.x);
  const auto strideLd = panelIdx + 1 < totalPanel ?MAX_BLOCK_N : (totalBlock - panelIdx * (MAX_BLOCK_N *gridDim.x)) / gridDim.x;
  const auto bx = (panelIdx & 1) ? gridDim.x -(baseBlockIdx - panelIdx * MAX_BLOCK_N * gridDim.x) /strideLd - 1 : (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) / strideLd;
  const auto by = (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) % strideLd + panelIdx * MAX_BLOCK_N;
  const auto bz = blockIdx.z;
  const dim3 blockIdx(bx, by, bz);
  
  for (int i_2_init = 0; i_2_init < 2; ++i_2_init) {
    for (int j_2_init = 0; j_2_init < 1; ++j_2_init) {
      for (int i = 0; i < 8; ++i) {
C_shared_warp[(i_2_init * 8) + i] = 0.0;}
;
    }
  }
  for (int k_0 = 0; k_0 < 448; ++k_0) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_ax2_ax3_0_fused_0 = 0; ax0_ax1_ax2_ax3_0_fused_0 < 2; ++ax0_ax1_ax2_ax3_0_fused_0) {
      *(uint4*)(A_shared + (((ax0_ax1_ax2_ax3_0_fused_0 * 1024) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(A + ((((ax0_ax1_ax2_ax3_0_fused_0 * 458752) + (k_0 * 1024)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8)));
    }
    for (int ax0_ax1_ax2_ax3_0_fused_0_1 = 0; ax0_ax1_ax2_ax3_0_fused_0_1 < 4; ++ax0_ax1_ax2_ax3_0_fused_0_1) {
      *(int*)(B_local + 0) = *(int*)(B + (((((((int)blockIdx.x) * 917504) + (ax0_ax1_ax2_ax3_0_fused_0_1 * 229376)) + (k_0 * 512)) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.x) * 4)));
      for (int ax0 = 0; ax0 < 8; ++ax0) {
        B_decode_local[ax0] = LUT_shared[((int)((B_local[(ax0 >> 1)] >> ((signed char)((ax0 & 1) * 4))) & (signed char)15))];
      }
      *(uint4*)(B_decode_shared + (((ax0_ax1_ax2_ax3_0_fused_0_1 * 1024) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(B_decode_local + 0);
    }
    __syncthreads();
    for (int k_1 = 0; k_1 < 4; ++k_1) {
      for (int ax0_1 = 0; ax0_1 < 2; ++ax0_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(A_shared[((ax0_1 * 1024) + (k_1 * 256))])) + (((int)threadIdx.x) * 8))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_shared[((ax0_1 * 1024) + (k_1 * 256))])) + (((int)threadIdx.x) * 8)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_shared_warp + (ax0_1 * 8)))[0]), "=r"(((unsigned *)(A_shared_warp + (ax0_1 * 8)))[1]), "=r"(((unsigned *)(A_shared_warp + (ax0_1 * 8)))[2]), "=r"(((unsigned *)(A_shared_warp + (ax0_1 * 8)))[3])
      : "r"(addr)
    );
  }
      }

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(B_decode_shared[((((int)threadIdx.z) * 1024) + (k_1 * 256))])) + (((int)threadIdx.x) * 8))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(B_decode_shared[((((int)threadIdx.z) * 1024) + (k_1 * 256))])) + (((int)threadIdx.x) * 8)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_decode_shared_warp + 0))[0]), "=r"(((unsigned *)(B_decode_shared_warp + 0))[1]), "=r"(((unsigned *)(B_decode_shared_warp + 0))[2]), "=r"(((unsigned *)(B_decode_shared_warp + 0))[3])
      : "r"(addr)
    );
  }
      for (int i_2 = 0; i_2 < 2; ++i_2) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(C_shared_warp + (i_2 * 8)))[0]), "=r"(((unsigned *)(C_shared_warp + (i_2 * 8)))[1])
      : "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[3]), "r"(((unsigned *)(B_decode_shared_warp + 0))[0]), "r"(((unsigned *)(B_decode_shared_warp + 0))[1]), "r"(((unsigned *)(C_shared_warp + (i_2 * 8)))[0]), "r"(((unsigned *)(C_shared_warp + (i_2 * 8)))[1]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(C_shared_warp + ((i_2 * 8) + 4)))[0]), "=r"(((unsigned *)(C_shared_warp + ((i_2 * 8) + 4)))[1])
      : "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[3]), "r"(((unsigned *)(B_decode_shared_warp + 4))[0]), "r"(((unsigned *)(B_decode_shared_warp + 4))[1]), "r"(((unsigned *)(C_shared_warp + ((i_2 * 8) + 4)))[0]), "r"(((unsigned *)(C_shared_warp + ((i_2 * 8) + 4)))[1]));
  }
      }
    }
  }
  for (int ax0_2 = 0; ax0_2 < 2; ++ax0_2) {
    __syncthreads();
    for (int local_id = 0; local_id < 8; local_id+=2) {
*((uint *)&(&(A_shared[(((int)threadIdx.z) * 256)]))[((((((local_id % 4) / 2) * 8) + (threadIdx.x / 4)) * 16) + ((((local_id / 4) * 8) + ((threadIdx.x % 4) * 2)) + (local_id % 2)))]) = *((uint *)&C_shared_warp[(ax0_2 * 8) + local_id]);
}
;
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_ax2_ax3_fused_0 = 0; ax0_ax1_ax2_ax3_fused_0 < 1; ++ax0_ax1_ax2_ax3_fused_0) {
      *(uint4*)(C + ((((ax0_2 * 131072) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(A_shared + ((((int)threadIdx.z) * 256) + (((int)threadIdx.x) * 8)));
    }
  }
}

