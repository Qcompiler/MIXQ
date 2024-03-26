#include <iostream>

__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void *pointer) {
  uint32_t address;
  asm("{\n\t"
      "  .reg .u64 u64addr;\n\t"
      "  cvta.to.shared.u64 u64addr, %1;\n\t"
      "  cvt.u32.u64 %0, u64addr;\n\t"
      "}"
      : "=r"(address)
      : "l"(pointer));
  return address;
}

__device__ __forceinline__ void ldmatrix_sync_aligned_m8n8_x4_b16(
    uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3,
    const uint32_t &address) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      : "r"(address));
}

__global__ void ldmatrix_test() {
  __shared__ uint16_t smem[4096];
  auto tid = threadIdx.x;
  // bank: tid * 64 * 2 / 4 % 32 = 0
  // T0 read: smem[0],   ..., smem[7]   -> bank 0 to bank 3
  // T1 read: smem[64],  ..., smem[71]  -> bank 0 to bank 3
  // T2 read: smem[128], ..., smem[135] -> bank 0 to bank 3
  // T3 read: smem[192], ..., smem[199] -> bank 0 to bank 3
  // T4 read: smem[256], ..., smem[263] -> bank 0 to bank 3
  // T5 read: smem[320], ..., smem[327] -> bank 0 to bank 3
  // T6 read: smem[384], ..., smem[391] -> bank 0 to bank 3
  // T7 read: smem[448], ..., smem[455] -> bank 0 to bank 3
  const uint32_t address =
      cvta_to_shared_u32(smem) + sizeof(uint16_t) * (64 * tid);
  for (uint32_t i = tid; i < 4096; i += blockDim.x) {
    smem[i] = i;
  }
  __syncthreads();

  uint32_t value[4];
  // each thread  uses 4 * uint32 RF 读取 8个 uint16_t
  ldmatrix_sync_aligned_m8n8_x4_b16(
      *reinterpret_cast<uint32_t *>(value + 0),
      *reinterpret_cast<uint32_t *>(value + 1),
      *reinterpret_cast<uint32_t *>(value + 2),
      *reinterpret_cast<uint32_t *>(value + 3),
      address);
 
    printf("value %d = %d\t",threadIdx.x, value[0]);
 
}

int main() {
  uint16_t *d_value;
  cudaMalloc(&d_value, sizeof(uint16_t));
  ldmatrix_test<<<1, 32>>>();
  cudaDeviceSynchronize();
  cudaFree(d_value);
  return 0;
}
