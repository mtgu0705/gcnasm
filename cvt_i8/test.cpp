#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <numeric>
#include <cmath>
#include <unordered_map>

using fp16 = _Float16;

#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                       \
} while(0)

template <int BLOCK_SIZE = 256>
__global__ void cvt_i4x8_fp8x8(const void* ptr_input_i8, void* ptr_out_f16, int pixels)
{
    int cur_pixel = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if(cur_pixel * 4 > pixels)
        return;
    uint32_t uint8_4 = reinterpret_cast<const uint32_t*>(ptr_input_i8)[cur_pixel];
    uint32_t fp16x2_0;
    uint32_t fp16x2_1;

    static constexpr uint32_t byte_selector_01 = 0x05010500;
    static constexpr uint32_t byte_selector_23 = 0x05030502;
    static constexpr uint32_t fp16_adder       = 0x64646464;
    fp16x2_0 = __builtin_amdgcn_perm(fp16_adder, uint8_4, byte_selector_01);
    fp16x2_1 = __builtin_amdgcn_perm(fp16_adder, uint8_4, byte_selector_23);

    printf("tid:%d, i4x8:0x%x, %x, %x\n", static_cast<int>(threadIdx.x), uint8_4, fp16x2_0, fp16x2_1);

    static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
    asm volatile("v_pk_add_f16 %0, %1, %2 neg_lo:[0,1] neg_hi:[0,1]"
                 : "=v"(fp16x2_0)
                 : "v"(fp16x2_0), "s"(I8s_TO_F16s_MAGIC_NUM));
    asm volatile("v_pk_add_f16 %0, %1, %2 neg_lo:[0,1] neg_hi:[0,1]"
                 : "=v"(fp16x2_1)
                 : "v"(fp16x2_1), "s"(I8s_TO_F16s_MAGIC_NUM));
    
    printf("tid:%d, 0x%x, 0x%x\n", static_cast<int>(threadIdx.x), fp16x2_0, fp16x2_1);

    reinterpret_cast<uint32_t*>(ptr_out_f16)[cur_pixel * 2 + 0] = fp16x2_0;
    reinterpret_cast<uint32_t*>(ptr_out_f16)[cur_pixel * 2 + 1] = fp16x2_1;
}

int main(int argc, char ** argv)
{
    int pixels = 256 * 4;
    int i8_bytes = pixels;
    int f16_bytes = pixels;

    uint8_t *host_src;
    fp16 *host_dst;
    void *device_src, *device_dst;

    //fp32 on host
    host_src = (uint8_t*)malloc(i8_bytes*sizeof(uint8_t));
    host_dst = (fp16*)malloc(f16_bytes*sizeof(fp16));

    //convert fp32 a and b into fp16 on host
    for(auto i = 0; i < i8_bytes; i++) {
        uint8_t u8 = static_cast<uint8_t>(i%256 & 0xff);
        host_src[i] = u8;
    }

    HIP_CALL(hipMalloc(&device_src, i8_bytes * sizeof(uint8_t)));
    HIP_CALL(hipMalloc(&device_dst, f16_bytes * sizeof(fp16)));

    HIP_CALL(hipMemcpy(device_src, host_src, i8_bytes * sizeof(uint8_t), hipMemcpyHostToDevice));
    constexpr int block_size = 256;
    constexpr int pixels_per_block  = block_size * 4;

    cvt_i4x8_fp8x8<<<(pixels + pixels_per_block - 1) / pixels_per_block, block_size>>>(device_src, device_dst, pixels);

    HIP_CALL(hipMemcpy(host_dst, device_dst, f16_bytes*sizeof(fp16), hipMemcpyDeviceToHost));

    for(auto i = 0 ;i < i8_bytes; i++) {
        uint8_t i8 = host_src[i] & 0xff;
        printf("[%3d]%d -> 0x%04x(%f)\n",i, i8, *reinterpret_cast<uint16_t*>(&host_dst[i]), static_cast<float>(host_dst[i]));
    }
}
