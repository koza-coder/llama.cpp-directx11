// Shared helpers for the ggml D3D12 backend kernels.
//
// All tensors are bound as raw (byte address) buffers through root UAV descriptors.
// Element offsets/strides in the constant buffer are in elements of the buffer's type.
// Every kernel is written to compile at cs_6_0; USE_16BIT adds 16-bit loads/stores
// (SM 6.2 with -enable-16bit-types, only when the device reports Native16BitShaderOps).

#ifndef WG_SIZE
#define WG_SIZE 256
#endif

// D3D11 binds one buffer to one UAV slot only. Kernels name their buffers through UAV_SLOT(n); when slots n and k
// hold tensors of one buffer, the host compiles a variant with ALIAS_n=k and binds only slot k (one view over both).
// Element offsets stay correct because both tensors are bound from the buffer start (ggml_d3d11_bind_tensor).
#ifndef ALIAS_0
#define ALIAS_0 0
#endif
#ifndef ALIAS_1
#define ALIAS_1 1
#endif
#ifndef ALIAS_2
#define ALIAS_2 2
#endif
#ifndef ALIAS_3
#define ALIAS_3 3
#endif
#ifndef ALIAS_4
#define ALIAS_4 4
#endif
#ifndef ALIAS_5
#define ALIAS_5 5
#endif
#ifndef ALIAS_6
#define ALIAS_6 6
#endif
#ifndef ALIAS_7
#define ALIAS_7 7
#endif
#ifndef ALIAS_8
#define ALIAS_8 8
#endif
#ifndef ALIAS_9
#define ALIAS_9 9
#endif
#ifndef ALIAS_10
#define ALIAS_10 10
#endif
#ifndef ALIAS_11
#define ALIAS_11 11
#endif
RWByteAddressBuffer g_uav0 : register(u0);
RWByteAddressBuffer g_uav1 : register(u1);
RWByteAddressBuffer g_uav2 : register(u2);
RWByteAddressBuffer g_uav3 : register(u3);
RWByteAddressBuffer g_uav4 : register(u4);
RWByteAddressBuffer g_uav5 : register(u5);
RWByteAddressBuffer g_uav6 : register(u6);
RWByteAddressBuffer g_uav7 : register(u7);
RWByteAddressBuffer g_uav8 : register(u8);
RWByteAddressBuffer g_uav9 : register(u9);
RWByteAddressBuffer g_uav10 : register(u10);
RWByteAddressBuffer g_uav11 : register(u11);
#define UAV_CAT2(a, b) a##b
#define UAV_CAT(a, b)  UAV_CAT2(a, b)
#define UAV_SLOT(n)    UAV_CAT(g_uav, UAV_CAT(ALIAS_, n))



// flat thread index over a 2D dispatch grid: nwg_x groups along x
uint flat_index(uint3 gid, uint nwg_x) {
    return gid.x + nwg_x * WG_SIZE * gid.y;
}

// typed load/store on raw buffers; indices are element indices
#define LOAD_F32(buf, i)     asfloat((buf).Load((i) * 4))
#define STORE_F32(buf, i, v) (buf).Store((i) * 4, asuint(v))
#define LOAD_I32(buf, i)     asint((buf).Load((i) * 4))
#define STORE_I32(buf, i, v) (buf).Store((i) * 4, asuint(v))
// f16 loads go through f16tof32 so the compiler cannot narrow float math back to half ops
// (half division is approximate on some drivers); stores convert in software with
// round-to-nearest-even, because neither f32tof16 nor the (float16_t) cast rounds the same
// way on every driver (the cast rounds toward zero on the Radeon AI PRO R9700 driver)
uint f32_to_f16_rne(float f) {
    const uint x    = asuint(f);
    const uint sign = (x >> 16) & 0x8000u;
    const uint ax   = x & 0x7fffffffu;
    if (ax >= 0x7f800000u) {                       // inf, nan
        return sign | 0x7c00u | (ax > 0x7f800000u ? 0x200u : 0u);
    }
    if (ax >= 0x477ff000u) {                       // >= 65520 rounds to inf
        return sign | 0x7c00u;
    }
    if (ax < 0x38800000u) {                        // below 2^-14: f16 subnormal or zero
        const uint sh = 126u - (ax >> 23);         // shift of the 24-bit mantissa, >= 14
        if (sh > 24u) {
            return sign;
        }
        const uint m    = (ax & 0x7fffffu) | 0x800000u;
        uint       mant = m >> sh;
        const uint rem  = m & ((1u << sh) - 1u);
        const uint half = 1u << (sh - 1u);
        if (rem > half || (rem == half && (mant & 1u))) {
            mant++;
        }
        return sign | mant;
    }
    const uint mant = (ax & 0x7fffffu) >> 13;
    const uint rem  = ax & 0x1fffu;
    uint h = (((ax >> 23) - 112u) << 10) | mant;   // a mantissa carry rolls into the exponent
    if (rem > 0x1000u || (rem == 0x1000u && (mant & 1u))) {
        h++;
    }
    return sign | h;
}
#ifdef USE_16BIT
#define LOAD_F16(buf, i)     f16tof32((uint) (buf).Load<uint16_t>((i) * 2))
#define STORE_F16(buf, i, v) (buf).Store<uint16_t>((i) * 2, (uint16_t) f32_to_f16_rne(v))
#elif defined(GGML_D3D11)
// cs_5_0 has no 16-bit loads: read the word, and write a half with two atomics so the other half is kept
#define LOAD_F16(buf, i)     f16tof32((buf).Load(((i) * 2) & ~3u) >> ((((i) * 2) & 2u) * 8u))
#define STORE_F16(buf, i, v) { \
    const uint _h  = f32_to_f16_rne(v); /* before the And: v may load this half (in-place ops) */ \
    const uint _sh = (((i) * 2) & 2u) * 8u; \
    (buf).InterlockedAnd(((i) * 2) & ~3u, ~(0xFFFFu << _sh)); \
    (buf).InterlockedOr(((i) * 2) & ~3u, _h << _sh); \
}
#endif

// unaligned raw loads (ByteAddressBuffer.Load ignores the low 2 address bits)
// The high half used to be shifted by (32 - _s), which is a shift by 32 when the address happens
// to be 4-byte aligned. A shift of 32 is undefined, and because the ternary around it is a select
// rather than a branch both arms get evaluated, so a driver free to assume it cannot happen may
// fold the whole expression wrongly. This showed up as wrong iq2_xxs and iq3_xxs scales on two
// different GPUs. Keep every shift below 32 and drop the high half with a mask instead.
#define LOAD_U32_UNALIGNED(buf, addr, out) { \
    const uint _a0 = (addr) & ~3u; \
    const uint _s  = ((addr) & 3u) * 8u; \
    const uint _lo = (buf).Load(_a0); \
    const uint _hi = (buf).Load(_a0 + 4); \
    out = (_lo >> _s) | ((_hi << ((32u - _s) & 31u)) & (_s == 0u ? 0u : 0xFFFFFFFFu)); \
}
#define LOAD_U16_UNALIGNED(buf, addr, out) { \
    const uint _w = (buf).Load((addr) & ~3u); \
    out = ((_w >> (((addr) & 2u) * 8u)) & 0xFFFFu); \
}
uint byte_of(uint v, uint k) {
    return (v >> (k * 8u)) & 0xFFu;
}
int sbyte_of(uint v, uint k) {
    return ((int) (v << ((3u - k) * 8u))) >> 24;
}
