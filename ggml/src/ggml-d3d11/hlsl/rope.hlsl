#include "common.hlsli"

// defines: TYPE_F32 or TYPE_F16 (+USE_16BIT), FF_FUNC when frequency factors are present,
// BACKWARD for ROPE_BACK, which is this same rotation with the sine negated

#ifdef BACKWARD
#define SIN_SIGN (-1.0f)
#else
#define SIN_SIGN (1.0f)
#endif

#if defined(TYPE_F16)
#define LOAD(buf, i)     LOAD_F16(buf, i)
#define STORE(buf, i, v) STORE_F16(buf, i, v)
#else
#define LOAD(buf, i)     LOAD_F32(buf, i)
#define STORE(buf, i, v) STORE_F32(buf, i, v)
#endif

#define src0  UAV_SLOT(0)
#define src1  UAV_SLOT(1)   // positions, i32
#define src2  UAV_SLOT(2)   // frequency factors, f32
#define dst   UAV_SLOT(3)

cbuffer Params : register(b0) {
    uint offset_src0;
    uint offset_src1;
    uint offset_src2;
    uint offset_dst;

    uint stride_src01;
    uint stride_src02;
    uint stride_src03;

    uint stride_dst1;
    uint stride_dst2;
    uint stride_dst3;

    uint n_threads;
    uint ne0;
    uint ne1;
    uint ne2;

    uint n_dims;
    uint mode;
    float theta_scale;
    float attn_factor;
    float freq_scale;
    float ext_factor;
    float corr_dim0;
    float corr_dim1;
    uint sections0;
    uint sections1;
    uint sections2;
    uint sections3;
    uint n_offs;

    uint nwg_x;
};

float freq_factor(uint i) {
#ifdef FF_FUNC
    return LOAD_F32(src2, offset_src2 + i / 2);
#else
    return 1.0f;
#endif
}

float rope_yarn_ramp(float low, float high, uint i) {
    const float y = ((float) (i / 2) - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

float2 rope_yarn(float theta_extrap, uint i) {
    float mscale = attn_factor;
    float theta  = freq_scale * theta_extrap;
    if (ext_factor != 0.0f) {
        const float ramp_mix = rope_yarn_ramp(corr_dim0, corr_dim1, i) * ext_factor;
        theta = theta * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    return float2(cos(theta) * mscale, SIN_SIGN * sin(theta) * mscale);
}

// one rotation unit t: dst element indices p0, p1 and their values
void rope_unit(uint t, out uint p0, out float v0, out uint p1, out float v1) {
    const bool is_neox   = (mode & 2u) != 0;
    const bool is_mrope  = (mode & 8u) != 0;
    const bool is_imrope = mode == 40u;
    const bool is_vision = mode == 24u;

    uint i = t * 2;
    const uint i3 = i / (ne2 * ne1 * ne0);
    i = i % (ne2 * ne1 * ne0);
    const uint i2 = i / (ne1 * ne0);
    i = i % (ne1 * ne0);
    const uint i1 = i / ne0;
    const uint i0 = i % ne0;

    const uint src_row = offset_src0 + i3 * stride_src03 + i2 * stride_src02 + i1 * stride_src01;
    const uint dst_row = offset_dst + i3 * stride_dst3 + i2 * stride_dst2 + i1 * stride_dst1;

    if ((i0 < n_offs || i0 >= n_offs + n_dims) && !is_vision) {
        p0 = dst_row + i0;
        v0 = LOAD(src0, src_row + i0);
        p1 = dst_row + i0 + 1;
        v1 = LOAD(src0, src_row + i0 + 1);
        return;
    }

    const uint iw = i0 - n_offs;
    uint theta_base_mult = 0;
    uint theta_scale_pwr = iw / 2;
    if (is_mrope) {
        const uint sect_dims = sections0 + sections1 + sections2 + sections3;
        const uint sec_w     = sections1 + sections0;
        const uint sec_e     = sections2 + sec_w;
        const uint sector    = (iw / 2) % sect_dims;
        if (is_imrope) {
            if (sector % 3 == 1 && sector < 3 * sections1) {
                theta_base_mult = 1;
            } else if (sector % 3 == 2 && sector < 3 * sections2) {
                theta_base_mult = 2;
            } else if (sector % 3 == 0 && sector < 3 * sections0) {
                theta_base_mult = 0;
            } else {
                theta_base_mult = 3;
            }
        } else {
            if (sector >= sections0 && sector < sec_w) {
                theta_base_mult = 1;
                if (is_vision) { theta_scale_pwr = sector - sections0; }
            } else if (sector >= sec_w && sector < sec_e) {
                theta_base_mult = 2;
                if (is_vision) { theta_scale_pwr = sector - sec_w; }
            } else if (sector >= sec_e) {
                if (is_vision) { theta_scale_pwr = (iw / 2) % sec_e; }
                theta_base_mult = 3;
            } else if (is_vision) {
                theta_scale_pwr = sector;
            }
        }
    }
    const float pos        = (float) LOAD_I32(src1, offset_src1 + i2 + ne2 * theta_base_mult);
    const float theta_base = pos * pow(abs(theta_scale), (float) theta_scale_pwr);
    const float2 thetas    = rope_yarn(theta_base / freq_factor(iw), iw);

    const bool div2 = is_neox || is_mrope || is_vision;
    const uint pair_base = div2 ? (i0 / 2 + n_offs / 2) : i0;
    const uint pair_off  = is_vision ? n_dims : ((is_neox || is_mrope) ? n_dims / 2 : 1);

    const float x0 = LOAD(src0, src_row + pair_base);
    const float x1 = LOAD(src0, src_row + pair_base + pair_off);
    p0 = dst_row + pair_base;
    v0 = x0 * thetas.x - x1 * thetas.y;
    p1 = dst_row + pair_base + pair_off;
    v1 = x0 * thetas.y + x1 * thetas.x;
}

#if defined(F16_WORDS)
// exp1105: two units per thread, whole 32-bit words, no atomics (host: ne0, n_dims, n_offs % 4 == 0, dst offset/strides even)
void store_word(uint p, float lo, float hi) {
    dst.Store(p * 2u, f32_to_f16_rne(lo) | (f32_to_f16_rne(hi) << 16));
}

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint t = 2u * flat_index(gid, nwg_x);
    if (t >= n_threads) {
        return;
    }
    uint  a0, a1, b0, b1;
    float va0, va1, vb0, vb1;
    rope_unit(t, a0, va0, a1, va1);
    rope_unit(t + 1u, b0, vb0, b1, vb1);
    if (a1 == a0 + 1u) {        // normal mode or pass-through: each unit owns one word
        store_word(a0, va0, va1);
        store_word(b0, vb0, vb1);
    } else {                    // neox / mrope / vision: b0 == a0 + 1, b1 == a1 + 1
        store_word(a0, va0, vb0);
        store_word(a1, va1, vb1);
    }
}
#else
[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint t = flat_index(gid, nwg_x);
    if (t >= n_threads) {
        return;
    }
    uint  p0, p1;
    float v0, v1;
    rope_unit(t, p0, v0, p1, v1);
    STORE(dst, p0, v0);
    STORE(dst, p1, v1);
}
#endif
