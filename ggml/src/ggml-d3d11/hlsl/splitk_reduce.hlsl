#include "common.hlsli"

// exp1055: split-K reduction. The tiled MUL_MAT wrote n_split partial products [m, n] one after another into tmp;
// dst[offset_dst + c * dst_ld + r] = sum over s of tmp[s * m * n + c * m + r]

#define tmp UAV_SLOT(0)
#define dst UAV_SLOT(1)

cbuffer Params : register(b0) {
    uint offset_dst;
    uint m;
    uint n;
    uint n_split;
    uint dst_ld;
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint i = flat_index(gid, nwg_x);
    if (i >= m * n) {
        return;
    }
    const uint r = i % m;
    const uint c = i / m;
    float s = 0.0f;
    for (uint z = 0; z < n_split; z++) {
        s += LOAD_F32(tmp, z * m * n + i);
    }
    STORE_F32(dst, offset_dst + c * dst_ld + r, s);
}
