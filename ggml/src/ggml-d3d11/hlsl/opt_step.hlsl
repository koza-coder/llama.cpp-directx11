#include "common.hlsli"

// OPT_STEP_SGD and OPT_STEP_ADAMW (f32): the optimiser step, applied in place to the weights.
//
//   SGD:   w = w*(1 - alpha*wd) - alpha*g
//   AdamW: m = m*beta1 + g*(1-beta1);  v = v*beta2 + g*g*(1-beta2)
//          w = w*(1 - alpha*wd) - alpha * (m*beta1h) / (sqrt(v*beta2h) + eps)
//
// The hyperparameters arrive in a tensor rather than in op_params, so they are read on the GPU and
// nothing has to be mapped back to the host. m and v are updated in place as well.
// Every element is independent: one thread each. defines: ADAMW

#define w_buf  UAV_SLOT(0)   // weights, written in place
#define g_buf  UAV_SLOT(1)   // gradients
#define m_buf  UAV_SLOT(2)   // first moment  (ADAMW only, else bound to g)
#define v_buf  UAV_SLOT(3)   // second moment (ADAMW only, else bound to g)
#define p_buf  UAV_SLOT(4)   // 7 floats for AdamW, 2 for SGD

cbuffer Params : register(b0) {
    uint offset_w;
    uint offset_g;
    uint offset_m;
    uint offset_v;
    uint offset_p;
    uint ne;
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const float alpha = LOAD_F32(p_buf, offset_p + 0);
    const float g     = LOAD_F32(g_buf, offset_g + i);
    float w = LOAD_F32(w_buf, offset_w + i);

#ifdef ADAMW
    const float beta1  = LOAD_F32(p_buf, offset_p + 1);
    const float beta2  = LOAD_F32(p_buf, offset_p + 2);
    const float eps    = LOAD_F32(p_buf, offset_p + 3);
    const float wd     = LOAD_F32(p_buf, offset_p + 4);
    const float beta1h = LOAD_F32(p_buf, offset_p + 5);
    const float beta2h = LOAD_F32(p_buf, offset_p + 6);

    const float m = LOAD_F32(m_buf, offset_m + i) * beta1 + g * (1.0f - beta1);
    const float v = LOAD_F32(v_buf, offset_v + i) * beta2 + g * g * (1.0f - beta2);
    STORE_F32(m_buf, offset_m + i, m);
    STORE_F32(v_buf, offset_v + i, v);

    const float mh = m * beta1h;
    const float vh = sqrt(v * beta2h) + eps;
    // the weight decay is applied independently of the momenta, which is not l2 regularisation
    w = w * (1.0f - alpha * wd) - alpha * mh / vh;
#else
    const float wd = LOAD_F32(p_buf, offset_p + 1);
    w = w * (1.0f - alpha * wd) - alpha * g;
#endif
    STORE_F32(w_buf, offset_w + i, w);
}
