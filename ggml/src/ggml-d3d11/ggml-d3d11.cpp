// ggml Direct3D 11 backend
//
// A port of the D3D12 backend to D3D11, for GPUs and drivers where D3D12 is missing or broken:
//  - the op encoders and the HLSL kernels are the D3D12 backend's; kernels compile at runtime to cs_5_0
//    with d3dcompiler_47.dll (FXC), which is part of Windows
//  - every tensor buffer gets a fake GPU address; a dispatch turns each address into a cached raw UAV
//  - one immediate context, no explicit barriers (D3D11 orders dependent dispatches itself)

#include "ggml-d3d11.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "ggml-d3d11-shaders.hpp"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d11_1.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef GGML_D3D11_DEBUG
#define D3D11_LOG_DEBUG(...) GGML_LOG_DEBUG("ggml_d3d11: " __VA_ARGS__)
#else
#define D3D11_LOG_DEBUG(...)
#endif

// Below 16 columns the matvec kernel is as fast or faster (MTT S80 pp8 -9%, R9700 pp2 -27%, 2026-09-21)
#ifndef GGML_D3D11_TILED_DEFAULT
#define GGML_D3D11_TILED_DEFAULT    16
#endif
#define D3D11_WG_SIZE               256
#define D3D11_MAX_WG_PER_DIM        65535
#define D3D11_MAX_ROOT_UAVS         12   // 2 DWORDs each in the root signature
#define D3D11_VA_WRITTEN            (1ull << 62)   // tag bit on a binding address: the dispatch writes this tensor
#define D3D11_BINDING_ALIGNMENT     256   // root CBV alignment, also used for tensor UAV base addresses
#define D3D11_PARAM_SLOT_SIZE       256
#define D3D11_PARAM_SLOT_COUNT      8192
#define D3D11_QUERY_CAPACITY        (2 * D3D11_PARAM_SLOT_COUNT)
#define D3D11_STAGING_SIZE          (64ull * 1024 * 1024)
#define D3D11_MEMSET_BYTES_PER_THREAD 16
// Windows moves whole buffers to system memory when a process is over its VRAM budget. With 256 MiB
// buffers less of the hot data moved (MTT S80, a 13.7 GB model over a 13 GB budget: tg32 5.5 -> 8.5 t/s).
#define D3D11_DEFAULT_MAX_ALLOC     (256ull * 1024 * 1024)
#define D3D11_DEFAULT_SUBMIT_BATCH  64    // dispatches per command list (GGML_D3D11_SUBMIT_BATCH)

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
// work budget per submission for kernels with long per-thread loops (flash attention, gated delta net,
// argsort): small enough that one command list stays well inside the Windows GPU timeout on slow cards
#define D3D11_FLASH_ATTN_WORK (1ull << 25)
#define D3D11_FLASH_ATTN_BLK      32                    // KV entries per flash attention block thread
#define D3D11_FLASH_ATTN_TMP_MAX  (64ull * 1024 * 1024)   // cap on the block results buffer

/* Minimal COM smart pointer (avoids a WRL dependency for MinGW builds) */

template <typename T> struct com_ptr {
    T * p = nullptr;

    com_ptr() = default;
    com_ptr(const com_ptr & o) : p(o.p) { if (p) { p->AddRef(); } }
    com_ptr(com_ptr && o) noexcept : p(o.p) { o.p = nullptr; }
    com_ptr & operator=(const com_ptr & o) {
        if (this != &o) { reset(); p = o.p; if (p) { p->AddRef(); } }
        return *this;
    }
    com_ptr & operator=(com_ptr && o) noexcept {
        if (this != &o) { reset(); p = o.p; o.p = nullptr; }
        return *this;
    }
    ~com_ptr() { reset(); }

    void reset() { if (p) { p->Release(); p = nullptr; } }
    T ** put() { reset(); return &p; }
    T * get() const { return p; }
    T * operator->() const { return p; }
    explicit operator bool() const { return p != nullptr; }
};

static double ggml_d3d11_time_us() {
    static LARGE_INTEGER freq = {};
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double) now.QuadPart * 1e6 / (double) freq.QuadPart;
}

static void ggml_d3d11_check(HRESULT hr, const char * what) {
    if (FAILED(hr)) {
        GGML_LOG_ERROR("ggml_d3d11: %s failed with HRESULT 0x%08lx\n", what, (unsigned long) hr);
        GGML_ABORT("ggml_d3d11: %s failed", what);
    }
}

static std::string ggml_d3d11_wide_to_utf8(const wchar_t * w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? n - 1 : 0, '\0');
    if (n > 1) {
        WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], n, nullptr, nullptr);
    }
    return s;
}

static std::wstring ggml_d3d11_utf8_to_wide(const std::string & s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(n > 0 ? n - 1 : 0, L'\0');
    if (n > 1) {
        MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    }
    return w;
}

// All device buffers report the same fake host base pointer; a tensor's byte offset inside its
// buffer is recovered from tensor->data (same scheme as the WebGPU backend).
static void * const d3d11_ptr_base = (void *) (uintptr_t) 0x1000;  // NOLINT

static size_t ggml_d3d11_tensor_offset(const ggml_tensor * tensor) {
    const ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
    return (size_t) ((uintptr_t) base_tensor->data - (uintptr_t) d3d11_ptr_base) + tensor->view_offs;
}

/* Structs */

// the D3D12 op encoders pass GPU addresses and shader model levels; D3D11 has neither, so both are emulated
typedef uint64_t D3D11_GPU_VIRTUAL_ADDRESS;
enum D3D_SHADER_MODEL { D3D_SHADER_MODEL_6_0 = 0x60 };
enum d3d11_heap_type { D3D11_HEAP_TYPE_DEFAULT };

struct d3d11_caps {
    D3D_SHADER_MODEL shader_model = D3D_SHADER_MODEL_6_0;
    bool             native_16bit = true;    // f16 loads and stores are emulated with 32-bit words
    bool             wave_ops     = false;
    uint32_t         wave_min     = 0;
    uint32_t         wave_max     = 0;
    bool             uma          = false;
};

struct d3d11_pipeline {
    com_ptr<ID3D11ComputeShader> cs;
    std::string                  name;
};

struct d3d11_device_ctx;

// a device buffer with a fake GPU address; reference counted so that com_ptr can hold it
struct d3d11_res {
    std::atomic<long>     refs{ 1 };
    d3d11_device_ctx *    dev  = nullptr;
    com_ptr<ID3D11Buffer> buf;
    uint64_t              va   = 0;
    size_t                size = 0;

    void     AddRef() { refs++; }
    void     Release();
    uint64_t GetGPUVirtualAddress() const { return va; }
};

struct d3d11_device_ctx {
    std::string name;   // "D3D110"
    std::string desc;   // adapter description
    uint32_t    vendor_id     = 0;
    size_t      dedicated_mem = 0;
    size_t      shared_mem    = 0;
    size_t      max_alloc     = D3D11_DEFAULT_MAX_ALLOC;
    d3d11_caps  caps;

    com_ptr<IDXGIAdapter1>       adapter;
    com_ptr<ID3D11Device>        device;
    com_ptr<ID3D11DeviceContext> ctx;
    com_ptr<ID3D11Buffer>        cbuf;        // kernel parameters, rewritten before every dispatch
    com_ptr<ID3D11Query>         done_query;  // event query for submit_and_wait
    uint32_t                     max_uavs = 8;
    com_ptr<ID3D11InfoQueue>     info;        // debug layer messages, with GGML_D3D11_DEBUG

    // fake address space: buffer base -> buffer, and address -> raw UAV starting there
    uint64_t                                                          next_va = 1ull << 40;
    std::map<uint64_t, d3d11_res *>                                   res_map;
    std::map<std::pair<uint64_t, uint64_t>, com_ptr<ID3D11UnorderedAccessView>> uavs;   // (start, end)
    std::unordered_map<uint64_t, uint64_t> bind_end;   // start -> end of the tensors bound for the next dispatch
    // outputs of the nodes being encoded: a dispatch at node i can only write nodes i and after (fused ones)
    const ggml_tensor *                    out_nodes[8] = {};
    com_ptr<d3d11_res> alias_copy[D3D11_MAX_ROOT_UAVS];   // per slot: copy of a slot that shares a buffer

    // staging for set/get tensor
    com_ptr<ID3D11Buffer> readback_buf;

    // flash attention block results, grown on demand
    com_ptr<d3d11_res> fa_tmp;
    size_t             fa_tmp_size = 0;
    uint64_t           fa_work     = D3D11_FLASH_ATTN_WORK;   // adapted to the measured submit time
    bool               fa_work_fixed = false;                 // GGML_D3D11_FA_WORK: no adaptation

    uint32_t submit_batch       = D3D11_DEFAULT_SUBMIT_BATCH;
    uint32_t dispatches_in_list = 0;

    std::unordered_map<std::string, d3d11_pipeline> pipelines;

    // parallel compilation: a collect pass over a new graph queues the missing pipelines, worker threads build them
    struct pipeline_job {
        std::string              key;
        const char *             source;
        std::vector<std::string> defines;
        D3D_SHADER_MODEL         min_sm;
    };
    bool                      collecting       = false;
    std::vector<pipeline_job> pipeline_jobs;
    std::mutex                pipelines_mutex;
    int                       last_graph_nodes = -1;

    bool recording = false;

    // counters printed at exit when GGML_D3D11_STATS is set
    bool     stats            = false;
    uint64_t n_graphs         = 0;
    uint64_t n_nodes          = 0;
    uint64_t n_dispatches     = 0;
    uint64_t n_submits        = 0;
    uint64_t n_set_tensor     = 0;
    uint64_t n_get_tensor     = 0;
    uint64_t bytes_set        = 0;
    uint64_t bytes_get        = 0;
    double   t_wait_us        = 0;
    double   t_submit_us      = 0;
    double   t_compile_us     = 0;
    uint64_t n_compiles       = 0;
    bool     no_fuse          = false;
    uint32_t mm_tpr_max       = D3D11_WG_SIZE;
    uint32_t tiled_min_cols   = GGML_D3D11_TILED_DEFAULT;
    com_ptr<d3d11_res> mmid_scratch;
    size_t             mmid_scratch_size = 0;
    std::string disable_ops;
    std::mutex  rejected_mutex;
    std::map<std::string, uint64_t> rejected;
    uint64_t n_flush_arena    = 0;
    uint64_t n_flush_batch    = 0;
    double   t_graph_us       = 0;
    double   t_graph_submit_us = 0;  // the part of t_submit_us spent inside graph_compute
    double   t_graph_wait_us   = 0;  // the part of t_wait_us spent inside graph_compute
    double   t_set_us         = 0;
    double   t_get_us         = 0;

    bool                                               profile = false;   // not implemented for D3D11
    std::map<std::string, std::pair<double, uint64_t>> prof;

    std::recursive_mutex mutex;

    ggml_backend_buffer_type buft = {};
};

void d3d11_res::Release() {
    if (--refs > 0) {
        return;
    }
    if (dev) {
        std::lock_guard<std::recursive_mutex> lock(dev->mutex);
        dev->res_map.erase(va);
        dev->uavs.erase(dev->uavs.lower_bound({ va, 0 }), dev->uavs.lower_bound({ va + size, 0 }));
    }
    delete this;
}

struct ggml_backend_d3d11_buffer_context {
    std::shared_ptr<d3d11_device_ctx> dev;
    com_ptr<d3d11_res>                res;
    D3D11_GPU_VIRTUAL_ADDRESS         va = 0;
    size_t                            size = 0;
};

struct ggml_backend_d3d11_context {
    std::shared_ptr<d3d11_device_ctx> dev;
    std::string                       name;
};

// the registry owns the device contexts; buffers and backends hold shared references
static std::shared_ptr<d3d11_device_ctx> ggml_d3d11_shared_dev(d3d11_device_ctx * dev);
static void ggml_d3d11_atexit();

/* Device helpers */

static com_ptr<d3d11_res> ggml_d3d11_create_buffer(d3d11_device_ctx & dev, size_t size, d3d11_heap_type heap_type,
                                                   const wchar_t * name) {
    GGML_UNUSED(heap_type);
    GGML_UNUSED(name);
    size = (size + 255) & ~(size_t) 255;
    D3D11_BUFFER_DESC desc = {};
    desc.ByteWidth         = (UINT) size;
    desc.Usage             = D3D11_USAGE_DEFAULT;
    desc.BindFlags         = D3D11_BIND_UNORDERED_ACCESS;
    desc.MiscFlags         = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    com_ptr<ID3D11Buffer> buf;
    HRESULT hr = dev.device->CreateBuffer(&desc, nullptr, buf.put());
    if (FAILED(hr)) {
        GGML_LOG_ERROR("ggml_d3d11: failed to allocate %zu bytes, HRESULT 0x%08lx\n", size, (unsigned long) hr);
        return {};
    }
    std::lock_guard<std::recursive_mutex> lock(dev.mutex);
    com_ptr<d3d11_res> res;
    res.p       = new d3d11_res();
    res.p->dev  = &dev;
    res.p->buf  = buf;
    res.p->size = size;
    res.p->va   = dev.next_va;
    dev.next_va += (size + 0xffffull) & ~0xffffull;
    dev.res_map[res.p->va] = res.p;
    return res;
}

static d3d11_res * ggml_d3d11_find_res(d3d11_device_ctx & dev, uint64_t va) {
    auto it = dev.res_map.upper_bound(va);
    GGML_ASSERT(it != dev.res_map.begin());
    --it;
    GGML_ASSERT(va < it->first + it->second->size);
    return it->second;
}

static void ggml_d3d11_print_debug(d3d11_device_ctx & dev);

// raw UAV over [start, end) of a buffer, by fake addresses
static ID3D11UnorderedAccessView * ggml_d3d11_uav(d3d11_device_ctx & dev, uint64_t start, uint64_t end) {
    auto it = dev.uavs.find({ start, end });
    if (it != dev.uavs.end()) {
        return it->second.get();
    }
    d3d11_res *    res = ggml_d3d11_find_res(dev, start);
    const uint64_t off = start - res->va;
    GGML_ASSERT(off % 4 == 0 && end > start && end <= res->va + res->size);
    D3D11_UNORDERED_ACCESS_VIEW_DESC d = {};
    d.Format              = DXGI_FORMAT_R32_TYPELESS;
    d.ViewDimension       = D3D11_UAV_DIMENSION_BUFFER;
    d.Buffer.FirstElement = (UINT) (off / 4);
    d.Buffer.NumElements  = (UINT) ((end - start + 3) / 4);
    d.Buffer.Flags        = D3D11_BUFFER_UAV_FLAG_RAW;
    com_ptr<ID3D11UnorderedAccessView> uav;
    const HRESULT hr = dev.device->CreateUnorderedAccessView(res->buf.get(), &d, uav.put());
    if (FAILED(hr)) {
        GGML_LOG_ERROR("ggml_d3d11: UAV of %u elements at element %u in a buffer of %llu bytes\n", d.Buffer.NumElements,
                       d.Buffer.FirstElement, (unsigned long long) res->size);
        ggml_d3d11_print_debug(dev);
    }
    ggml_d3d11_check(hr, "CreateUnorderedAccessView");
    return (dev.uavs[{ start, end }] = uav).get();
}

static void ggml_d3d11_begin(d3d11_device_ctx & dev, bool compute) {
    GGML_UNUSED(compute);
    GGML_ASSERT(!dev.recording);
    dev.recording = true;
}

// GGML_D3D11_DEBUG: print the debug layer messages stored since the last call
static void ggml_d3d11_print_debug(d3d11_device_ctx & dev) {
    if (!dev.info) {
        return;
    }
    const UINT64 n = dev.info->GetNumStoredMessages();
    for (UINT64 i = 0; i < n; i++) {
        SIZE_T len = 0;
        dev.info->GetMessage(i, nullptr, &len);
        std::vector<char> buf(len);
        auto * msg = (D3D11_MESSAGE *) buf.data();
        if (SUCCEEDED(dev.info->GetMessage(i, msg, &len))) {
            GGML_LOG_WARN("ggml_d3d11: debug layer: %.*s\n", (int) msg->DescriptionByteLength, msg->pDescription);
        }
    }
    dev.info->ClearStoredMessages();
}

// flush the queued work and wait until the GPU has finished it
static void ggml_d3d11_submit_and_wait(d3d11_device_ctx & dev) {
    GGML_ASSERT(dev.recording);
    const double ts = ggml_d3d11_time_us();
    dev.ctx->End(dev.done_query.get());
    dev.ctx->Flush();
    const double t0 = ggml_d3d11_time_us();
    dev.t_submit_us += t0 - ts;
    BOOL done = FALSE;
    while (dev.ctx->GetData(dev.done_query.get(), &done, sizeof(done), 0) == S_FALSE) {
        SwitchToThread();
    }
    dev.t_wait_us += ggml_d3d11_time_us() - t0;
    dev.n_submits++;
    ggml_d3d11_print_debug(dev);
    dev.recording          = false;
    dev.dispatches_in_list = 0;

    HRESULT removed = dev.device->GetDeviceRemovedReason();
    if (FAILED(removed)) {
        GGML_LOG_ERROR("ggml_d3d11: device removed, reason 0x%08lx\n", (unsigned long) removed);
        fflush(stderr);
        GGML_ABORT("ggml_d3d11: device removed");
    }
}

/* Shader compilation */

// compiled DXBC is cached in d3d11-shader-cache next to ggml-d3d11.dll; GGML_D3D11_NO_SHADER_CACHE disables it.
// Unwritable folders just skip the cache.
static const std::wstring & ggml_d3d11_shader_cache_dir() {
    static const std::wstring dir = []() -> std::wstring {
        if (getenv("GGML_D3D11_NO_SHADER_CACHE") != nullptr) {
            return L"";
        }
        HMODULE module = nullptr;
        wchar_t path[MAX_PATH];
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                (LPCWSTR) &ggml_d3d11_shader_cache_dir, &module) ||
            GetModuleFileNameW(module, path, MAX_PATH) - 1 >= MAX_PATH - 1) {
            return L"";
        }
        std::wstring d = path;
        d = d.substr(0, d.find_last_of(L"\\/") + 1) + L"d3d11-shader-cache";
        CreateDirectoryW(d.c_str(), nullptr);
        return d;
    }();
    return dir;
}

// the file name is a hash of the source, the defines and the FXC flags, so a change recompiles only
// the shaders it touches. Do not add a global salt: a full recompile is slow on weak CPUs.
static std::wstring ggml_d3d11_shader_cache_file(const char * source, const std::vector<std::string> & args, UINT flags) {
    const std::wstring & dir = ggml_d3d11_shader_cache_dir();
    if (dir.empty()) {
        return L"";
    }
    uint64_t h = 0xcbf29ce484222325ULL;
    auto hash_bytes = [&h](const void * data, size_t size) {
        const unsigned char * b = (const unsigned char *) data;
        for (size_t i = 0; i < size; i++) {
            h = (h ^ b[i]) * 0x100000001b3ULL;
        }
    };
    hash_bytes("fxc-4", 5);
    if (flags != D3DCOMPILE_OPTIMIZATION_LEVEL3) {   // default flags are not hashed: keeps older cache files valid
        hash_bytes(&flags, sizeof(flags));
    }
    hash_bytes(source, strlen(source));
    for (const auto & a : args) {
        hash_bytes(a.c_str(), a.size() + 1);
    }
    std::wstring name = L"\\0000000000000000.dxbc";
    for (int i = 16; i > 0; i--, h >>= 4) {
        name[i] = L"0123456789abcdef"[h & 0xf];
    }
    return dir + name;
}

static std::vector<uint8_t> ggml_d3d11_read_file(const std::wstring & file) {
    std::vector<uint8_t> data;
    FILE * f = file.empty() ? nullptr : _wfopen(file.c_str(), L"rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (size > 0) {
            data.resize((size_t) size);
            if (fread(data.data(), 1, data.size(), f) != data.size()) {
                data.clear();
            }
        }
        fclose(f);
    }
    if (data.size() < 32 || memcmp(data.data(), "DXBC", 4) != 0) {
        data.clear();
    }
    return data;
}

// write to a per-process temp name and rename, so concurrent processes never see a partial file
static void ggml_d3d11_write_file(const std::wstring & file, const void * data, size_t size) {
    if (file.empty()) {
        return;
    }
    const std::wstring tmp = file + L"." + std::to_wstring(GetCurrentProcessId()) + L".tmp";
    FILE * f = _wfopen(tmp.c_str(), L"wb");
    if (!f) {
        return;
    }
    const bool ok = fwrite(data, 1, size, f) == size;
    fclose(f);
    if (!ok || !MoveFileExW(tmp.c_str(), file.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        DeleteFileW(tmp.c_str());
    }
}

// FXC at level 3 gives wrong values for these types (GET_ROWS on R9700 and Iris Xe; right with the
// optimizer off). iq1_s does not fit in the register limit without the optimizer, so it gets level 1.
static UINT ggml_d3d11_compile_flags(const std::string & key) {
    UINT flags = D3DCOMPILE_OPTIMIZATION_LEVEL3;
    for (const char * t : { "Q3_K", "IQ2_S", "IQ3_S" }) {
        if (key.find(std::string("SRC0_") + t) != std::string::npos) {
            flags = D3DCOMPILE_SKIP_OPTIMIZATION;
        }
    }
    if (key.find("SRC0_IQ1_S") != std::string::npos) {
        flags = D3DCOMPILE_OPTIMIZATION_LEVEL1;
    }
    return flags;
}

// each used shader adds a line "source<TAB>key<TAB>define..." to this file; the next process
// compiles the listed shaders that are not in the cache on all CPU cores (see ggml_d3d11_prewarm)
static std::wstring ggml_d3d11_shader_list_file() {
    const std::wstring & dir = ggml_d3d11_shader_cache_dir();
    return dir.empty() ? L"" : dir + L"\\shaders.txt";
}

// counts shaders compiled from source (cache hits are not counted) and says so at most once a second,
// so a long first run shows it is compiling rather than hung
static std::mutex g_compile_mutex;
static size_t     g_compile_done  = 0;
static size_t     g_compile_total = 0;   // cache misses queued so far, for "N of M"

static void ggml_d3d11_note_queued(size_t n) {
    std::lock_guard<std::mutex> lock(g_compile_mutex);
    g_compile_total += n;
}

static void ggml_d3d11_note_compile() {
    static std::chrono::steady_clock::time_point t_print;
    std::lock_guard<std::mutex> lock(g_compile_mutex);
    const auto now = std::chrono::steady_clock::now();
    const size_t n = ++g_compile_done;
    g_compile_total = std::max(g_compile_total, n);
    if (n == 1) {
        t_print = now;
    } else if (now - t_print >= std::chrono::seconds(1)) {
        t_print = now;
        // straight to stderr: llama-cli hides every log below error level once it has started
        fprintf(stderr, "\nggml_d3d11: compiling shaders, %zu of %zu\n", n, g_compile_total);
        fflush(stderr);
    }
}

// FXC macros for one shader; they are also part of its cache key
static std::vector<std::string> ggml_d3d11_compile_args(const std::vector<std::string> & defines) {
    // USE_16BIT selects SM 6.2 16-bit loads; without it common.hlsli emulates them with 32-bit words
    std::vector<std::string> args = { "WG_SIZE=" + std::to_string(D3D11_WG_SIZE), "GGML_D3D11" };
    for (const auto & d : defines) {
        if (d != "USE_16BIT") {
            args.push_back(d);
        }
    }
    return args;
}

// returns the DXBC from the disk cache, or compiles it with FXC and caches it; thread safe
static std::vector<uint8_t> ggml_d3d11_get_dxbc(const std::string &              key,
                                                const char *                     source,
                                                const std::vector<std::string> & defines,
                                                std::wstring &                   cache_file) {
    const std::vector<std::string> args = ggml_d3d11_compile_args(defines);
    const UINT flags = ggml_d3d11_compile_flags(key);
    cache_file       = ggml_d3d11_shader_cache_file(source, args, flags);
    std::vector<uint8_t> dxbc = ggml_d3d11_read_file(cache_file);
    if (!dxbc.empty()) {
        return dxbc;
    }
    std::vector<std::string> names, values;
    for (const auto & a : args) {
        const size_t eq = a.find('=');
        names.push_back(a.substr(0, eq));
        values.push_back(eq == std::string::npos ? "1" : a.substr(eq + 1));
    }
    std::vector<D3D_SHADER_MACRO> macros;
    for (size_t i = 0; i < args.size(); i++) {
        macros.push_back({ names[i].c_str(), values[i].c_str() });
    }
    macros.push_back({ nullptr, nullptr });

    com_ptr<ID3DBlob> code, errors;
    HRESULT hr = D3DCompile(source, strlen(source), key.c_str(), macros.data(), nullptr, "main", "cs_5_0",
                            flags, 0, code.put(), errors.put());
    if (FAILED(hr)) {
        GGML_LOG_ERROR("ggml_d3d11: shader compilation failed for %s:\n%s\n", key.c_str(),
                       errors ? (const char *) errors->GetBufferPointer() : "(no output)");
        fflush(stderr);
        GGML_ABORT("ggml_d3d11: shader compilation failed");
    }
    if (errors && errors->GetBufferSize() > 1) {
        GGML_LOG_WARN("ggml_d3d11: FXC warnings for %s:\n%s\n", key.c_str(), (const char *) errors->GetBufferPointer());
    }
    ggml_d3d11_note_compile();
    const uint8_t * bytes = (const uint8_t *) code->GetBufferPointer();
    dxbc.assign(bytes, bytes + code->GetBufferSize());
    ggml_d3d11_write_file(cache_file, dxbc.data(), dxbc.size());
    return dxbc;
}

// keys already in the list file, read by ggml_d3d11_prewarm
static std::mutex                   g_listed_mutex;
static std::map<std::string, bool>  g_listed;

// adds a shader to the list file once; one short append per line, so parallel processes do not mix lines
static void ggml_d3d11_list_shader(const std::string & key, const char * source, const std::vector<std::string> & defines) {
    const std::wstring file = ggml_d3d11_shader_list_file();
    {
        std::lock_guard<std::mutex> lock(g_listed_mutex);
        if (g_listed[key]) {
            return;
        }
        g_listed[key] = true;
    }
    for (const auto & t : hlsl_table) {
        if (t.source != source || file.empty()) {
            continue;
        }
        std::string line = std::string(t.name) + "\t" + key;
        for (const auto & d : defines) {
            line += "\t" + d;
        }
        line += "\n";
        std::lock_guard<std::mutex> lock(g_listed_mutex);
        if (FILE * f = _wfopen(file.c_str(), L"ab")) {
            fwrite(line.data(), 1, line.size(), f);
            fclose(f);
        }
        return;
    }
}

// compiles the listed shaders that are missing from the cache, on all CPU cores, in the background.
// test-backend-ops makes one small graph per case and needs about one new shader each time, so without
// this all compilation runs on one core.
static void ggml_d3d11_prewarm() {
    struct item {
        std::string              key;
        const char *             source;
        std::vector<std::string> defines;
    };
    std::vector<item> items;
    std::lock_guard<std::mutex> lock(g_listed_mutex);
    std::map<std::string, bool> & seen = g_listed;
    FILE * f = nullptr;
    const std::wstring file = ggml_d3d11_shader_list_file();
    if (file.empty() || (f = _wfopen(file.c_str(), L"rb")) == nullptr) {
        return;
    }
    char buf[4096];
    while (fgets(buf, sizeof(buf), f)) {
        std::vector<std::string> fields;
        std::string              line = buf;
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        for (size_t pos = 0; pos <= line.size();) {
            const size_t tab = std::min(line.find('\t', pos), line.size());
            fields.push_back(line.substr(pos, tab - pos));
            pos = tab + 1;
        }
        if (fields.size() < 2 || seen[fields[1]]) {
            continue;
        }
        seen[fields[1]] = true;
        for (const auto & t : hlsl_table) {
            if (fields[0] == t.name) {
                items.push_back({ fields[1], t.source, std::vector<std::string>(fields.begin() + 2, fields.end()) });
            }
        }
    }
    fclose(f);

    // drop the ones already in the cache
    std::vector<item> missing;
    for (auto & it : items) {
        std::vector<std::string> args = { "WG_SIZE=" + std::to_string(D3D11_WG_SIZE), "GGML_D3D11" };
        for (const auto & d : it.defines) {
            if (d != "USE_16BIT") {
                args.push_back(d);
            }
        }
        const std::wstring cf = ggml_d3d11_shader_cache_file(it.source, args, ggml_d3d11_compile_flags(it.key));
        if (GetFileAttributesW(cf.c_str()) == INVALID_FILE_ATTRIBUTES) {
            missing.push_back(std::move(it));
        }
    }
    if (missing.empty()) {
        return;
    }
    const size_t n_threads = std::min<size_t>(missing.size(), std::max(1u, std::thread::hardware_concurrency()));
    GGML_LOG_INFO("ggml_d3d11: compiling %zu listed shaders on %zu threads in the background\n", missing.size(), n_threads);
    auto shared = std::make_shared<std::vector<item>>(std::move(missing));
    auto next   = std::make_shared<std::atomic<size_t>>(0);
    for (size_t t = 0; t < n_threads; t++) {
        std::thread([shared, next]() {
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
            for (size_t i; (i = (*next)++) < shared->size();) {
                const item & it = (*shared)[i];
                std::wstring cf;
                ggml_d3d11_get_dxbc(it.key, it.source, it.defines, cf);
            }
        }).detach();
    }
}

// builds one pipeline from the disk cache or by compiling with FXC; thread safe
static d3d11_pipeline ggml_d3d11_build_pipeline(d3d11_device_ctx &               dev,
                                               const std::string &              key,
                                               const char *                     source,
                                               const std::vector<std::string> & defines) {
    const double t_compile0 = ggml_d3d11_time_us();

    std::wstring               cache_file;
    const std::vector<uint8_t> dxbc = ggml_d3d11_get_dxbc(key, source, defines, cache_file);
    ggml_d3d11_list_shader(key, source, defines);
    d3d11_pipeline pipeline;
    pipeline.name = key;
    HRESULT hr = dev.device->CreateComputeShader(dxbc.data(), dxbc.size(), nullptr, pipeline.cs.put());
    if (FAILED(hr)) {
        GGML_LOG_ERROR("ggml_d3d11: CreateComputeShader failed for %s (HRESULT 0x%08lx)\n", key.c_str(), (unsigned long) hr);
        DeleteFileW(cache_file.c_str());
        fflush(stderr);
        GGML_ABORT("ggml_d3d11: pipeline creation failed");
    }
    std::lock_guard<std::mutex> lock(dev.pipelines_mutex);
    dev.n_compiles++;
    dev.t_compile_us += ggml_d3d11_time_us() - t_compile0;
    return pipeline;
}

// defines: list of "NAME" or "NAME=VALUE"
static d3d11_pipeline & ggml_d3d11_get_pipeline(d3d11_device_ctx &               dev,
                                                const char *                     shader_name,
                                                const char *                     source,
                                                const std::vector<std::string> & defines,
                                                D3D_SHADER_MODEL                 min_sm = D3D_SHADER_MODEL_6_0) {
    std::string key = shader_name;
    for (const auto & d : defines) {
        key += " -D" + d;
    }
    auto it = dev.pipelines.find(key);
    if (it != dev.pipelines.end()) {
        return it->second;
    }
    if (dev.collecting) {
        static d3d11_pipeline placeholder;
        const bool queued = std::any_of(dev.pipeline_jobs.begin(), dev.pipeline_jobs.end(),
                                        [&](const d3d11_device_ctx::pipeline_job & j) { return j.key == key; });
        if (!queued) {
            dev.pipeline_jobs.push_back({ key, source, defines, min_sm });
        }
        return placeholder;
    }
    d3d11_pipeline pipeline = ggml_d3d11_build_pipeline(dev, key, source, defines);
    return dev.pipelines.emplace(key, std::move(pipeline)).first->second;
}

// user-mode driver version as a.b.c.d, the number Device Manager shows
static std::string ggml_d3d11_driver_version(IDXGIAdapter1 * adapter) {
    LARGE_INTEGER v;
    if (FAILED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &v))) {
        return "unknown";
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "%u.%u.%u.%u", (unsigned) HIWORD(v.HighPart), (unsigned) LOWORD(v.HighPart),
             (unsigned) HIWORD(v.LowPart), (unsigned) LOWORD(v.LowPart));
    return buf;
}

// builds the queued pipelines on worker threads (FXC and CreateComputeShader are thread safe)
static void ggml_d3d11_build_pipeline_jobs(d3d11_device_ctx & dev) {
    std::vector<d3d11_device_ctx::pipeline_job> jobs;
    jobs.swap(dev.pipeline_jobs);
    if (jobs.empty()) {
        return;
    }
    size_t n_missing = 0;
    for (const auto & j : jobs) {
        const std::wstring cf = ggml_d3d11_shader_cache_file(j.source, ggml_d3d11_compile_args(j.defines),
                                                             ggml_d3d11_compile_flags(j.key));
        n_missing += GetFileAttributesW(cf.c_str()) == INVALID_FILE_ATTRIBUTES;
    }
    ggml_d3d11_note_queued(n_missing);
    const size_t        n_threads = std::min<size_t>(jobs.size(), std::max(1u, std::thread::hardware_concurrency()));
    std::atomic<size_t> next{ 0 };
    auto worker = [&]() {
        for (size_t i; (i = next++) < jobs.size();) {
            const auto &   j = jobs[i];
            d3d11_pipeline p = ggml_d3d11_build_pipeline(dev, j.key, j.source, j.defines);
            std::lock_guard<std::mutex> lock(dev.pipelines_mutex);
            dev.pipelines.emplace(j.key, std::move(p));
        }
    };
    std::vector<std::thread> threads;
    for (size_t t = 1; t < n_threads; t++) {
        threads.emplace_back(worker);
    }
    worker();
    for (auto & th : threads) {
        th.join();
    }
}

/* Dispatch encoding */

static inline void ggml_d3d11_workgroups_2d(uint32_t total_wg, uint32_t & wg_x, uint32_t & wg_y) {
    wg_y = std::max(1u, CEIL_DIV(total_wg, (uint32_t) D3D11_MAX_WG_PER_DIM));
    wg_x = CEIL_DIV(total_wg, wg_y);
}

static inline uint32_t ggml_d3d11_u32_from_f32(float value) {
    uint32_t u;
    memcpy(&u, &value, sizeof(u));
    return u;
}

struct d3d11_binding {
    D3D11_GPU_VIRTUAL_ADDRESS va;
    uint32_t                  elem_offset;   // misalignment in elements, passed to the kernel
};

static D3D11_GPU_VIRTUAL_ADDRESS ggml_d3d11_tensor_va(const ggml_tensor * t) {
    auto * buf_ctx = (ggml_backend_d3d11_buffer_context *) t->buffer->context;
    return buf_ctx->va;
}

// Bind at the largest 256-byte aligned offset at or before the tensor such that the distance is a
// whole number of type blocks, so the kernel can index the remainder in elements.
static d3d11_binding ggml_d3d11_bind_tensor(const ggml_tensor * t) {
    const size_t offset    = ggml_d3d11_tensor_offset(t);
    const size_t type_size = ggml_type_size(t->type);
    size_t       aligned   = offset & ~((size_t) D3D11_BINDING_ALIGNMENT - 1);
    while ((offset - aligned) % type_size != 0) {
        GGML_ASSERT(aligned >= D3D11_BINDING_ALIGNMENT);
        aligned -= D3D11_BINDING_ALIGNMENT;
    }
    d3d11_binding b;
    b.va          = ggml_d3d11_tensor_va(t) + aligned;
    b.elem_offset = (uint32_t) ((offset - aligned) / type_size);
    // D3D11 refuses overlapping UAVs of one buffer, so each view covers only its tensor
    auto *         buf_ctx = (ggml_backend_d3d11_buffer_context *) t->buffer->context;
    const uint64_t end     = std::min<uint64_t>(ggml_d3d11_tensor_va(t) + ((offset + ggml_nbytes(t) + 3) & ~(size_t) 3),
                                                buf_ctx->va + buf_ctx->size);
    uint64_t &     e       = buf_ctx->dev->bind_end[b.va];
    e = std::max(e, end);
    // a written tensor (a node output) is tagged in the address, so the dispatch knows the written slot
    // even when a read slot has the same address (an in-place op)
    for (const ggml_tensor * o : buf_ctx->dev->out_nodes) {
        if (o != nullptr && o == t) {
            b.va |= D3D11_VA_WRITTEN;
        }
    }
    return b;
}

// records one dispatch on the immediate context (params get nwg_x appended)
static void ggml_d3d11_dispatch(d3d11_device_ctx &                             dev,
                                d3d11_pipeline &                               pipeline,
                                std::vector<uint32_t>                          params,
                                const std::vector<D3D11_GPU_VIRTUAL_ADDRESS> & uavs,
                                uint32_t                                       total_wg) {
    GGML_ASSERT(dev.recording);
    GGML_ASSERT(uavs.size() <= dev.max_uavs && uavs.size() <= D3D11_MAX_ROOT_UAVS);
    if (dev.collecting) {
        dev.bind_end.clear();
        return;
    }

    uint32_t wg_x, wg_y;
    ggml_d3d11_workgroups_2d(total_wg, wg_x, wg_y);
    params.push_back(wg_x);
    GGML_ASSERT(params.size() * sizeof(uint32_t) <= D3D11_PARAM_SLOT_SIZE);

    if (dev.dispatches_in_list >= dev.submit_batch) {
        // keep the GPU busy on long graphs
        dev.ctx->Flush();
        dev.n_flush_batch++;
        dev.dispatches_in_list = 0;
    }
    dev.dispatches_in_list++;

    D3D11_MAPPED_SUBRESOURCE m = {};
    ggml_d3d11_check(dev.ctx->Map(dev.cbuf.get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m), "Map (params)");
    memcpy(m.pData, params.data(), params.size() * sizeof(uint32_t));
    dev.ctx->Unmap(dev.cbuf.get(), 0);

    // ranges: a bound tensor ends where its data ends, other buffers (scratch) at the buffer end
    const size_t n = uavs.size();
    uint64_t     lo[D3D11_MAX_ROOT_UAVS], hi[D3D11_MAX_ROOT_UAVS];
    bool         out[D3D11_MAX_ROOT_UAVS] = {};
    d3d11_res *  rs[D3D11_MAX_ROOT_UAVS];
    // address 0: an unused slot (a D3D12 placeholder), bound as a null UAV
    for (size_t i = 0; i < n; i++) {
        if (uavs[i] == 0) {
            rs[i] = nullptr;
            lo[i] = hi[i] = 0;
            continue;
        }
        const uint64_t va = uavs[i] & ~D3D11_VA_WRITTEN;
        out[i]  = (uavs[i] & D3D11_VA_WRITTEN) != 0;
        rs[i]   = ggml_d3d11_find_res(dev, va);
        lo[i]   = va;
        auto it = dev.bind_end.find(va);
        hi[i]   = it != dev.bind_end.end() ? it->second : rs[i]->va + rs[i]->size;
    }
    dev.bind_end.clear();
    // D3D11 refuses two UAVs of one buffer in one dispatch, even when their ranges do not overlap.
    // Of the slots that share a buffer, one keeps it: the written one (a node output), else the last.
    // The others use copies; a written copy is copied back after the dispatch.
    uint64_t back_lo[D3D11_MAX_ROOT_UAVS] = {};
    for (size_t i = 0; i < n; i++) {
        if (!rs[i]) {
            continue;
        }
        size_t keep = i;
        for (size_t j = 0; j < n; j++) {
            if (rs[j] == rs[i] && (out[j] ? (!out[keep] || j > keep) : (!out[keep] && j > keep))) {
                keep = j;
            }
        }
        if (keep == i) {
            continue;
        }
        // one copy buffer per slot: two slots in one copy buffer would conflict again
        const uint64_t size = hi[i] - lo[i];
        com_ptr<d3d11_res> & cp = dev.alias_copy[i];
        if (!cp || cp->size < size) {
            cp = ggml_d3d11_create_buffer(dev, std::max<size_t>(size, 1 << 20), D3D11_HEAP_TYPE_DEFAULT, nullptr);
            GGML_ASSERT(cp);
        }
        D3D11_BOX box = { (UINT) (lo[i] - rs[i]->va), 0, 0, (UINT) (hi[i] - rs[i]->va), 1, 1 };
        dev.ctx->CopySubresourceRegion(cp->buf.get(), 0, 0, 0, 0, rs[i]->buf.get(), 0, &box);
        back_lo[i] = out[i] ? lo[i] : 0;
        lo[i] = cp->va;
        hi[i] = cp->va + size;
    }
    ID3D11UnorderedAccessView * views[D3D11_MAX_ROOT_UAVS] = {};
    for (size_t i = 0; i < n; i++) {
        views[i] = rs[i] ? ggml_d3d11_uav(dev, lo[i], hi[i]) : nullptr;
    }
    ID3D11Buffer * cb = dev.cbuf.get();
    dev.ctx->CSSetShader(pipeline.cs.get(), nullptr, 0);
    dev.ctx->CSSetConstantBuffers(0, 1, &cb);
    dev.ctx->CSSetUnorderedAccessViews(0, std::min<UINT>(dev.max_uavs, D3D11_MAX_ROOT_UAVS), views, nullptr);
    dev.ctx->Dispatch(wg_x, wg_y, 1);
    dev.n_dispatches++;
    for (size_t i = 0; i < n; i++) {
        if (back_lo[i] != 0) {
            // tensors start on D3D11_BINDING_ALIGNMENT, so [lo, hi) holds no bytes of other tensors
            const D3D11_BOX box = { 0, 0, 0, (UINT) (hi[i] - lo[i]), 1, 1 };
            dev.ctx->CopySubresourceRegion(rs[i]->buf.get(), 0, (UINT) (back_lo[i] - rs[i]->va), 0, 0,
                                           dev.alias_copy[i]->buf.get(), 0, &box);
        }
    }
}

// memset [offset, offset+size) bytes of a buffer with a replicated byte value; standalone submission
static void ggml_d3d11_buffer_memset(d3d11_device_ctx & dev, D3D11_GPU_VIRTUAL_ADDRESS va, size_t offset, size_t size, uint8_t value) {
    if (size == 0) {
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(dev.mutex);
    const uint32_t val32 = (uint32_t) value * 0x01010101u;

    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "memset", hlsl_memset,
                                                        { "BYTES_PER_THREAD=" + std::to_string(D3D11_MEMSET_BYTES_PER_THREAD) });
    ggml_d3d11_begin(dev, true);
    const size_t chunk_max = 1ull << 30;
    size_t done = 0;
    while (done < size) {
        const size_t   abs_off   = offset + done;
        const size_t   base      = abs_off & ~((size_t) D3D11_BINDING_ALIGNMENT - 1);
        const uint32_t rel_off   = (uint32_t) (abs_off - base);
        const uint32_t n         = (uint32_t) std::min(size - done, chunk_max);
        const uint32_t span      = rel_off + n;
        const uint32_t threads   = CEIL_DIV(span, (uint32_t) D3D11_MEMSET_BYTES_PER_THREAD);
        ggml_d3d11_dispatch(dev, pipeline, { rel_off, n, val32 }, { va + base }, CEIL_DIV(threads, (uint32_t) D3D11_WG_SIZE));
        done += n;
    }
    ggml_d3d11_submit_and_wait(dev);
}

/* Op encoders */

static std::string ggml_d3d11_type_define(ggml_type type, const char * prefix) {
    std::string s = prefix;
    switch (type) {
        case GGML_TYPE_F32: s += "_F32"; break;
        case GGML_TYPE_F16: s += "_F16"; break;
        case GGML_TYPE_I32: s += "_I32"; break;
        default: GGML_ABORT("ggml_d3d11: unsupported type %s", ggml_type_name(type));
    }
    return s;
}

static void ggml_d3d11_cpy(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<std::string> defines = { ggml_d3d11_type_define(src->type, "SRC"), ggml_d3d11_type_define(dst->type, "DST") };
    if (src->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F16) {
        defines.push_back("USE_16BIT");
    }
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "cpy", hlsl_cpy, defines);

    const d3d11_binding bsrc = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bdst = ggml_d3d11_bind_tensor(dst);
    const uint32_t      ne   = (uint32_t) ggml_nelements(dst);
    const size_t        ts   = ggml_type_size(src->type);
    const size_t        td   = ggml_type_size(dst->type);

    std::vector<uint32_t> params = {
        ne, bsrc.elem_offset, bdst.elem_offset,
        (uint32_t) (src->nb[0] / ts), (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts), (uint32_t) (src->nb[3] / ts),
        (uint32_t) (dst->nb[0] / td), (uint32_t) (dst->nb[1] / td), (uint32_t) (dst->nb[2] / td), (uint32_t) (dst->nb[3] / td),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2],
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bsrc.va, bdst.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_binary_op(d3d11_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
    const char * op_define = nullptr;
    switch (dst->op) {
        case GGML_OP_ADD: op_define = "OP_ADD"; break;
        case GGML_OP_SUB: op_define = "OP_SUB"; break;
        case GGML_OP_MUL: op_define = "OP_MUL"; break;
        case GGML_OP_DIV: op_define = "OP_DIV"; break;
        default: GGML_ABORT("ggml_d3d11: unexpected binary op");
    }
    std::vector<std::string> defines = { ggml_d3d11_type_define(dst->type, "TYPE"), op_define };
    if (dst->type == GGML_TYPE_F16) {
        defines.push_back("USE_16BIT");
    }
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "binary", hlsl_binary, defines);

    const d3d11_binding b0 = ggml_d3d11_bind_tensor(src0);
    const d3d11_binding b1 = ggml_d3d11_bind_tensor(src1);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const size_t        t0 = ggml_type_size(src0->type);
    const size_t        t1 = ggml_type_size(src1->type);

    std::vector<uint32_t> params = {
        ne, b0.elem_offset, b1.elem_offset, bd.elem_offset,
        (uint32_t) (src0->nb[0] / t0), (uint32_t) (src0->nb[1] / t0), (uint32_t) (src0->nb[2] / t0), (uint32_t) (src0->nb[3] / t0),
        (uint32_t) (src1->nb[0] / t1), (uint32_t) (src1->nb[1] / t1), (uint32_t) (src1->nb[2] / t1), (uint32_t) (src1->nb[3] / t1),
        (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], (uint32_t) src0->ne[2],
        (uint32_t) src1->ne[0], (uint32_t) src1->ne[1], (uint32_t) src1->ne[2], (uint32_t) src1->ne[3],
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { b0.va, b1.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_scale(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "scale", hlsl_scale, {});

    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const size_t        ts = ggml_type_size(src->type);
    const size_t        td = ggml_type_size(dst->type);

    std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts), (uint32_t) (src->nb[3] / ts),
        (uint32_t) (dst->nb[1] / td), (uint32_t) (dst->nb[2] / td), (uint32_t) (dst->nb[3] / td),
        ne, (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2],
        ggml_d3d11_u32_from_f32(ggml_get_op_params_f32(dst, 0)),   // scale
        ggml_d3d11_u32_from_f32(ggml_get_op_params_f32(dst, 1)),   // bias
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

// SET_ROWS into q8_0 (quantized KV cache): two dispatches, rows of even and odd dst row numbers
static void ggml_d3d11_set_rows_q8_0(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * idx, ggml_tensor * dst) {
    std::vector<std::string> defines;
    if (idx->type == GGML_TYPE_I64) {
        defines.push_back("I64_IDX");
    }
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "set_rows_q8", hlsl_set_rows_q8, defines);

    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bi = ggml_d3d11_bind_tensor(idx);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const size_t        ts = ggml_type_size(src->type);
    const size_t        ti = ggml_type_size(idx->type);
    const size_t        td = ggml_type_size(dst->type);
    const uint32_t      n_rows = (uint32_t) (src->ne[1] * src->ne[2] * src->ne[3]);

    std::vector<uint32_t> params = {
        bs.elem_offset, bi.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts), (uint32_t) (src->nb[3] / ts),
        (uint32_t) (idx->nb[0] / ti), (uint32_t) (idx->nb[1] / ti), (uint32_t) (idx->nb[2] / ti),
        (uint32_t) (dst->nb[1] / td), (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) src->ne[3],
        (uint32_t) idx->ne[1], (uint32_t) idx->ne[2], 0,
    };
    for (uint32_t parity = 0; parity < 2; parity++) {
        params.back() = parity;
        ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bi.va, bd.va }, CEIL_DIV(n_rows, (uint32_t) D3D11_WG_SIZE));
    }
}

static void ggml_d3d11_set_rows(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * idx, ggml_tensor * dst) {
    if (ggml_is_empty(src) || ggml_is_empty(idx)) {
        return;
    }
    if (dst->type == GGML_TYPE_Q8_0) {
        ggml_d3d11_set_rows_q8_0(dev, src, idx, dst);
        return;
    }
    std::vector<std::string> defines = { ggml_d3d11_type_define(dst->type, "DST") };
    if (dst->type == GGML_TYPE_F16) {
        defines.push_back("USE_16BIT");
    }
    if (idx->type == GGML_TYPE_I64) {
        defines.push_back("I64_IDX");
    }
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "set_rows", hlsl_set_rows, defines);

    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bi = ggml_d3d11_bind_tensor(idx);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const size_t        ts = ggml_type_size(src->type);
    const size_t        ti = ggml_type_size(idx->type);
    const size_t        td = ggml_type_size(dst->type);
    const uint32_t      ne = (uint32_t) ggml_nelements(src);

    std::vector<uint32_t> params = {
        bs.elem_offset, bi.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts), (uint32_t) (src->nb[3] / ts),
        (uint32_t) (idx->nb[0] / ti), (uint32_t) (idx->nb[1] / ti), (uint32_t) (idx->nb[2] / ti),
        (uint32_t) (dst->nb[1] / td), (uint32_t) (dst->nb[2] / td), (uint32_t) (dst->nb[3] / td),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) src->ne[3],
        (uint32_t) idx->ne[1], (uint32_t) idx->ne[2],
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bi.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

static bool ggml_d3d11_mul_mat_vec_type(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
            return true;
        default:
            return false;
    }
}

// one matrix of a fused mul_mat_vec dispatch: dst = src0 * src1 (+ add, broadcast like ggml_add)
struct d3d11_mat_slot {
    ggml_tensor * src0;
    ggml_tensor * dst;   // the MUL_MAT node, or the fused ADD node
    ggml_tensor * add;   // addend of the fused ADD, or null
};

// types the tiled prompt kernel dequantizes (mul_mat_tiled.hlsl)
static bool ggml_d3d11_tiled_type(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
        // these reuse dot_row from dequant_row.hlsli
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
            return true;
        default:
            return false;
    }
}

// Is this product worth the tiled kernel, and can that kernel express it? Long prompts only: for a
// handful of columns the matvec kernel wins, because a tile of 32 columns would be mostly padding.
static bool ggml_d3d11_use_tiled(const d3d11_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1,
                                 ggml_tensor * dst) {
    return dev.tiled_min_cols != 0 && (uint32_t) dst->ne[1] >= dev.tiled_min_cols &&
           ggml_d3d11_tiled_type(src0->type) && src0->ne[0] % 32 == 0 &&
           (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) && dst->type == GGML_TYPE_F32;
}

// dst = src0 * src1 with a TILE_M x TILE_N tile of dst per workgroup; see mul_mat_tiled.hlsl
static void ggml_d3d11_mul_mat_tiled(d3d11_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1,
                                     ggml_tensor * dst) {
    std::string define = "SRC0_";
    define += ggml_type_name(src0->type);
    for (auto & ch : define) {
        ch = (char) toupper((unsigned char) ch);
    }
    std::vector<std::string> defines = { define };
    if (src1->type == GGML_TYPE_F16) {
        defines.push_back("SRC1_F16");
    }
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "mul_mat_tiled", hlsl_mul_mat_tiled, defines);

    const d3d11_binding b0 = ggml_d3d11_bind_tensor(src0);
    const d3d11_binding b1 = ggml_d3d11_bind_tensor(src1);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const size_t        t0 = ggml_type_size(src0->type);
    const size_t        t1 = ggml_type_size(src1->type);

    const uint32_t broadcast2 = (uint32_t) (src1->ne[2] / src0->ne[2]);
    const uint32_t broadcast3 = (uint32_t) (src1->ne[3] / src0->ne[3]);

    std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, bd.elem_offset,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) src0->ne[0],
        (uint32_t) (src0->nb[1] / t0), (uint32_t) (src0->nb[2] / t0), (uint32_t) (src0->nb[3] / t0),
        (uint32_t) (src1->nb[1] / t1), (uint32_t) (src1->nb[2] / t1), (uint32_t) (src1->nb[3] / t1),
        (uint32_t) dst->ne[2], broadcast2, broadcast3,
        (uint32_t) (dst->ne[2] * dst->ne[3]),
    };

    // tile sizes must match TILE_M and TILE_N in mul_mat_tiled.hlsl
    const uint32_t tiles_m  = CEIL_DIV((uint32_t) dst->ne[0], 64u);
    const uint32_t tiles_n  = CEIL_DIV((uint32_t) dst->ne[1], 32u);
    const uint32_t batches  = (uint32_t) (dst->ne[2] * dst->ne[3]);
    ggml_d3d11_dispatch(dev, pipeline, params, { b1.va, b0.va, bd.va }, tiles_m * tiles_n * batches);
}

// up to 3 matrices sharing src1 in one dispatch; every matrix owns a range of workgroups
static void ggml_d3d11_mul_mat_group(d3d11_device_ctx & dev, ggml_tensor * src1, const std::vector<d3d11_mat_slot> & mats) {
    GGML_ASSERT(!mats.empty() && mats.size() <= 3);
    ggml_tensor * src0 = mats[0].src0;
    // a single unfused product with many columns goes to the tiled kernel instead
    if (mats.size() == 1 && mats[0].add == nullptr &&
        ggml_d3d11_use_tiled(dev, src0, src1, mats[0].dst)) {
        ggml_d3d11_mul_mat_tiled(dev, src0, src1, mats[0].dst);
        return;
    }
    std::string define = "SRC0_";
    define += ggml_type_name(src0->type);
    for (auto & ch : define) {
        ch = (char) toupper((unsigned char) ch);
    }
    // threads per row: one unit of work (4 floats or one 32-wide block) per thread, power of two, at most WG_SIZE
    const uint32_t units = (uint32_t) (ggml_is_quantized(src0->type) ? src0->ne[0] / 32 : src0->ne[0] / 4);
    uint32_t       tpr   = 1;
    while (tpr < units && tpr < D3D11_WG_SIZE && tpr < dev.mm_tpr_max) {
        tpr *= 2;
    }
    bool fuse_add = false;
    for (const auto & m : mats) {
        fuse_add = fuse_add || m.add != nullptr;
    }
    // single-column (decode) products skip the per-element column loop (+50% decode on the MTT S80).
    // Not for F32 weights: on the S80 that variant failed MUL_MAT f32 k=256 bs=[3,2] nr=[1,2] (ERR 0.3,
    // deterministic) while F16 and the quant types passed the same shape; cause not found yet.
    const bool     one_col  = mats[0].dst->ne[1] == 1 && src0->type != GGML_TYPE_F32;
    const uint32_t max_cols = one_col ? 1 : 4;
    std::vector<std::string> defines = { define, "TPR=" + std::to_string(tpr), "N_MATS=" + std::to_string(mats.size()) };
    if (one_col) {
        defines.push_back("ONE_COL");
    }
    // On the AMD Vega iGPU every N_MATS=2 kernel without FUSE_ADD hung the GPU (DEVICE_HUNG, 2026-09-19); taking
    // the FUSE_ADD variant for groups (add_flag 0 skips the add at run time) avoids it. Why the variant matters is
    // not known. It is not free - about 7% of tg128 on an Intel Iris Xe - so it is limited to AMD integrated GPUs;
    // the R9700, the S80 and the Intel pass MUL_MAT_VEC_FUSION and generate correctly without it.
    if (fuse_add || (mats.size() > 1 && dev.vendor_id == 0x1002 && dev.caps.uma)) {
        defines.push_back("FUSE_ADD");
    }
    if (src1->type == GGML_TYPE_F16) {
        defines.push_back("SRC1_F16");
    }
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "mul_mat_vec", hlsl_mul_mat_vec, defines);
    const uint32_t rows_per_wg = D3D11_WG_SIZE / tpr;

    const d3d11_binding b1 = ggml_d3d11_bind_tensor(src1);
    const size_t        t0 = ggml_type_size(src0->type);   // block size in bytes for quant types
    const size_t        t1 = ggml_type_size(src1->type);
    ggml_tensor * dst0 = mats[0].dst;

    std::vector<uint32_t> params = {
        b1.elem_offset, (uint32_t) dst0->ne[1], (uint32_t) src0->ne[0],
        (uint32_t) (src1->nb[1] / t1), (uint32_t) (src1->nb[2] / t1), (uint32_t) (src1->nb[3] / t1),
        (uint32_t) src0->ne[2], (uint32_t) src0->ne[3],
        (uint32_t) (src1->ne[2] / src0->ne[2]), (uint32_t) (src1->ne[3] / src0->ne[3]),
        0,   // col0, set per chunk below
        0, 0 // wg_start_1, wg_start_2
    };
    const size_t col0_idx = 10;
    std::vector<D3D11_GPU_VIRTUAL_ADDRESS> uavs = { b1.va };
    uint32_t total_wg = 0;
    for (size_t i = 0; i < 3; i++) {
        if (i >= mats.size()) {
            for (int j = 0; j < 14; j++) {
                params.push_back(0);
            }
            continue;
        }
        const d3d11_mat_slot & m  = mats[i];
        const d3d11_binding    b0 = ggml_d3d11_bind_tensor(m.src0);
        const d3d11_binding    bd = ggml_d3d11_bind_tensor(m.dst);
        const d3d11_binding    ba = m.add ? ggml_d3d11_bind_tensor(m.add) : d3d11_binding{ 0, 0 };
        if (i > 0) {
            params[10 + i] = total_wg;
        }
        params.insert(params.end(), {
            b0.elem_offset, bd.elem_offset, (uint32_t) m.dst->ne[0],
            (uint32_t) (m.src0->nb[1] / t0), (uint32_t) (m.src0->nb[2] / t0), (uint32_t) (m.src0->nb[3] / t0),
            m.add ? 1u : 0u, ba.elem_offset,
            m.add ? (uint32_t) m.add->ne[1] : 1u, m.add ? (uint32_t) m.add->ne[2] : 1u, m.add ? (uint32_t) m.add->ne[3] : 1u,
            m.add ? (uint32_t) (m.add->nb[1] / 4) : 0u, m.add ? (uint32_t) (m.add->nb[2] / 4) : 0u, m.add ? (uint32_t) (m.add->nb[3] / 4) : 0u,
        });
        uavs.insert(uavs.end(), { b0.va, bd.va, ba.va });
        total_wg += CEIL_DIV((uint32_t) m.dst->ne[0], rows_per_wg) * (uint32_t) (m.dst->ne[2] * m.dst->ne[3]);
    }
    // columns are processed 4 at a time; every chunk re-reads src0 (fine for decode, slow for long prompts)
    for (uint32_t col0 = 0; col0 < (uint32_t) dst0->ne[1]; col0 += max_cols) {
        params[col0_idx] = col0;
        ggml_d3d11_dispatch(dev, pipeline, params, uavs, total_wg);
    }
}

// MUL_MAT_ID: dst[:, slot, token] = as[:, :, ids[slot, token]]^T * src1[:, slot % ne11, token]
// mul_mat_id for long prompts: group the (token, slot) pairs by expert, then one tiled product per expert
// tile. The same plan as mm_ids_helper + mmq in the CUDA backend. Returns false when not eligible.
static bool ggml_d3d11_mul_mat_id_tiled(d3d11_device_ctx & dev, ggml_tensor * as, ggml_tensor * src1, ggml_tensor * ids,
                                        ggml_tensor * dst) {
    const uint32_t n_used    = (uint32_t) ids->ne[0];
    const uint32_t n_tokens  = (uint32_t) ids->ne[1];
    const uint32_t n_experts = (uint32_t) as->ne[2];
    // MAX_EXPERTS in mul_mat_id_prep.hlsl
    if (dev.tiled_min_cols == 0 || n_tokens < dev.tiled_min_cols || n_experts > 1024 || as->ne[3] != 1 ||
        !ggml_d3d11_tiled_type(as->type) || as->ne[0] % 32 != 0 || src1->type != GGML_TYPE_F32 ||
        dst->nb[0] != sizeof(float)) {
        return false;
    }
    const uint32_t n_pairs   = n_used * n_tokens;
    const uint32_t max_tiles = CEIL_DIV(n_pairs, 32u) + std::min(n_experts, n_pairs);
    const uint32_t list_base = 1 + 3 * max_tiles;
    const size_t   need      = (size_t) (list_base + n_pairs) * 4;
    if (!dev.collecting && need > dev.mmid_scratch_size) {
        // commands already recorded may still use the old buffer: run them before it is replaced
        ggml_d3d11_submit_and_wait(dev);
        ggml_d3d11_begin(dev, true);
        const size_t size = std::max(need, dev.mmid_scratch_size * 2);
        dev.mmid_scratch  = ggml_d3d11_create_buffer(dev, size, D3D11_HEAP_TYPE_DEFAULT, L"ggml_d3d11 mmid scratch");
        GGML_ASSERT(dev.mmid_scratch);
        dev.mmid_scratch_size = size;
    }
    const D3D11_GPU_VIRTUAL_ADDRESS scratch_va = dev.mmid_scratch ? dev.mmid_scratch->GetGPUVirtualAddress() : 0;

    const d3d11_binding bi = ggml_d3d11_bind_tensor(ids);
    d3d11_pipeline & prep = ggml_d3d11_get_pipeline(dev, "mul_mat_id_prep", hlsl_mul_mat_id_prep, {});
    ggml_d3d11_dispatch(dev, prep, { bi.elem_offset, (uint32_t) (ids->nb[1] / 4), n_used, n_tokens, n_experts, list_base },
                        { bi.va, scratch_va }, 1);

    std::string define = "SRC0_";
    define += ggml_type_name(as->type);
    for (auto & ch : define) {
        ch = (char) toupper((unsigned char) ch);
    }
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "mul_mat_tiled", hlsl_mul_mat_tiled, { define, "MMID" });

    const d3d11_binding b0 = ggml_d3d11_bind_tensor(as);
    const d3d11_binding b1 = ggml_d3d11_bind_tensor(src1);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const size_t        t0 = ggml_type_size(as->type);

    std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, bd.elem_offset,
        (uint32_t) dst->ne[0], 0u, (uint32_t) as->ne[0],
        (uint32_t) (as->nb[1] / t0), (uint32_t) (as->nb[2] / t0), 0u,
        (uint32_t) (src1->nb[1] / 4), (uint32_t) (src1->nb[2] / 4), 0u,
        1u, 1u, 1u, 1u,
        n_used, (uint32_t) src1->ne[1], (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), list_base,
    };
    // TILE_M in mul_mat_tiled.hlsl; workgroups past the real tile count do nothing
    const uint32_t tiles_m = CEIL_DIV((uint32_t) dst->ne[0], 64u);
    ggml_d3d11_dispatch(dev, pipeline, params, { b1.va, b0.va, bd.va, scratch_va }, tiles_m * max_tiles);
    return true;
}

static void ggml_d3d11_mul_mat_id(d3d11_device_ctx & dev, ggml_tensor * as, ggml_tensor * src1, ggml_tensor * ids,
                                  ggml_tensor * dst) {
    std::string define = "SRC0_";
    define += ggml_type_name(as->type);
    for (auto & ch : define) {
        ch = (char) toupper((unsigned char) ch);
    }
    const uint32_t units = (uint32_t) (ggml_is_quantized(as->type) ? as->ne[0] / 32 : as->ne[0] / 4);
    uint32_t       tpr   = 1;
    while (tpr < units && tpr < D3D11_WG_SIZE && tpr < dev.mm_tpr_max) {
        tpr *= 2;
    }
    if (ggml_d3d11_mul_mat_id_tiled(dev, as, src1, ids, dst)) {
        return;
    }
    std::vector<std::string> defines = { define, "TPR=" + std::to_string(tpr), "N_MATS=1", "MMID" };
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "mul_mat_vec", hlsl_mul_mat_vec, defines);

    const d3d11_binding b0 = ggml_d3d11_bind_tensor(as);
    const d3d11_binding b1 = ggml_d3d11_bind_tensor(src1);
    const d3d11_binding bi = ggml_d3d11_bind_tensor(ids);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const size_t        t0 = ggml_type_size(as->type);
    const size_t        t1 = ggml_type_size(src1->type);

    std::vector<uint32_t> params = {
        b1.elem_offset, 1u, (uint32_t) as->ne[0],
        (uint32_t) (src1->nb[1] / t1), (uint32_t) (src1->nb[2] / t1), (uint32_t) (src1->nb[3] / t1),
        1u, 1u, 1u, 1u,
        0u,                              // col0
        (uint32_t) ids->ne[1], 0u,       // wg_start_1 carries the token count for MMID, wg_start_2 unused
    };
    // matrix slot 0 describes the expert matrices; slots 1 and 2 stay empty
    params.insert(params.end(), {
        b0.elem_offset, bd.elem_offset, (uint32_t) as->ne[1],
        (uint32_t) (as->nb[1] / t0), (uint32_t) (as->nb[2] / t0), (uint32_t) (as->nb[3] / t0),
        0u, bd.elem_offset, 1u, 1u, 1u, 0u, 0u, 0u,
    });
    for (int i = 0; i < 28; i++) {
        params.push_back(0);
    }
    params.insert(params.end(), {
        bi.elem_offset, (uint32_t) (ids->nb[1] / 4), (uint32_t) ids->ne[0], (uint32_t) src1->ne[1],
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4),
    });

    const uint32_t rows_per_wg = D3D11_WG_SIZE / tpr;
    const uint32_t total_wg    = CEIL_DIV((uint32_t) as->ne[1], rows_per_wg) *
                                 (uint32_t) (ids->ne[0] * ids->ne[1]);
    ggml_d3d11_dispatch(dev, pipeline, params, { b1.va, b0.va, bd.va, 0, bi.va }, total_wg);
}

static void ggml_d3d11_mul_mat(d3d11_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
    ggml_d3d11_mul_mat_group(dev, src1, { { src0, dst, nullptr } });
}

// can `add_node` (an ADD consuming `mm`) be folded into the matvec that produces `mm`? returns the addend
static ggml_tensor * ggml_d3d11_fusable_addend(const ggml_tensor * mm, const ggml_tensor * add_node) {
    if (add_node->src[0] != mm && add_node->src[1] != mm) {
        return nullptr;
    }
    ggml_tensor * other = add_node->src[0] == mm ? add_node->src[1] : add_node->src[0];
    if (other == mm || other->type != GGML_TYPE_F32 || other->nb[0] != sizeof(float) || other->ne[0] != mm->ne[0]) {
        return nullptr;
    }
    // ggml_add broadcasts src1 onto src0: when mm is src1, the shapes must match outright
    if (add_node->src[1] == mm && !ggml_are_same_shape(mm, other)) {
        return nullptr;
    }
    for (int d = 1; d < 4; d++) {
        if (other->ne[d] != 1 && other->ne[d] != mm->ne[d]) {
            return nullptr;
        }
    }
    if (add_node->type != GGML_TYPE_F32 || !ggml_is_contiguous(add_node)) {
        return nullptr;
    }
    return other;
}

// encode the MUL_MAT at node i together with up to two following MUL_MATs that share src1, each with its
// optional bias/residual ADD; returns the number of graph nodes consumed
static int ggml_d3d11_encode_mul_mat_group(d3d11_device_ctx & dev, const ggml_cgraph * cgraph, int i) {
    ggml_tensor * first = cgraph->nodes[i];
    ggml_tensor * src1  = first->src[1];
    // long prompts: the tiled kernel alone, no grouping or ADD fusion (the matvec path made pp512 up to 25x slower)
    if (ggml_d3d11_use_tiled(dev, first->src[0], src1, first)) {
        ggml_d3d11_mul_mat_tiled(dev, first->src[0], src1, first);
        return 1;
    }
    std::vector<d3d11_mat_slot> mats;
    int j = i;
    // D3D11: a group writes several dst tensors that share a buffer, and all but the last slot would be copies
    const size_t max_mats = 1;
    while (j < cgraph->n_nodes && mats.size() < max_mats) {
        ggml_tensor * node = cgraph->nodes[j];
        if (node->op != GGML_OP_MUL_MAT || node->src[1] != src1 || node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node) ||
            node->src[0]->type != first->src[0]->type || node->src[0]->ne[0] != first->src[0]->ne[0] ||
            node->src[0]->ne[2] != first->src[0]->ne[2] || node->src[0]->ne[3] != first->src[0]->ne[3] ||
            node->src[0]->nb[0] != ggml_type_size(node->src[0]->type)) {
            break;
        }
        bool independent = true;
        for (const auto & m : mats) {
            independent = independent && node->src[0] != m.dst;
        }
        if (!independent) {
            break;
        }
        d3d11_mat_slot slot = { node->src[0], node, nullptr };
        int consumed = 1;
        if (!dev.no_fuse && ggml_can_fuse(cgraph, j, { GGML_OP_MUL_MAT, GGML_OP_ADD })) {
            ggml_tensor * addend = ggml_d3d11_fusable_addend(node, cgraph->nodes[j + 1]);
            bool ok = addend != nullptr;
            for (const auto & m : mats) {
                ok = ok && addend != m.dst && addend != m.src0;
            }
            if (ok) {
                slot.dst = cgraph->nodes[j + 1];
                slot.add = addend;
                consumed = 2;
            }
        }
        mats.push_back(slot);
        j += consumed;
    }
    if (mats.empty()) {
        // first node did not pass the group checks (e.g. non-contiguous dst): plain path
        ggml_d3d11_mul_mat(dev, first->src[0], src1, first);
        return 1;
    }
    ggml_d3d11_mul_mat_group(dev, src1, mats);
    return j - i;
}

static const char * ggml_d3d11_float_type_define(ggml_type t, std::vector<std::string> & defines) {
    defines.push_back(t == GGML_TYPE_F16 ? "TYPE_F16" : "TYPE_F32");
    if (t == GGML_TYPE_F16) {
        defines.push_back("USE_16BIT");
    }
    return nullptr;
}

// rms_norm, optionally fused with a following MUL by `wgt` (then `dst` is the MUL node and `norm` the RMS_NORM node)
static void ggml_d3d11_rms_norm(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * norm, ggml_tensor * dst, ggml_tensor * wgt) {
    std::vector<std::string> defines;
    if (wgt) {
        defines.push_back("FUSE_MUL");
    }
    if (norm->op == GGML_OP_L2_NORM) {
        defines.push_back("L2_NORM");
    }
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "rms_norm", hlsl_rms_norm, defines);
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const d3d11_binding bw = wgt ? ggml_d3d11_bind_tensor(wgt) : d3d11_binding{ 0, 0 };
    const uint32_t n_rows  = (uint32_t) ggml_nrows(dst);
    std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], n_rows,
        ggml_d3d11_u32_from_f32(ggml_get_op_params_f32(norm, 0)),
        bw.elem_offset,
        wgt ? (uint32_t) wgt->ne[1] : 1u, wgt ? (uint32_t) wgt->ne[2] : 1u, wgt ? (uint32_t) wgt->ne[3] : 1u,
        wgt ? (uint32_t) (wgt->nb[1] / 4) : 0u, wgt ? (uint32_t) (wgt->nb[2] / 4) : 0u, wgt ? (uint32_t) (wgt->nb[3] / 4) : 0u,
        ggml_d3d11_u32_from_f32(1.0f),   // post_scale
    };
    std::vector<D3D11_GPU_VIRTUAL_ADDRESS> uavs = { bs.va, bd.va };
    if (wgt) {
        uavs.push_back(bw.va);
    }
    ggml_d3d11_dispatch(dev, pipeline, params, uavs, n_rows);
}

// ARGSORT and TOP_K share one kernel; TOP_K keeps the k largest (dst->ne[0]) in no particular order
static void ggml_d3d11_argsort(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<std::string> defines;
    if (dst->op == GGML_OP_TOP_K || (ggml_sort_order) ggml_get_op_params_i32(dst, 0) == GGML_SORT_ORDER_DESC) {
        defines.push_back("SORT_DESC");
    }
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "argsort", hlsl_argsort, defines);
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t n_rows  = (uint32_t) ggml_nrows(src);
    std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset, (uint32_t) (src->nb[1] / 4), (uint32_t) src->ne[0], (uint32_t) dst->ne[0], 0, 0,
    };
    // every element is compared with its whole row: ne0^2 work per row, bounded per command list
    const uint64_t row_work = (uint64_t) src->ne[0] * (uint64_t) src->ne[0];
    const uint32_t chunk    = (uint32_t) std::max<uint64_t>(1, D3D11_FLASH_ATTN_WORK / row_work);
    const bool     big      = (uint64_t) n_rows * row_work > D3D11_FLASH_ATTN_WORK;
    for (uint32_t row0 = 0; row0 < n_rows; row0 += chunk) {
        params[5] = row0;
        params[6] = std::min(chunk, n_rows - row0);
        if (big && dev.dispatches_in_list > 0) {
            dev.n_flush_batch++;
            ggml_d3d11_submit_and_wait(dev);
            ggml_d3d11_begin(dev, true);
        }
        ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, params[6]);
    }
}

static void ggml_d3d11_repeat(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "repeat", hlsl_repeat, {});
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[0] / 4), (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[0] / 4), (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) src->ne[3],
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], ne,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_im2col(d3d11_device_ctx & dev, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * src    = dst->src[1];
    const int32_t *     op     = (const int32_t *) dst->op_params;
    const bool          is_2D  = op[6] == 1;

    std::vector<std::string> defines;
    if (dst->type == GGML_TYPE_F16) {
        defines = { "DST_F16", "USE_16BIT" };
    }
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "im2col", hlsl_im2col, defines);
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[is_2D ? 3 : 2] / 4), (uint32_t) (src->nb[is_2D ? 2 : 1] / 4),
        is_2D ? (uint32_t) (src->nb[1] / 4) : 0u, (uint32_t) (src->nb[0] / 4),
        (uint32_t) op[0], (uint32_t) op[1], (uint32_t) op[2], (uint32_t) op[3], (uint32_t) op[4], (uint32_t) op[5],
        (uint32_t) (is_2D ? src->ne[2] : src->ne[1]), (uint32_t) (is_2D ? src->ne[1] : 1), (uint32_t) src->ne[0],
        (uint32_t) (is_2D ? kernel->ne[1] : 1), (uint32_t) kernel->ne[0],
        (uint32_t) (is_2D ? dst->ne[2] : 1), (uint32_t) dst->ne[1], ne,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_im2col_3d(d3d11_device_ctx & dev, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * src    = dst->src[1];
    const int32_t *     op     = (const int32_t *) dst->op_params;
    const uint32_t      IC     = (uint32_t) op[9];

    std::vector<std::string> defines;
    if (dst->type == GGML_TYPE_F16) {
        defines = { "DST_F16", "USE_16BIT" };
    }
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "im2col_3d", hlsl_im2col_3d, defines);
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      N  = (uint32_t) (src->ne[3] / IC);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[3] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[1] / 4),
        (uint32_t) op[0], (uint32_t) op[1], (uint32_t) op[2], (uint32_t) op[3], (uint32_t) op[4],
        (uint32_t) op[5], (uint32_t) op[6], (uint32_t) op[7], (uint32_t) op[8],
        IC, (uint32_t) src->ne[2], (uint32_t) src->ne[1], (uint32_t) src->ne[0],
        (uint32_t) kernel->ne[2], (uint32_t) kernel->ne[1], (uint32_t) kernel->ne[0],
        (uint32_t) (dst->ne[3] / N), (uint32_t) dst->ne[2], (uint32_t) dst->ne[1], ne,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_upscale(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    const int32_t mode_flags = ggml_get_op_params_i32(dst, 0);
    float sf0 = (float) dst->ne[0] / src->ne[0];
    float sf1 = (float) dst->ne[1] / src->ne[1];
    const float sf2 = (float) dst->ne[2] / src->ne[2];
    const float sf3 = (float) dst->ne[3] / src->ne[3];
    float pixel_offset = 0.5f;
    if (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        pixel_offset = 0.0f;
        sf0 = dst->ne[0] > 1 && src->ne[0] > 1 ? (float) (dst->ne[0] - 1) / (src->ne[0] - 1) : sf0;
        sf1 = dst->ne[1] > 1 && src->ne[1] > 1 ? (float) (dst->ne[1] - 1) / (src->ne[1] - 1) : sf1;
    }
    std::vector<std::string> defines;
    if ((mode_flags & 0xFF) == GGML_SCALE_MODE_BILINEAR) {
        defines.push_back("BILINEAR");
    }
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "upscale", hlsl_upscale, defines);
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[0] / 4), (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[0] / 4), (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1],
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], ne,
        ggml_d3d11_u32_from_f32(sf0), ggml_d3d11_u32_from_f32(sf1), ggml_d3d11_u32_from_f32(sf2), ggml_d3d11_u32_from_f32(sf3),
        ggml_d3d11_u32_from_f32(pixel_offset),
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_pool_2d(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    const int32_t * op = (const int32_t *) dst->op_params;
    std::vector<std::string> defines;
    if (src->type == GGML_TYPE_F16) {
        defines.push_back("SRC_F16");
    }
    if ((ggml_op_pool) op[0] == GGML_OP_POOL_MAX) {
        defines.push_back("POOL_MAX");
    }
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "pool_2d", hlsl_pool_2d, defines);
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const size_t        ts = ggml_type_size(src->type);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
        (uint32_t) op[1], (uint32_t) op[2], (uint32_t) op[3], (uint32_t) op[4], (uint32_t) op[5], (uint32_t) op[6],
        ne,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

// FILL: the memset kernel with the constant's bit pattern (f16: the half value twice per word)
static void ggml_d3d11_fill(d3d11_device_ctx & dev, ggml_tensor * dst) {
    const float c = ggml_get_op_params_f32(dst, 0);
    uint32_t pattern;
    if (dst->type == GGML_TYPE_F16) {
        const uint32_t h = ggml_fp32_to_fp16(c);
        pattern = h | (h << 16);
    } else {
        pattern = ggml_d3d11_u32_from_f32(c);
    }
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "memset", hlsl_memset,
                                                        { "BYTES_PER_THREAD=" + std::to_string(D3D11_MEMSET_BYTES_PER_THREAD) });
    const size_t   offset = ggml_d3d11_tensor_offset(dst);
    const size_t   base   = offset & ~((size_t) D3D11_BINDING_ALIGNMENT - 1);
    const uint32_t rel    = (uint32_t) (offset - base);
    const uint32_t n      = (uint32_t) ggml_nbytes(dst);
    const uint32_t threads = CEIL_DIV(rel + n, (uint32_t) D3D11_MEMSET_BYTES_PER_THREAD);
    ggml_d3d11_dispatch(dev, pipeline, { rel, n, pattern }, { ggml_d3d11_tensor_va(dst) + base },
                        CEIL_DIV(threads, (uint32_t) D3D11_WG_SIZE));
}

// MEAN reuses the sum_rows kernel, which divides by the row length when MEAN is defined
static void ggml_d3d11_sum_rows(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst, bool mean = false) {
    std::vector<std::string> defines;
    if (mean) {
        defines.push_back("MEAN");
    }
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, mean ? "sum_rows_mean" : "sum_rows", hlsl_sum_rows, defines);
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t n_rows  = (uint32_t) ggml_nrows(src);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], n_rows,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, n_rows);
}

// SUM: every element of src into one scalar, a single workgroup walking the whole tensor
static void ggml_d3d11_sum(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "sum", hlsl_sum, {});
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2],
        (uint32_t) ggml_nrows(src),
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, 1);
}

// ARGMAX: index of the largest element of each row, one workgroup per row
static void ggml_d3d11_argmax(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "argmax", hlsl_argmax, {});
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      n_rows = (uint32_t) src->ne[1];
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / 4),
        (uint32_t) src->ne[0], n_rows,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, n_rows);
}

// ARANGE: no input tensor, dst[i] = start + step * i
static void ggml_d3d11_arange(d3d11_device_ctx & dev, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "arange", hlsl_arange, {});
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const float start = ggml_get_op_params_f32(dst, 0);
    const float step  = ggml_get_op_params_f32(dst, 2);
    const std::vector<uint32_t> params = {
        bd.elem_offset, ne, ggml_d3d11_u32_from_f32(start), ggml_d3d11_u32_from_f32(step),
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

// DIAG_MASK_INF / DIAG_MASK_ZERO: copy src and replace everything right of the shifted diagonal
static void ggml_d3d11_diag_mask(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst, float value) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "diag_mask", hlsl_diag_mask, {});
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset, ne,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
        (uint32_t) ggml_get_op_params_i32(dst, 0), ggml_d3d11_u32_from_f32(value),
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

// ROLL: cyclic shift along all four axes, one workgroup per destination row
static void ggml_d3d11_roll(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "roll", hlsl_roll, {});
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      n_rows = (uint32_t) ggml_nrows(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], (uint32_t) dst->ne[3],
        (uint32_t) ggml_get_op_params_i32(dst, 0), (uint32_t) ggml_get_op_params_i32(dst, 1),
        (uint32_t) ggml_get_op_params_i32(dst, 2), (uint32_t) ggml_get_op_params_i32(dst, 3),
        n_rows,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, n_rows);
}

// PAD: src copied into a larger dst at the left pads; outside, zero or a wrapped source element
static void ggml_d3d11_pad(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    const bool circular = ggml_get_op_params_i32(dst, 8) != 0;
    std::vector<std::string> defines;
    if (circular) {
        defines.push_back("CIRCULAR");
    }
    d3d11_pipeline & pipeline =
        ggml_d3d11_get_pipeline(dev, circular ? "pad_circular" : "pad", hlsl_pad, defines);
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[0] / 4), (uint32_t) (src->nb[1] / 4),
        (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) src->ne[3],
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], ne,
        (uint32_t) ggml_get_op_params_i32(dst, 0), (uint32_t) ggml_get_op_params_i32(dst, 2),
        (uint32_t) ggml_get_op_params_i32(dst, 4), (uint32_t) ggml_get_op_params_i32(dst, 6),
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

// PAD_REFLECT_1D: mirror the row into both margins, one workgroup per row
static void ggml_d3d11_pad_reflect_1d(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "pad_reflect_1d", hlsl_pad_reflect_1d, {});
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      n_rows = (uint32_t) ggml_nrows(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],
        (uint32_t) src->ne[0], (uint32_t) ggml_get_op_params_i32(dst, 0), n_rows,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, n_rows);
}

// TIMESTEP_EMBEDDING: cos/sin ladder per input timestep, one workgroup per timestep
static void ggml_d3d11_timestep_embedding(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    d3d11_pipeline & pipeline =
        ggml_d3d11_get_pipeline(dev, "timestep_embedding", hlsl_timestep_embedding, {});
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      dim        = (uint32_t) ggml_get_op_params_i32(dst, 0);
    const uint32_t      max_period = (uint32_t) ggml_get_op_params_i32(dst, 1);
    const uint32_t      ne00       = (uint32_t) src->ne[0];
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset, (uint32_t) (dst->nb[1] / 4),
        ne00, dim / 2, dim, ggml_d3d11_u32_from_f32(-logf((float) max_period)),
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, ne00);
}

// SET / ACC: dst takes a copy of src0, then src1 lands in the view described by op_params.
// Two dispatches, because the copy has to be complete before the scatter overwrites part of it.
static void ggml_d3d11_set_acc(d3d11_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1,
                               ggml_tensor * dst, bool acc) {
    if (src0->data != dst->data) {
        ggml_d3d11_cpy(dev, src0, dst);
    }
    std::vector<std::string> defines;
    if (acc) {
        defines.push_back("ACC");
    }
    d3d11_pipeline & pipeline =
        ggml_d3d11_get_pipeline(dev, acc ? "set_acc_add" : "set_acc", hlsl_set_acc, defines);
    const d3d11_binding b1 = ggml_d3d11_bind_tensor(src1);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      n_rows = (uint32_t) ggml_nrows(src1);
    // the view strides and offset are byte counts in op_params; the kernel indexes in elements
    const std::vector<uint32_t> params = {
        b1.elem_offset, bd.elem_offset,
        (uint32_t) (src1->nb[1] / 4), (uint32_t) (src1->nb[2] / 4), (uint32_t) (src1->nb[3] / 4),
        (uint32_t) (ggml_get_op_params_i32(dst, 0) / 4), (uint32_t) (ggml_get_op_params_i32(dst, 1) / 4),
        (uint32_t) (ggml_get_op_params_i32(dst, 2) / 4), (uint32_t) (ggml_get_op_params_i32(dst, 3) / 4),
        (uint32_t) src1->ne[0], (uint32_t) src1->ne[1], (uint32_t) src1->ne[2],
        n_rows,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { b1.va, bd.va }, n_rows);
}

// CUMSUM: inclusive prefix sum along each row
static void ggml_d3d11_cumsum(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "cumsum", hlsl_cumsum, {});
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      n_rows = (uint32_t) ggml_nrows(src);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], n_rows,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, n_rows);
}

// TRI: keep one triangle of each matrix, zero the rest
static void ggml_d3d11_tri(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "tri", hlsl_tri, {});
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      n_rows = (uint32_t) ggml_nrows(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], n_rows,
        (uint32_t) ggml_get_op_params_i32(dst, 0),
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, n_rows);
}

// COUNT_EQUAL: number of positions where the two i32 tensors agree, into an i64 scalar
static void ggml_d3d11_count_equal(d3d11_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1,
                                   ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "count_equal", hlsl_count_equal, {});
    const d3d11_binding b0 = ggml_d3d11_bind_tensor(src0);
    const d3d11_binding b1 = ggml_d3d11_bind_tensor(src1);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, bd.elem_offset * (uint32_t) ggml_type_size(dst->type),
        (uint32_t) (src0->nb[1] / 4), (uint32_t) (src0->nb[2] / 4), (uint32_t) (src0->nb[3] / 4),
        (uint32_t) (src1->nb[1] / 4), (uint32_t) (src1->nb[2] / 4), (uint32_t) (src1->nb[3] / 4),
        (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], (uint32_t) src0->ne[2],
        (uint32_t) ggml_nrows(src0),
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { b0.va, b1.va, bd.va }, 1);
}

// ADD1: dst = src0 plus a scalar that lives in device memory
static void ggml_d3d11_add1(d3d11_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1,
                            ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "add1", hlsl_add1, {});
    const d3d11_binding b0 = ggml_d3d11_bind_tensor(src0);
    const d3d11_binding b1 = ggml_d3d11_bind_tensor(src1);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      n_rows = (uint32_t) ggml_nrows(dst);
    const std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, bd.elem_offset,
        (uint32_t) (src0->nb[1] / 4), (uint32_t) (src0->nb[2] / 4), (uint32_t) (src0->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], n_rows,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { b0.va, b1.va, bd.va }, n_rows);
}

// LEAKY_RELU: negative inputs scaled by the slope in op_params
static void ggml_d3d11_leaky_relu(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "leaky_relu", hlsl_leaky_relu, {});
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      n_rows = (uint32_t) ggml_nrows(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / 4), (uint32_t) (dst->nb[1] / 4),
        (uint32_t) dst->ne[0], n_rows,
        ggml_d3d11_u32_from_f32(ggml_get_op_params_f32(dst, 0)),
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, n_rows);
}

// GROUP_NORM: one workgroup per (channel group, batch)
static void ggml_d3d11_group_norm(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "group_norm", hlsl_group_norm, {});
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t n_channels = (uint32_t) src->ne[2];
    const uint32_t n_groups   = (uint32_t) ggml_get_op_params_i32(dst, 0);
    const uint32_t n_batches  = (uint32_t) src->ne[3];
    float eps;
    memcpy(&eps, (const int32_t *) dst->op_params + 1, sizeof(float));
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1],
        n_channels, n_groups, CEIL_DIV(n_channels, n_groups), n_batches,
        ggml_d3d11_u32_from_f32(eps),
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, n_groups * n_batches);
}

// DIAG: vector to diagonal matrix, one workgroup per destination row
static void ggml_d3d11_diag(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "diag", hlsl_diag, {});
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      n_rows = (uint32_t) ggml_nrows(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], n_rows,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, n_rows);
}

// POOL_1D: sliding window along the row only
static void ggml_d3d11_pool_1d(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<std::string> defines;
    if (src->type == GGML_TYPE_F16) {
        defines = { "SRC_F16", "USE_16BIT" };
    }
    d3d11_pipeline & pipeline =
        ggml_d3d11_get_pipeline(dev, src->type == GGML_TYPE_F16 ? "pool_1d_f16" : "pool_1d",
                                hlsl_pool_1d, defines);
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const int32_t * opts   = (const int32_t *) dst->op_params;
    const uint32_t  n_rows = (uint32_t) ggml_nrows(src);
    const size_t    ts     = ggml_type_size(src->type);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / ts), (uint32_t) (dst->nb[1] / 4),
        (uint32_t) src->ne[0], (uint32_t) dst->ne[0],
        (uint32_t) opts[1], (uint32_t) opts[2], (uint32_t) opts[3],
        (uint32_t) (opts[0] == GGML_OP_POOL_MAX ? 1 : 0),
        n_rows,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, n_rows);
}

static void ggml_d3d11_dsv4_hc_pre(d3d11_device_ctx & dev, ggml_tensor * dst) {
    ggml_tensor * x = dst->src[0];
    ggml_tensor * w = dst->src[1];
    const bool gated = ggml_get_op_params_i32(dst, 1) != 0;

    std::vector<std::string> defines;
    if (gated) {
        defines.push_back("GATED");
    }
    d3d11_pipeline & pipeline =
        ggml_d3d11_get_pipeline(dev, gated ? "dsv4_hc_pre_gated" : "dsv4_hc_pre", hlsl_dsv4_hc_pre, defines);
    const d3d11_binding bx = ggml_d3d11_bind_tensor(x);
    const d3d11_binding bw = ggml_d3d11_bind_tensor(w);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        bx.elem_offset, bw.elem_offset, bd.elem_offset,
        (uint32_t) (x->nb[0] / 4), (uint32_t) (x->nb[1] / 4), (uint32_t) (x->nb[2] / 4),
        (uint32_t) (w->nb[0] / 4), (uint32_t) (w->nb[1] / 4), (uint32_t) (w->nb[2] / 4),
        (uint32_t) (dst->nb[0] / 4), (uint32_t) (dst->nb[1] / 4),
        (uint32_t) x->ne[0], (uint32_t) x->ne[1],
        ggml_d3d11_u32_from_f32(ggml_get_op_params_f32(dst, 0)),
        ne,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bx.va, bw.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_dsv4_hc_post(d3d11_device_ctx & dev, ggml_tensor * dst) {
    ggml_tensor * x = dst->src[0];
    ggml_tensor * r = dst->src[1];
    ggml_tensor * p = dst->src[2];
    ggml_tensor * c = dst->src[3];

    std::vector<std::string> defines;
    if (c) {
        defines.push_back("HAS_COMB");
    }
    d3d11_pipeline & pipeline =
        ggml_d3d11_get_pipeline(dev, c ? "dsv4_hc_post_comb" : "dsv4_hc_post", hlsl_dsv4_hc_post, defines);
    const d3d11_binding bx = ggml_d3d11_bind_tensor(x);
    const d3d11_binding br = ggml_d3d11_bind_tensor(r);
    const d3d11_binding bp = ggml_d3d11_bind_tensor(p);
    // with no comb matrix the slot still has to be bound; the residual stands in and is never read
    const d3d11_binding bc = ggml_d3d11_bind_tensor(c ? c : r);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        bx.elem_offset, br.elem_offset, bp.elem_offset, bc.elem_offset, bd.elem_offset,
        (uint32_t) (x->nb[0] / 4), (uint32_t) (x->nb[1] / 4),
        (uint32_t) (r->nb[0] / 4), (uint32_t) (r->nb[1] / 4), (uint32_t) (r->nb[2] / 4),
        (uint32_t) (p->nb[0] / 4), (uint32_t) (p->nb[1] / 4),
        (uint32_t) (c ? c->nb[0] / 4 : 0), (uint32_t) (c ? c->nb[1] / 4 : 0), (uint32_t) (c ? c->nb[2] / 4 : 0),
        (uint32_t) (dst->nb[0] / 4), (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4),
        (uint32_t) x->ne[0], (uint32_t) r->ne[1],
        ne,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bx.va, br.va, bp.va, bc.va, bd.va },
                        CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_dsv4_hc_comb(d3d11_device_ctx & dev, ggml_tensor * dst) {
    ggml_tensor * m = dst->src[0];
    ggml_tensor * s = dst->src[1];
    ggml_tensor * b = dst->src[2];

    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "dsv4_hc_comb", hlsl_dsv4_hc_comb, {});
    const d3d11_binding bm = ggml_d3d11_bind_tensor(m);
    const d3d11_binding bs = ggml_d3d11_bind_tensor(s);
    const d3d11_binding bb = ggml_d3d11_bind_tensor(b);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t n_tokens = (uint32_t) m->ne[1];
    const std::vector<uint32_t> params = {
        bm.elem_offset, bs.elem_offset, bb.elem_offset, bd.elem_offset,
        (uint32_t) (m->nb[0] / 4), (uint32_t) (m->nb[1] / 4),
        (uint32_t) (s->nb[0] / 4), (uint32_t) (b->nb[0] / 4),
        (uint32_t) (dst->nb[0] / 4), (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4),
        ggml_d3d11_u32_from_f32(ggml_get_op_params_f32(dst, 0)),
        (uint32_t) ggml_get_op_params_i32(dst, 1),
        n_tokens,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bm.va, bs.va, bb.va, bd.va },
                        CEIL_DIV(n_tokens, (uint32_t) D3D11_WG_SIZE));
}

// OPT_STEP_SGD / OPT_STEP_ADAMW: dst is a view of src0, so the update lands in src0's own buffer
static void ggml_d3d11_opt_step(d3d11_device_ctx & dev, ggml_tensor * dst, bool adamw) {
    ggml_tensor * w = dst->src[0];
    ggml_tensor * g = dst->src[1];
    ggml_tensor * m = adamw ? dst->src[2] : g;
    ggml_tensor * v = adamw ? dst->src[3] : g;
    ggml_tensor * p = adamw ? dst->src[4] : dst->src[2];

    std::vector<std::string> defines;
    if (adamw) {
        defines.push_back("ADAMW");
    }
    d3d11_pipeline & pipeline =
        ggml_d3d11_get_pipeline(dev, adamw ? "opt_step_adamw" : "opt_step_sgd", hlsl_opt_step, defines);
    // w (and m, v for AdamW) are written in place and are not node outputs: tag them as written.
    // SGD has no m and v; their slots are null.
    d3d11_binding       bw = ggml_d3d11_bind_tensor(w);
    const d3d11_binding bg = ggml_d3d11_bind_tensor(g);
    d3d11_binding       bm = adamw ? ggml_d3d11_bind_tensor(m) : d3d11_binding{ 0, 0 };
    d3d11_binding       bv = adamw ? ggml_d3d11_bind_tensor(v) : d3d11_binding{ 0, 0 };
    const d3d11_binding bp = ggml_d3d11_bind_tensor(p);
    bw.va |= D3D11_VA_WRITTEN;
    if (adamw) {
        bm.va |= D3D11_VA_WRITTEN;
        bv.va |= D3D11_VA_WRITTEN;
    }
    const uint32_t ne = (uint32_t) ggml_nelements(w);
    const std::vector<uint32_t> params = {
        bw.elem_offset, bg.elem_offset, bm.elem_offset, bv.elem_offset, bp.elem_offset, ne,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bw.va, bg.va, bm.va, bv.va, bp.va },
                        CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_silu_back(d3d11_device_ctx & dev, ggml_tensor * dy, ggml_tensor * x, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "silu_back", hlsl_silu_back, {});
    const d3d11_binding bg = ggml_d3d11_bind_tensor(dy);
    const d3d11_binding bx = ggml_d3d11_bind_tensor(x);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = { bg.elem_offset, bx.elem_offset, bd.elem_offset, ne };
    ggml_d3d11_dispatch(dev, pipeline, params, { bg.va, bx.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_repeat_back(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "repeat_back", hlsl_repeat_back, {});
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], (uint32_t) dst->ne[3],
        (uint32_t) (src->ne[0] / dst->ne[0]), (uint32_t) (src->ne[1] / dst->ne[1]),
        (uint32_t) (src->ne[2] / dst->ne[2]), (uint32_t) (src->ne[3] / dst->ne[3]),
        ne,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_rms_norm_back(d3d11_device_ctx & dev, ggml_tensor * dz, ggml_tensor * x, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "rms_norm_back", hlsl_rms_norm_back, {});
    const d3d11_binding bz = ggml_d3d11_bind_tensor(dz);
    const d3d11_binding bx = ggml_d3d11_bind_tensor(x);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t n_rows  = (uint32_t) ggml_nrows(dst);
    const std::vector<uint32_t> params = {
        bz.elem_offset, bx.elem_offset, bd.elem_offset,
        (uint32_t) (dz->nb[1] / 4), (uint32_t) (dz->nb[2] / 4), (uint32_t) (dz->nb[3] / 4),
        (uint32_t) (x->nb[1] / 4), (uint32_t) (x->nb[2] / 4), (uint32_t) (x->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], n_rows,
        ggml_d3d11_u32_from_f32(ggml_get_op_params_f32(dst, 0)),
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bz.va, bx.va, bd.va }, n_rows);
}

static void ggml_d3d11_soft_max_back(d3d11_device_ctx & dev, ggml_tensor * dy, ggml_tensor * y, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "soft_max_back", hlsl_soft_max_back, {});
    const d3d11_binding bg = ggml_d3d11_bind_tensor(dy);
    const d3d11_binding by = ggml_d3d11_bind_tensor(y);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t n_rows  = (uint32_t) ggml_nrows(dst);
    const std::vector<uint32_t> params = {
        bg.elem_offset, by.elem_offset, bd.elem_offset, (uint32_t) dst->ne[0], n_rows,
        ggml_d3d11_u32_from_f32(ggml_get_op_params_f32(dst, 0)),
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bg.va, by.va, bd.va }, n_rows);
}

// the loss is a single scalar, so the whole reduction runs in one workgroup
static void ggml_d3d11_cross_entropy_loss(d3d11_device_ctx & dev, ggml_tensor * s0, ggml_tensor * s1, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "cross_entropy_loss", hlsl_cross_entropy_loss, {});
    const d3d11_binding b0 = ggml_d3d11_bind_tensor(s0);
    const d3d11_binding b1 = ggml_d3d11_bind_tensor(s1);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, bd.elem_offset, (uint32_t) s0->ne[0], (uint32_t) ggml_nrows(s0),
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { b0.va, b1.va, bd.va }, 1);
}

static void ggml_d3d11_cross_entropy_loss_back(d3d11_device_ctx & dev, ggml_tensor * dst) {
    d3d11_pipeline & pipeline =
        ggml_d3d11_get_pipeline(dev, "cross_entropy_loss_back", hlsl_cross_entropy_loss_back, {});
    const d3d11_binding bg = ggml_d3d11_bind_tensor(dst->src[0]);
    const d3d11_binding b0 = ggml_d3d11_bind_tensor(dst->src[1]);
    const d3d11_binding b1 = ggml_d3d11_bind_tensor(dst->src[2]);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t n_rows  = (uint32_t) ggml_nrows(dst);
    const std::vector<uint32_t> params = {
        bg.elem_offset, b0.elem_offset, b1.elem_offset, bd.elem_offset, (uint32_t) dst->ne[0], n_rows,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bg.va, b0.va, b1.va, bd.va }, n_rows);
}

static void ggml_d3d11_get_rows_back(d3d11_device_ctx & dev, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    ggml_tensor * ids = dst->src[1];

    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "get_rows_back", hlsl_get_rows_back, {});
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bi = ggml_d3d11_bind_tensor(ids);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t n_rows  = (uint32_t) ggml_nrows(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bi.elem_offset, bd.elem_offset,
        (uint32_t) dst->ne[0], n_rows, (uint32_t) ggml_nelements(ids),
        (uint32_t) (src->nb[1] / 4), (uint32_t) (dst->nb[1] / 4),
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bi.va, bd.va }, n_rows);
}

static void ggml_d3d11_im2col_back(d3d11_device_ctx & dev, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];   // gradients of the im2col output
    ggml_tensor * src1 = dst->src[1];   // the convolution kernel, only its shape is used

    const int32_t s0 = ggml_get_op_params_i32(dst, 0);
    const int32_t s1 = ggml_get_op_params_i32(dst, 1);
    const int32_t p0 = ggml_get_op_params_i32(dst, 2);
    const int32_t p1 = ggml_get_op_params_i32(dst, 3);
    const int32_t d0 = ggml_get_op_params_i32(dst, 4);
    const int32_t d1 = ggml_get_op_params_i32(dst, 5);
    const bool is_2D = ggml_get_op_params_i32(dst, 6) == 1;

    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "im2col_back", hlsl_im2col_back, {});
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src0);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t ne = (uint32_t) ggml_nelements(dst);
    // in the 1D case the CPU pins ioh to 0 and ignores s1/p1/d1, so neutral values give the same result
    const std::vector<uint32_t> params = {
        (uint32_t) (is_2D ? dst->ne[3] : dst->ne[2]),
        (uint32_t) (is_2D ? dst->ne[2] : dst->ne[1]),
        (uint32_t) (is_2D ? dst->ne[1] : 1),
        (uint32_t) dst->ne[0],
        (uint32_t) (is_2D ? src1->ne[1] : 1),
        (uint32_t) src1->ne[0],
        (uint32_t) (is_2D ? src0->ne[2] : 1),
        (uint32_t) src0->ne[1],
        (uint32_t) s0, (uint32_t) (is_2D ? s1 : 1),
        (uint32_t) p0, (uint32_t) (is_2D ? p1 : 0),
        (uint32_t) d0, (uint32_t) (is_2D ? d1 : 1),
        bs.elem_offset, bd.elem_offset, ne,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_lightning_indexer(d3d11_device_ctx & dev, ggml_tensor * dst) {
    ggml_tensor * q = dst->src[0];
    ggml_tensor * k = dst->src[1];
    ggml_tensor * w = dst->src[2];
    ggml_tensor * m = dst->src[3];

    std::vector<std::string> defines = { "USE_16BIT" };
    if (k->type == GGML_TYPE_F16) {
        defines.push_back("K_F16");
    }
    d3d11_pipeline & pipeline =
        ggml_d3d11_get_pipeline(dev, k->type == GGML_TYPE_F16 ? "lightning_indexer_f16" : "lightning_indexer",
                                hlsl_lightning_indexer, defines);
    const d3d11_binding bq = ggml_d3d11_bind_tensor(q);
    const d3d11_binding bk = ggml_d3d11_bind_tensor(k);
    const d3d11_binding bw = ggml_d3d11_bind_tensor(w);
    const d3d11_binding bm = ggml_d3d11_bind_tensor(m);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const size_t   kts = ggml_type_size(k->type);
    const uint32_t ne  = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        bq.elem_offset, bk.elem_offset, bw.elem_offset, bm.elem_offset, bd.elem_offset,
        (uint32_t) (q->nb[1] / 4), (uint32_t) (q->nb[2] / 4), (uint32_t) (q->nb[3] / 4),
        (uint32_t) (k->nb[2] / kts), (uint32_t) (k->nb[3] / kts),
        (uint32_t) (w->nb[1] / 4), (uint32_t) (w->nb[3] / 4),
        (uint32_t) (m->nb[1] / 2), (uint32_t) (m->nb[3] / 2),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) q->ne[0], (uint32_t) q->ne[1], (uint32_t) q->ne[2], (uint32_t) k->ne[2],
        (uint32_t) m->ne[3],
        ne,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bq.va, bk.va, bw.va, bm.va, bd.va },
                        CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

// RWKV_WKV6 / GATED_LINEAR_ATTN / RWKV_WKV7 all carry a per-sequence state through the tokens and
// lay dst out as C*T outputs followed by that state, so they share these shape helpers.
struct d3d11_wkv_shape {
    uint32_t cc;       // C
    uint32_t hs;       // head size
    uint32_t tps;      // tokens per sequence
    uint32_t s_off;    // C * T
    uint32_t n_jobs;   // n_seqs * C
};

static d3d11_wkv_shape ggml_d3d11_wkv_shape(const ggml_tensor * dst, const ggml_tensor * state) {
    const uint32_t T      = (uint32_t) dst->src[1]->ne[2];
    const uint32_t cc     = (uint32_t) dst->ne[0];
    const uint32_t heads  = (uint32_t) dst->src[1]->ne[1];
    const uint32_t n_seqs = (uint32_t) state->ne[1];
    return { cc, cc / heads, T / n_seqs, cc * T, n_seqs * cc };
}

static void ggml_d3d11_rwkv_wkv6(d3d11_device_ctx & dev, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "rwkv_wkv6", hlsl_rwkv_wkv6, {});
    const d3d11_binding bk = ggml_d3d11_bind_tensor(dst->src[0]);
    const d3d11_binding bv = ggml_d3d11_bind_tensor(dst->src[1]);
    const d3d11_binding br = ggml_d3d11_bind_tensor(dst->src[2]);
    const d3d11_binding bf = ggml_d3d11_bind_tensor(dst->src[3]);
    const d3d11_binding bt = ggml_d3d11_bind_tensor(dst->src[4]);
    const d3d11_binding bs = ggml_d3d11_bind_tensor(dst->src[5]);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const d3d11_wkv_shape sh = ggml_d3d11_wkv_shape(dst, dst->src[5]);
    const std::vector<uint32_t> params = {
        bk.elem_offset, bv.elem_offset, br.elem_offset, bf.elem_offset, bt.elem_offset,
        bs.elem_offset, bd.elem_offset,
        sh.cc, sh.hs, sh.tps, sh.s_off, sh.n_jobs,
    };
    ggml_d3d11_dispatch(dev, pipeline, params,
                        { bk.va, bv.va, br.va, bf.va, bt.va, bs.va, bd.va },
                        CEIL_DIV(sh.n_jobs, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_gated_linear_attn(d3d11_device_ctx & dev, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "gated_linear_attn", hlsl_gated_linear_attn, {});
    const d3d11_binding bk = ggml_d3d11_bind_tensor(dst->src[0]);
    const d3d11_binding bv = ggml_d3d11_bind_tensor(dst->src[1]);
    const d3d11_binding bq = ggml_d3d11_bind_tensor(dst->src[2]);
    const d3d11_binding bg = ggml_d3d11_bind_tensor(dst->src[3]);
    const d3d11_binding bs = ggml_d3d11_bind_tensor(dst->src[4]);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const d3d11_wkv_shape sh = ggml_d3d11_wkv_shape(dst, dst->src[4]);
    const std::vector<uint32_t> params = {
        bk.elem_offset, bv.elem_offset, bq.elem_offset, bg.elem_offset, bs.elem_offset, bd.elem_offset,
        sh.cc, sh.hs, sh.tps, sh.s_off,
        ggml_d3d11_u32_from_f32(ggml_get_op_params_f32(dst, 0)),
        sh.n_jobs,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bk.va, bv.va, bq.va, bg.va, bs.va, bd.va },
                        CEIL_DIV(sh.n_jobs, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_rwkv_wkv7(d3d11_device_ctx & dev, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "rwkv_wkv7", hlsl_rwkv_wkv7, {});
    const d3d11_binding br = ggml_d3d11_bind_tensor(dst->src[0]);
    const d3d11_binding bw = ggml_d3d11_bind_tensor(dst->src[1]);
    const d3d11_binding bk = ggml_d3d11_bind_tensor(dst->src[2]);
    const d3d11_binding bv = ggml_d3d11_bind_tensor(dst->src[3]);
    const d3d11_binding ba = ggml_d3d11_bind_tensor(dst->src[4]);
    const d3d11_binding bb = ggml_d3d11_bind_tensor(dst->src[5]);
    const d3d11_binding bs = ggml_d3d11_bind_tensor(dst->src[6]);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const d3d11_wkv_shape sh = ggml_d3d11_wkv_shape(dst, dst->src[6]);
    const std::vector<uint32_t> params = {
        br.elem_offset, bw.elem_offset, bk.elem_offset, bv.elem_offset, ba.elem_offset,
        bb.elem_offset, bs.elem_offset, bd.elem_offset,
        sh.cc, sh.hs, sh.tps, sh.s_off, sh.n_jobs,
    };
    ggml_d3d11_dispatch(dev, pipeline, params,
                        { br.va, bw.va, bk.va, bv.va, ba.va, bb.va, bs.va, bd.va },
                        CEIL_DIV(sh.n_jobs, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_conv_2d(d3d11_device_ctx & dev, ggml_tensor * knl, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<std::string> defines;
    if (knl->type == GGML_TYPE_F16) {
        defines = { "KNL_F16", "USE_16BIT" };
    }
    d3d11_pipeline & pipeline =
        ggml_d3d11_get_pipeline(dev, knl->type == GGML_TYPE_F16 ? "conv_2d_f16" : "conv_2d", hlsl_conv_2d, defines);
    const d3d11_binding bk = ggml_d3d11_bind_tensor(knl);
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const int32_t * op = (const int32_t *) dst->op_params;
    const uint32_t  ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        bk.elem_offset, bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) knl->ne[0], (uint32_t) knl->ne[1], (uint32_t) src->ne[0], (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],
        (uint32_t) op[0], (uint32_t) op[1], (uint32_t) op[2], (uint32_t) op[3], (uint32_t) op[4], (uint32_t) op[5],
        ne,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bk.va, bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_conv_3d(d3d11_device_ctx & dev, ggml_tensor * knl, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<std::string> defines;
    if (knl->type == GGML_TYPE_F16) {
        defines = { "KNL_F16", "USE_16BIT" };
    }
    d3d11_pipeline & pipeline =
        ggml_d3d11_get_pipeline(dev, knl->type == GGML_TYPE_F16 ? "conv_3d_f16" : "conv_3d", hlsl_conv_3d, defines);
    const d3d11_binding bk = ggml_d3d11_bind_tensor(knl);
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const int32_t * op = (const int32_t *) dst->op_params;
    const uint32_t  ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        bk.elem_offset, bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) knl->ne[0], (uint32_t) knl->ne[1], (uint32_t) knl->ne[2],
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2],
        (uint32_t) op[9], (uint32_t) op[11],
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],
        (uint32_t) op[0], (uint32_t) op[1], (uint32_t) op[2], (uint32_t) op[3], (uint32_t) op[4],
        (uint32_t) op[5], (uint32_t) op[6], (uint32_t) op[7], (uint32_t) op[8],
        ne,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bk.va, bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_conv_transpose_2d(d3d11_device_ctx & dev, ggml_tensor * knl, ggml_tensor * src,
                                         ggml_tensor * dst) {
    std::vector<std::string> defines;
    if (knl->type == GGML_TYPE_F16) {
        defines = { "KNL_F16", "USE_16BIT" };
    }
    d3d11_pipeline & pipeline =
        ggml_d3d11_get_pipeline(dev, knl->type == GGML_TYPE_F16 ? "conv_transpose_2d_f16" : "conv_transpose_2d",
                                hlsl_conv_transpose_2d, defines);
    const d3d11_binding bk = ggml_d3d11_bind_tensor(knl);
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const size_t   ts = ggml_type_size(knl->type);
    const uint32_t ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        bk.elem_offset, bs.elem_offset, bd.elem_offset,
        (uint32_t) (knl->nb[1] / ts), (uint32_t) (knl->nb[2] / ts), (uint32_t) (knl->nb[3] / ts),
        (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) knl->ne[0], (uint32_t) knl->ne[1], (uint32_t) src->ne[0], (uint32_t) src->ne[1],
        (uint32_t) knl->ne[3],
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],
        (uint32_t) ggml_get_op_params_i32(dst, 0),
        ne,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bk.va, bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_col2im_1d(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "col2im_1d", hlsl_col2im_1d, {});
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t k_oc = (uint32_t) src->ne[0];
    const uint32_t oc   = (uint32_t) ggml_get_op_params_i32(dst, 1);
    const uint32_t ne   = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        k_oc, (uint32_t) src->ne[1], k_oc / oc, (uint32_t) dst->ne[0],
        (uint32_t) ggml_get_op_params_i32(dst, 0), (uint32_t) ggml_get_op_params_i32(dst, 2),
        ne,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_conv_transpose_1d(d3d11_device_ctx & dev, ggml_tensor * knl, ggml_tensor * src,
                                         ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "conv_transpose_1d", hlsl_conv_transpose_1d, {});
    const d3d11_binding bk = ggml_d3d11_bind_tensor(knl);
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        bk.elem_offset, bs.elem_offset, bd.elem_offset,
        (uint32_t) (knl->nb[1] / 4), (uint32_t) (knl->nb[2] / 4),
        (uint32_t) (src->nb[1] / 4), (uint32_t) (dst->nb[1] / 4),
        (uint32_t) knl->ne[0], (uint32_t) knl->ne[2], (uint32_t) src->ne[0],
        (uint32_t) dst->ne[0], (uint32_t) ggml_get_op_params_i32(dst, 0),
        ne,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bk.va, bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_conv_2d_dw(d3d11_device_ctx & dev, ggml_tensor * knl, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<std::string> defines;
    if (knl->type == GGML_TYPE_F16) {
        defines = { "KNL_F16", "USE_16BIT" };
    }
    d3d11_pipeline & pipeline =
        ggml_d3d11_get_pipeline(dev, knl->type == GGML_TYPE_F16 ? "conv_2d_dw_f16" : "conv_2d_dw",
                                hlsl_conv_2d_dw, defines);
    const d3d11_binding bk = ggml_d3d11_bind_tensor(knl);
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const int32_t * opts   = (const int32_t *) dst->op_params;
    const uint32_t  ne     = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        bk.elem_offset, bs.elem_offset, bd.elem_offset,
        (uint32_t) src->ne[0], (uint32_t) src->ne[1],
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
        (uint32_t) knl->ne[0], (uint32_t) knl->ne[1],
        (uint32_t) src->ne[2],
        (uint32_t) opts[0], (uint32_t) opts[1], (uint32_t) opts[2],
        (uint32_t) opts[3], (uint32_t) opts[4], (uint32_t) opts[5],
        ne,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bk.va, bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

// WIN_PART / WIN_UNPART: index remap between an image and its window tiling
static void ggml_d3d11_win_part(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst, bool unpart) {
    std::vector<std::string> defines;
    if (unpart) {
        defines.push_back("UNPART");
    }
    d3d11_pipeline & pipeline =
        ggml_d3d11_get_pipeline(dev, unpart ? "win_unpart" : "win_part", hlsl_win_part, defines);
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    uint32_t w, windows_across;
    if (unpart) {
        w = (uint32_t) ggml_get_op_params_i32(dst, 0);
        // npx: how many windows span a padded row, which is the stride between window rows in src
        const uint32_t pad = (w - (uint32_t) dst->ne[1] % w) % w;
        windows_across = (pad + (uint32_t) dst->ne[1]) / w;
    } else {
        windows_across = (uint32_t) ggml_get_op_params_i32(dst, 0);
        w              = (uint32_t) ggml_get_op_params_i32(dst, 2);
    }
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2],
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],
        ne, windows_across, w,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

// GET_REL_POS: f16 relative position lookup
static void ggml_d3d11_get_rel_pos(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    d3d11_pipeline & pipeline =
        ggml_d3d11_get_pipeline(dev, "get_rel_pos", hlsl_get_rel_pos, { "USE_16BIT" });
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) src->ne[0], (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], ne,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

// ADD_REL_POS: both biases added in one gather, so no copy pass is needed
static void ggml_d3d11_add_rel_pos(d3d11_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1,
                                   ggml_tensor * src2, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "add_rel_pos", hlsl_add_rel_pos, {});
    const d3d11_binding b0 = ggml_d3d11_bind_tensor(src0);
    const d3d11_binding b1 = ggml_d3d11_bind_tensor(src1);
    const d3d11_binding b2 = ggml_d3d11_bind_tensor(src2);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, b2.elem_offset, bd.elem_offset,
        (uint32_t) src1->ne[0], ne,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { b0.va, b1.va, b2.va, bd.va },
                        CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

// SOLVE_TRI: forward substitution, one right-hand-side column per thread
static void ggml_d3d11_solve_tri(d3d11_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1,
                                 ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "solve_tri", hlsl_solve_tri, {});
    const d3d11_binding ba = ggml_d3d11_bind_tensor(src0);
    const d3d11_binding bb = ggml_d3d11_bind_tensor(src1);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t n      = (uint32_t) src0->ne[0];
    const uint32_t k      = (uint32_t) src1->ne[0];
    const uint32_t n_jobs = (uint32_t) (src0->ne[2] * src0->ne[3]) * k;
    const std::vector<uint32_t> params = {
        ba.elem_offset, bb.elem_offset, bd.elem_offset,
        (uint32_t) (src0->nb[2] / 4), (uint32_t) (src0->nb[3] / 4),
        (uint32_t) (src1->nb[2] / 4), (uint32_t) (src1->nb[3] / 4),
        (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        n, k, (uint32_t) src0->ne[2], n_jobs,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { ba.va, bb.va, bd.va },
                        CEIL_DIV(n_jobs, (uint32_t) D3D11_WG_SIZE));
}

// OUT_PROD: contraction over src0's second dimension, one destination element per thread
static void ggml_d3d11_out_prod(d3d11_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1,
                                ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "out_prod", hlsl_out_prod, {});
    const d3d11_binding b0 = ggml_d3d11_bind_tensor(src0);
    const d3d11_binding b1 = ggml_d3d11_bind_tensor(src1);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, bd.elem_offset,
        (uint32_t) (src0->nb[1] / 4), (uint32_t) (src0->nb[2] / 4), (uint32_t) (src0->nb[3] / 4),
        (uint32_t) (src1->nb[0] / 4), (uint32_t) (src1->nb[1] / 4),
        (uint32_t) (src1->nb[2] / 4), (uint32_t) (src1->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],
        (uint32_t) src0->ne[1],
        (uint32_t) (dst->ne[2] / src0->ne[2]), (uint32_t) (dst->ne[3] / src0->ne[3]),
        ne,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { b0.va, b1.va, bd.va },
                        CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

// SSM_SCAN: one thread per (sequence, head, dim); the token loop stays inside the thread
static void ggml_d3d11_ssm_scan(d3d11_device_ctx & dev, ggml_tensor * dst) {
    ggml_tensor * s0  = dst->src[0];
    ggml_tensor * x   = dst->src[1];
    ggml_tensor * dt  = dst->src[2];
    ggml_tensor * A   = dst->src[3];
    ggml_tensor * B   = dst->src[4];
    ggml_tensor * C   = dst->src[5];
    ggml_tensor * ids = dst->src[6];

    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "ssm_scan", hlsl_ssm_scan, {});
    const d3d11_binding b0 = ggml_d3d11_bind_tensor(s0);
    const d3d11_binding b1 = ggml_d3d11_bind_tensor(x);
    const d3d11_binding b2 = ggml_d3d11_bind_tensor(dt);
    const d3d11_binding b3 = ggml_d3d11_bind_tensor(A);
    const d3d11_binding b4 = ggml_d3d11_bind_tensor(B);
    const d3d11_binding b5 = ggml_d3d11_bind_tensor(C);
    const d3d11_binding b6 = ggml_d3d11_bind_tensor(ids);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);

    const uint32_t nc = (uint32_t) s0->ne[0];
    const uint32_t nr = (uint32_t) s0->ne[1];
    const uint32_t nh = (uint32_t) x->ne[1];
    const uint32_t ng = (uint32_t) B->ne[1];
    const uint32_t nt = (uint32_t) x->ne[2];
    const uint32_t ns = (uint32_t) x->ne[3];
    const uint32_t n_jobs = ns * nh * nr;

    const std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, b2.elem_offset, b3.elem_offset,
        b4.elem_offset, b5.elem_offset, b6.elem_offset, bd.elem_offset,
        (uint32_t) (s0->nb[3] / 4),
        (uint32_t) (x->nb[2] / 4), (uint32_t) (x->nb[3] / 4),
        (uint32_t) (dt->nb[1] / 4), (uint32_t) (dt->nb[2] / 4),
        (uint32_t) (B->nb[2] / 4), (uint32_t) (B->nb[3] / 4),
        (uint32_t) (C->nb[2] / 4), (uint32_t) (C->nb[3] / 4),
        nc, nr, nh, ng, nt, ns,
        (uint32_t) ggml_get_op_params_i32(dst, 0),
        (uint32_t) ggml_nelements(x),          // s_off, in elements
        (uint32_t) (A->ne[0] == 1 ? 1 : 0),    // Mamba-2 has a scalar decay per head
        n_jobs,
    };
    ggml_d3d11_dispatch(dev, pipeline, params,
                        { b0.va, b1.va, b2.va, b3.va, b4.va, b5.va, b6.va, bd.va },
                        CEIL_DIV(n_jobs, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_add_id(d3d11_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * ids,
                              ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "add_id", hlsl_add_id, {});
    const d3d11_binding b0 = ggml_d3d11_bind_tensor(src0);
    const d3d11_binding b1 = ggml_d3d11_bind_tensor(src1);
    const d3d11_binding bi = ggml_d3d11_bind_tensor(ids);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, bi.elem_offset, bd.elem_offset,
        (uint32_t) (src0->nb[1] / 4), (uint32_t) (src0->nb[2] / 4), (uint32_t) (src1->nb[1] / 4),
        (uint32_t) (ids->nb[1] / 4), (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4),
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], ne,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { b0.va, b1.va, bi.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_norm(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "norm", hlsl_norm, {});
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t n_rows  = (uint32_t) ggml_nrows(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], n_rows,
        ggml_d3d11_u32_from_f32(ggml_get_op_params_f32(dst, 0)),
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, n_rows);
}

// RMS_NORM at node i followed by a MUL with a broadcastable f32 weight: one dispatch; returns nodes consumed
static int ggml_d3d11_encode_rms_norm(d3d11_device_ctx & dev, const ggml_cgraph * cgraph, int i) {
    ggml_tensor * norm = cgraph->nodes[i];
    if (!dev.no_fuse && ggml_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
        ggml_tensor * mul = cgraph->nodes[i + 1];
        ggml_tensor * w   = mul->src[0] == norm ? mul->src[1] : mul->src[0];
        bool ok = w != norm && w->type == GGML_TYPE_F32 && w->nb[0] == sizeof(float) && w->ne[0] == norm->ne[0] &&
                  mul->type == GGML_TYPE_F32 && (mul->src[0] == norm || ggml_are_same_shape(norm, w));
        for (int d = 1; d < 4; d++) {
            ok = ok && (w->ne[d] == 1 || w->ne[d] == norm->ne[d]);
        }
        if (ok) {
            ggml_d3d11_rms_norm(dev, norm->src[0], norm, mul, w);
            return 2;
        }
    }
    ggml_d3d11_rms_norm(dev, norm->src[0], norm, norm, nullptr);
    return 1;
}

static void ggml_d3d11_soft_max(d3d11_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * src2, ggml_tensor * dst) {
    std::vector<std::string> defines;
    if (src1) {
        defines.push_back("HAS_MASK");
        defines.push_back(src1->type == GGML_TYPE_F16 ? "MASK_F16" : "MASK_F32");
    }
    if (src2) {
        defines.push_back("HAS_SINK");
    }
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "soft_max", hlsl_soft_max, defines);

    const d3d11_binding b0 = ggml_d3d11_bind_tensor(src0);
    const d3d11_binding b1 = src1 ? ggml_d3d11_bind_tensor(src1) : d3d11_binding{ 0, 0 };
    const d3d11_binding b2 = src2 ? ggml_d3d11_bind_tensor(src2) : d3d11_binding{ 0, 0 };
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const size_t        t1 = src1 ? ggml_type_size(src1->type) : 4;

    const float max_bias    = ggml_get_op_params_f32(dst, 1);
    const float n_head_log2 = (float) (1u << (uint32_t) floor(log2((double) src0->ne[2])));
    const float m0          = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    const uint32_t n_rows   = (uint32_t) ggml_nrows(dst);

    std::vector<uint32_t> params = {
        b0.elem_offset, src1 ? b1.elem_offset : 0u, src2 ? b2.elem_offset : 0u, bd.elem_offset,
        (uint32_t) (src0->nb[1] / 4), (uint32_t) (src0->nb[2] / 4), (uint32_t) (src0->nb[3] / 4),
        src1 ? (uint32_t) (src1->nb[1] / t1) : 0u, src1 ? (uint32_t) (src1->nb[2] / t1) : 0u, src1 ? (uint32_t) (src1->nb[3] / t1) : 0u,
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], (uint32_t) src0->ne[2],
        src1 ? (uint32_t) src1->ne[2] : 1u, src1 ? (uint32_t) src1->ne[3] : 1u,
        ggml_d3d11_u32_from_f32(ggml_get_op_params_f32(dst, 0)),
        ggml_d3d11_u32_from_f32(max_bias), ggml_d3d11_u32_from_f32(n_head_log2),
        ggml_d3d11_u32_from_f32(m0), ggml_d3d11_u32_from_f32(m1),
        n_rows,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { b0.va, b1.va, b2.va, bd.va }, n_rows);
}

static void ggml_d3d11_concat(d3d11_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "concat", hlsl_concat, {});

    const d3d11_binding b0  = ggml_d3d11_bind_tensor(src0);
    const d3d11_binding b1  = ggml_d3d11_bind_tensor(src1);
    const d3d11_binding bd  = ggml_d3d11_bind_tensor(dst);
    const int32_t       dim = ggml_get_op_params_i32(dst, 0);
    const uint32_t      ne  = (uint32_t) ggml_nelements(dst);

    const std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, bd.elem_offset,
        (uint32_t) (src0->nb[0] / 4), (uint32_t) (src0->nb[1] / 4), (uint32_t) (src0->nb[2] / 4), (uint32_t) (src0->nb[3] / 4),
        (uint32_t) (src1->nb[0] / 4), (uint32_t) (src1->nb[1] / 4), (uint32_t) (src1->nb[2] / 4), (uint32_t) (src1->nb[3] / 4),
        (uint32_t) (dst->nb[0] / 4), (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        ne, (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],
        (uint32_t) dim, (uint32_t) src0->ne[dim],
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { b0.va, b1.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_ssm_conv(d3d11_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "ssm_conv", hlsl_ssm_conv, {});

    const d3d11_binding b0 = ggml_d3d11_bind_tensor(src0);
    const d3d11_binding b1 = ggml_d3d11_bind_tensor(src1);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);

    const std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, bd.elem_offset,
        (uint32_t) (src0->nb[1] / 4), (uint32_t) (src0->nb[2] / 4), (uint32_t) (src1->nb[1] / 4),
        (uint32_t) (dst->nb[0] / 4), (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4),
        (uint32_t) src1->ne[0], (uint32_t) src0->ne[1], (uint32_t) dst->ne[1], ne,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { b0.va, b1.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}


static void ggml_d3d11_flash_attn_ext(d3d11_device_ctx & dev, ggml_tensor * dst) {
    ggml_tensor * q     = dst->src[0];
    ggml_tensor * k     = dst->src[1];
    ggml_tensor * v     = dst->src[2];
    ggml_tensor * mask  = dst->src[3];
    ggml_tensor * sinks = dst->src[4];

    float       scale         = ggml_get_op_params_f32(dst, 0);
    const float max_bias      = ggml_get_op_params_f32(dst, 1);
    const float logit_softcap = ggml_get_op_params_f32(dst, 2);
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    std::vector<std::string> defines = {
        "DK=" + std::to_string(k->ne[0]), "DV=" + std::to_string(v->ne[0]),
        k->type == GGML_TYPE_F16 ? "K_F16" : k->type == GGML_TYPE_Q8_0 ? "K_Q8_0" : "K_F32",
        v->type == GGML_TYPE_F16 ? "V_F16" : v->type == GGML_TYPE_Q8_0 ? "V_Q8_0" : "V_F32",
    };
    if (mask) {
        defines.push_back("HAS_MASK");
    }
    if (sinks) {
        defines.push_back("HAS_SINKS");
    }
    if (logit_softcap != 0.0f) {
        defines.push_back("SOFTCAP");
    }
    const d3d11_binding bq = ggml_d3d11_bind_tensor(q);
    const d3d11_binding bk = ggml_d3d11_bind_tensor(k);
    const d3d11_binding bv = ggml_d3d11_bind_tensor(v);
    const d3d11_binding bm = mask ? ggml_d3d11_bind_tensor(mask) : d3d11_binding{ 0, 0 };
    const d3d11_binding bs = sinks ? ggml_d3d11_bind_tensor(sinks) : d3d11_binding{ 0, 0 };
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const size_t        tk = ggml_type_size(k->type);
    const size_t        tv = ggml_type_size(v->type);

    // f16 rows start on 4-byte boundaries when the element offset and every stride are even
    auto f16_aligned = [](const d3d11_binding & b, const ggml_tensor * t) {
        return b.elem_offset % 2 == 0 && (t->nb[1] / 2) % 2 == 0 && (t->nb[2] / 2) % 2 == 0 && (t->nb[3] / 2) % 2 == 0;
    };
    if (k->type == GGML_TYPE_F16 && f16_aligned(bk, k)) {
        defines.push_back("K_ALIGNED");
    }
    if (v->type == GGML_TYPE_F16 && f16_aligned(bv, v)) {
        defines.push_back("V_ALIGNED");
    }
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "flash_attn", hlsl_flash_attn, defines);

    const uint32_t n_head      = (uint32_t) q->ne[2];
    const float    n_head_log2 = (float) (1u << (uint32_t) floor(log2((double) n_head)));
    const float    m0          = powf(2.0f, -(max_bias) / n_head_log2);
    const float    m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    std::vector<uint32_t> params = {
        bq.elem_offset, bk.elem_offset, bv.elem_offset, mask ? bm.elem_offset : 0u, sinks ? bs.elem_offset : 0u, bd.elem_offset,
        (uint32_t) (q->nb[1] / 4), (uint32_t) (q->nb[2] / 4), (uint32_t) (q->nb[3] / 4),
        (uint32_t) (k->nb[1] / tk), (uint32_t) (k->nb[2] / tk), (uint32_t) (k->nb[3] / tk),
        (uint32_t) (v->nb[1] / tv), (uint32_t) (v->nb[2] / tv), (uint32_t) (v->nb[3] / tv),
        mask ? (uint32_t) (mask->nb[1] / 2) : 0u, mask ? (uint32_t) (mask->nb[2] / 2) : 0u, mask ? (uint32_t) (mask->nb[3] / 2) : 0u,
        mask ? (uint32_t) mask->ne[2] : 1u, mask ? (uint32_t) mask->ne[3] : 1u,
        (uint32_t) q->ne[1], n_head, (uint32_t) k->ne[1],
        (uint32_t) (q->ne[2] / k->ne[2]), (uint32_t) (q->ne[3] / k->ne[3]),
        (uint32_t) (q->ne[2] / v->ne[2]), (uint32_t) (q->ne[3] / v->ne[3]),
        ggml_d3d11_u32_from_f32(scale), ggml_d3d11_u32_from_f32(max_bias), ggml_d3d11_u32_from_f32(logit_softcap),
        ggml_d3d11_u32_from_f32(n_head_log2), ggml_d3d11_u32_from_f32(m0), ggml_d3d11_u32_from_f32(m1),
        D3D11_FLASH_ATTN_BLK, 0, 0, 0,   // blk_size, n_blocks, row0, n_rows
    };
    const size_t row0_idx = params.size() - 2;

    const uint32_t n_kv     = (uint32_t) k->ne[1];
    const uint32_t n_blocks = CEIL_DIV(n_kv, (uint32_t) D3D11_FLASH_ATTN_BLK);
    params[row0_idx - 1]    = n_blocks;

    const uint64_t n_rows    = (uint64_t) ggml_nrows(dst);   // dst is [DV, n_head, n_q, n_batch]
    const uint64_t row_work  = std::max<uint64_t>(1, (uint64_t) n_kv * (uint64_t) (k->ne[0] + v->ne[0]));
    const uint64_t row_bytes = (uint64_t) n_blocks * (uint64_t) (v->ne[0] + 1) * sizeof(float);
    const uint64_t tmp_rows  = std::max<uint64_t>(1, D3D11_FLASH_ATTN_TMP_MAX / row_bytes);
    const bool     big       = n_rows * row_work > dev.fa_work;

    const size_t tmp_need = (size_t) (std::min(tmp_rows, n_rows) * row_bytes);
    if (dev.fa_tmp_size < tmp_need) {
        // the old buffer may still be referenced by recorded dispatches: run them before releasing it
        if (dev.dispatches_in_list > 0) {
            ggml_d3d11_submit_and_wait(dev);
            ggml_d3d11_begin(dev, true);
        }
        size_t size = 1ull << 20;
        while (size < tmp_need) {
            size *= 2;
        }
        dev.fa_tmp      = ggml_d3d11_create_buffer(dev, size, D3D11_HEAP_TYPE_DEFAULT, L"ggml_d3d11_fa_tmp");
        dev.fa_tmp_size = size;
    }
    const D3D11_GPU_VIRTUAL_ADDRESS tmp_va = dev.fa_tmp->GetGPUVirtualAddress();

    std::vector<std::string> combine_defines = { "DV=" + std::to_string(v->ne[0]), "COMBINE" };
    if (sinks) {
        combine_defines.push_back("HAS_SINKS");
    }
    d3d11_pipeline & combine = ggml_d3d11_get_pipeline(dev, "flash_attn", hlsl_flash_attn, combine_defines);

    for (uint64_t row0 = 0, n = 0; row0 < n_rows; row0 += n) {
        n = std::min(n_rows - row0, std::max<uint64_t>(1, std::min(dev.fa_work / row_work, tmp_rows)));
        params[row0_idx]     = (uint32_t) row0;
        params[row0_idx + 1] = (uint32_t) n;
        if (big && dev.dispatches_in_list > 0) {
            // one bounded chunk per command list; the budget follows the submit time of the previous chunk
            // (target 15..60 ms, far below the Windows GPU timeout, short enough to keep the desktop responsive)
            dev.n_flush_batch++;
            const double t0 = ggml_d3d11_time_us();
            ggml_d3d11_submit_and_wait(dev);
            const double ms = (ggml_d3d11_time_us() - t0) / 1000.0;
            if (row0 > 0 && !dev.fa_work_fixed) {
                if (ms < 15.0) {
                    dev.fa_work = std::min<uint64_t>(dev.fa_work * 2, 1ull << 34);
                } else if (ms > 60.0) {
                    dev.fa_work = std::max<uint64_t>(dev.fa_work / 2, 1ull << 20);
                }
            }
            ggml_d3d11_begin(dev, true);
        }
        const std::vector<D3D11_GPU_VIRTUAL_ADDRESS> uavs = { bq.va, bk.va, bv.va, bm.va, bs.va, bd.va, tmp_va };
        ggml_d3d11_dispatch(dev, pipeline, params, uavs, CEIL_DIV((uint32_t) n * n_blocks, (uint32_t) D3D11_WG_SIZE));
        ggml_d3d11_dispatch(dev, combine, params, uavs, CEIL_DIV((uint32_t) n, (uint32_t) D3D11_WG_SIZE));
    }
}

static void ggml_d3d11_gated_delta_net(d3d11_device_ctx & dev, ggml_tensor * dst) {
    ggml_tensor * q = dst->src[0];
    ggml_tensor * k = dst->src[1];
    ggml_tensor * v = dst->src[2];
    ggml_tensor * g = dst->src[3];
    ggml_tensor * b = dst->src[4];
    ggml_tensor * s = dst->src[5];

    const uint32_t S_v      = (uint32_t) v->ne[0];
    const uint32_t H        = (uint32_t) v->ne[1];
    const uint32_t n_tokens = (uint32_t) v->ne[2];
    const uint32_t n_seqs   = (uint32_t) v->ne[3];
    const uint32_t K        = (uint32_t) ggml_get_op_params_i32(dst, 0);

    std::vector<std::string> defines = { "S_V=" + std::to_string(S_v) };
    if (g->ne[0] == S_v) {
        defines.push_back("KDA");
    }
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "gated_delta_net", hlsl_gated_delta_net, defines);

    const d3d11_binding bq = ggml_d3d11_bind_tensor(q);
    const d3d11_binding bk = ggml_d3d11_bind_tensor(k);
    const d3d11_binding bv = ggml_d3d11_bind_tensor(v);
    const d3d11_binding bg = ggml_d3d11_bind_tensor(g);
    const d3d11_binding bb = ggml_d3d11_bind_tensor(b);
    const d3d11_binding bs = ggml_d3d11_bind_tensor(s);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const uint32_t n_rows  = n_seqs * H * S_v;

    std::vector<uint32_t> params = {
        bq.elem_offset, bk.elem_offset, bv.elem_offset, bg.elem_offset, bb.elem_offset, bs.elem_offset, bd.elem_offset,
        (uint32_t) (q->nb[1] / 4), (uint32_t) (q->nb[2] / 4), (uint32_t) (q->nb[3] / 4),
        (uint32_t) (k->nb[1] / 4), (uint32_t) (k->nb[2] / 4), (uint32_t) (k->nb[3] / 4),
        (uint32_t) (v->nb[1] / 4), (uint32_t) (v->nb[2] / 4), (uint32_t) (v->nb[3] / 4),
        (uint32_t) (g->nb[1] / 4), (uint32_t) (g->nb[2] / 4), (uint32_t) (g->nb[3] / 4),
        (uint32_t) (b->nb[1] / 4), (uint32_t) (b->nb[2] / 4), (uint32_t) (b->nb[3] / 4),
        H, n_tokens, (uint32_t) q->ne[1], (uint32_t) k->ne[1],
        (uint32_t) (v->ne[3] / q->ne[3]), (uint32_t) (v->ne[3] / k->ne[3]),
        (uint32_t) (s->nb[3] / 4), K,
        0, 0,   // t0, t1
        n_rows, ggml_d3d11_u32_from_f32(1.0f / sqrtf((float) S_v)),
        bd.elem_offset + S_v * H * n_tokens * n_seqs, S_v * S_v * H * n_seqs,   // offset_st, st_slot_stride
    };
    const size_t t0_idx = 30;

    // with one snapshot the state carries over between token ranges, so long prompts are split to bound the
    // work per command list (same budget as flash attention)
    const uint64_t tok_work = std::max<uint64_t>(1, (uint64_t) n_rows * 3 * S_v);
    const uint32_t chunk    = K == 1 ? (uint32_t) std::max<uint64_t>(1, D3D11_FLASH_ATTN_WORK / tok_work) : n_tokens;
    const bool     big      = (uint64_t) n_tokens * tok_work > D3D11_FLASH_ATTN_WORK;
    for (uint32_t t0 = 0; t0 < n_tokens; t0 += chunk) {
        params[t0_idx]     = t0;
        params[t0_idx + 1] = std::min(n_tokens, t0 + chunk);
        if (big && dev.dispatches_in_list > 0) {
            dev.n_flush_batch++;
            ggml_d3d11_submit_and_wait(dev);
            ggml_d3d11_begin(dev, true);
        }
        ggml_d3d11_dispatch(dev, pipeline, params, { bq.va, bk.va, bv.va, bg.va, bb.va, bs.va, bd.va },
                            CEIL_DIV(n_rows, (uint32_t) D3D11_WG_SIZE));
    }
}

static void ggml_d3d11_rope(d3d11_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * src2,
                            ggml_tensor * dst, bool backward = false) {
    std::vector<std::string> defines;
    ggml_d3d11_float_type_define(dst->type, defines);
    if (src2) {
        defines.push_back("FF_FUNC");
    }
    if (backward) {
        defines.push_back("BACKWARD");
    }
    d3d11_pipeline & pipeline =
        ggml_d3d11_get_pipeline(dev, backward ? "rope_back" : "rope", hlsl_rope, defines);

    const d3d11_binding b0 = ggml_d3d11_bind_tensor(src0);
    const d3d11_binding b1 = ggml_d3d11_bind_tensor(src1);
    const d3d11_binding b2 = src2 ? ggml_d3d11_bind_tensor(src2) : d3d11_binding{ 0, 0 };
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const size_t        ts = ggml_type_size(src0->type);

    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    const int n_offs     = ((int32_t *) dst->op_params)[15];
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (int32_t *) dst->op_params + 5,  sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params + 6,  sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params + 7,  sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 8,  sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params + 9,  sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    int sections[4];
    memcpy(sections, (int32_t *) dst->op_params + 11, 4 * sizeof(int));
    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const uint32_t n_threads = (uint32_t) (ggml_nelements(src0) / 2);
    std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, src2 ? b2.elem_offset : 0u, bd.elem_offset,
        (uint32_t) (src0->nb[1] / ts), (uint32_t) (src0->nb[2] / ts), (uint32_t) (src0->nb[3] / ts),
        (uint32_t) (dst->nb[1] / ts), (uint32_t) (dst->nb[2] / ts), (uint32_t) (dst->nb[3] / ts),
        n_threads, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], (uint32_t) src0->ne[2],
        (uint32_t) n_dims, (uint32_t) mode,
        ggml_d3d11_u32_from_f32(theta_scale), ggml_d3d11_u32_from_f32(attn_factor),
        ggml_d3d11_u32_from_f32(freq_scale), ggml_d3d11_u32_from_f32(ext_factor),
        ggml_d3d11_u32_from_f32(corr_dims[0]), ggml_d3d11_u32_from_f32(corr_dims[1]),
        (uint32_t) sections[0], (uint32_t) sections[1], (uint32_t) sections[2], (uint32_t) sections[3],
        (uint32_t) n_offs,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { b0.va, b1.va, b2.va, bd.va }, CEIL_DIV(n_threads, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_glu(d3d11_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
    std::vector<std::string> defines;
    ggml_d3d11_float_type_define(dst->type, defines);
    defines.push_back(std::string("OP_") + ggml_glu_op_name(ggml_get_glu_op(dst)));
    if (!src1) {
        defines.push_back("NO_SPLIT");
    }
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "glu", hlsl_glu, defines);

    const d3d11_binding b0 = ggml_d3d11_bind_tensor(src0);
    const d3d11_binding b1 = src1 ? ggml_d3d11_bind_tensor(src1) : d3d11_binding{ 0, 0 };
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const size_t        ts = ggml_type_size(dst->type);
    const ggml_tensor * s1 = src1 ? src1 : src0;
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        b0.elem_offset, src1 ? b1.elem_offset : 0u, bd.elem_offset,
        (uint32_t) (src0->nb[1] / ts), (uint32_t) (src0->nb[2] / ts), (uint32_t) (src0->nb[3] / ts),
        (uint32_t) (s1->nb[1] / ts), (uint32_t) (s1->nb[2] / ts), (uint32_t) (s1->nb[3] / ts),
        (uint32_t) (dst->nb[1] / ts), (uint32_t) (dst->nb[2] / ts), (uint32_t) (dst->nb[3] / ts),
        ne, (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],
        (uint32_t) ((int32_t *) dst->op_params)[1],
        ggml_d3d11_u32_from_f32(ggml_get_op_params_f32(dst, 2)),
        ggml_d3d11_u32_from_f32(ggml_get_op_params_f32(dst, 3)),
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { b0.va, b1.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_unary(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<std::string> defines;
    ggml_d3d11_float_type_define(dst->type, defines);
    defines.push_back(dst->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(dst)) : ggml_op_name(dst->op));
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "unary", hlsl_unary, defines);

    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const size_t        ts = ggml_type_size(src->type);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const bool          clamp = dst->op == GGML_OP_CLAMP;

    std::vector<uint32_t> params = {
        ne, bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[0] / ts), (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts), (uint32_t) (src->nb[3] / ts),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2],
        clamp ? ggml_d3d11_u32_from_f32(ggml_get_op_params_f32(dst, 0)) : 0u,
        clamp ? ggml_d3d11_u32_from_f32(ggml_get_op_params_f32(dst, 1)) : 0u,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D11_WG_SIZE));
}

// GET_ROWS of a quantized source: the dequant paths of the matrix-vector kernel, TPR threads per row
static void ggml_d3d11_get_rows_quant(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * idx, ggml_tensor * dst) {
    const uint32_t tpr = std::min<uint32_t>(32, dev.mm_tpr_max);
    std::string    define = "SRC0_";
    define += ggml_type_name(src->type);
    for (auto & ch : define) {
        ch = (char) toupper((unsigned char) ch);
    }
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "get_rows_q", hlsl_get_rows_q,
                                                        { define, "TPR=" + std::to_string(tpr) });
    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bi = ggml_d3d11_bind_tensor(idx);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const size_t        ts = ggml_type_size(src->type);
    const size_t        td = ggml_type_size(dst->type);
    const uint32_t      n_rows = (uint32_t) ggml_nrows(dst);

    const std::vector<uint32_t> params = {
        bs.elem_offset, bi.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts), (uint32_t) (src->nb[3] / ts),
        (uint32_t) (idx->nb[0] / 4), (uint32_t) (idx->nb[1] / 4), (uint32_t) (idx->nb[2] / 4),
        (uint32_t) (dst->nb[1] / td), (uint32_t) (dst->nb[2] / td), (uint32_t) (dst->nb[3] / td),
        (uint32_t) dst->ne[0], (uint32_t) idx->ne[0], (uint32_t) idx->ne[1], n_rows,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bi.va, bd.va }, CEIL_DIV(n_rows * tpr, (uint32_t) D3D11_WG_SIZE));
}

static void ggml_d3d11_get_rows(d3d11_device_ctx & dev, ggml_tensor * src, ggml_tensor * idx, ggml_tensor * dst) {
    if (ggml_is_quantized(src->type)) {
        ggml_d3d11_get_rows_quant(dev, src, idx, dst);
        return;
    }
    std::string define = "SRC_";
    define += ggml_type_name(src->type);
    for (auto & ch : define) {
        ch = (char) toupper((unsigned char) ch);
    }
    d3d11_pipeline & pipeline = ggml_d3d11_get_pipeline(dev, "get_rows", hlsl_get_rows, { define });

    const d3d11_binding bs = ggml_d3d11_bind_tensor(src);
    const d3d11_binding bi = ggml_d3d11_bind_tensor(idx);
    const d3d11_binding bd = ggml_d3d11_bind_tensor(dst);
    const size_t        ts = ggml_type_size(src->type);
    const size_t        td = ggml_type_size(dst->type);
    const bool          quant   = ggml_is_quantized(src->type);
    const uint32_t      n_units = (uint32_t) (quant ? ggml_nrows(dst) * (dst->ne[0] / 32) : ggml_nelements(dst));

    std::vector<uint32_t> params = {
        bs.elem_offset, bi.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts), (uint32_t) (src->nb[3] / ts),
        (uint32_t) (idx->nb[0] / 4), (uint32_t) (idx->nb[1] / 4), (uint32_t) (idx->nb[2] / 4),
        (uint32_t) (dst->nb[1] / td), (uint32_t) (dst->nb[2] / td), (uint32_t) (dst->nb[3] / td),
        (uint32_t) dst->ne[0], (uint32_t) idx->ne[0], (uint32_t) idx->ne[1], n_units,
    };
    ggml_d3d11_dispatch(dev, pipeline, params, { bs.va, bi.va, bd.va }, CEIL_DIV(n_units, (uint32_t) D3D11_WG_SIZE));
}

static bool ggml_d3d11_unary_supported(ggml_unary_op op) {
    switch (op) {
        case GGML_UNARY_OP_ABS: case GGML_UNARY_OP_SGN: case GGML_UNARY_OP_NEG: case GGML_UNARY_OP_STEP:
        case GGML_UNARY_OP_TANH: case GGML_UNARY_OP_ELU: case GGML_UNARY_OP_RELU: case GGML_UNARY_OP_SIGMOID:
        case GGML_UNARY_OP_GELU: case GGML_UNARY_OP_GELU_QUICK: case GGML_UNARY_OP_GELU_ERF: case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_HARDSWISH: case GGML_UNARY_OP_HARDSIGMOID: case GGML_UNARY_OP_EXP: case GGML_UNARY_OP_SOFTPLUS:
        case GGML_UNARY_OP_EXPM1: case GGML_UNARY_OP_FLOOR: case GGML_UNARY_OP_CEIL: case GGML_UNARY_OP_ROUND:
        case GGML_UNARY_OP_TRUNC:
            return true;
        default:
            return false;
    }
}

static void ggml_d3d11_encode_node(d3d11_device_ctx & dev, ggml_tensor * node);

// encodes the node at index i, fused with following nodes when possible; returns the number of nodes consumed
static int ggml_d3d11_encode_nodes_inner(d3d11_device_ctx & dev, const ggml_cgraph * cgraph, int i);

static int ggml_d3d11_encode_nodes(d3d11_device_ctx & dev, const ggml_cgraph * cgraph, int i) {
    for (int k = 0; k < 8; k++) {
        dev.out_nodes[k] = i + k < cgraph->n_nodes ? cgraph->nodes[i + k] : nullptr;
    }
    const int n = ggml_d3d11_encode_nodes_inner(dev, cgraph, i);
    std::fill(std::begin(dev.out_nodes), std::end(dev.out_nodes), nullptr);
    return n;
}

static int ggml_d3d11_encode_nodes_inner(d3d11_device_ctx & dev, const ggml_cgraph * cgraph, int i) {
    ggml_tensor * node = cgraph->nodes[i];
    if (!ggml_is_empty(node)) {
        if (node->op == GGML_OP_MUL_MAT) {
            return ggml_d3d11_encode_mul_mat_group(dev, cgraph, i);
        }
        if (node->op == GGML_OP_RMS_NORM) {
            return ggml_d3d11_encode_rms_norm(dev, cgraph, i);
        }
    }
    ggml_d3d11_encode_node(dev, node);
    return 1;
}

static void ggml_d3d11_encode_node(d3d11_device_ctx & dev, ggml_tensor * node) {
    if (ggml_is_empty(node)) {
        return;
    }
    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_RESHAPE:
            return;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
            ggml_d3d11_cpy(dev, node->src[0], node);
            return;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            ggml_d3d11_binary_op(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_SCALE:
            ggml_d3d11_scale(dev, node->src[0], node);
            return;
        case GGML_OP_SET_ROWS:
            ggml_d3d11_set_rows(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_MUL_MAT:
            ggml_d3d11_mul_mat(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_MUL_MAT_ID:
            ggml_d3d11_mul_mat_id(dev, node->src[0], node->src[1], node->src[2], node);
            return;
        case GGML_OP_RMS_NORM:
            ggml_d3d11_rms_norm(dev, node->src[0], node, node, nullptr);
            return;
        case GGML_OP_SOFT_MAX:
            ggml_d3d11_soft_max(dev, node->src[0], node->src[1], node->src[2], node);
            return;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_d3d11_flash_attn_ext(dev, node);
            return;
        case GGML_OP_CONCAT:
            ggml_d3d11_concat(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_NORM:
            ggml_d3d11_norm(dev, node->src[0], node);
            return;
        case GGML_OP_L2_NORM:
            ggml_d3d11_rms_norm(dev, node->src[0], node, node, nullptr);
            return;
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            ggml_d3d11_argsort(dev, node->src[0], node);
            return;
        case GGML_OP_REPEAT:
            ggml_d3d11_repeat(dev, node->src[0], node);
            return;
        case GGML_OP_IM2COL:
            ggml_d3d11_im2col(dev, node);
            return;
        case GGML_OP_UPSCALE:
            ggml_d3d11_upscale(dev, node->src[0], node);
            return;
        case GGML_OP_POOL_2D:
            ggml_d3d11_pool_2d(dev, node->src[0], node);
            return;
        case GGML_OP_FILL:
            ggml_d3d11_fill(dev, node);
            return;
        case GGML_OP_SUM_ROWS:
            ggml_d3d11_sum_rows(dev, node->src[0], node);
            return;
        case GGML_OP_MEAN:
            ggml_d3d11_sum_rows(dev, node->src[0], node, true);
            return;
        case GGML_OP_SUM:
            ggml_d3d11_sum(dev, node->src[0], node);
            return;
        case GGML_OP_ARGMAX:
            ggml_d3d11_argmax(dev, node->src[0], node);
            return;
        case GGML_OP_ARANGE:
            ggml_d3d11_arange(dev, node);
            return;
        case GGML_OP_DIAG_MASK_INF:
            ggml_d3d11_diag_mask(dev, node->src[0], node, -INFINITY);
            return;
        case GGML_OP_DIAG_MASK_ZERO:
            ggml_d3d11_diag_mask(dev, node->src[0], node, 0.0f);
            return;
        case GGML_OP_ROLL:
            ggml_d3d11_roll(dev, node->src[0], node);
            return;
        case GGML_OP_PAD:
            ggml_d3d11_pad(dev, node->src[0], node);
            return;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_d3d11_pad_reflect_1d(dev, node->src[0], node);
            return;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_d3d11_timestep_embedding(dev, node->src[0], node);
            return;
        case GGML_OP_SET:
            ggml_d3d11_set_acc(dev, node->src[0], node->src[1], node, false);
            return;
        case GGML_OP_ACC:
            ggml_d3d11_set_acc(dev, node->src[0], node->src[1], node, true);
            return;
        case GGML_OP_CUMSUM:
            ggml_d3d11_cumsum(dev, node->src[0], node);
            return;
        case GGML_OP_TRI:
            ggml_d3d11_tri(dev, node->src[0], node);
            return;
        case GGML_OP_COUNT_EQUAL:
            ggml_d3d11_count_equal(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_ADD1:
            ggml_d3d11_add1(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_LEAKY_RELU:
            ggml_d3d11_leaky_relu(dev, node->src[0], node);
            return;
        case GGML_OP_GROUP_NORM:
            ggml_d3d11_group_norm(dev, node->src[0], node);
            return;
        case GGML_OP_DIAG:
            ggml_d3d11_diag(dev, node->src[0], node);
            return;
        case GGML_OP_POOL_1D:
            ggml_d3d11_pool_1d(dev, node->src[0], node);
            return;
        case GGML_OP_WIN_PART:
            ggml_d3d11_win_part(dev, node->src[0], node, false);
            return;
        case GGML_OP_WIN_UNPART:
            ggml_d3d11_win_part(dev, node->src[0], node, true);
            return;
        case GGML_OP_GET_REL_POS:
            ggml_d3d11_get_rel_pos(dev, node->src[0], node);
            return;
        case GGML_OP_ADD_REL_POS:
            ggml_d3d11_add_rel_pos(dev, node->src[0], node->src[1], node->src[2], node);
            return;
        case GGML_OP_SOLVE_TRI:
            ggml_d3d11_solve_tri(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_OUT_PROD:
            ggml_d3d11_out_prod(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_ADD_ID:
            ggml_d3d11_add_id(dev, node->src[0], node->src[1], node->src[2], node);
            return;
        case GGML_OP_GATED_DELTA_NET:
            ggml_d3d11_gated_delta_net(dev, node);
            return;
        case GGML_OP_SSM_CONV:
            ggml_d3d11_ssm_conv(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_SSM_SCAN:
            ggml_d3d11_ssm_scan(dev, node);
            return;
        case GGML_OP_CONV_2D_DW:
            ggml_d3d11_conv_2d_dw(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_d3d11_conv_transpose_1d(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_COL2IM_1D:
            ggml_d3d11_col2im_1d(dev, node->src[0], node);
            return;
        case GGML_OP_IM2COL_3D:
            ggml_d3d11_im2col_3d(dev, node);
            return;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_d3d11_conv_transpose_2d(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_CONV_2D:
            ggml_d3d11_conv_2d(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_CONV_3D:
            ggml_d3d11_conv_3d(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_d3d11_lightning_indexer(dev, node);
            return;
        case GGML_OP_DSV4_HC_PRE:
            ggml_d3d11_dsv4_hc_pre(dev, node);
            return;
        case GGML_OP_DSV4_HC_POST:
            ggml_d3d11_dsv4_hc_post(dev, node);
            return;
        case GGML_OP_DSV4_HC_COMB:
            ggml_d3d11_dsv4_hc_comb(dev, node);
            return;
        case GGML_OP_RWKV_WKV6:
            ggml_d3d11_rwkv_wkv6(dev, node);
            return;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_d3d11_gated_linear_attn(dev, node);
            return;
        case GGML_OP_RWKV_WKV7:
            ggml_d3d11_rwkv_wkv7(dev, node);
            return;
        case GGML_OP_ROPE:
            ggml_d3d11_rope(dev, node->src[0], node->src[1], node->src[2], node);
            return;
        case GGML_OP_ROPE_BACK:
            ggml_d3d11_rope(dev, node->src[0], node->src[1], node->src[2], node, true);
            return;
        case GGML_OP_OPT_STEP_SGD:
            ggml_d3d11_opt_step(dev, node, false);
            return;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_d3d11_opt_step(dev, node, true);
            return;
        case GGML_OP_SILU_BACK:
            ggml_d3d11_silu_back(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_REPEAT_BACK:
            ggml_d3d11_repeat_back(dev, node->src[0], node);
            return;
        case GGML_OP_RMS_NORM_BACK:
            ggml_d3d11_rms_norm_back(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_d3d11_soft_max_back(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_d3d11_cross_entropy_loss(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_d3d11_cross_entropy_loss_back(dev, node);
            return;
        case GGML_OP_IM2COL_BACK:
            ggml_d3d11_im2col_back(dev, node);
            return;
        case GGML_OP_GET_ROWS_BACK:
            ggml_d3d11_get_rows_back(dev, node);
            return;
        case GGML_OP_GLU:
            ggml_d3d11_glu(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_UNARY:
        case GGML_OP_CLAMP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
            ggml_d3d11_unary(dev, node->src[0], node);
            return;
        case GGML_OP_GET_ROWS:
            ggml_d3d11_get_rows(dev, node->src[0], node->src[1], node);
            return;
        default:
            GGML_ABORT("ggml_d3d11: unsupported op %s", ggml_op_name(node->op));
    }
}

/* GGML Backend Interface */

static const char * ggml_backend_d3d11_name(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_d3d11_context *) backend->context;
    return ctx->name.c_str();
}

static void ggml_d3d11_print_stats(d3d11_device_ctx & dev) {
    fprintf(stderr, "ggml_d3d11 stats [%s]: graphs %llu, nodes %llu, dispatches %llu, submits %llu (mid-graph flushes: "
                    "batch %llu, arena %llu) | graph_compute %.1f ms = encode %.1f (shader compile %.1f in %llu pipelines) + submit calls %.1f + fence wait %.1f | "
                    "set_tensor %llu calls %.1f MB %.1f ms | get_tensor %llu calls %.1f MB %.1f ms\n",
            dev.name.c_str(), (unsigned long long) dev.n_graphs, (unsigned long long) dev.n_nodes,
            (unsigned long long) dev.n_dispatches, (unsigned long long) dev.n_submits,
            (unsigned long long) dev.n_flush_batch, (unsigned long long) dev.n_flush_arena,
            dev.t_graph_us / 1000.0, (dev.t_graph_us - dev.t_graph_submit_us - dev.t_graph_wait_us) / 1000.0,
            dev.t_compile_us / 1000.0, (unsigned long long) dev.n_compiles,
            dev.t_graph_submit_us / 1000.0, dev.t_graph_wait_us / 1000.0,
            (unsigned long long) dev.n_set_tensor, dev.bytes_set / 1e6, dev.t_set_us / 1000.0,
            (unsigned long long) dev.n_get_tensor, dev.bytes_get / 1e6, dev.t_get_us / 1000.0);
    for (const auto & r : dev.rejected) {
        fprintf(stderr, "ggml_d3d11 rejected [%s]: %6llu x %s\n", dev.name.c_str(), (unsigned long long) r.second, r.first.c_str());
    }
    if (dev.profile && !dev.prof.empty()) {
        std::vector<std::pair<std::string, std::pair<double, uint64_t>>> rows(dev.prof.begin(), dev.prof.end());
        std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) { return a.second.first > b.second.first; });
        double total = 0;
        for (const auto & r : rows) {
            total += r.second.first;
        }
        fprintf(stderr, "ggml_d3d11 gpu time by pipeline (total %.1f ms over %llu graphs):\n", total / 1000.0,
                (unsigned long long) dev.n_graphs);
        for (const auto & r : rows) {
            fprintf(stderr, "  %9.1f ms %5.1f%% %8llu x %7.1f us  %s\n", r.second.first / 1000.0,
                    100.0 * r.second.first / total, (unsigned long long) r.second.second,
                    r.second.first / (double) r.second.second, r.first.c_str());
        }
    }
    fflush(stderr);
}

static void ggml_backend_d3d11_free(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_d3d11_context *) backend->context;
    delete ctx;
    delete backend;
}

static ggml_status ggml_backend_d3d11_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto *             ctx = (ggml_backend_d3d11_context *) backend->context;
    d3d11_device_ctx & dev = *ctx->dev;
    D3D11_LOG_DEBUG("graph_compute(%d nodes)\n", cgraph->n_nodes);

    std::lock_guard<std::recursive_mutex> lock(dev.mutex);
    const double t0 = ggml_d3d11_time_us();
    const double submit0 = dev.t_submit_us, wait0 = dev.t_wait_us;
    ggml_d3d11_begin(dev, true);
    if (dev.n_graphs < 4 || cgraph->n_nodes != dev.last_graph_nodes) {
        // a graph of a new shape: find the pipelines it needs first and compile the missing ones in parallel
        dev.collecting = true;
        for (int i = 0; i < cgraph->n_nodes;) {
            i += ggml_d3d11_encode_nodes(dev, cgraph, i);
        }
        dev.collecting = false;
        ggml_d3d11_build_pipeline_jobs(dev);
    }
    dev.last_graph_nodes = cgraph->n_nodes;
    for (int i = 0; i < cgraph->n_nodes;) {
        i += ggml_d3d11_encode_nodes(dev, cgraph, i);
    }
    ggml_d3d11_submit_and_wait(dev);
    dev.n_graphs++;
    dev.n_nodes += cgraph->n_nodes;
    dev.t_graph_us += ggml_d3d11_time_us() - t0;
    dev.t_graph_submit_us += dev.t_submit_us - submit0;
    dev.t_graph_wait_us += dev.t_wait_us - wait0;
    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i ggml_backend_d3d11_i = {
    /* .get_name                = */ ggml_backend_d3d11_name,
    /* .free                    = */ ggml_backend_d3d11_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_d3d11_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_d3d11_guid(void) {
    static ggml_guid guid = { 0xd3, 0xd1, 0x2b, 0xac, 0x4e, 0x6d, 0x47, 0x1a,
                              0x9c, 0x0f, 0x8a, 0x21, 0x5b, 0x77, 0xe0, 0x39 };
    return &guid;
}

/* GGML Backend Buffer Interface */

static void ggml_backend_d3d11_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_d3d11_buffer_context *) buffer->context;
    delete ctx;
}

static void * ggml_backend_d3d11_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return d3d11_ptr_base;
}

static void ggml_backend_d3d11_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_d3d11_buffer_context *) buffer->context;
    ggml_d3d11_buffer_memset(*ctx->dev, ctx->va, ggml_d3d11_tensor_offset(tensor) + offset, size, value);
}

static void ggml_backend_d3d11_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto *             ctx = (ggml_backend_d3d11_buffer_context *) buffer->context;
    d3d11_device_ctx & dev = *ctx->dev;
    if (size == 0) {
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(dev.mutex);
    const double t0         = ggml_d3d11_time_us();
    const size_t dst_offset = ggml_d3d11_tensor_offset(tensor) + offset;
    size_t       done       = 0;
    while (done < size) {
        const size_t n   = std::min(size - done, (size_t) D3D11_STAGING_SIZE);
        D3D11_BOX    box = { (UINT) (dst_offset + done), 0, 0, (UINT) (dst_offset + done + n), 1, 1 };
        dev.ctx->UpdateSubresource(ctx->res->buf.get(), 0, &box, (const char *) data + done, 0, 0);
        done += n;
    }
    dev.n_set_tensor++;
    dev.bytes_set += size;
    dev.t_set_us += ggml_d3d11_time_us() - t0;
}

static void ggml_backend_d3d11_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto *             ctx = (ggml_backend_d3d11_buffer_context *) buffer->context;
    d3d11_device_ctx & dev = *ctx->dev;
    if (size == 0) {
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(dev.mutex);

    if (!dev.readback_buf) {
        D3D11_BUFFER_DESC desc = {};
        desc.ByteWidth         = (UINT) D3D11_STAGING_SIZE;
        desc.Usage             = D3D11_USAGE_STAGING;
        desc.CPUAccessFlags    = D3D11_CPU_ACCESS_READ;
        ggml_d3d11_check(dev.device->CreateBuffer(&desc, nullptr, dev.readback_buf.put()), "CreateBuffer (readback)");
    }

    const double t0         = ggml_d3d11_time_us();
    const size_t src_offset = ggml_d3d11_tensor_offset(tensor) + offset;
    size_t       done       = 0;
    while (done < size) {
        const size_t n   = std::min(size - done, (size_t) D3D11_STAGING_SIZE);
        D3D11_BOX    box = { (UINT) (src_offset + done), 0, 0, (UINT) (src_offset + done + n), 1, 1 };
        dev.ctx->CopySubresourceRegion(dev.readback_buf.get(), 0, 0, 0, 0, ctx->res->buf.get(), 0, &box);
        D3D11_MAPPED_SUBRESOURCE m = {};
        ggml_d3d11_check(dev.ctx->Map(dev.readback_buf.get(), 0, D3D11_MAP_READ, 0, &m), "Map (readback)");
        memcpy((char *) data + done, m.pData, n);
        dev.ctx->Unmap(dev.readback_buf.get(), 0);
        ggml_d3d11_print_debug(dev);
        done += n;
    }
    dev.n_get_tensor++;
    dev.bytes_get += size;
    dev.t_get_us += ggml_d3d11_time_us() - t0;
}
static void ggml_backend_d3d11_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_backend_d3d11_buffer_context *) buffer->context;
    ggml_d3d11_buffer_memset(*ctx->dev, ctx->va, 0, ctx->size, value);
}

static ggml_backend_buffer_i ggml_backend_d3d11_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_d3d11_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_d3d11_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ ggml_backend_d3d11_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_d3d11_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_d3d11_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_d3d11_buffer_clear,
    /* .reset           = */ NULL,
};

/* GGML Backend Buffer Type Interface */

static const char * ggml_backend_d3d11_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    auto * dev = (d3d11_device_ctx *) buft->context;
    return dev->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_d3d11_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * dev = (d3d11_device_ctx *) buft->context;
    std::lock_guard<std::recursive_mutex> lock(dev->mutex);

    // one binding alignment of slack past the end: quant blocks that end mid-word are read as whole words
    const size_t alloc_size = std::max((size_t) D3D11_BINDING_ALIGNMENT,
                                       (size + D3D11_BINDING_ALIGNMENT - 1) & ~((size_t) D3D11_BINDING_ALIGNMENT - 1))
                              + D3D11_BINDING_ALIGNMENT;
    com_ptr<d3d11_res> res = ggml_d3d11_create_buffer(*dev, alloc_size, D3D11_HEAP_TYPE_DEFAULT, nullptr);
    if (!res) {
        return nullptr;
    }
    auto * ctx = new ggml_backend_d3d11_buffer_context();
    ctx->res  = res;
    ctx->va   = res->GetGPUVirtualAddress();
    ctx->size = alloc_size;
    ctx->dev  = ggml_d3d11_shared_dev(dev);
    return ggml_backend_buffer_init(buft, ggml_backend_d3d11_buffer_interface, ctx, size);
}
static size_t ggml_backend_d3d11_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return D3D11_BINDING_ALIGNMENT;
}

static size_t ggml_backend_d3d11_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * dev = (d3d11_device_ctx *) buft->context;
    return dev->max_alloc;
}

/* GGML Backend Device Interface */

static const char * ggml_backend_d3d11_device_get_name(ggml_backend_dev_t dev) {
    auto * ctx = (d3d11_device_ctx *) dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_d3d11_device_get_description(ggml_backend_dev_t dev) {
    auto * ctx = (d3d11_device_ctx *) dev->context;
    return ctx->desc.c_str();
}

static void ggml_backend_d3d11_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * ctx = (d3d11_device_ctx *) dev->context;
    *total     = ctx->caps.uma ? ctx->shared_mem : ctx->dedicated_mem;
    *free      = *total;
    com_ptr<IDXGIAdapter3> adapter3;
    if (SUCCEEDED(ctx->adapter->QueryInterface(IID_PPV_ARGS(adapter3.put())))) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
        // LOCAL is the segment that carries the budget on both kinds of adapter: dedicated VRAM on a
        // discrete card, and the share of system memory the driver grants on a UMA one. NON_LOCAL is
        // the system-memory fallback of a discrete card and reads back as a zero budget on UMA, which
        // made an integrated GPU report no free memory at all.
        const DXGI_MEMORY_SEGMENT_GROUP group = DXGI_MEMORY_SEGMENT_GROUP_LOCAL;
        if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, group, &info))) {
            *free = info.Budget > info.CurrentUsage ? (size_t) (info.Budget - info.CurrentUsage) : 0;
            if (info.Budget > 0) {
                *total = (size_t) info.Budget;
            }
        }
    }
}

static enum ggml_backend_dev_type ggml_backend_d3d11_device_get_type(ggml_backend_dev_t dev) {
    auto * ctx = (d3d11_device_ctx *) dev->context;
    return ctx->caps.uma ? GGML_BACKEND_DEVICE_TYPE_IGPU : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_d3d11_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_d3d11_device_get_name(dev);
    props->description = ggml_backend_d3d11_device_get_description(dev);
    props->type        = ggml_backend_d3d11_device_get_type(dev);
    ggml_backend_d3d11_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_d3d11_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    auto * dev_ctx = (d3d11_device_ctx *) dev->context;

    auto * ctx = new ggml_backend_d3d11_context();
    ctx->dev   = ggml_d3d11_shared_dev(dev_ctx);
    ctx->name  = dev_ctx->name;

    auto * backend = new ggml_backend();
    *backend       = {
        /* .guid      = */ ggml_backend_d3d11_guid(),
        /* .interface = */ ggml_backend_d3d11_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };
    return backend;
}

static ggml_backend_buffer_type_t ggml_backend_d3d11_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto * ctx = (d3d11_device_ctx *) dev->context;
    return &ctx->buft;
}

static bool ggml_backend_d3d11_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_d3d11_buffer_type_get_name && buft->device == dev;
}

static bool ggml_d3d11_supports_op(d3d11_device_ctx * ctx, const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    if (!ctx->disable_ops.empty() && op->op != GGML_OP_NONE && op->op != GGML_OP_VIEW &&
        op->op != GGML_OP_RESHAPE && op->op != GGML_OP_PERMUTE && op->op != GGML_OP_TRANSPOSE) {
        const std::string name = std::string(",") + ggml_op_name(op->op) + ",";
        // ALL keeps SET_ROWS, otherwise the KV cache cannot be allocated on this device at all
        const bool all = ctx->disable_ops == ",ALL," && op->op != GGML_OP_SET_ROWS;
        if (all || ctx->disable_ops.find(name) != std::string::npos) {
            return false;
        }
    }

    auto type_ok = [&](ggml_type t) {
        return t == GGML_TYPE_F32 || t == GGML_TYPE_I32 || (t == GGML_TYPE_F16 && ctx->caps.native_16bit);
    };

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_RESHAPE:
            return true;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
            return type_ok(op->type) && type_ok(src0->type);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return (op->type == GGML_TYPE_F32 || (op->type == GGML_TYPE_F16 && ctx->caps.native_16bit)) &&
                   src0->type == op->type && src1->type == op->type;
        case GGML_OP_SCALE:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
        case GGML_OP_SET_ROWS:
            if (op->type == GGML_TYPE_Q8_0) {
                // rows are quantized in place: contiguous dst, contiguous source rows
                return src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32) &&
                       ggml_is_contiguous(op) && src0->nb[0] == sizeof(float) && op->ne[0] % 32 == 0 &&
                       ggml_nbytes(op) < (1ull << 31);
            }
            return (op->type == GGML_TYPE_F32 || (op->type == GGML_TYPE_F16 && ctx->caps.native_16bit)) &&
                   src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);
        case GGML_OP_MUL_MAT:
            {
                // mat-vec kernel, any column count in chunks of 4; contiguous rows required
                const bool quant = ggml_is_quantized(src0->type);
                return ggml_d3d11_mul_mat_vec_type(src0->type) &&
                       (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) && op->type == GGML_TYPE_F32 &&
                       src0->nb[0] == ggml_type_size(src0->type) && src1->nb[0] == ggml_type_size(src1->type) &&
                       (quant ? src0->ne[0] % 256 == 0 || (src0->ne[0] % 32 == 0 && ggml_blck_size(src0->type) == 32)
                              : src0->ne[0] % 4 == 0);
            }
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            // the rank kernel is quadratic in the row length: long rows (e.g. vocab sorts) stay on the CPU
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_I32 && ggml_is_contiguous(op) &&
                   ggml_is_contiguous_rows(src0) && src0->ne[0] <= 1024;
        case GGML_OP_REPEAT:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_IM2COL:
            return src1->type == GGML_TYPE_F32 && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                   ggml_is_contiguous(op) && ggml_nelements(op) <= UINT32_MAX && ggml_nelements(src1) <= UINT32_MAX;
        case GGML_OP_IM2COL_3D:
            {
                // IC comes from op_params, not from a shape, so it has to be sane before N is derived from it
                const int32_t ic = ggml_get_op_params_i32(op, 9);
                return src1->type == GGML_TYPE_F32 && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                       src1->nb[0] == sizeof(float) && ggml_is_contiguous(op) &&
                       ic > 0 && src1->ne[3] % ic == 0 &&
                       ggml_nelements(op) <= UINT32_MAX && ggml_nelements(src1) <= UINT32_MAX;
            }
        case GGML_OP_POOL_2D: {
            const ggml_op_pool pool = (ggml_op_pool) ggml_get_op_params_i32(op, 0);
            return op->type == GGML_TYPE_F32 && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                   (pool == GGML_OP_POOL_AVG || pool == GGML_OP_POOL_MAX) && ggml_is_contiguous(op) &&
                   src0->nb[0] == ggml_type_size(src0->type) && ggml_nelements(op) <= UINT32_MAX;
        }
        case GGML_OP_UPSCALE: {
            const int32_t mode_flags = ggml_get_op_params_i32(op, 0);
            const int32_t mode       = mode_flags & 0xFF;
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && !(mode_flags & GGML_SCALE_FLAG_ANTIALIAS) &&
                   (mode == GGML_SCALE_MODE_NEAREST || mode == GGML_SCALE_MODE_BILINEAR) &&
                   ggml_nelements(op) <= UINT32_MAX && ggml_nelements(src0) <= UINT32_MAX;
        }
        case GGML_OP_FILL:
            return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && ggml_is_contiguous(op) &&
                   ggml_nbytes(op) < (1ull << 30);
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(src0);
        case GGML_OP_SUM:
            // one scalar out; the kernel accumulates in f32 where the CPU uses f64
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_scalar(op) &&
                   ggml_is_contiguous_rows(src0) && ggml_nelements(src0) <= UINT32_MAX;
        case GGML_OP_ARGMAX:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_I32 &&
                   ggml_is_contiguous_rows(src0) && ggml_is_contiguous(op);
        case GGML_OP_ARANGE:
            return op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_DIAG_MASK_ZERO:
            // the flat index assumes both sides are contiguous, which is what the CPU asserts too
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(src0) && ggml_is_contiguous(op) &&
                   ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_ROLL:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) &&
                   ggml_are_same_shape(src0, op);
        case GGML_OP_PAD:
            // dst is addressed by a flat index, as the CPU does; src keeps its strides
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) &&
                   ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_PAD_REFLECT_1D:
            // reflection may not read past the far edge of the source row
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) &&
                   ggml_get_op_params_i32(op, 0) < src0->ne[0] && ggml_get_op_params_i32(op, 1) < src0->ne[0];
        case GGML_OP_TIMESTEP_EMBEDDING:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && op->nb[0] == sizeof(float);
        case GGML_OP_SOLVE_TRI:
            // A, B and X are indexed with n and k directly, so their rows must be packed
            return src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(op);
        case GGML_OP_OUT_PROD:
            return src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) &&
                   ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_WIN_PART:
        case GGML_OP_WIN_UNPART:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(src0) && ggml_is_contiguous(op) &&
                   ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_GET_REL_POS:
            return src0->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F16 && ctx->caps.native_16bit &&
                   ggml_is_contiguous(src0) && ggml_is_contiguous(op) &&
                   ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_ADD_REL_POS:
            return src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                   op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
                   ggml_is_contiguous(op->src[2]) && ggml_is_contiguous(op) &&
                   ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_GROUP_NORM:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) &&
                   ggml_are_same_shape(src0, op);
        case GGML_OP_DIAG:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && op->nb[0] == sizeof(float);
        case GGML_OP_POOL_1D: {
            const ggml_op_pool pool = (ggml_op_pool) ggml_get_op_params_i32(op, 0);
            return op->type == GGML_TYPE_F32 && (src0->type == GGML_TYPE_F32 ||
                   (src0->type == GGML_TYPE_F16 && ctx->caps.native_16bit)) &&
                   (pool == GGML_OP_POOL_AVG || pool == GGML_OP_POOL_MAX) &&
                   src0->nb[0] == ggml_type_size(src0->type) && op->nb[0] == sizeof(float);
        }
        case GGML_OP_CUMSUM:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && op->nb[0] == sizeof(float);
        case GGML_OP_TRI:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && op->nb[0] == sizeof(float);
        case GGML_OP_COUNT_EQUAL:
            // the count goes into the low word of the i64 result, so it must fit in 32 bits
            return src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 &&
                   op->type == GGML_TYPE_I64 && ggml_is_scalar(op) &&
                   ggml_are_same_shape(src0, src1) && ggml_is_contiguous_rows(src0) &&
                   ggml_is_contiguous_rows(src1) && ggml_nelements(src0) <= UINT32_MAX;
        case GGML_OP_ADD1:
            return src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 && ggml_is_scalar(src1) &&
                   src0->nb[0] == sizeof(float) && op->nb[0] == sizeof(float);
        case GGML_OP_LEAKY_RELU:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && op->nb[0] == sizeof(float);
        case GGML_OP_SET:
        case GGML_OP_ACC:
            // the CPU asserts the same: src0 and dst are contiguous and the same shape
            return src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(src0) && ggml_is_contiguous(op) && ggml_are_same_shape(src0, op) &&
                   src1->nb[0] == sizeof(float);
        case GGML_OP_ADD_ID:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                   op->src[2]->type == GGML_TYPE_I32 && ggml_is_contiguous(src1) &&
                   ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_MUL_MAT_ID:
            {
                const ggml_tensor * ids = op->src[2];
                const bool quant = ggml_is_quantized(src0->type);
                return ggml_d3d11_mul_mat_vec_type(src0->type) && src1->type == GGML_TYPE_F32 &&
                       op->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 &&
                       src0->nb[0] == ggml_type_size(src0->type) && src1->nb[0] == sizeof(float) &&
                       (quant ? src0->ne[0] % 256 == 0 || (src0->ne[0] % 32 == 0 && ggml_blck_size(src0->type) == 32)
                              : src0->ne[0] % 4 == 0);
            }
        case GGML_OP_RMS_NORM:
        case GGML_OP_NORM:
        case GGML_OP_L2_NORM:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(src0);
        case GGML_OP_SOFT_MAX:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 &&
                   (!src1 || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
                   (!op->src[2] || op->src[2]->type == GGML_TYPE_F32);
        case GGML_OP_GATED_DELTA_NET:
            {
                // f32 everywhere, rows of S_v (a multiple of 4, <= 512) elements, q/k head size equal to S_v
                const ggml_tensor * v = op->src[2];
                bool ok = op->type == GGML_TYPE_F32 && v->ne[0] % 4 == 0 && v->ne[0] <= 512 &&
                          op->src[0]->ne[0] == v->ne[0] && op->src[1]->ne[0] == v->ne[0] &&
                          ggml_nelements(op) <= UINT32_MAX;
                for (int i = 0; i < 6; i++) {
                    ok = ok && op->src[i]->type == GGML_TYPE_F32 && op->src[i]->nb[0] == sizeof(float);
                }
                return ok;
            }
        case GGML_OP_SSM_CONV:
            // same layout requirements as the CPU kernel
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && src1->nb[0] == sizeof(float) &&
                   src0->nb[1] == src0->ne[0] * sizeof(float) && src1->nb[1] == src1->ne[0] * sizeof(float) &&
                   ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_OPT_STEP_ADAMW:
            {
                // the update is indexed flat, so every operand must be contiguous and the same shape
                const bool adamw = op->op == GGML_OP_OPT_STEP_ADAMW;
                const ggml_tensor * p = adamw ? op->src[4] : op->src[2];
                bool ok = src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                          p && p->type == GGML_TYPE_F32 &&
                          ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(p) &&
                          ggml_are_same_shape(src0, src1) &&
                          ggml_nelements(p) == (adamw ? 7 : 2) && ggml_nelements(src0) <= UINT32_MAX;
                if (ok && adamw) {
                    for (int i = 2; i <= 3; i++) {
                        ok = ok && op->src[i] && op->src[i]->type == GGML_TYPE_F32 &&
                             ggml_is_contiguous(op->src[i]) && ggml_are_same_shape(src0, op->src[i]);
                    }
                }
                return ok;
            }
        case GGML_OP_SILU_BACK:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(op) &&
                   ggml_are_same_shape(src1, op) && ggml_are_same_shape(src1, src0) &&
                   ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_REPEAT_BACK:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) &&
                   // dst is what repeats up to src0, the same direction the CPU asserts
                   ggml_can_repeat(op, src0) && ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_RMS_NORM_BACK:
            // rows are addressed by nb[1..3], so only the rows themselves have to be packed
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && src1->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) &&
                   ggml_are_same_shape(src0, op) && ggml_are_same_shape(src0, src1);
        case GGML_OP_SOFT_MAX_BACK:
            {
                // the kernel indexes rows flat, and it has no ALiBi slope
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
                return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                       max_bias == 0.0f &&
                       ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(op) &&
                       ggml_are_same_shape(src0, op) && ggml_are_same_shape(src1, op);
            }
        case GGML_OP_CROSS_ENTROPY_LOSS:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_scalar(op) &&
                   ggml_are_same_shape(src0, src1);
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                   op->src[2] && op->src[2]->type == GGML_TYPE_F32 && ggml_is_scalar(src0) &&
                   ggml_is_contiguous(src1) && ggml_is_contiguous(op->src[2]) && ggml_is_contiguous(op) &&
                   ggml_are_same_shape(src1, op->src[2]) && ggml_are_same_shape(src1, op);
        case GGML_OP_IM2COL_BACK:
            // src1 is only read for its shape, so its type does not matter here
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(src0) && ggml_is_contiguous(op) &&
                   ggml_get_op_params_i32(op, 0) > 0 && ggml_get_op_params_i32(op, 1) > 0 &&
                   ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_GET_ROWS_BACK:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32 &&
                   ggml_is_contiguous(op) && src0->nb[0] == sizeof(float) &&
                   src0->ne[0] == op->ne[0] && ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_DSV4_HC_PRE:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                   op->ne[0] == src0->ne[0] && op->ne[1] == src0->ne[2] &&
                   ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_DSV4_HC_POST:
            {
                const ggml_tensor * p = op->src[2];
                const ggml_tensor * c = op->src[3];
                return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                       p->type == GGML_TYPE_F32 && (!c || c->type == GGML_TYPE_F32) &&
                       op->ne[0] == src0->ne[0] && op->ne[1] == src1->ne[1] && op->ne[2] == src0->ne[1] &&
                       ggml_nelements(op) <= UINT32_MAX;
            }
        case GGML_OP_DSV4_HC_COMB:
            // the kernel fixes hc at 4, exactly as the CPU reference does
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                   op->src[2]->type == GGML_TYPE_F32 && op->ne[0] == 4 && op->ne[1] == 4 &&
                   op->ne[2] == src0->ne[1] && src1->ne[0] >= 3 && ggml_get_op_params_i32(op, 1) > 0 &&
                   ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_LIGHTNING_INDEXER:
            {
                // the mask is f16 whatever K is, so this op always needs 16 bit loads
                const ggml_tensor * w = op->src[2];
                const ggml_tensor * m = op->src[3];
                return ctx->caps.native_16bit && op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 &&
                       (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
                       w->type == GGML_TYPE_F32 && m->type == GGML_TYPE_F16 &&
                       op->nb[0] == sizeof(float) && src0->nb[0] == sizeof(float) &&
                       src1->nb[0] == ggml_type_size(src1->type) && w->nb[0] == sizeof(float) &&
                       m->nb[0] == sizeof(ggml_fp16_t) &&
                       op->ne[0] == src1->ne[2] && op->ne[1] == src0->ne[2] && op->ne[2] == 1 &&
                       op->ne[3] == src0->ne[3] && m->ne[3] > 0 &&
                       ggml_nelements(op) <= UINT32_MAX;
            }
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            {
                // every tensor is indexed flat, exactly as the CPU kernels do
                const int n_src  = op->op == GGML_OP_RWKV_WKV7 ? 7 : (op->op == GGML_OP_RWKV_WKV6 ? 6 : 5);
                const ggml_tensor * state = op->src[n_src - 1];
                const int64_t T      = op->src[1]->ne[2];
                const int64_t heads  = op->src[1]->ne[1];
                const int64_t n_seqs = state->ne[1];
                bool ok = op->type == GGML_TYPE_F32 && ggml_is_contiguous(op);
                for (int i = 0; i < n_src; i++) {
                    ok = ok && op->src[i] && op->src[i]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[i]);
                }
                ok = ok && heads > 0 && n_seqs > 0 && op->ne[0] % heads == 0 && T % n_seqs == 0 &&
                     ggml_nelements(op) <= UINT32_MAX;
                return ok;
            }
        case GGML_OP_CONV_2D:
            return op->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                   (src0->type == GGML_TYPE_F32 || (src0->type == GGML_TYPE_F16 && ctx->caps.native_16bit)) &&
                   ggml_is_contiguous(src0) && src1->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) &&
                   src0->ne[2] == src1->ne[2] && ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_CONV_3D:
            {
                // C and OC come from op_params rather than a shape, so check them against the tensors
                const int32_t c_in  = ggml_get_op_params_i32(op, 9);
                const int32_t c_out = ggml_get_op_params_i32(op, 11);
                return op->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                       (src0->type == GGML_TYPE_F32 || (src0->type == GGML_TYPE_F16 && ctx->caps.native_16bit)) &&
                       ggml_is_contiguous(src0) && src1->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) &&
                       c_in > 0 && c_out > 0 && src0->ne[3] == (int64_t) c_in * c_out &&
                       src1->ne[3] % c_in == 0 && op->ne[3] % c_out == 0 &&
                       ggml_nelements(op) <= UINT32_MAX;
            }
        case GGML_OP_CONV_TRANSPOSE_2D:
            // Cin is walked across both inputs, so the kernel's ne[3] must match src1's ne[2]
            return op->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                   (src0->type == GGML_TYPE_F32 || (src0->type == GGML_TYPE_F16 && ctx->caps.native_16bit)) &&
                   src0->nb[0] == ggml_type_size(src0->type) && src1->nb[0] == sizeof(float) &&
                   op->nb[0] == sizeof(float) && src0->ne[3] == src1->ne[2] &&
                   ggml_get_op_params_i32(op, 0) > 0 && ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_COL2IM_1D:
            // f32 only; the CPU also takes f16 and bf16 columns
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(src0) && ggml_is_contiguous(op) &&
                   ggml_get_op_params_i32(op, 1) > 0 && ggml_get_op_params_i32(op, 0) > 0 &&
                   ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_CONV_TRANSPOSE_1D:
            // the gather walks Cin across both inputs, so the kernel's Cin count must match src1's rows
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && src1->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) &&
                   src0->ne[2] == src1->ne[1] && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
                   ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_CONV_2D_DW:
            // only the WHCN path; the CWHN variant the CPU also handles has a different kernel layout
            return op->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                   (src0->type == GGML_TYPE_F32 || (src0->type == GGML_TYPE_F16 && ctx->caps.native_16bit)) &&
                   ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(op) &&
                   ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_SSM_SCAN:
            {
                // one thread per (sequence, head, dim); the rows it indexes directly must be packed
                const ggml_tensor * dt  = op->src[2];
                const ggml_tensor * A   = op->src[3];
                const ggml_tensor * B   = op->src[4];
                const ggml_tensor * C   = op->src[5];
                const ggml_tensor * ids = op->src[6];
                const int64_t nc = src0->ne[0];
                const int64_t nr = src0->ne[1];
                const int64_t nh = src1->ne[1];
                const int64_t ng = B->ne[1];
                bool ok = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                          dt->type == GGML_TYPE_F32 && A->type == GGML_TYPE_F32 && B->type == GGML_TYPE_F32 &&
                          C->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32;
                ok = ok && ggml_is_contiguous(src0) && src1->nb[0] == sizeof(float) && dt->nb[0] == sizeof(float) &&
                     src1->nb[1] == nr * sizeof(float) && A->nb[1] == A->ne[0] * sizeof(float) &&
                     B->nb[1] == nc * sizeof(float) && C->nb[1] == nc * sizeof(float);
                // every stride is passed in float units, and the flat job index is 32 bit
                ok = ok && ng != 0 && nh % ng == 0 && ggml_nelements(op) <= UINT32_MAX;
                return ok;
            }
        case GGML_OP_CONCAT:
            return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32) && src0->type == op->type &&
                   src1->type == op->type && ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                // f32/f16 KV, f16 mask, head sizes up to 576 and multiples of 4; contiguous rows and a contiguous dst
                const ggml_tensor * k = op->src[1];
                const ggml_tensor * v = op->src[2];
                const ggml_tensor * m = op->src[3];
                const ggml_tensor * s = op->src[4];
                auto kv_ok = [](const ggml_tensor * t) {
                    if (t->type == GGML_TYPE_Q8_0) {
                        return t->ne[0] <= 576 && t->ne[0] % 32 == 0 && t->nb[1] % ggml_type_size(t->type) == 0 &&
                               t->nb[2] % ggml_type_size(t->type) == 0 && t->nb[3] % ggml_type_size(t->type) == 0;
                    }
                    return (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) && t->nb[0] == ggml_type_size(t->type) &&
                           t->ne[0] <= 576 && t->ne[0] % 4 == 0;
                };
                return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src0->nb[0] == sizeof(float) &&
                       kv_ok(k) && kv_ok(v) && (!m || (m->type == GGML_TYPE_F16 && m->nb[0] == 2)) &&
                       (!s || s->type == GGML_TYPE_F32) && ggml_is_contiguous(op);
            }
        case GGML_OP_ROPE:
        // ROPE_BACK is the same rotation with the sine negated, and DeepSeek-V4-Flash uses it in its
        // forward graph, so it is not a training-only op
        case GGML_OP_ROPE_BACK:
            return (op->type == GGML_TYPE_F32 || (op->type == GGML_TYPE_F16 && ctx->caps.native_16bit)) &&
                   src0->type == op->type && src1->type == GGML_TYPE_I32 && src0->ne[0] % 2 == 0 &&
                   (!op->src[2] || op->src[2]->type == GGML_TYPE_F32);
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU: case GGML_GLU_OP_GEGLU: case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_GEGLU_ERF: case GGML_GLU_OP_GEGLU_QUICK: case GGML_GLU_OP_SWIGLU_CLAMP:
                    return (op->type == GGML_TYPE_F32 || (op->type == GGML_TYPE_F16 && ctx->caps.native_16bit)) &&
                           src0->type == op->type && (!src1 || src1->type == op->type);
                case GGML_GLU_OP_SWIGLU_OAI:
                    return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && (!src1 || src1->type == GGML_TYPE_F32);
                default:
                    return false;
            }
        case GGML_OP_UNARY:
            return ggml_d3d11_unary_supported(ggml_get_unary_op(op)) && src0->type == op->type &&
                   (op->type == GGML_TYPE_F32 || (op->type == GGML_TYPE_F16 && ctx->caps.native_16bit));
        case GGML_OP_CLAMP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
            return src0->type == op->type &&
                   (op->type == GGML_TYPE_F32 || (op->type == GGML_TYPE_F16 && ctx->caps.native_16bit));
        case GGML_OP_GET_ROWS:
            if (src1->type != GGML_TYPE_I32) {
                return false;
            }
            switch (src0->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                    return op->type == GGML_TYPE_F32;
                case GGML_TYPE_I32:
                    return op->type == GGML_TYPE_I32;
                default:
                    // quantized sources go through the dequant paths of the matrix-vector kernel
                    return ggml_is_quantized(src0->type) && ggml_d3d11_mul_mat_vec_type(src0->type) &&
                           op->type == GGML_TYPE_F32 && op->ne[0] % ggml_blck_size(src0->type) == 0;
            }
        default:
            return false;
    }
}

static bool ggml_backend_d3d11_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    auto *     ctx = (d3d11_device_ctx *) dev->context;
    // iq1_s comes out wrong from FXC at every optimization level that fits the register limit
    bool iq1_s = op->type == GGML_TYPE_IQ1_S;
    for (int i = 0; i < GGML_MAX_SRC && op->src[i]; i++) {
        iq1_s = iq1_s || op->src[i]->type == GGML_TYPE_IQ1_S;
    }
    // AMD (R9700): flash attention with V head size >= 192 gives wrong values; the same DXBC is right on
    // Intel and the MTT S80, and /Od or unrolled loops did not fix it
    const bool amd_fa = ctx->vendor_id == 0x1002 && op->op == GGML_OP_FLASH_ATTN_EXT && op->src[2]->ne[0] >= 192;
    const bool ok  = !iq1_s && !amd_fa && ggml_d3d11_supports_op(ctx, op);
    if (!ok && ctx->stats) {
        std::string key = ggml_op_desc(op);
        for (int i = 0; i < GGML_MAX_SRC && op->src[i]; i++) {
            key += (i == 0 ? " " : ",");
            key += ggml_type_name(op->src[i]->type);
        }
        key += std::string(" -> ") + ggml_type_name(op->type);
        std::lock_guard<std::mutex> lock(ctx->rejected_mutex);
        ctx->rejected[key]++;
    }
    return ok;
}

static struct ggml_backend_device_i ggml_backend_d3d11_device_i = {
    /* .get_name             = */ ggml_backend_d3d11_device_get_name,
    /* .get_description      = */ ggml_backend_d3d11_device_get_description,
    /* .get_memory           = */ ggml_backend_d3d11_device_get_memory,
    /* .get_type             = */ ggml_backend_d3d11_device_get_type,
    /* .get_props            = */ ggml_backend_d3d11_device_get_props,
    /* .init_backend         = */ ggml_backend_d3d11_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_d3d11_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_d3d11_device_supports_op,
    /* .supports_buft        = */ ggml_backend_d3d11_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

/* Registry: adapter enumeration and device initialization */

struct ggml_backend_d3d11_reg_context {
    std::vector<std::shared_ptr<d3d11_device_ctx>> devs;
    std::vector<ggml_backend_device>               devices;
};

static ggml_backend_d3d11_reg_context * g_reg_ctx = nullptr;

static void ggml_d3d11_atexit() {
    if (!g_reg_ctx) {
        return;
    }
    for (auto & dev : g_reg_ctx->devs) {
        if (dev->stats) {
            ggml_d3d11_print_stats(*dev);
        }
    }
}

static std::shared_ptr<d3d11_device_ctx> ggml_d3d11_shared_dev(d3d11_device_ctx * dev) {
    for (auto & d : g_reg_ctx->devs) {
        if (d.get() == dev) {
            return d;
        }
    }
    GGML_ABORT("ggml_d3d11: unknown device context");
}

// Loads a DLL from the folder this backend DLL sits in, then from the normal search path. An application
// that loads the backend from a folder of its own (LM Studio, for one) does not put that folder on the DLL
// search path, so dxcompiler.dll and dxil.dll next to ggml-d3d11.dll would otherwise not be found.
static bool ggml_d3d11_init_device(d3d11_device_ctx & dev, ggml_backend_dev_t ggml_dev) {
    static std::once_flag prewarm_once;
    std::call_once(prewarm_once, ggml_d3d11_prewarm);
    D3D11_FEATURE_DATA_D3D11_OPTIONS2 o2 = {};
    if (SUCCEEDED(dev.device->CheckFeatureSupport(D3D11_FEATURE_D3D11_OPTIONS2, &o2, sizeof(o2)))) {
        dev.caps.uma = o2.UnifiedMemoryArchitecture;
    }
    dev.max_uavs = dev.device->GetFeatureLevel() >= D3D_FEATURE_LEVEL_11_1 ? D3D11_MAX_ROOT_UAVS : 8;
    if (const char * env = getenv("GGML_D3D11_MAX_ALLOC_MB")) {
        dev.max_alloc = (size_t) atoll(env) * 1024 * 1024;
    }
    if (const char * env = getenv("GGML_D3D11_SUBMIT_BATCH")) {
        dev.submit_batch = std::max(1, atoi(env));
    }
    // the test runner passes only GGML_D3D12_* variables, so these two are read under both names
    if (getenv("GGML_D3D11_NO_FUSE") != nullptr || getenv("GGML_D3D12_NO_FUSE") != nullptr) {
        dev.no_fuse = true;
    }
    // fixed flash attention work budget per submission (no adaptation); for tests of the chunked path
    if (const char * env = getenv("GGML_D3D11_FA_WORK")) {
        dev.fa_work       = std::max<uint64_t>(1, strtoull(env, nullptr, 10));
        dev.fa_work_fixed = true;
    }
    if (const char * env = getenv("GGML_D3D11_MM_TPR")) {
        dev.mm_tpr_max = (uint32_t) std::max(1, atoi(env));
    }
    if (const char * env = getenv("GGML_D3D11_TILED")) {
        dev.tiled_min_cols = (uint32_t) std::max(0, atoi(env));
    }
    const char * disable_ops = getenv("GGML_D3D11_DISABLE_OPS") ? getenv("GGML_D3D11_DISABLE_OPS") : getenv("GGML_D3D12_DISABLE_OPS");
    if (const char * env = disable_ops) {
        dev.disable_ops = std::string(",") + env + ",";
        for (char & c : dev.disable_ops) {
            if (c == '.') { c = ','; }
        }
    }

    if (getenv("GGML_D3D11_DEBUG") != nullptr) {
        dev.device->QueryInterface(IID_PPV_ARGS(dev.info.put()));
    }
    D3D11_BUFFER_DESC cb = {};
    cb.ByteWidth         = D3D11_PARAM_SLOT_SIZE;
    cb.Usage             = D3D11_USAGE_DYNAMIC;
    cb.BindFlags         = D3D11_BIND_CONSTANT_BUFFER;
    cb.CPUAccessFlags    = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(dev.device->CreateBuffer(&cb, nullptr, dev.cbuf.put()))) {
        GGML_LOG_ERROR("ggml_d3d11: CreateBuffer (params) failed\n");
        return false;
    }
    D3D11_QUERY_DESC qd = { D3D11_QUERY_EVENT, 0 };
    if (FAILED(dev.device->CreateQuery(&qd, dev.done_query.put()))) {
        GGML_LOG_ERROR("ggml_d3d11: CreateQuery failed\n");
        return false;
    }

    dev.stats = getenv("GGML_D3D11_STATS") != nullptr;
    if (dev.stats) {
        static bool registered = false;
        if (!registered) {
            registered = true;
            atexit(ggml_d3d11_atexit);
        }
    }

    dev.buft = {
        /* .iface = */ {
            /* .get_name       = */ ggml_backend_d3d11_buffer_type_get_name,
            /* .alloc_buffer   = */ ggml_backend_d3d11_buffer_type_alloc_buffer,
            /* .get_alignment  = */ ggml_backend_d3d11_buffer_type_get_alignment,
            /* .get_max_size   = */ ggml_backend_d3d11_buffer_type_get_max_size,
            /* .get_alloc_size = */ NULL,
            /* .is_host        = */ NULL,
        },
        /* .device  = */ ggml_dev,
        /* .context = */ &dev,
    };

    GGML_LOG_WARN("ggml_d3d11: %s = %s | Direct3D 11 | driver %s | feature level %x, cs_5_0 | %u UAVs | %s | %zu MiB | build %s %s\n",
                  dev.name.c_str(), dev.desc.c_str(), ggml_d3d11_driver_version(dev.adapter.get()).c_str(),
                  (unsigned) dev.device->GetFeatureLevel(), dev.max_uavs,
                  dev.caps.uma ? "UMA" : "discrete", (dev.caps.uma ? dev.shared_mem : dev.dedicated_mem) / (1024 * 1024),
                  __DATE__, __TIME__);
    return true;
}

static void ggml_d3d11_enumerate(ggml_backend_d3d11_reg_context & reg_ctx, ggml_backend_reg_t reg) {
    // off unless asked for, so that a box with both backends keeps running D3D12 alone.
    // GGML_D3D12_DISABLE turns it on too: with D3D12 off, D3D11 takes its place.
    // A build without the D3D12 backend has it on by default.
#ifndef GGML_D3D11_DEFAULT_ON
    if (getenv("GGML_D3D11_ENABLE") == nullptr && getenv("GGML_D3D12_DISABLE") == nullptr) {
        return;
    }
#endif
    if (getenv("GGML_D3D11_DISABLE") != nullptr) {
        return;
    }
    const bool debug      = getenv("GGML_D3D11_DEBUG") != nullptr;
    const bool allow_warp = getenv("GGML_D3D11_WARP") != nullptr;

    com_ptr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(factory.put())))) {
        GGML_LOG_WARN("ggml_d3d11: CreateDXGIFactory1 failed\n");
        return;
    }
    com_ptr<IDXGIFactory6> factory6;
    factory->QueryInterface(IID_PPV_ARGS(factory6.put()));

    std::vector<com_ptr<IDXGIAdapter1>> adapters;
    for (UINT i = 0;; i++) {
        com_ptr<IDXGIAdapter1> adapter;
        HRESULT hr;
        if (factory6) {
            hr = factory6->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(adapter.put()));
        } else {
            hr = factory->EnumAdapters1(i, adapter.put());
        }
        if (hr == DXGI_ERROR_NOT_FOUND || FAILED(hr)) {
            break;
        }
        adapters.push_back(adapter);
    }
    if (const char * only = getenv("GGML_D3D11_ONLY")) {
        std::vector<com_ptr<IDXGIAdapter1>> kept;
        for (auto & adapter : adapters) {
            DXGI_ADAPTER_DESC1 desc = {};
            adapter->GetDesc1(&desc);
            if (ggml_d3d11_wide_to_utf8(desc.Description).find(only) != std::string::npos) {
                kept.push_back(adapter);
            }
        }
        adapters = std::move(kept);
    }

    for (auto & adapter : adapters) {
        DXGI_ADAPTER_DESC1 desc = {};
        adapter->GetDesc1(&desc);
        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && !allow_warp) {
            continue;
        }
        auto dev = std::make_shared<d3d11_device_ctx>();
        const D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
        UINT flags = debug ? D3D11_CREATE_DEVICE_DEBUG : 0;
        HRESULT hr = D3D11CreateDevice(adapter.get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags, levels, 2,
                                       D3D11_SDK_VERSION, dev->device.put(), nullptr, dev->ctx.put());
        if (FAILED(hr) && debug) {
            GGML_LOG_WARN("ggml_d3d11: debug layer unavailable (install the Graphics Tools feature)\n");
            flags = 0;
            hr = D3D11CreateDevice(adapter.get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags, levels, 2,
                                   D3D11_SDK_VERSION, dev->device.put(), nullptr, dev->ctx.put());
        }
        if (FAILED(hr)) {
            // Windows without the 11.1 runtime rejects the list that names 11_1
            hr = D3D11CreateDevice(adapter.get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags, levels + 1, 1,
                                   D3D11_SDK_VERSION, dev->device.put(), nullptr, dev->ctx.put());
        }
        if (FAILED(hr)) {
            GGML_LOG_INFO("ggml_d3d11: skipped %s - no feature level 11_0 device (0x%08x)\n",
                          ggml_d3d11_wide_to_utf8(desc.Description).c_str(), (unsigned) hr);
            continue;
        }
        dev->adapter       = adapter;
        dev->desc          = ggml_d3d11_wide_to_utf8(desc.Description);
        dev->vendor_id     = desc.VendorId;
        dev->dedicated_mem = desc.DedicatedVideoMemory;
        dev->shared_mem    = desc.SharedSystemMemory;
        dev->name          = GGML_D3D11_NAME + std::to_string(reg_ctx.devs.size());
        reg_ctx.devs.push_back(dev);
    }

    reg_ctx.devices.reserve(reg_ctx.devs.size());
    std::vector<std::shared_ptr<d3d11_device_ctx>> kept;
    for (auto & dev : reg_ctx.devs) {
        ggml_backend_device ggml_dev = {
            /* .iface   = */ ggml_backend_d3d11_device_i,
            /* .reg     = */ reg,
            /* .context = */ dev.get(),
        };
        reg_ctx.devices.push_back(ggml_dev);
        if (!ggml_d3d11_init_device(*dev, &reg_ctx.devices.back())) {
            reg_ctx.devices.pop_back();
            continue;
        }
        kept.push_back(dev);
    }
    reg_ctx.devs = std::move(kept);
    for (size_t i = 0; i < reg_ctx.devs.size(); i++) {
        reg_ctx.devs[i]->name = GGML_D3D11_NAME + std::to_string(i);
    }
}
static const char * ggml_backend_d3d11_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_D3D11_NAME;
}

static size_t ggml_backend_d3d11_reg_get_device_count(ggml_backend_reg_t reg) {
    auto * ctx = (ggml_backend_d3d11_reg_context *) reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_d3d11_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * ctx = (ggml_backend_d3d11_reg_context *) reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return &ctx->devices[index];
}

static const struct ggml_backend_reg_i ggml_backend_d3d11_reg_i = {
    /* .get_name         = */ ggml_backend_d3d11_reg_get_name,
    /* .get_device_count = */ ggml_backend_d3d11_reg_get_device_count,
    /* .get_device       = */ ggml_backend_d3d11_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_d3d11_reg() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    // leaked on purpose: D3D11 objects must not be torn down during static destruction
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_d3d11_reg_i,
        /* .context     = */ nullptr,
    };
    if (g_reg_ctx == nullptr) {
        g_reg_ctx   = new ggml_backend_d3d11_reg_context();
        reg.context = g_reg_ctx;
        ggml_d3d11_enumerate(*g_reg_ctx, &reg);
    }
    return &reg;
}

ggml_backend_t ggml_backend_d3d11_init(int device) {
    ggml_backend_reg_t reg = ggml_backend_d3d11_reg();
    if (device < 0 || (size_t) device >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    return ggml_backend_d3d11_device_init_backend(ggml_backend_reg_dev_get(reg, device), nullptr);
}

GGML_BACKEND_DL_IMPL(ggml_backend_d3d11_reg)
