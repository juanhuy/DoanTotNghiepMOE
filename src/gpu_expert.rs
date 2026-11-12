//! A1: INT8 SwiGLU expert compute on the GPU for experts already resident in VRAM.
//!
//! Kernels are compiled at runtime with NVRTC (loaded dynamically, like libcuda/libcublas), so the
//! build needs no CUDA toolkit. Layout matches `Int8Expert`: weights = [gate I×H; up I×H; down H×I]
//! as signed INT8 bytes, scales = [gate I; up I; down H] per row. Only the hidden vector goes up and
//! one H-vector per expert comes back; expert weights never cross PCIe on this path.
//!
//! Invariants:
//! - Every call sets the engine's primary context current on the calling thread.
//! - Device scratch (input, activations, outputs) is owned here and guarded by a mutex; kernels run on
//!   the legacy default stream and the final synchronous cuMemcpyDtoH orders after them.
//! - Weight/scale pointers come from `GpuExpertWeights` borrowed for the whole call, so the pool slot
//!   cannot be freed or reused while kernels read it.
//! - Drop frees scratch and unloads the module with the context current; `GpuDenseStorage` declares
//!   this field before `engine`, so it drops before the primary context is released.

use crate::gpu::{CUcontext, CUdeviceptr, CudaDriverApi, GpuComputeEngine, GpuExpertWeights};
use std::ffi::{c_char, c_int, c_uint, c_void, CString};
use std::sync::{Arc, Mutex};

type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUresult = c_int;
type NvrtcProgram = *mut c_void;
type NvrtcResult = c_int;

const CUDA_SUCCESS: CUresult = 0;
const NVRTC_SUCCESS: NvrtcResult = 0;
const ATTR_CC_MAJOR: c_int = 75;
const ATTR_CC_MINOR: c_int = 76;
const BLOCK: c_uint = 256;

extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
const RTLD_LAZY: c_int = 1;

pub(crate) const KERNEL_SOURCE: &str = r#"
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}
// Sum of v over the block, valid in thread 0. `sh` holds 32 floats.
__device__ __forceinline__ float block_sum(float v, float* sh) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nw) ? sh[lane] : 0.0f;
    if (wid == 0) v = warp_sum(v);
    return v;
}
__device__ __forceinline__ float dot_row(const signed char* w, const float* x, int n, int vec4) {
    float acc = 0.0f;
    if (vec4) {
        const char4* w4 = (const char4*)w;
        const float4* x4 = (const float4*)x;
        for (int c = threadIdx.x; c < (n >> 2); c += blockDim.x) {
            char4 a = w4[c];
            float4 b = x4[c];
            acc += (float)a.x * b.x + (float)a.y * b.y + (float)a.z * b.z + (float)a.w * b.w;
        }
    } else {
        for (int c = threadIdx.x; c < n; c += blockDim.x) acc += (float)w[c] * x[c];
    }
    return acc;
}
// grid.x = I rows; act[row] = silu(gate·x * sg) * (up·x * su)
extern "C" __global__ void moe_gate_up(const signed char* w, const float* s, const float* x,
                                       float* act, int H, int I, int vec4) {
    __shared__ float sh_g[32];
    __shared__ float sh_u[32];
    int row = blockIdx.x;
    float g = block_sum(dot_row(w + (size_t)row * H, x, H, vec4), sh_g);
    float u = block_sum(dot_row(w + (size_t)(I + row) * H, x, H, vec4), sh_u);
    if (threadIdx.x == 0) {
        float gg = g * s[row];
        float uu = u * s[I + row];
        act[row] = gg / (1.0f + expf(-gg)) * uu;
    }
}
// grid.x = H rows; out[row] = down·act * sd
extern "C" __global__ void moe_down(const signed char* w, const float* s, const float* act,
                                    float* out, int H, int I, int vec4) {
    __shared__ float sh[32];
    int row = blockIdx.x;
    float d = block_sum(dot_row(w + (size_t)2 * I * H + (size_t)row * I, act, I, vec4), sh);
    if (threadIdx.x == 0) out[row] = d * s[2 * I + row];
}
"#;

struct KernelDriver {
    _handle: *mut c_void,
    device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> CUresult,
    module_load_data: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
    module_get_function: unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
    module_unload: unsafe extern "C" fn(CUmodule) -> CUresult,
    #[allow(clippy::type_complexity)]
    launch_kernel: unsafe extern "C" fn(
        CUfunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CUresult,
}

struct Nvrtc {
    _handle: *mut c_void,
    create: unsafe extern "C" fn(
        *mut NvrtcProgram,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> NvrtcResult,
    compile: unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> NvrtcResult,
    ptx_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
    ptx: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
    log_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
    log: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
    destroy: unsafe extern "C" fn(*mut NvrtcProgram) -> NvrtcResult,
}

unsafe fn open_first(names: &[&str]) -> Option<*mut c_void> {
    names.iter().find_map(|name| {
        let name = CString::new(*name).ok()?;
        let handle = dlopen(name.as_ptr(), RTLD_LAZY);
        (!handle.is_null()).then_some(handle)
    })
}

macro_rules! sym {
    ($handle:expr, $name:literal) => {{
        let ptr = dlsym($handle, concat!($name, "\0").as_ptr() as *const c_char);
        if ptr.is_null() {
            return Err(concat!("missing symbol ", $name).to_string());
        }
        std::mem::transmute::<*mut c_void, _>(ptr)
    }};
}

impl KernelDriver {
    // Each target type is the struct field the symbol is assigned to.
    #[allow(clippy::missing_transmute_annotations)]
    fn load() -> Result<Self, String> {
        unsafe {
            let handle = open_first(&["libcuda.so.1", "libcuda.so"]).ok_or("libcuda not found")?;
            Ok(Self {
                _handle: handle,
                device_get_attribute: sym!(handle, "cuDeviceGetAttribute"),
                module_load_data: sym!(handle, "cuModuleLoadData"),
                module_get_function: sym!(handle, "cuModuleGetFunction"),
                module_unload: sym!(handle, "cuModuleUnload"),
                launch_kernel: sym!(handle, "cuLaunchKernel"),
            })
        }
    }
}

impl Nvrtc {
    // Each target type is the struct field the symbol is assigned to.
    #[allow(clippy::missing_transmute_annotations)]
    fn load() -> Result<Self, String> {
        unsafe {
            let handle = open_first(&[
                "libnvrtc.so.13",
                "libnvrtc.so.12",
                "libnvrtc.so",
                "/opt/cuda/lib64/libnvrtc.so",
                "/usr/local/cuda/lib64/libnvrtc.so",
            ])
            .ok_or("libnvrtc not found")?;
            Ok(Self {
                _handle: handle,
                create: sym!(handle, "nvrtcCreateProgram"),
                compile: sym!(handle, "nvrtcCompileProgram"),
                ptx_size: sym!(handle, "nvrtcGetPTXSize"),
                ptx: sym!(handle, "nvrtcGetPTX"),
                log_size: sym!(handle, "nvrtcGetProgramLogSize"),
                log: sym!(handle, "nvrtcGetProgramLog"),
                destroy: sym!(handle, "nvrtcDestroyProgram"),
            })
        }
    }

    /// Compile to PTX for `arch` ("compute_75" ...); returns NUL-terminated PTX.
    fn compile_ptx(&self, arch: &str) -> Result<Vec<u8>, String> {
        unsafe {
            let src = CString::new(KERNEL_SOURCE).unwrap();
            let name = CString::new("moe_expert.cu").unwrap();
            let mut program = std::ptr::null_mut();
            if (self.create)(
                &mut program,
                src.as_ptr(),
                name.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            ) != NVRTC_SUCCESS
            {
                return Err("nvrtcCreateProgram failed".into());
            }
            let option = CString::new(format!("--gpu-architecture={arch}")).unwrap();
            let options = [option.as_ptr()];
            let status = (self.compile)(program, 1, options.as_ptr());
            let result = if status == NVRTC_SUCCESS {
                let mut size = 0;
                (self.ptx_size)(program, &mut size);
                let mut ptx = vec![0u8; size];
                if (self.ptx)(program, ptx.as_mut_ptr() as *mut c_char) == NVRTC_SUCCESS {
                    Ok(ptx)
                } else {
                    Err("nvrtcGetPTX failed".into())
                }
            } else {
                let mut size = 0;
                (self.log_size)(program, &mut size);
                let mut log = vec![0u8; size.max(1)];
                (self.log)(program, log.as_mut_ptr() as *mut c_char);
                Err(format!(
                    "NVRTC {arch} compile error {status}: {}",
                    String::from_utf8_lossy(&log).trim_end_matches('\0')
                ))
            };
            (self.destroy)(&mut program);
            result
        }
    }
}

struct Scratch {
    d_input: CUdeviceptr,
    d_act: CUdeviceptr,
    d_out: CUdeviceptr,
    host_in: Vec<f32>,
    host_out: Vec<f32>,
}

/// Compiled INT8 SwiGLU kernels plus device scratch for up to `max_experts` per call.
pub struct GpuExpertKernels {
    kernel_driver: KernelDriver,
    driver: Arc<CudaDriverApi>,
    ctx: CUcontext,
    module: CUmodule,
    gate_up: CUfunction,
    down: CUfunction,
    hidden: usize,
    intermediate: usize,
    max_experts: usize,
    pub arch: String,
    scratch: Mutex<Scratch>,
}

unsafe impl Send for GpuExpertKernels {}
unsafe impl Sync for GpuExpertKernels {}

impl GpuExpertKernels {
    pub fn init(
        engine: &GpuComputeEngine,
        hidden: usize,
        intermediate: usize,
        max_experts: usize,
    ) -> Result<Self, String> {
        if hidden == 0 || intermediate == 0 || max_experts == 0 {
            return Err("expert kernel dimensions must be positive".into());
        }
        let kernel_driver = KernelDriver::load()?;
        let nvrtc = Nvrtc::load()?;
        let driver = engine.driver_api();
        unsafe {
            let _ = (driver.cu_ctx_set_current)(engine.ctx);
            let (mut major, mut minor) = (0, 0);
            (kernel_driver.device_get_attribute)(&mut major, ATTR_CC_MAJOR, engine.dev);
            (kernel_driver.device_get_attribute)(&mut minor, ATTR_CC_MINOR, engine.dev);
            // Device arch first; older virtual archs JIT forward if this NVRTC lacks the new one.
            let mut errors = Vec::new();
            let mut built = None;
            for arch in [
                format!("compute_{major}{minor}"),
                "compute_75".into(),
                "compute_52".into(),
            ] {
                match nvrtc.compile_ptx(&arch) {
                    Ok(ptx) => {
                        let mut module = std::ptr::null_mut();
                        let status = (kernel_driver.module_load_data)(
                            &mut module,
                            ptx.as_ptr() as *const c_void,
                        );
                        if status == CUDA_SUCCESS {
                            built = Some((arch, module));
                            break;
                        }
                        errors.push(format!("cuModuleLoadData {arch}: {status}"));
                    }
                    Err(error) => errors.push(error),
                }
            }
            let (arch, module) = built.ok_or_else(|| errors.join("; "))?;
            let mut gate_up = std::ptr::null_mut();
            let mut down = std::ptr::null_mut();
            let names = (
                CString::new("moe_gate_up").unwrap(),
                CString::new("moe_down").unwrap(),
            );
            if (kernel_driver.module_get_function)(&mut gate_up, module, names.0.as_ptr())
                != CUDA_SUCCESS
                || (kernel_driver.module_get_function)(&mut down, module, names.1.as_ptr())
                    != CUDA_SUCCESS
            {
                (kernel_driver.module_unload)(module);
                return Err("cuModuleGetFunction failed".into());
            }
            let sizes = [
                max_experts * hidden * 4,
                max_experts * intermediate * 4,
                max_experts * hidden * 4,
            ];
            let mut ptrs = [0u64; 3];
            for (i, bytes) in sizes.iter().enumerate() {
                if (driver.cu_mem_alloc)(&mut ptrs[i], *bytes) != CUDA_SUCCESS {
                    for p in &ptrs[..i] {
                        (driver.cu_mem_free)(*p);
                    }
                    (kernel_driver.module_unload)(module);
                    return Err("cuMemAlloc expert kernel scratch failed".into());
                }
            }
            Ok(Self {
                kernel_driver,
                ctx: engine.ctx,
                driver,
                module,
                gate_up,
                down,
                hidden,
                intermediate,
                max_experts,
                arch,
                scratch: Mutex::new(Scratch {
                    d_input: ptrs[0],
                    d_act: ptrs[1],
                    d_out: ptrs[2],
                    host_in: vec![0.0; max_experts * hidden],
                    host_out: vec![0.0; max_experts * hidden],
                }),
            })
        }
    }

    pub fn max_experts(&self) -> usize {
        self.max_experts
    }

    /// Decode: run every expert on the same `input` (length H); `outputs[j]` gets expert j's result.
    pub fn forward(
        &self,
        experts: &[&GpuExpertWeights],
        input: &[f32],
        outputs: &mut [&mut [f32]],
    ) -> Result<(), String> {
        self.forward_pairs(experts, &[input], outputs)
    }

    /// Run pair j = (experts[j], inputs[j]); one shared input when `inputs.len() == 1`.
    /// Prefill passes one expert repeated over the tokens routed to it.
    pub fn forward_pairs(
        &self,
        experts: &[&GpuExpertWeights],
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> Result<(), String> {
        let (h, i) = (self.hidden, self.intermediate);
        if inputs.len() != 1 && inputs.len() != experts.len() {
            return Err("expert/input count mismatch".into());
        }
        let weight_bytes = 3 * h * i;
        let scale_bytes = (2 * i + h) * 4;
        if experts.len() != outputs.len() || experts.len() > self.max_experts {
            return Err("expert/output count mismatch".into());
        }
        if inputs.iter().any(|x| x.len() != h) || outputs.iter().any(|o| o.len() != h) {
            return Err("expert kernel dimension mismatch".into());
        }
        if experts
            .iter()
            .any(|e| e.weight_bytes != weight_bytes || e.scale_bytes != scale_bytes)
        {
            return Err("GPU expert compute supports INT8 experts of the loaded shape only".into());
        }
        if experts.is_empty() {
            return Ok(());
        }
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| "expert kernel mutex poisoned")?;
        let vec4: c_int = (h % 4 == 0 && i % 4 == 0) as c_int;
        let (hc, ic) = (h as c_int, i as c_int);
        unsafe {
            let _ = (self.driver.cu_ctx_set_current)(self.ctx);
            let Scratch { host_in, .. } = &mut *scratch;
            for (j, x) in inputs.iter().enumerate() {
                host_in[j * h..(j + 1) * h].copy_from_slice(x);
            }
            let src = scratch.host_in.as_ptr() as *const c_void;
            if (self.driver.cu_memcpy_h_to_d)(scratch.d_input, src, inputs.len() * h * 4)
                != CUDA_SUCCESS
            {
                return Err("cuMemcpyHtoD expert input failed".into());
            }
            for (j, expert) in experts.iter().enumerate() {
                let mut w = expert.d_weights;
                let mut s = expert.d_scales;
                let input_slot = if inputs.len() == 1 { 0 } else { j };
                let mut x = scratch.d_input + (input_slot * h * 4) as u64;
                let mut act = scratch.d_act + (j * i * 4) as u64;
                let mut out = scratch.d_out + (j * h * 4) as u64;
                let (mut hv, mut iv, mut v4) = (hc, ic, vec4);
                let mut args: [*mut c_void; 7] = [
                    &mut w as *mut _ as *mut c_void,
                    &mut s as *mut _ as *mut c_void,
                    &mut x as *mut _ as *mut c_void,
                    &mut act as *mut _ as *mut c_void,
                    &mut hv as *mut _ as *mut c_void,
                    &mut iv as *mut _ as *mut c_void,
                    &mut v4 as *mut _ as *mut c_void,
                ];
                if (self.kernel_driver.launch_kernel)(
                    self.gate_up,
                    i as c_uint,
                    1,
                    1,
                    BLOCK,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                ) != CUDA_SUCCESS
                {
                    return Err("cuLaunchKernel moe_gate_up failed".into());
                }
                let mut down_args: [*mut c_void; 7] = [
                    &mut w as *mut _ as *mut c_void,
                    &mut s as *mut _ as *mut c_void,
                    &mut act as *mut _ as *mut c_void,
                    &mut out as *mut _ as *mut c_void,
                    &mut hv as *mut _ as *mut c_void,
                    &mut iv as *mut _ as *mut c_void,
                    &mut v4 as *mut _ as *mut c_void,
                ];
                if (self.kernel_driver.launch_kernel)(
                    self.down,
                    h as c_uint,
                    1,
                    1,
                    BLOCK,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    down_args.as_mut_ptr(),
                    std::ptr::null_mut(),
                ) != CUDA_SUCCESS
                {
                    return Err("cuLaunchKernel moe_down failed".into());
                }
            }
            let n = experts.len() * h;
            let host = scratch.host_out.as_mut_ptr();
            if (self.driver.cu_memcpy_d_to_h)(host as *mut c_void, scratch.d_out, n * 4)
                != CUDA_SUCCESS
            {
                return Err("cuMemcpyDtoH expert outputs failed (kernel fault?)".into());
            }
        }
        for (j, out) in outputs.iter_mut().enumerate() {
            out.copy_from_slice(&scratch.host_out[j * h..(j + 1) * h]);
        }
        crate::backend::finite(&scratch.host_out[..experts.len() * h])
            .map_err(|_| "GPU expert output is not finite".to_string())
    }
}

impl Drop for GpuExpertKernels {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.driver.cu_ctx_set_current)(self.ctx);
            if let Ok(scratch) = self.scratch.get_mut() {
                for p in [scratch.d_input, scratch.d_act, scratch.d_out] {
                    let _ = (self.driver.cu_mem_free)(p);
                }
            }
            let _ = (self.kernel_driver.module_unload)(self.module);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuComputeEngine;
    use crate::int8_expert::Int8Expert;

    fn expert(h: usize, i: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let mut state = seed.wrapping_mul(2_654_435_761).max(1);
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let weights = (0..3 * h * i).map(|_| (next() % 255) as u8).collect();
        let scales = (0..2 * i + h)
            .map(|_| 0.002 + (next() % 1000) as f32 * 1e-5)
            .collect();
        (weights, scales)
    }

    #[test]
    fn gpu_int8_swiglu_matches_cpu_expert() {
        let Some(engine) = GpuComputeEngine::init() else {
            eprintln!("SKIPPED GPU expert kernels: GPU unavailable");
            return;
        };
        // Real OLMoE shape (vec4 path), small vec4 shape, and odd shape (scalar path).
        for (h, i) in [(2048, 1024), (16, 8), (6, 10)] {
            let kernels = match GpuExpertKernels::init(&engine, h, i, 2) {
                Ok(k) => k,
                Err(error) => {
                    eprintln!("SKIPPED GPU expert kernels: {error}");
                    return;
                }
            };
            let input: Vec<f32> = (0..h)
                .map(|c| ((c * 37 % 101) as f32 - 50.0) / 25.0)
                .collect();
            let mut uploaded = Vec::new();
            let mut expected = Vec::new();
            for seed in [1, 2] {
                let (w, s) = expert(h, i, seed + h as u32);
                uploaded.push(engine.upload_expert(0, seed as usize, &w, &s).unwrap());
                expected.push(Int8Expert::from_raw(h, i, w, s).forward(&input).unwrap());
            }
            let refs: Vec<&GpuExpertWeights> = uploaded.iter().collect();
            let mut out = vec![vec![0.0f32; h]; 2];
            let mut outs: Vec<&mut [f32]> = out.iter_mut().map(|o| &mut o[..]).collect();
            kernels.forward(&refs, &input, &mut outs).unwrap();
            for (gpu, cpu) in out.iter().zip(&expected) {
                let scale = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
                let error = gpu
                    .iter()
                    .zip(cpu)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    error / scale < 1e-4,
                    "h={h} i={i} rel error={}",
                    error / scale
                );
            }
            // Wrong count/shape is rejected rather than read out of bounds.
            let mut one: Vec<&mut [f32]> = out.iter_mut().take(1).map(|o| &mut o[..]).collect();
            assert!(kernels.forward(&refs, &input, &mut one).is_err());
            eprintln!(
                "EXECUTED GPU INT8 SwiGLU h={h} i={i} arch={} matches CPU",
                kernels.arch
            );
        }
    }
}
