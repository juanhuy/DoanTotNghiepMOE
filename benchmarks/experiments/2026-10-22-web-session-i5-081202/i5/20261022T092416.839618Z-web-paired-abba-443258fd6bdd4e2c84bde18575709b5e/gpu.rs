//! GPU hardware abstraction and hybrid VRAM offloading tier.
//! Provides dynamic detection of NVIDIA GPU via CUDA Driver API (`libcuda.so`),
//! cuBLAS SGEMV compute engine (`libcublas.so`) for Dense Layer acceleration,
//! adaptive VRAM budget calculation, and graceful zero-cost CPU fallback.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::sync::Arc;
use tracing::{info, warn};

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CublasHandle = *mut c_void;
pub type CublasStatus = c_int;

const CUDA_SUCCESS: CUresult = 0;
const CUBLAS_STATUS_SUCCESS: CublasStatus = 0;
const CUBLAS_OP_N: c_int = 0;
const CUBLAS_OP_T: c_int = 1;

/// Thông tin chi tiết phần cứng GPU được nhận diện
#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuDeviceInfo {
    pub name: String,
    pub total_vram_bytes: u64,
    pub free_vram_bytes: u64,
    pub driver_version: i32,
    pub is_available: bool,
}

/// Cấu hình phân bổ ngân sách VRAM theo nguyên lý thác nước
#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuTierBudget {
    pub total_vram_bytes: u64,
    pub reserved_os_bytes: u64,
    pub usable_vram_bytes: u64,
    pub dense_vram_bytes: u64,
    pub expert_cache_vram_bytes: u64,
    pub is_dense_offloaded: bool,
}

/// Trạng thái giám sát thời gian thực của tầng GPU
#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuTierStatus {
    pub enabled: bool,
    pub device_name: String,
    pub total_vram_bytes: u64,
    pub allocated_vram_bytes: u64,
    pub free_vram_bytes: u64,
    pub dense_offloaded: bool,
    pub vram_experts_resident: usize,
    pub message: String,
}

// Bảng con trỏ hàm CUDA Driver API được liên kết động
pub(crate) struct CudaDriverApi {
    _handle: *mut c_void,
    pub cu_init: unsafe extern "C" fn(flags: c_int) -> CUresult,
    pub cu_driver_get_version: unsafe extern "C" fn(version: *mut c_int) -> CUresult,
    pub cu_device_get: unsafe extern "C" fn(device: *mut CUdevice, ordinal: c_int) -> CUresult,
    pub cu_device_get_name:
        unsafe extern "C" fn(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult,
    pub cu_device_total_mem: unsafe extern "C" fn(bytes: *mut usize, dev: CUdevice) -> CUresult,
    pub cu_mem_get_info: unsafe extern "C" fn(free: *mut usize, total: *mut usize) -> CUresult,
    pub cu_device_primary_ctx_retain:
        unsafe extern "C" fn(ctx: *mut CUcontext, dev: CUdevice) -> CUresult,
    pub cu_device_primary_ctx_release: unsafe extern "C" fn(dev: CUdevice) -> CUresult,
    pub cu_ctx_set_current: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
    pub cu_mem_alloc: unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult,
    pub cu_mem_free: unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult,
    pub cu_memcpy_h_to_d:
        unsafe extern "C" fn(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CUresult,
    pub cu_memcpy_d_to_h:
        unsafe extern "C" fn(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult,
    /// Cấp phát Pinned Host Memory — DMA không cần bounce buffer trung gian (16-C)
    pub cu_mem_host_alloc:
        unsafe extern "C" fn(pp: *mut *mut c_void, byte_count: usize, flags: c_uint) -> CUresult,
    /// Giải phóng Pinned Host Memory (16-C)
    pub cu_mem_free_host: unsafe extern "C" fn(p: *mut c_void) -> CUresult,
}

unsafe impl Send for CudaDriverApi {}
unsafe impl Sync for CudaDriverApi {}

extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

const RTLD_LAZY: c_int = 1;
const RTLD_LOCAL: c_int = 0;

impl CudaDriverApi {
    fn load() -> Option<Self> {
        unsafe {
            let lib_names: &[&[u8]] = &[b"libcuda.so.1\0", b"libcuda.so\0"];
            let mut handle = std::ptr::null_mut();
            for name in lib_names {
                handle = dlopen(name.as_ptr() as *const c_char, RTLD_LAZY | RTLD_LOCAL);
                if !handle.is_null() {
                    break;
                }
            }

            if handle.is_null() {
                return None;
            }

            macro_rules! get_sym {
                ($sym:expr, $ty:ty) => {{
                    let ptr = dlsym(handle, $sym.as_ptr() as *const c_char);
                    if ptr.is_null() {
                        let err_ptr = dlerror();
                        let err_str = if !err_ptr.is_null() {
                            CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
                        } else {
                            "Unknown dlsym error".to_string()
                        };
                        warn!(
                            symbol = ?std::str::from_utf8($sym).unwrap_or("?"),
                            error = %err_str,
                            "Không tìm thấy CUDA driver symbol"
                        );
                        return None;
                    }
                    std::mem::transmute::<*mut c_void, $ty>(ptr)
                }};
            }

            let cu_init = get_sym!(b"cuInit\0", unsafe extern "C" fn(c_int) -> CUresult);
            let cu_driver_get_version = get_sym!(
                b"cuDriverGetVersion\0",
                unsafe extern "C" fn(*mut c_int) -> CUresult
            );
            let cu_device_get = get_sym!(
                b"cuDeviceGet\0",
                unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult
            );
            let cu_device_get_name = get_sym!(
                b"cuDeviceGetName\0",
                unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult
            );
            let cu_device_total_mem = get_sym!(
                b"cuDeviceTotalMem_v2\0",
                unsafe extern "C" fn(*mut usize, CUdevice) -> CUresult
            );
            let cu_mem_get_info = get_sym!(
                b"cuMemGetInfo_v2\0",
                unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult
            );
            let cu_device_primary_ctx_retain = get_sym!(
                b"cuDevicePrimaryCtxRetain\0",
                unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult
            );
            let cu_device_primary_ctx_release = get_sym!(
                b"cuDevicePrimaryCtxRelease_v2\0",
                unsafe extern "C" fn(CUdevice) -> CUresult
            );
            let cu_ctx_set_current = get_sym!(
                b"cuCtxSetCurrent\0",
                unsafe extern "C" fn(CUcontext) -> CUresult
            );
            let cu_mem_alloc = get_sym!(
                b"cuMemAlloc_v2\0",
                unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult
            );
            let cu_mem_free = get_sym!(
                b"cuMemFree_v2\0",
                unsafe extern "C" fn(CUdeviceptr) -> CUresult
            );
            let cu_memcpy_h_to_d = get_sym!(
                b"cuMemcpyHtoD_v2\0",
                unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult
            );
            let cu_memcpy_d_to_h = get_sym!(
                b"cuMemcpyDtoH_v2\0",
                unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult
            );
            // Pinned host memory — 16-C: loại bỏ bounce-buffer overhead khi H2D DMA
            let cu_mem_host_alloc = get_sym!(
                b"cuMemHostAlloc\0",
                unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CUresult
            );
            let cu_mem_free_host = get_sym!(
                b"cuMemFreeHost\0",
                unsafe extern "C" fn(*mut c_void) -> CUresult
            );

            Some(Self {
                _handle: handle,
                cu_init,
                cu_driver_get_version,
                cu_device_get,
                cu_device_get_name,
                cu_device_total_mem,
                cu_mem_get_info,
                cu_device_primary_ctx_retain,
                cu_device_primary_ctx_release,
                cu_ctx_set_current,
                cu_mem_alloc,
                cu_mem_free,
                cu_memcpy_h_to_d,
                cu_memcpy_d_to_h,
                cu_mem_host_alloc,
                cu_mem_free_host,
            })
        }
    }
}

// Bảng con trỏ hàm cuBLAS API được liên kết động
pub(crate) struct CudaCublasApi {
    _handle: *mut c_void,
    pub cublas_create: unsafe extern "C" fn(handle: *mut CublasHandle) -> CublasStatus,
    pub cublas_destroy: unsafe extern "C" fn(handle: CublasHandle) -> CublasStatus,
    pub cublas_sgemv: unsafe extern "C" fn(
        handle: CublasHandle,
        trans: c_int,
        m: c_int,
        n: c_int,
        alpha: *const f32,
        a: CUdeviceptr,
        lda: c_int,
        x: CUdeviceptr,
        incx: c_int,
        beta: *const f32,
        y: CUdeviceptr,
        incy: c_int,
    ) -> CublasStatus,
    pub cublas_sgemm: unsafe extern "C" fn(
        handle: CublasHandle,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        a: CUdeviceptr,
        lda: c_int,
        b: CUdeviceptr,
        ldb: c_int,
        beta: *const f32,
        c: CUdeviceptr,
        ldc: c_int,
    ) -> CublasStatus,
}

unsafe impl Send for CudaCublasApi {}
unsafe impl Sync for CudaCublasApi {}

impl CudaCublasApi {
    fn load() -> Option<Self> {
        unsafe {
            let lib_names: &[&[u8]] = &[
                b"libcublas.so.13\0",
                b"libcublas.so\0",
                b"/opt/cuda/lib64/libcublas.so.13\0",
                b"/opt/cuda/lib64/libcublas.so\0",
            ];
            let mut handle = std::ptr::null_mut();
            for name in lib_names {
                handle = dlopen(name.as_ptr() as *const c_char, RTLD_LAZY | RTLD_LOCAL);
                if !handle.is_null() {
                    break;
                }
            }

            if handle.is_null() {
                return None;
            }

            macro_rules! get_sym {
                ($sym:expr, $ty:ty) => {{
                    let ptr = dlsym(handle, $sym.as_ptr() as *const c_char);
                    if ptr.is_null() {
                        let err_ptr = dlerror();
                        let err_str = if !err_ptr.is_null() {
                            CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
                        } else {
                            "Unknown dlsym error".to_string()
                        };
                        warn!(
                            symbol = ?std::str::from_utf8($sym).unwrap_or("?"),
                            error = %err_str,
                            "Không tìm thấy cuBLAS symbol"
                        );
                        return None;
                    }
                    std::mem::transmute::<*mut c_void, $ty>(ptr)
                }};
            }

            let cublas_create = get_sym!(
                b"cublasCreate_v2\0",
                unsafe extern "C" fn(*mut CublasHandle) -> CublasStatus
            );
            let cublas_destroy = get_sym!(
                b"cublasDestroy_v2\0",
                unsafe extern "C" fn(CublasHandle) -> CublasStatus
            );
            let cublas_sgemv = get_sym!(
                b"cublasSgemv_v2\0",
                unsafe extern "C" fn(
                    CublasHandle,
                    c_int,
                    c_int,
                    c_int,
                    *const f32,
                    CUdeviceptr,
                    c_int,
                    CUdeviceptr,
                    c_int,
                    *const f32,
                    CUdeviceptr,
                    c_int,
                ) -> CublasStatus
            );

            let cublas_sgemm = get_sym!(
                b"cublasSgemm_v2\0",
                unsafe extern "C" fn(
                    CublasHandle,
                    c_int,
                    c_int,
                    c_int,
                    c_int,
                    c_int,
                    *const f32,
                    CUdeviceptr,
                    c_int,
                    CUdeviceptr,
                    c_int,
                    *const f32,
                    CUdeviceptr,
                    c_int,
                ) -> CublasStatus
            );

            Some(Self {
                _handle: handle,
                cublas_create,
                cublas_destroy,
                cublas_sgemv,
                cublas_sgemm,
            })
        }
    }
}

/// GPU Matrix nằm thường trực trong VRAM
pub struct GpuMatrix {
    pub rows: usize,
    pub columns: usize,
    pub d_ptr: CUdeviceptr,
    pub bytes: usize,
    driver: Arc<CudaDriverApi>,
    ctx: CUcontext,
}

unsafe impl Send for GpuMatrix {}
unsafe impl Sync for GpuMatrix {}

impl GpuMatrix {
    pub fn from_matrix(matrix: &crate::backend::Matrix, engine: &GpuComputeEngine) -> Option<Self> {
        let d_ptr = engine.alloc_and_upload(&matrix.values)?;
        let bytes = std::mem::size_of_val(&matrix.values[..]);
        Some(Self {
            rows: matrix.rows,
            columns: matrix.columns,
            d_ptr,
            bytes,
            driver: Arc::clone(&engine.driver),
            ctx: engine.ctx,
        })
    }
}

impl Drop for GpuMatrix {
    fn drop(&mut self) {
        if self.d_ptr != 0 {
            unsafe {
                let _ = (self.driver.cu_ctx_set_current)(self.ctx);
                let _ = (self.driver.cu_mem_free)(self.d_ptr);
                self.d_ptr = 0;
            }
        }
    }
}

struct GpuExecutionState {
    handle: CublasHandle,
    d_scratch_in: CUdeviceptr,
    d_scratch_in_cap: usize,
    d_scratch_out: CUdeviceptr,
    d_scratch_out_cap: usize,
}

/// Bộ thực thi tính toán ma trận GPU sử dụng cuBLAS (Thread-safe)
pub struct GpuComputeEngine {
    driver: Arc<CudaDriverApi>,
    cublas: Arc<CudaCublasApi>,
    pub dev: CUdevice,
    pub ctx: CUcontext,
    state: std::sync::Mutex<GpuExecutionState>,
    circuit_broken: std::sync::atomic::AtomicBool,
}

unsafe impl Send for GpuComputeEngine {}
unsafe impl Sync for GpuComputeEngine {}

impl GpuComputeEngine {
    pub fn init() -> Option<Self> {
        let driver = Arc::new(CudaDriverApi::load()?);
        let cublas = Arc::new(CudaCublasApi::load()?);
        unsafe {
            if (driver.cu_init)(0) != CUDA_SUCCESS {
                return None;
            }
            let mut dev = 0;
            if (driver.cu_device_get)(&mut dev, 0) != CUDA_SUCCESS {
                return None;
            }
            let mut ctx = std::ptr::null_mut();
            if (driver.cu_device_primary_ctx_retain)(&mut ctx, dev) != CUDA_SUCCESS {
                return None;
            }
            if (driver.cu_ctx_set_current)(ctx) != CUDA_SUCCESS {
                let _ = (driver.cu_device_primary_ctx_release)(dev);
                return None;
            }
            let mut handle = std::ptr::null_mut();
            if (cublas.cublas_create)(&mut handle) != CUBLAS_STATUS_SUCCESS {
                let _ = (driver.cu_device_primary_ctx_release)(dev);
                return None;
            }

            let in_cap = 8192 * std::mem::size_of::<f32>();
            let mut d_scratch_in = 0;
            if (driver.cu_mem_alloc)(&mut d_scratch_in, in_cap) != CUDA_SUCCESS {
                let _ = (cublas.cublas_destroy)(handle);
                let _ = (driver.cu_device_primary_ctx_release)(dev);
                return None;
            }
            let out_cap = 65536 * std::mem::size_of::<f32>();
            let mut d_scratch_out = 0;
            if (driver.cu_mem_alloc)(&mut d_scratch_out, out_cap) != CUDA_SUCCESS {
                let _ = (driver.cu_mem_free)(d_scratch_in);
                let _ = (cublas.cublas_destroy)(handle);
                let _ = (driver.cu_device_primary_ctx_release)(dev);
                return None;
            }

            info!("Đã khởi tạo GPU cuBLAS Compute Engine thành công!");
            Some(Self {
                driver,
                cublas,
                dev,
                ctx,
                state: std::sync::Mutex::new(GpuExecutionState {
                    handle,
                    d_scratch_in,
                    d_scratch_in_cap: in_cap,
                    d_scratch_out,
                    d_scratch_out_cap: out_cap,
                }),
                circuit_broken: std::sync::atomic::AtomicBool::new(false),
            })
        }
    }

    pub fn alloc_and_upload(&self, values: &[f32]) -> Option<CUdeviceptr> {
        unsafe {
            let _ = (self.driver.cu_ctx_set_current)(self.ctx);
            let bytes = std::mem::size_of_val(values);
            let mut d_ptr = 0;
            if (self.driver.cu_mem_alloc)(&mut d_ptr, bytes) != CUDA_SUCCESS {
                return None;
            }
            if (self.driver.cu_memcpy_h_to_d)(d_ptr, values.as_ptr() as *const c_void, bytes)
                != CUDA_SUCCESS
            {
                let _ = (self.driver.cu_mem_free)(d_ptr);
                return None;
            }
            Some(d_ptr)
        }
    }

    /// Kiểm tra xem GPU engine có đang khỏe mạnh và sẵn sàng tính toán không
    pub fn is_healthy(&self) -> bool {
        !self
            .circuit_broken
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Kích hoạt Circuit Breaker khi phát hiện lỗi GPU runtime
    pub fn trip_circuit_breaker(&self) {
        self.circuit_broken
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Khôi phục trạng thái Circuit Breaker
    pub fn reset_circuit_breaker(&self) {
        self.circuit_broken
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Tự động mở rộng kích thước d_scratch_in và d_scratch_out khi batch size lớn
    fn ensure_scratch_caps(
        &self,
        state: &mut GpuExecutionState,
        in_bytes: usize,
        out_bytes: usize,
    ) -> Result<(), String> {
        unsafe {
            let _ = (self.driver.cu_ctx_set_current)(self.ctx);
            if in_bytes > state.d_scratch_in_cap {
                let new_cap = in_bytes
                    .next_power_of_two()
                    .max(8192 * std::mem::size_of::<f32>());
                let mut new_ptr = 0;
                let status = (self.driver.cu_mem_alloc)(&mut new_ptr, new_cap);
                if status != CUDA_SUCCESS {
                    self.circuit_broken
                        .store(true, std::sync::atomic::Ordering::SeqCst);
                    return Err(format!(
                        "Lỗi cuMemAlloc mở rộng scratch_in: {} bytes (mã lỗi CUDA: {})",
                        new_cap, status
                    ));
                }
                if state.d_scratch_in != 0 {
                    let _ = (self.driver.cu_mem_free)(state.d_scratch_in);
                }
                state.d_scratch_in = new_ptr;
                state.d_scratch_in_cap = new_cap;
            }
            if out_bytes > state.d_scratch_out_cap {
                let new_cap = out_bytes
                    .next_power_of_two()
                    .max(65536 * std::mem::size_of::<f32>());
                let mut new_ptr = 0;
                let status = (self.driver.cu_mem_alloc)(&mut new_ptr, new_cap);
                if status != CUDA_SUCCESS {
                    self.circuit_broken
                        .store(true, std::sync::atomic::Ordering::SeqCst);
                    return Err(format!(
                        "Lỗi cuMemAlloc mở rộng scratch_out: {} bytes (mã lỗi CUDA: {})",
                        new_cap, status
                    ));
                }
                if state.d_scratch_out != 0 {
                    let _ = (self.driver.cu_mem_free)(state.d_scratch_out);
                }
                state.d_scratch_out = new_ptr;
                state.d_scratch_out_cap = new_cap;
            }
            Ok(())
        }
    }

    /// Thực thi phép nhân ma trận - vector y = A * x trên GPU bằng cuBLAS (Thread-safe)
    pub fn gemv(
        &self,
        matrix: &GpuMatrix,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        if self
            .circuit_broken
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return Err("GPU Circuit Breaker đang ngắt do lỗi trước đó".into());
        }
        if input.len() != matrix.columns {
            return Err("Kích thước vector input không khớp số cột ma trận".into());
        }
        if output.len() != matrix.rows {
            return Err("Kích thước vector output không khớp số hàng ma trận".into());
        }
        if matrix.columns == 0 || matrix.rows == 0 {
            return Ok(());
        }
        let in_bytes = std::mem::size_of_val(input);
        let out_bytes = std::mem::size_of_val(output);
        let mut state = self
            .state
            .lock()
            .map_err(|_| "GpuComputeEngine mutex poisoned".to_string())?;

        unsafe {
            let _ = (self.driver.cu_ctx_set_current)(self.ctx);
        }
        self.ensure_scratch_caps(&mut state, in_bytes, out_bytes)?;
        unsafe {
            if (self.driver.cu_memcpy_h_to_d)(
                state.d_scratch_in,
                input.as_ptr() as *const c_void,
                in_bytes,
            ) != CUDA_SUCCESS
            {
                self.circuit_broken
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err("Lỗi cuMemcpyHtoD scratch_in".into());
            }

            let alpha = 1.0f32;
            let beta = 0.0f32;
            let status = (self.cublas.cublas_sgemv)(
                state.handle,
                CUBLAS_OP_T,
                matrix.columns as c_int,
                matrix.rows as c_int,
                &alpha,
                matrix.d_ptr,
                matrix.columns as c_int,
                state.d_scratch_in,
                1,
                &beta,
                state.d_scratch_out,
                1,
            );
            if status != CUBLAS_STATUS_SUCCESS {
                self.circuit_broken
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err(format!("Lỗi cublasSgemv: {}", status));
            }

            if (self.driver.cu_memcpy_d_to_h)(
                output.as_mut_ptr() as *mut c_void,
                state.d_scratch_out,
                out_bytes,
            ) != CUDA_SUCCESS
            {
                self.circuit_broken
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err("Lỗi cuMemcpyDtoH scratch_out".into());
            }

            if crate::backend::finite(output).is_err() {
                self.circuit_broken
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err("cuBLAS GEMV output chứa giá trị không hợp lệ (NaN/Inf)".into());
            }

            Ok(())
        }
    }

    /// Thực thi tối ưu gộp 3 phép nhân ma trận Attention (Q, K, V) từ cùng một vector đầu vào `hidden`:
    /// 1. Copy `hidden` lên GPU một lần duy nhất (giảm 66% lưu lượng H2D).
    /// 2. Tuần tự thực thi 3 kernel cuBLAS SGEMV cho Q, K, V trên cùng stream.
    /// 3. Copy toàn bộ kết quả ghép [Q; K; V] về Host trong một lần truyền D2H duy nhất.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_qkv(
        &self,
        q: &GpuMatrix,
        k: &GpuMatrix,
        v: &GpuMatrix,
        hidden: &[f32],
        out_buf: &mut [f32],
        query: &mut [f32],
        key: &mut [f32],
        value: &mut [f32],
    ) -> Result<(), String> {
        if self
            .circuit_broken
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return Err("GPU Circuit Breaker đang ngắt do lỗi trước đó".into());
        }
        let d = hidden.len();
        if q.columns != d || k.columns != d || v.columns != d {
            return Err("Kích thước hidden không khớp số cột của ma trận Q/K/V".into());
        }
        if query.len() != q.rows || key.len() != k.rows || value.len() != v.rows {
            return Err("Kích thước output không khớp số hàng của ma trận Q/K/V".into());
        }
        let total_out_len = q.rows + k.rows + v.rows;
        if out_buf.len() < total_out_len {
            return Err("out_buf không đủ kích thước chứa kết quả ghép QKV".into());
        }

        let in_bytes = std::mem::size_of_val(hidden);
        let total_out_bytes = total_out_len * std::mem::size_of::<f32>();

        let mut state = self
            .state
            .lock()
            .map_err(|_| "GpuComputeEngine mutex poisoned".to_string())?;

        unsafe {
            let _ = (self.driver.cu_ctx_set_current)(self.ctx);
        }
        self.ensure_scratch_caps(&mut state, in_bytes, total_out_bytes)?;

        unsafe {
            // 1. Copy H2D hidden một lần duy nhất
            if (self.driver.cu_memcpy_h_to_d)(
                state.d_scratch_in,
                hidden.as_ptr() as *const c_void,
                in_bytes,
            ) != CUDA_SUCCESS
            {
                self.circuit_broken
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err("Lỗi cuMemcpyHtoD scratch_in QKV".into());
            }

            let alpha = 1.0f32;
            let beta = 0.0f32;

            let d_out_q = state.d_scratch_out;
            let d_out_k = state.d_scratch_out + (q.rows * std::mem::size_of::<f32>()) as u64;
            let d_out_v =
                state.d_scratch_out + ((q.rows + k.rows) * std::mem::size_of::<f32>()) as u64;

            // 2. Chạy 3 phép SGEMV tuần tự trên cùng CUDA stream
            let status_q = (self.cublas.cublas_sgemv)(
                state.handle,
                CUBLAS_OP_T,
                q.columns as c_int,
                q.rows as c_int,
                &alpha,
                q.d_ptr,
                q.columns as c_int,
                state.d_scratch_in,
                1,
                &beta,
                d_out_q,
                1,
            );
            if status_q != CUBLAS_STATUS_SUCCESS {
                self.circuit_broken
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err(format!("Lỗi cublasSgemv Q: {}", status_q));
            }

            let status_k = (self.cublas.cublas_sgemv)(
                state.handle,
                CUBLAS_OP_T,
                k.columns as c_int,
                k.rows as c_int,
                &alpha,
                k.d_ptr,
                k.columns as c_int,
                state.d_scratch_in,
                1,
                &beta,
                d_out_k,
                1,
            );
            if status_k != CUBLAS_STATUS_SUCCESS {
                self.circuit_broken
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err(format!("Lỗi cublasSgemv K: {}", status_k));
            }

            let status_v = (self.cublas.cublas_sgemv)(
                state.handle,
                CUBLAS_OP_T,
                v.columns as c_int,
                v.rows as c_int,
                &alpha,
                v.d_ptr,
                v.columns as c_int,
                state.d_scratch_in,
                1,
                &beta,
                d_out_v,
                1,
            );
            if status_v != CUBLAS_STATUS_SUCCESS {
                self.circuit_broken
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err(format!("Lỗi cublasSgemv V: {}", status_v));
            }

            // 3. Copy D2H toàn bộ vùng đệm ghép 1 lần duy nhất
            if (self.driver.cu_memcpy_d_to_h)(
                out_buf.as_mut_ptr() as *mut c_void,
                state.d_scratch_out,
                total_out_bytes,
            ) != CUDA_SUCCESS
            {
                self.circuit_broken
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err("Lỗi cuMemcpyDtoH scratch_out QKV".into());
            }

            query.copy_from_slice(&out_buf[..q.rows]);
            key.copy_from_slice(&out_buf[q.rows..q.rows + k.rows]);
            value.copy_from_slice(&out_buf[q.rows + k.rows..total_out_len]);

            if crate::backend::finite(query).is_err()
                || crate::backend::finite(key).is_err()
                || crate::backend::finite(value).is_err()
            {
                self.circuit_broken
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err("cuBLAS GEMV QKV output chứa giá trị không hợp lệ (NaN/Inf)".into());
            }

            Ok(())
        }
    }

    /// Thực thi phép nhân ma trận - ma trận Y = X * W^T trên GPU bằng cuBLAS SGEMM (Row-major)
    /// - `matrix`: Ma trận trọng số W (kích thước rows x columns = M x K)
    /// - `batch_input`: Dữ liệu đầu vào X (kích thước batch_size x columns = B x K)
    /// - `batch_output`: Dữ liệu đầu ra Y (kích thước batch_size x rows = B x M)
    /// - `batch_size`: Số lượng tokens (B)
    pub fn gemm(
        &self,
        matrix: &GpuMatrix,
        batch_input: &[f32],
        batch_output: &mut [f32],
        batch_size: usize,
    ) -> Result<(), String> {
        if self
            .circuit_broken
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return Err("GPU Circuit Breaker đang ngắt do lỗi trước đó".into());
        }
        if batch_size == 0 || matrix.columns == 0 || matrix.rows == 0 {
            return Ok(());
        }
        if batch_input.len() != batch_size * matrix.columns {
            return Err("Kích thước batch_input không khớp batch_size * matrix.columns".into());
        }
        if batch_output.len() != batch_size * matrix.rows {
            return Err("Kích thước batch_output không khớp batch_size * matrix.rows".into());
        }

        let in_bytes = std::mem::size_of_val(batch_input);
        let out_bytes = std::mem::size_of_val(batch_output);
        let mut state = self
            .state
            .lock()
            .map_err(|_| "GpuComputeEngine mutex poisoned".to_string())?;

        unsafe {
            let _ = (self.driver.cu_ctx_set_current)(self.ctx);
        }
        self.ensure_scratch_caps(&mut state, in_bytes, out_bytes)?;

        unsafe {
            if (self.driver.cu_memcpy_h_to_d)(
                state.d_scratch_in,
                batch_input.as_ptr() as *const c_void,
                in_bytes,
            ) != CUDA_SUCCESS
            {
                self.circuit_broken
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err("Lỗi cuMemcpyHtoD scratch_in GEMM".into());
            }

            let alpha = 1.0f32;
            let beta = 0.0f32;
            let status = (self.cublas.cublas_sgemm)(
                state.handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                matrix.rows as c_int,
                batch_size as c_int,
                matrix.columns as c_int,
                &alpha,
                matrix.d_ptr,
                matrix.columns as c_int,
                state.d_scratch_in,
                matrix.columns as c_int,
                &beta,
                state.d_scratch_out,
                matrix.rows as c_int,
            );
            if status != CUBLAS_STATUS_SUCCESS {
                self.circuit_broken
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err(format!("Lỗi cublasSgemm: {}", status));
            }

            if (self.driver.cu_memcpy_d_to_h)(
                batch_output.as_mut_ptr() as *mut c_void,
                state.d_scratch_out,
                out_bytes,
            ) != CUDA_SUCCESS
            {
                self.circuit_broken
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err("Lỗi cuMemcpyDtoH scratch_out GEMM".into());
            }

            if crate::backend::finite(batch_output).is_err() {
                self.circuit_broken
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err("cuBLAS GEMM output chứa giá trị không hợp lệ (NaN/Inf)".into());
            }

            Ok(())
        }
    }

    /// Thực thi phép nhân ma trận Attention (Q, K, V) cho cả batch đầu vào:
    /// 1. Copy toàn bộ `batch_hidden` (kích thước B x D) lên GPU trong 1 lần H2D.
    /// 2. Tuần tự thực thi 3 kernel cuBLAS SGEMM cho Q, K, V trên cùng CUDA stream.
    /// 3. Copy các mảng kết quả về Host qua D2H.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv(
        &self,
        q: &GpuMatrix,
        k: &GpuMatrix,
        v: &GpuMatrix,
        batch_hidden: &[f32],
        batch_size: usize,
        all_q: &mut [f32],
        all_k: &mut [f32],
        all_v: &mut [f32],
    ) -> Result<(), String> {
        if self
            .circuit_broken
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return Err("GPU Circuit Breaker đang ngắt do lỗi trước đó".into());
        }
        if batch_size == 0 {
            return Ok(());
        }
        let d = q.columns;
        if k.columns != d || v.columns != d {
            return Err("Kích thước số cột của ma trận Q/K/V không đồng nhất".into());
        }
        if batch_hidden.len() != batch_size * d {
            return Err("Kích thước batch_hidden không khớp batch_size * d".into());
        }
        if all_q.len() != batch_size * q.rows
            || all_k.len() != batch_size * k.rows
            || all_v.len() != batch_size * v.rows
        {
            return Err("Kích thước all_q / all_k / all_v không khớp batch_size * rows".into());
        }

        let q_floats = batch_size * q.rows;
        let k_floats = batch_size * k.rows;
        let v_floats = batch_size * v.rows;
        let total_out_floats = q_floats + k_floats + v_floats;
        let in_bytes = std::mem::size_of_val(batch_hidden);
        let total_out_bytes = total_out_floats * std::mem::size_of::<f32>();

        let mut state = self
            .state
            .lock()
            .map_err(|_| "GpuComputeEngine mutex poisoned".to_string())?;

        unsafe {
            let _ = (self.driver.cu_ctx_set_current)(self.ctx);
        }
        self.ensure_scratch_caps(&mut state, in_bytes, total_out_bytes)?;

        unsafe {
            // 1. Copy H2D batch_hidden một lần duy nhất
            if (self.driver.cu_memcpy_h_to_d)(
                state.d_scratch_in,
                batch_hidden.as_ptr() as *const c_void,
                in_bytes,
            ) != CUDA_SUCCESS
            {
                self.circuit_broken
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err("Lỗi cuMemcpyHtoD scratch_in QKV GEMM".into());
            }

            let alpha = 1.0f32;
            let beta = 0.0f32;

            let d_out_q = state.d_scratch_out;
            let d_out_k = state.d_scratch_out + (q_floats * std::mem::size_of::<f32>()) as u64;
            let d_out_v =
                state.d_scratch_out + ((q_floats + k_floats) * std::mem::size_of::<f32>()) as u64;

            // 2. Chạy 3 phép SGEMM trên cùng CUDA stream
            let status_q = (self.cublas.cublas_sgemm)(
                state.handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                q.rows as c_int,
                batch_size as c_int,
                d as c_int,
                &alpha,
                q.d_ptr,
                d as c_int,
                state.d_scratch_in,
                d as c_int,
                &beta,
                d_out_q,
                q.rows as c_int,
            );
            if status_q != CUBLAS_STATUS_SUCCESS {
                self.circuit_broken
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err(format!("Lỗi cublasSgemm Q: {}", status_q));
            }

            let status_k = (self.cublas.cublas_sgemm)(
                state.handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                k.rows as c_int,
                batch_size as c_int,
                d as c_int,
                &alpha,
                k.d_ptr,
                d as c_int,
                state.d_scratch_in,
                d as c_int,
                &beta,
                d_out_k,
                k.rows as c_int,
            );
            if status_k != CUBLAS_STATUS_SUCCESS {
                self.circuit_broken
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err(format!("Lỗi cublasSgemm K: {}", status_k));
            }

            let status_v = (self.cublas.cublas_sgemm)(
                state.handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                v.rows as c_int,
                batch_size as c_int,
                d as c_int,
                &alpha,
                v.d_ptr,
                d as c_int,
                state.d_scratch_in,
                d as c_int,
                &beta,
                d_out_v,
                v.rows as c_int,
            );
            if status_v != CUBLAS_STATUS_SUCCESS {
                self.circuit_broken
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err(format!("Lỗi cublasSgemm V: {}", status_v));
            }

            // 3. Copy D2H các mảng kết quả
            if (self.driver.cu_memcpy_d_to_h)(
                all_q.as_mut_ptr() as *mut c_void,
                d_out_q,
                q_floats * std::mem::size_of::<f32>(),
            ) != CUDA_SUCCESS
            {
                self.circuit_broken
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err("Lỗi cuMemcpyDtoH all_q GEMM".into());
            }

            if (self.driver.cu_memcpy_d_to_h)(
                all_k.as_mut_ptr() as *mut c_void,
                d_out_k,
                k_floats * std::mem::size_of::<f32>(),
            ) != CUDA_SUCCESS
            {
                self.circuit_broken
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err("Lỗi cuMemcpyDtoH all_k GEMM".into());
            }

            if (self.driver.cu_memcpy_d_to_h)(
                all_v.as_mut_ptr() as *mut c_void,
                d_out_v,
                v_floats * std::mem::size_of::<f32>(),
            ) != CUDA_SUCCESS
            {
                self.circuit_broken
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err("Lỗi cuMemcpyDtoH all_v GEMM".into());
            }

            if crate::backend::finite(all_q).is_err()
                || crate::backend::finite(all_k).is_err()
                || crate::backend::finite(all_v).is_err()
            {
                self.circuit_broken
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                return Err("cuBLAS GEMM QKV output chứa giá trị không hợp lệ (NaN/Inf)".into());
            }

            Ok(())
        }
    }

    /// Nạp một expert INT8 (weights + scales) lên VRAM
    pub fn upload_expert(
        &self,
        layer: usize,
        expert_id: usize,
        weights: &[u8],
        scales: &[f32],
    ) -> Option<GpuExpertWeights> {
        let weight_bytes = weights.len();
        let scale_bytes = std::mem::size_of_val(scales);
        unsafe {
            let _ = (self.driver.cu_ctx_set_current)(self.ctx);
            let mut d_weights = 0;
            if (self.driver.cu_mem_alloc)(&mut d_weights, weight_bytes) != CUDA_SUCCESS {
                return None;
            }
            if (self.driver.cu_memcpy_h_to_d)(
                d_weights,
                weights.as_ptr() as *const c_void,
                weight_bytes,
            ) != CUDA_SUCCESS
            {
                let _ = (self.driver.cu_mem_free)(d_weights);
                return None;
            }

            let mut d_scales = 0;
            if (self.driver.cu_mem_alloc)(&mut d_scales, scale_bytes) != CUDA_SUCCESS {
                let _ = (self.driver.cu_mem_free)(d_weights);
                return None;
            }
            if (self.driver.cu_memcpy_h_to_d)(
                d_scales,
                scales.as_ptr() as *const c_void,
                scale_bytes,
            ) != CUDA_SUCCESS
            {
                let _ = (self.driver.cu_mem_free)(d_scales);
                let _ = (self.driver.cu_mem_free)(d_weights);
                return None;
            }

            Some(GpuExpertWeights {
                layer,
                expert_id,
                d_weights,
                d_scales,
                weight_bytes,
                scale_bytes,
                pool_slot: None,
                driver: Arc::clone(&self.driver),
                ctx: self.ctx,
            })
        }
    }

    /// Tải dữ liệu expert từ VRAM về Host RAM qua PCIe DMA tốc độ cao
    pub fn download_expert(
        &self,
        expert: &GpuExpertWeights,
        weights_out: &mut [u8],
        scales_out: &mut [f32],
    ) -> Result<(), String> {
        if weights_out.len() != expert.weight_bytes {
            return Err("Kích thước weights_out không khớp".into());
        }
        let scale_bytes = std::mem::size_of_val(scales_out);
        if scale_bytes != expert.scale_bytes {
            return Err("Kích thước scales_out không khớp".into());
        }
        unsafe {
            let _ = (self.driver.cu_ctx_set_current)(self.ctx);
            if (self.driver.cu_memcpy_d_to_h)(
                weights_out.as_mut_ptr() as *mut c_void,
                expert.d_weights,
                expert.weight_bytes,
            ) != CUDA_SUCCESS
            {
                return Err("cuMemcpyDtoH d_weights thất bại".into());
            }
            if (self.driver.cu_memcpy_d_to_h)(
                scales_out.as_mut_ptr() as *mut c_void,
                expert.d_scales,
                expert.scale_bytes,
            ) != CUDA_SUCCESS
            {
                return Err("cuMemcpyDtoH d_scales thất bại".into());
            }
            Ok(())
        }
    }
}

impl Drop for GpuComputeEngine {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.driver.cu_ctx_set_current)(self.ctx);
            if let Ok(mut state) = self.state.lock() {
                if state.d_scratch_in != 0 {
                    let _ = (self.driver.cu_mem_free)(state.d_scratch_in);
                    state.d_scratch_in = 0;
                }
                if state.d_scratch_out != 0 {
                    let _ = (self.driver.cu_mem_free)(state.d_scratch_out);
                    state.d_scratch_out = 0;
                }
                if !state.handle.is_null() {
                    let _ = (self.cublas.cublas_destroy)(state.handle);
                    state.handle = std::ptr::null_mut();
                }
            }
            let _ = (self.driver.cu_device_primary_ctx_release)(self.dev);
        }
    }
}

/// Lưu trữ các tầng Dense Attention của 1 layer trên VRAM (Router tính trên CPU để tối ưu PCIe)
pub struct LayerDenseGpu {
    pub q_proj: GpuMatrix,
    pub k_proj: GpuMatrix,
    pub v_proj: GpuMatrix,
    pub o_proj: GpuMatrix,
}

/// Bộ đệm máy chủ được ghim (Pinned Host Memory) — Sub-step 16-C
///
/// `cuMemHostAlloc` cấp phát bộ nhớ page-locked trên RAM máy chủ. CUDA DMA engine
/// có thể chuyển dữ liệu trực tiếp từ vùng nhớ này sang VRAM mà không cần bounce
/// buffer trung gian, nhân đôi băng thông PCIe hiệu quả (từ ~6 GB/s lên ~12 GB/s).
pub struct PinnedBuffer {
    ptr: *mut c_void,
    pub len: usize,
    driver: Arc<CudaDriverApi>,
}

unsafe impl Send for PinnedBuffer {}
unsafe impl Sync for PinnedBuffer {}

impl PinnedBuffer {
    /// Cấp phát `len` byte trong vùng nhớ pinned. Trả về `None` nếu GPU không sẵn sàng
    /// hoặc vượt quá bộ nhớ hệ thống.
    pub fn alloc(engine: &GpuComputeEngine, len: usize) -> Option<Self> {
        unsafe {
            let _ = (engine.driver.cu_ctx_set_current)(engine.ctx);
            let mut ptr: *mut c_void = std::ptr::null_mut();
            // CU_MEMHOSTALLOC_PORTABLE (0x01) | CU_MEMHOSTALLOC_DEVICEMAP (0x02)
            if (engine.driver.cu_mem_host_alloc)(&mut ptr, len, 0x01) != CUDA_SUCCESS {
                return None;
            }
            Some(Self {
                ptr,
                len,
                driver: Arc::clone(&engine.driver),
            })
        }
    }

    /// Truy cập slice có thể ghi được vào vùng nhớ pinned.
    /// # Safety
    /// Caller phải đảm bảo không có GPU kernel nào đang đọc/ghi vùng nhớ này đồng thời.
    pub unsafe fn as_mut_slice<T>(&mut self) -> &mut [T] {
        assert_eq!(self.len % std::mem::size_of::<T>(), 0);
        assert_eq!(
            (self.ptr as usize) % std::mem::align_of::<T>(),
            0,
            "PinnedBuffer con trỏ không căn chỉnh theo alignment của T"
        );
        std::slice::from_raw_parts_mut(self.ptr as *mut T, self.len / std::mem::size_of::<T>())
    }

    /// Truy cập slice chỉ đọc vào vùng nhớ pinned.
    /// # Safety
    /// Caller phải đảm bảo không có GPU kernel nào đang ghi vào vùng nhớ này đồng thời.
    pub unsafe fn as_slice<T>(&self) -> &[T] {
        assert_eq!(self.len % std::mem::size_of::<T>(), 0);
        assert_eq!(
            (self.ptr as usize) % std::mem::align_of::<T>(),
            0,
            "PinnedBuffer con trỏ không căn chỉnh theo alignment của T"
        );
        std::slice::from_raw_parts(self.ptr as *const T, self.len / std::mem::size_of::<T>())
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.ptr
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = (self.driver.cu_mem_free_host)(self.ptr);
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// VRAM Memory Pool — Sub-step 16-A
///
/// Cấp phát **1 block VRAM lớn duy nhất** ngay khi khởi động (thay vì gọi `cuMemAlloc`
/// mỗi lần upload expert). Block được chia thành `max_slots` slot có kích thước cố định
/// `slot_bytes` bằng pointer arithmetic. Khi evict expert, slot được trả lại pool để
/// tái sử dụng ngay lập tức — không cần `cuMemFree`/`cuMemAlloc` lại.
///
/// Lợi ích: Giảm số lần gọi `cuMemAlloc` từ **N × 2 = hàng trăm** lần xuống còn
/// **1 lần duy nhất**, loại bỏ overhead đồng bộ kernel ~7.5s ở cold start.
pub struct GpuExpertPool {
    base_weights: CUdeviceptr, // Một block lớn cho tất cả weight slots
    base_scales: CUdeviceptr,  // Một block lớn cho tất cả scale slots
    weight_slot_bytes: usize,
    scale_slot_bytes: usize,
    pub max_slots: usize,
    free_slots: std::collections::VecDeque<usize>,
    slot_in_use: Vec<bool>,
    driver: Arc<CudaDriverApi>,
    ctx: CUcontext,
}

unsafe impl Send for GpuExpertPool {}
unsafe impl Sync for GpuExpertPool {}

impl GpuExpertPool {
    /// Khởi tạo pool với `max_slots` slot, mỗi slot chứa `weight_slot_bytes` byte weight
    /// và `scale_slot_bytes` byte scale. Trả về `None` nếu không đủ VRAM.
    pub fn new(
        engine: &GpuComputeEngine,
        max_slots: usize,
        weight_slot_bytes: usize,
        scale_slot_bytes: usize,
    ) -> Option<Self> {
        if max_slots == 0 {
            return None;
        }
        // Đảm bảo kích thước mỗi slot căn chỉnh theo bội số 256 byte để thỏa mãn alignment phần cứng GPU
        let weight_slot_bytes = (weight_slot_bytes + 255) & !255;
        let scale_slot_bytes = (scale_slot_bytes + 255) & !255;
        let total_weight_bytes = max_slots * weight_slot_bytes;
        let total_scale_bytes = max_slots * scale_slot_bytes;
        unsafe {
            let _ = (engine.driver.cu_ctx_set_current)(engine.ctx);
            let mut base_weights = 0u64;
            if (engine.driver.cu_mem_alloc)(&mut base_weights, total_weight_bytes) != CUDA_SUCCESS {
                return None;
            }
            let mut base_scales = 0u64;
            if (engine.driver.cu_mem_alloc)(&mut base_scales, total_scale_bytes) != CUDA_SUCCESS {
                let _ = (engine.driver.cu_mem_free)(base_weights);
                return None;
            }
            let free_slots = (0..max_slots).collect();
            let slot_in_use = vec![false; max_slots];
            Some(Self {
                base_weights,
                base_scales,
                weight_slot_bytes,
                scale_slot_bytes,
                max_slots,
                free_slots,
                slot_in_use,
                driver: Arc::clone(&engine.driver),
                ctx: engine.ctx,
            })
        }
    }

    /// Số slot còn rảnh trong pool.
    pub fn available_slots(&self) -> usize {
        self.free_slots.len()
    }

    /// Lấy địa chỉ VRAM của weight slot `i`.
    fn weight_ptr(&self, slot: usize) -> CUdeviceptr {
        self.base_weights + (slot * self.weight_slot_bytes) as u64
    }

    /// Lấy địa chỉ VRAM của scale slot `i`.
    fn scale_ptr(&self, slot: usize) -> CUdeviceptr {
        self.base_scales + (slot * self.scale_slot_bytes) as u64
    }

    /// Upload expert lên một slot rảnh bằng DMA, trả về slot index đã dùng.
    /// Dữ liệu nguồn `weights_src` nên là pinned memory để đạt băng thông tối đa.
    ///
    /// # Safety
    /// Caller phải đảm bảo `weights_src` hợp lệ cho việc đọc ít nhất `weight_bytes` byte,
    /// và `scales_src` hợp lệ cho việc đọc ít nhất `scale_bytes` byte.
    pub unsafe fn upload(
        &mut self,
        weights_src: *const c_void,
        weight_bytes: usize,
        scales_src: *const c_void,
        scale_bytes: usize,
    ) -> Option<usize> {
        if weight_bytes > self.weight_slot_bytes || scale_bytes > self.scale_slot_bytes {
            return None;
        }
        let slot = self.free_slots.pop_front()?;
        self.slot_in_use[slot] = true;
        let _ = (self.driver.cu_ctx_set_current)(self.ctx);
        if (self.driver.cu_memcpy_h_to_d)(self.weight_ptr(slot), weights_src, weight_bytes)
            != CUDA_SUCCESS
        {
            self.slot_in_use[slot] = false;
            self.free_slots.push_front(slot);
            return None;
        }
        if (self.driver.cu_memcpy_h_to_d)(self.scale_ptr(slot), scales_src, scale_bytes)
            != CUDA_SUCCESS
        {
            self.slot_in_use[slot] = false;
            self.free_slots.push_front(slot);
            return None;
        }
        Some(slot)
    }

    /// Trả slot về pool để tái sử dụng (không gọi cuMemFree).
    /// Kiểm tra trạng thái slot_in_use để chống double-free và memory aliasing.
    pub fn free_slot(&mut self, slot: usize) {
        if slot < self.max_slots && self.slot_in_use[slot] {
            self.slot_in_use[slot] = false;
            self.free_slots.push_back(slot);
        }
    }

    /// Địa chỉ VRAM weight của slot đã cấp phát (dùng cho cuBLAS GEMV hoặc download).
    pub fn weight_device_ptr(&self, slot: usize) -> CUdeviceptr {
        self.weight_ptr(slot)
    }

    /// Địa chỉ VRAM scale của slot đã cấp phát.
    pub fn scale_device_ptr(&self, slot: usize) -> CUdeviceptr {
        self.scale_ptr(slot)
    }

    /// Download expert từ VRAM về host memory.
    ///
    /// # Safety
    /// Caller phải đảm bảo `weights_dst` có thể ghi ít nhất `weight_bytes` byte,
    /// và `scales_dst` có thể ghi ít nhất `scale_bytes` byte.
    pub unsafe fn download(
        &self,
        slot: usize,
        weights_dst: *mut c_void,
        weight_bytes: usize,
        scales_dst: *mut c_void,
        scale_bytes: usize,
    ) -> Result<(), String> {
        let _ = (self.driver.cu_ctx_set_current)(self.ctx);
        if (self.driver.cu_memcpy_d_to_h)(weights_dst, self.weight_ptr(slot), weight_bytes)
            != CUDA_SUCCESS
        {
            return Err("cuMemcpyDtoH weights thất bại".into());
        }
        if (self.driver.cu_memcpy_d_to_h)(scales_dst, self.scale_ptr(slot), scale_bytes)
            != CUDA_SUCCESS
        {
            return Err("cuMemcpyDtoH scales thất bại".into());
        }
        Ok(())
    }
}

impl Drop for GpuExpertPool {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.driver.cu_ctx_set_current)(self.ctx);
            if self.base_weights != 0 {
                let _ = (self.driver.cu_mem_free)(self.base_weights);
            }
            if self.base_scales != 0 {
                let _ = (self.driver.cu_mem_free)(self.base_scales);
            }
        }
    }
}

/// Trọng số một Expert INT8 được lưu trực tiếp trên GPU VRAM (L1 VRAM Expert Cache)
pub struct GpuExpertWeights {
    pub layer: usize,
    pub expert_id: usize,
    pub d_weights: CUdeviceptr,
    pub d_scales: CUdeviceptr,
    pub weight_bytes: usize,
    pub scale_bytes: usize,
    /// Slot index trong GpuExpertPool nếu được cấp phát từ pool (16-A)
    pub pool_slot: Option<usize>,
    driver: Arc<CudaDriverApi>,
    ctx: CUcontext,
}

unsafe impl Send for GpuExpertWeights {}
unsafe impl Sync for GpuExpertWeights {}

impl Drop for GpuExpertWeights {
    fn drop(&mut self) {
        // Nếu được cấp phát từ pool, bộ nhớ thuộc về base_weights/base_scales của pool,
        // không được gọi cu_mem_free (16-A)
        if self.pool_slot.is_some() {
            return;
        }
        unsafe {
            let _ = (self.driver.cu_ctx_set_current)(self.ctx);
            if self.d_weights != 0 {
                let _ = (self.driver.cu_mem_free)(self.d_weights);
            }
            if self.d_scales != 0 {
                let _ = (self.driver.cu_mem_free)(self.d_scales);
            }
        }
    }
}

/// Bộ đệm chuyên gia L1 trên GPU VRAM (L1 VRAM Expert Cache)
pub struct GpuExpertCache {
    pub max_bytes: usize,
    pub used_bytes: usize,
    pub entries: std::collections::HashMap<(usize, usize), GpuExpertWeights>,
    pub lru: std::collections::VecDeque<(usize, usize)>,
}

impl GpuExpertCache {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            used_bytes: 0,
            entries: std::collections::HashMap::new(),
            lru: std::collections::VecDeque::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, layer: usize, expert: usize) -> bool {
        self.entries.contains_key(&(layer, expert))
    }

    pub fn get(&mut self, layer: usize, expert: usize) -> Option<&GpuExpertWeights> {
        let key = (layer, expert);
        if self.entries.contains_key(&key) {
            if let Some(pos) = self.lru.iter().position(|k| *k == key) {
                self.lru.remove(pos);
            }
            self.lru.push_back(key);
            self.entries.get(&key)
        } else {
            None
        }
    }

    pub fn insert_with_pool(
        &mut self,
        weights: GpuExpertWeights,
        mut pool: Option<&mut GpuExpertPool>,
    ) -> usize {
        let expert_bytes = weights.weight_bytes + weights.scale_bytes;
        if expert_bytes > self.max_bytes {
            if let (Some(p), Some(slot)) = (pool.as_mut(), weights.pool_slot) {
                p.free_slot(slot);
            }
            return 0;
        }
        let key = (weights.layer, weights.expert_id);
        let mut evicted_count = 0;

        if let Some(old_exp) = self.entries.remove(&key) {
            let old_bytes = old_exp.weight_bytes + old_exp.scale_bytes;
            self.used_bytes = self.used_bytes.saturating_sub(old_bytes);
            if let Some(pos) = self.lru.iter().position(|k| *k == key) {
                self.lru.remove(pos);
            }
            if let (Some(p), Some(slot)) = (pool.as_mut(), old_exp.pool_slot) {
                p.free_slot(slot);
            }
        }

        while self.used_bytes + expert_bytes > self.max_bytes && !self.lru.is_empty() {
            if let Some(victim_key) = self.lru.pop_front() {
                if let Some(victim) = self.entries.remove(&victim_key) {
                    let v_bytes = victim.weight_bytes + victim.scale_bytes;
                    self.used_bytes = self.used_bytes.saturating_sub(v_bytes);
                    if let (Some(p), Some(slot)) = (pool.as_mut(), victim.pool_slot) {
                        p.free_slot(slot);
                    }
                    evicted_count += 1;
                }
            }
        }

        if self.used_bytes + expert_bytes > self.max_bytes {
            if let (Some(p), Some(slot)) = (pool.as_mut(), weights.pool_slot) {
                p.free_slot(slot);
            }
            return evicted_count;
        }

        self.used_bytes += expert_bytes;
        self.lru.push_back(key);
        self.entries.insert(key, weights);
        evicted_count
    }

    pub fn insert(&mut self, weights: GpuExpertWeights) -> usize {
        self.insert_with_pool(weights, None)
    }

    /// Remove a cached expert after its synchronous transfer has completed.
    /// The caller exclusively borrows the cache/pool, so no slot can be reused
    /// while another cache lookup still borrows this entry.
    pub(crate) fn remove_with_pool(
        &mut self,
        layer: usize,
        expert: usize,
        pool: Option<&mut GpuExpertPool>,
    ) -> bool {
        let key = (layer, expert);
        let Some(entry) = self.entries.remove(&key) else {
            return false;
        };
        self.used_bytes = self
            .used_bytes
            .saturating_sub(entry.weight_bytes + entry.scale_bytes);
        if let Some(position) = self.lru.iter().position(|item| *item == key) {
            self.lru.remove(position);
        }
        if let (Some(pool), Some(slot)) = (pool, entry.pool_slot) {
            pool.free_slot(slot);
        }
        true
    }

    /// Xóa toàn bộ chuyên gia trong L1 VRAM Expert Cache và hoàn trả slot về Pool
    pub fn clear(&mut self, mut pool: Option<&mut GpuExpertPool>) {
        for (_, entry) in self.entries.drain() {
            if let (Some(p), Some(slot)) = (pool.as_mut(), entry.pool_slot) {
                p.free_slot(slot);
            }
        }
        self.lru.clear();
        self.used_bytes = 0;
    }
}

/// Toàn bộ cấu trúc Dense Offload và L1 Expert Cache trên VRAM.
/// Chú ý thứ tự khai báo: `engine` phải được khai báo CUỐI CÙNG để khi Drop,
/// toàn bộ các buffer VRAM (layers, lm_head, expert_cache, expert_pool, pinned_upload_buf)
/// được giải phóng (cuMemFree) TRƯỚC KHI CUDA Primary Context bị giải phóng trong `engine.drop()`.
pub struct GpuDenseStorage {
    pub layers: Vec<LayerDenseGpu>,
    pub lm_head: GpuMatrix,
    pub total_allocated_bytes: usize,
    pub expert_cache: GpuExpertCache,
    /// Pool VRAM pre-allocated — Sub-step 16-A: 1 cuMemAlloc thay cho N×2 lần (16-A)
    pub expert_pool: Option<GpuExpertPool>,
    /// Pinned host buffer tái sử dụng cho upload expert — Sub-step 16-C
    pub pinned_upload_buf: Option<PinnedBuffer>,
    pub engine: GpuComputeEngine,
}

impl GpuDenseStorage {
    /// Nạp và thăng hạng expert lên VRAM sử dụng Memory Pool và Pinned Memory buffer (16-A + 16-C)
    pub fn upload_and_cache(
        &mut self,
        layer_id: usize,
        expert_id: usize,
        weights: &[u8],
        scales: &[f32],
    ) -> bool {
        if self.expert_cache.contains(layer_id, expert_id) {
            return false;
        }

        let weight_bytes = weights.len();
        let scale_bytes = std::mem::size_of_val(scales);

        // 1. Thử dùng GpuExpertPool trước (16-A)
        if let Some(pool) = self.expert_pool.as_mut() {
            // Nếu pool hết slot rảnh, kích hoạt evict từ cache để lấy slot
            if pool.available_slots() == 0 {
                if let Some(victim_key) = self.expert_cache.lru.pop_front() {
                    if let Some(victim) = self.expert_cache.entries.remove(&victim_key) {
                        let v_bytes = victim.weight_bytes + victim.scale_bytes;
                        self.expert_cache.used_bytes =
                            self.expert_cache.used_bytes.saturating_sub(v_bytes);
                        if let Some(slot) = victim.pool_slot {
                            pool.free_slot(slot);
                        }
                    }
                }
            }

            // Dùng PinnedBuffer nếu có để DMA transfer đạt tốc độ tối đa (16-C)
            let (weights_ptr, scales_ptr) = if let Some(pinned) = self.pinned_upload_buf.as_mut() {
                let aligned_weight_bytes = (weight_bytes + 255) & !255;
                let total_needed = aligned_weight_bytes + scale_bytes;
                if pinned.len >= total_needed {
                    unsafe {
                        let slice = pinned.as_mut_slice::<u8>();
                        slice[..weight_bytes].copy_from_slice(weights);
                        let scales_u8 =
                            std::slice::from_raw_parts(scales.as_ptr() as *const u8, scale_bytes);
                        slice[aligned_weight_bytes..aligned_weight_bytes + scale_bytes]
                            .copy_from_slice(scales_u8);
                        (pinned.as_ptr(), pinned.as_ptr().add(aligned_weight_bytes))
                    }
                } else {
                    (
                        weights.as_ptr() as *const c_void,
                        scales.as_ptr() as *const c_void,
                    )
                }
            } else {
                (
                    weights.as_ptr() as *const c_void,
                    scales.as_ptr() as *const c_void,
                )
            };

            let maybe_slot =
                unsafe { pool.upload(weights_ptr, weight_bytes, scales_ptr, scale_bytes) };
            if let Some(slot) = maybe_slot {
                let exp_weights = GpuExpertWeights {
                    layer: layer_id,
                    expert_id,
                    d_weights: pool.weight_device_ptr(slot),
                    d_scales: pool.scale_device_ptr(slot),
                    weight_bytes,
                    scale_bytes,
                    pool_slot: Some(slot),
                    driver: Arc::clone(&self.engine.driver),
                    ctx: self.engine.ctx,
                };
                self.expert_cache
                    .insert_with_pool(exp_weights, self.expert_pool.as_mut());
                return true;
            }
        }

        // 2. Fallback sang upload thông thường nếu không có pool
        if let Some(uploaded) = self
            .engine
            .upload_expert(layer_id, expert_id, weights, scales)
        {
            self.expert_cache.insert(uploaded);
            return true;
        }

        false
    }

    /// Xóa toàn bộ chuyên gia đang lưu trong VRAM Expert Cache và hoàn trả slot về Pool
    pub fn clear_expert_cache(&mut self) {
        self.expert_cache.clear(self.expert_pool.as_mut());
    }
}

/// Trình quản lý tầng phần cứng GPU (Thread-safe)
pub struct GpuTierManager {
    _api: Option<CudaDriverApi>,
    device_info: Option<GpuDeviceInfo>,
    budget: Option<GpuTierBudget>,
    allocated_vram_bytes: usize,
    vram_experts: Vec<usize>,
    is_manually_disabled: bool,
}

impl GpuTierManager {
    /// Khởi tạo và tự động phát hiện GPU trên máy
    pub fn probe() -> Self {
        let api = match CudaDriverApi::load() {
            Some(api) => api,
            None => {
                info!("CUDA Driver API (libcuda.so) không có sẵn. Chạy chế độ CPU/RAM/SSD.");
                return Self {
                    _api: None,
                    device_info: None,
                    budget: None,
                    allocated_vram_bytes: 0,
                    vram_experts: Vec::new(),
                    is_manually_disabled: false,
                };
            }
        };

        unsafe {
            if (api.cu_init)(0) != CUDA_SUCCESS {
                warn!("cuInit thất bại. Chạy chế độ CPU/RAM/SSD.");
                return Self {
                    _api: None,
                    device_info: None,
                    budget: None,
                    allocated_vram_bytes: 0,
                    vram_experts: Vec::new(),
                    is_manually_disabled: false,
                };
            }

            let mut version = 0;
            let _ = (api.cu_driver_get_version)(&mut version);

            let mut dev = 0;
            if (api.cu_device_get)(&mut dev, 0) != CUDA_SUCCESS {
                warn!("Không tìm thấy thiết bị CUDA (ordinal 0).");
                return Self {
                    _api: None,
                    device_info: None,
                    budget: None,
                    allocated_vram_bytes: 0,
                    vram_experts: Vec::new(),
                    is_manually_disabled: false,
                };
            }

            let mut name_buf = [0 as c_char; 256];
            let name = if (api.cu_device_get_name)(name_buf.as_mut_ptr(), 256, dev) == CUDA_SUCCESS
            {
                CStr::from_ptr(name_buf.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            } else {
                "NVIDIA GPU".to_string()
            };

            let mut free = 0;
            let mut total = 0;
            if (api.cu_mem_get_info)(&mut free, &mut total) != CUDA_SUCCESS {
                let mut total_bytes = 0;
                let _ = (api.cu_device_total_mem)(&mut total_bytes, dev);
                total = total_bytes;
                free = total_bytes;
            }

            let device_info = GpuDeviceInfo {
                name: name.clone(),
                total_vram_bytes: total as u64,
                free_vram_bytes: free as u64,
                driver_version: version,
                is_available: true,
            };

            info!(
                name = %name,
                total_mb = total / 1048576,
                free_mb = free / 1048576,
                driver_version = version,
                "Đã phát hiện phần cứng GPU NVIDIA"
            );

            // Dự trữ an toàn 512 MB cho hệ điều hành
            let reserved = 512 * 1024 * 1024;
            let usable = free.saturating_sub(reserved);

            let dense_target = 1536 * 1024 * 1024; // 1.5 GB
            let is_dense_offloaded = usable >= dense_target;
            let dense_allocated = if is_dense_offloaded { dense_target } else { 0 };
            let expert_cache_vram = usable.saturating_sub(dense_allocated);

            let budget = GpuTierBudget {
                total_vram_bytes: total as u64,
                reserved_os_bytes: reserved as u64,
                usable_vram_bytes: usable as u64,
                dense_vram_bytes: dense_allocated as u64,
                expert_cache_vram_bytes: expert_cache_vram as u64,
                is_dense_offloaded,
            };

            Self {
                _api: Some(api),
                device_info: Some(device_info),
                budget: Some(budget),
                allocated_vram_bytes: dense_allocated,
                vram_experts: Vec::new(),
                is_manually_disabled: false,
            }
        }
    }

    /// Lấy báo cáo trạng thái giám sát tầng GPU theo thời gian thực
    pub fn status(&self) -> GpuTierStatus {
        if self.is_manually_disabled {
            return GpuTierStatus {
                enabled: false,
                device_name: self
                    .device_info
                    .as_ref()
                    .map(|d| format!("{} (Đã tắt)", d.name))
                    .unwrap_or_else(|| "GPU (Đã tắt)".to_string()),
                total_vram_bytes: self
                    .device_info
                    .as_ref()
                    .map(|d| d.total_vram_bytes)
                    .unwrap_or(0),
                allocated_vram_bytes: 0,
                free_vram_bytes: self
                    .device_info
                    .as_ref()
                    .map(|d| d.total_vram_bytes)
                    .unwrap_or(0),
                dense_offloaded: false,
                vram_experts_resident: 0,
                message: "Chế độ GPU đã tắt thủ công (Chạy 100% CPU/RAM/SSD)".to_string(),
            };
        }

        match (&self.device_info, &self.budget) {
            (Some(dev), Some(b)) => GpuTierStatus {
                enabled: true,
                device_name: dev.name.clone(),
                total_vram_bytes: dev.total_vram_bytes,
                allocated_vram_bytes: self.allocated_vram_bytes as u64,
                free_vram_bytes: dev
                    .free_vram_bytes
                    .saturating_sub(self.allocated_vram_bytes as u64),
                dense_offloaded: b.is_dense_offloaded,
                vram_experts_resident: self.vram_experts.len(),
                message: format!(
                    "Hoạt động ({:.2} GB / {:.2} GB VRAM · cuBLAS SGEMV Active)",
                    self.allocated_vram_bytes as f64 / 1073741824.0,
                    dev.total_vram_bytes as f64 / 1073741824.0
                ),
            },
            _ => GpuTierStatus {
                enabled: false,
                device_name: "Không có GPU CUDA".to_string(),
                total_vram_bytes: 0,
                allocated_vram_bytes: 0,
                free_vram_bytes: 0,
                dense_offloaded: false,
                vram_experts_resident: 0,
                message: "Chạy chế độ CPU/RAM/SSD thuần túy".to_string(),
            },
        }
    }

    /// Cập nhật dung lượng VRAM thực tế đã cấp phát cho Dense
    pub fn set_dense_allocated_bytes(&mut self, bytes: usize) {
        self.allocated_vram_bytes = bytes;
    }

    /// Bật hoặc tắt tầng GPU theo yêu cầu người dùng từ API/Web
    pub fn set_enabled(&mut self, enabled: bool) -> bool {
        self.is_manually_disabled = !enabled;
        self.is_enabled()
    }

    /// Đảo trạng thái Bật / Tắt tầng GPU
    pub fn toggle_enabled(&mut self) -> bool {
        self.is_manually_disabled = !self.is_manually_disabled;
        self.is_enabled()
    }

    /// Kiểm tra tầng GPU có đang kích hoạt thực tế hay không
    pub fn is_enabled(&self) -> bool {
        !self.is_manually_disabled && self.is_available()
    }

    /// Kiểm tra xem GPU có sẵn sàng hoạt động hay không
    pub fn is_available(&self) -> bool {
        self.device_info.as_ref().is_some_and(|d| d.is_available)
    }

    /// Lấy lượng VRAM hiện tại đang chiếm dụng (cho telemetry)
    pub fn allocated_vram_mib(&self) -> u64 {
        (self.allocated_vram_bytes / 1048576) as u64
    }

    /// Ghi nhận 1 expert được nạp vào VRAM Cache
    pub fn pin_expert(&mut self, expert_id: usize, size_bytes: usize) -> bool {
        if let Some(b) = &self.budget {
            if self.allocated_vram_bytes + size_bytes <= b.usable_vram_bytes as usize {
                self.allocated_vram_bytes += size_bytes;
                self.vram_experts.push(expert_id);
                return true;
            }
        }
        false
    }

    /// Lấy cấu hình ngân sách VRAM của GPU
    pub fn budget(&self) -> Option<&GpuTierBudget> {
        self.budget.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_probe() {
        let mgr = GpuTierManager::probe();
        let status = mgr.status();
        eprintln!("\n=== GPU PROBE TEST RESULT ===");
        eprintln!("Available: {}", mgr.is_available());
        eprintln!("Device: {}", status.device_name);
        eprintln!(
            "Total VRAM: {:.2} GB",
            status.total_vram_bytes as f64 / 1073741824.0
        );
        eprintln!("Dense Offloaded: {}", status.dense_offloaded);
        eprintln!("Message: {}", status.message);
        eprintln!("=============================\n");

        if mgr.is_available() {
            assert!(status.total_vram_bytes > 0);
            assert!(!status.device_name.is_empty());
        }
    }

    #[test]
    fn test_gpu_compute_gemv() {
        if let Some(engine) = GpuComputeEngine::init() {
            let m = 3;
            let n = 4;
            let values = vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ];
            let matrix = crate::backend::Matrix {
                rows: m,
                columns: n,
                values: values.clone(),
            };
            let gpu_matrix = GpuMatrix::from_matrix(&matrix, &engine).expect("upload failed");
            let x = vec![1.0, 2.0, 3.0, 4.0];
            let mut y = vec![0.0; m];
            engine.gemv(&gpu_matrix, &x, &mut y).expect("gemv failed");
            eprintln!("cuBLAS GEMV result: {:?}", y);
            assert_eq!(y, vec![30.0, 70.0, 110.0]);
        }
    }

    #[test]
    fn test_gpu_compute_gemv_qkv() {
        if let Some(engine) = GpuComputeEngine::init() {
            let d = 4;
            let q_rows = 3;
            let k_rows = 2;
            let v_rows = 2;

            let q_mat = crate::backend::Matrix {
                rows: q_rows,
                columns: d,
                values: vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ],
            };
            let k_mat = crate::backend::Matrix {
                rows: k_rows,
                columns: d,
                values: vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
            };
            let v_mat = crate::backend::Matrix {
                rows: v_rows,
                columns: d,
                values: vec![2.0, 1.0, 0.0, -1.0, -2.0, -3.0, 1.0, 2.0],
            };

            let q_gpu = GpuMatrix::from_matrix(&q_mat, &engine).expect("upload q failed");
            let k_gpu = GpuMatrix::from_matrix(&k_mat, &engine).expect("upload k failed");
            let v_gpu = GpuMatrix::from_matrix(&v_mat, &engine).expect("upload v failed");

            let hidden = vec![1.0, 2.0, 3.0, 4.0];

            // 1. Tính toán riêng lẻ
            let mut q_expected = vec![0.0; q_rows];
            let mut k_expected = vec![0.0; k_rows];
            let mut v_expected = vec![0.0; v_rows];
            engine.gemv(&q_gpu, &hidden, &mut q_expected).unwrap();
            engine.gemv(&k_gpu, &hidden, &mut k_expected).unwrap();
            engine.gemv(&v_gpu, &hidden, &mut v_expected).unwrap();

            // 2. Tính toán gộp qua gemv_qkv
            let mut q_actual = vec![0.0; q_rows];
            let mut k_actual = vec![0.0; k_rows];
            let mut v_actual = vec![0.0; v_rows];
            let mut out_buf = vec![0.0; q_rows + k_rows + v_rows];

            engine
                .gemv_qkv(
                    &q_gpu,
                    &k_gpu,
                    &v_gpu,
                    &hidden,
                    &mut out_buf,
                    &mut q_actual,
                    &mut k_actual,
                    &mut v_actual,
                )
                .expect("gemv_qkv failed");

            assert_eq!(q_actual, q_expected);
            assert_eq!(k_actual, k_expected);
            assert_eq!(v_actual, v_expected);
        }
    }

    #[test]
    fn test_gpu_compute_gemm() {
        if let Some(engine) = GpuComputeEngine::init() {
            let m = 3;
            let k = 4;
            let values = vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ];
            let matrix = crate::backend::Matrix {
                rows: m,
                columns: k,
                values,
            };
            let gpu_matrix = GpuMatrix::from_matrix(&matrix, &engine).expect("upload failed");
            let batch_input = vec![
                1.0, 2.0, 3.0, 4.0, // token 0
                2.0, 0.0, 1.0, -1.0, // token 1
            ];
            let mut batch_output = vec![0.0; 2 * m];
            engine
                .gemm(&gpu_matrix, &batch_input, &mut batch_output, 2)
                .expect("gemm failed");
            eprintln!("cuBLAS SGEMM result: {:?}", batch_output);
            assert_eq!(batch_output, vec![30.0, 70.0, 110.0, 1.0, 9.0, 17.0]);
        }
    }

    #[test]
    fn test_gpu_compute_gemm_qkv() {
        if let Some(engine) = GpuComputeEngine::init() {
            let d = 4;
            let q_rows = 3;
            let k_rows = 2;
            let v_rows = 2;
            let batch_size = 2;

            let q_mat = crate::backend::Matrix {
                rows: q_rows,
                columns: d,
                values: vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ],
            };
            let k_mat = crate::backend::Matrix {
                rows: k_rows,
                columns: d,
                values: vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
            };
            let v_mat = crate::backend::Matrix {
                rows: v_rows,
                columns: d,
                values: vec![2.0, 1.0, 0.0, -1.0, -2.0, -3.0, 1.0, 2.0],
            };

            let q_gpu = GpuMatrix::from_matrix(&q_mat, &engine).expect("upload q failed");
            let k_gpu = GpuMatrix::from_matrix(&k_mat, &engine).expect("upload k failed");
            let v_gpu = GpuMatrix::from_matrix(&v_mat, &engine).expect("upload v failed");

            let batch_hidden = vec![
                1.0, 2.0, 3.0, 4.0, // token 0
                2.0, 0.0, 1.0, -1.0, // token 1
            ];

            // 1. Tính toán riêng lẻ bằng gemm
            let mut q_expected = vec![0.0; batch_size * q_rows];
            let mut k_expected = vec![0.0; batch_size * k_rows];
            let mut v_expected = vec![0.0; batch_size * v_rows];
            engine
                .gemm(&q_gpu, &batch_hidden, &mut q_expected, batch_size)
                .unwrap();
            engine
                .gemm(&k_gpu, &batch_hidden, &mut k_expected, batch_size)
                .unwrap();
            engine
                .gemm(&v_gpu, &batch_hidden, &mut v_expected, batch_size)
                .unwrap();

            // 2. Tính toán gộp qua gemm_qkv
            let mut q_actual = vec![0.0; batch_size * q_rows];
            let mut k_actual = vec![0.0; batch_size * k_rows];
            let mut v_actual = vec![0.0; batch_size * v_rows];

            engine
                .gemm_qkv(
                    &q_gpu,
                    &k_gpu,
                    &v_gpu,
                    &batch_hidden,
                    batch_size,
                    &mut q_actual,
                    &mut k_actual,
                    &mut v_actual,
                )
                .expect("gemm_qkv failed");

            assert_eq!(q_actual, q_expected);
            assert_eq!(k_actual, k_expected);
            assert_eq!(v_actual, v_expected);
        }
    }

    #[test]
    fn test_gpu_compute_gemm_qkv_across_threads_and_scratch_expansion() {
        if let Some(engine) = GpuComputeEngine::init() {
            let engine = std::sync::Arc::new(engine);
            let d = 256;
            let q_rows = 128;
            let k_rows = 64;
            let v_rows = 64;
            // Batch lớn: 200 tokens x 256 floats = 51,200 floats = 204,800 bytes
            // Vượt quá d_scratch_in_cap mặc định (32,768 bytes), buộc mở rộng lên 262,144 bytes!
            let batch_size = 200;

            let q_mat = crate::backend::Matrix {
                rows: q_rows,
                columns: d,
                values: vec![0.01; q_rows * d],
            };
            let k_mat = crate::backend::Matrix {
                rows: k_rows,
                columns: d,
                values: vec![0.02; k_rows * d],
            };
            let v_mat = crate::backend::Matrix {
                rows: v_rows,
                columns: d,
                values: vec![0.03; v_rows * d],
            };

            let q_gpu = GpuMatrix::from_matrix(&q_mat, &engine).expect("upload q failed");
            let k_gpu = GpuMatrix::from_matrix(&k_mat, &engine).expect("upload k failed");
            let v_gpu = GpuMatrix::from_matrix(&v_mat, &engine).expect("upload v failed");

            let batch_hidden = vec![0.5f32; batch_size * d];

            // Gọi trên một OS thread hoàn toàn mới (mô phỏng worker thread của Tokio / Axum)
            let handle = std::thread::spawn(move || {
                let mut q_actual = vec![0.0; batch_size * q_rows];
                let mut k_actual = vec![0.0; batch_size * k_rows];
                let mut v_actual = vec![0.0; batch_size * v_rows];

                engine
                    .gemm_qkv(
                        &q_gpu,
                        &k_gpu,
                        &v_gpu,
                        &batch_hidden,
                        batch_size,
                        &mut q_actual,
                        &mut k_actual,
                        &mut v_actual,
                    )
                    .expect("gemm_qkv on new thread with scratch expansion must succeed");

                assert!(!q_actual.is_empty());
                assert!((q_actual[0] - (d as f32) * 0.5 * 0.01).abs() < 1e-4);
            });

            handle.join().expect("thread join failed");
        }
    }

    #[test]
    fn test_gpu_expert_weights_upload_download() {
        if let Some(engine) = GpuComputeEngine::init() {
            let layer = 0;
            let expert_id = 42;
            let weight_count = 1024 * 16;
            let scale_count = 64;
            let weights: Vec<u8> = (0..weight_count).map(|i| (i % 256) as u8).collect();
            let scales: Vec<f32> = (0..scale_count).map(|i| 0.01 * (i + 1) as f32).collect();

            let gpu_exp = engine
                .upload_expert(layer, expert_id, &weights, &scales)
                .expect("upload_expert failed");
            assert_eq!(gpu_exp.layer, layer);
            assert_eq!(gpu_exp.expert_id, expert_id);
            assert_eq!(gpu_exp.weight_bytes, weight_count);

            let mut dl_weights = vec![0u8; weight_count];
            let mut dl_scales = vec![0.0f32; scale_count];
            engine
                .download_expert(&gpu_exp, &mut dl_weights, &mut dl_scales)
                .expect("download_expert failed");

            assert_eq!(weights, dl_weights);
            assert_eq!(scales, dl_scales);
        }
    }

    #[test]
    fn test_gpu_expert_cache_remove_accounts_bytes_and_is_idempotent() {
        if let Some(engine) = GpuComputeEngine::init() {
            let weights = vec![13u8; 1024];
            let scales = vec![0.25f32; 16];
            let mut cache = GpuExpertCache::new(4096);
            let expert = engine.upload_expert(0, 7, &weights, &scales).unwrap();
            cache.insert(expert);
            assert_eq!(cache.used_bytes, 1088);
            assert!(cache.remove_with_pool(0, 7, None));
            assert!(!cache.remove_with_pool(0, 7, None));
            assert_eq!(cache.used_bytes, 0);
            assert!(cache.lru.is_empty());
            assert!(cache.is_empty());
        }
    }

    #[test]
    fn test_gpu_expert_cache_lru_eviction() {
        if let Some(engine) = GpuComputeEngine::init() {
            let weight_count = 1024;
            let scale_count = 16;
            let weights = vec![1u8; weight_count];
            let scales = vec![1.0f32; scale_count];
            let single_exp_bytes = weight_count + scale_count * 4;

            // Budget chỉ đủ cho 2 experts
            let mut cache = GpuExpertCache::new(single_exp_bytes * 2 + 10);

            let exp0 = engine.upload_expert(0, 0, &weights, &scales).unwrap();
            let exp1 = engine.upload_expert(0, 1, &weights, &scales).unwrap();
            let exp2 = engine.upload_expert(0, 2, &weights, &scales).unwrap();

            cache.insert(exp0);
            assert_eq!(cache.len(), 1);
            cache.insert(exp1);
            assert_eq!(cache.len(), 2);

            // Insert expert thứ 3 sẽ evict exp0
            let evicted = cache.insert(exp2);
            assert_eq!(evicted, 1);
            assert_eq!(cache.len(), 2);
            assert!(!cache.contains(0, 0));
            assert!(cache.contains(0, 1));
            assert!(cache.contains(0, 2));
        }
    }

    #[test]
    fn test_gpu_pinned_buffer() {
        if let Some(engine) = GpuComputeEngine::init() {
            let len = 4096;
            let mut pinned = PinnedBuffer::alloc(&engine, len).expect("alloc pinned buffer");
            assert_eq!(pinned.len, len);
            unsafe {
                let slice = pinned.as_mut_slice::<u32>();
                for (i, val) in slice.iter_mut().enumerate() {
                    *val = (i * 7) as u32;
                }
                let read_slice = pinned.as_slice::<u32>();
                for (i, val) in read_slice.iter().enumerate() {
                    assert_eq!(*val, (i * 7) as u32);
                }
            }
        }
    }

    #[test]
    fn test_gpu_expert_pool_lifecycle() {
        if let Some(engine) = GpuComputeEngine::init() {
            let max_slots = 3;
            let weight_slot_bytes = 2048;
            let scale_slot_bytes = 64;
            let mut pool =
                GpuExpertPool::new(&engine, max_slots, weight_slot_bytes, scale_slot_bytes)
                    .expect("alloc GpuExpertPool");
            assert_eq!(pool.available_slots(), 3);

            let w = vec![42u8; weight_slot_bytes];
            let s = vec![1.5f32; scale_slot_bytes / 4];

            let slot0 = unsafe {
                pool.upload(
                    w.as_ptr() as *const c_void,
                    weight_slot_bytes,
                    s.as_ptr() as *const c_void,
                    scale_slot_bytes,
                )
                .unwrap()
            };
            let slot1 = unsafe {
                pool.upload(
                    w.as_ptr() as *const c_void,
                    weight_slot_bytes,
                    s.as_ptr() as *const c_void,
                    scale_slot_bytes,
                )
                .unwrap()
            };
            let _slot2 = unsafe {
                pool.upload(
                    w.as_ptr() as *const c_void,
                    weight_slot_bytes,
                    s.as_ptr() as *const c_void,
                    scale_slot_bytes,
                )
                .unwrap()
            };
            assert_eq!(pool.available_slots(), 0);

            // Pool đầy, upload thêm phải trả về None
            assert!(unsafe {
                pool.upload(
                    w.as_ptr() as *const c_void,
                    weight_slot_bytes,
                    s.as_ptr() as *const c_void,
                    scale_slot_bytes,
                )
                .is_none()
            });

            // Download kiểm tra dữ liệu đúng
            let mut dl_w = vec![0u8; weight_slot_bytes];
            let mut dl_s = vec![0.0f32; scale_slot_bytes / 4];
            unsafe {
                pool.download(
                    slot0,
                    dl_w.as_mut_ptr() as *mut c_void,
                    weight_slot_bytes,
                    dl_s.as_mut_ptr() as *mut c_void,
                    scale_slot_bytes,
                )
                .unwrap();
            }
            assert_eq!(w, dl_w);
            assert_eq!(s, dl_s);

            // Free 1 slot và upload lại thành công
            pool.free_slot(slot1);
            assert_eq!(pool.available_slots(), 1);
            // Thử free_slot lần 2 trên cùng slot1 (chống double-free)
            pool.free_slot(slot1);
            assert_eq!(pool.available_slots(), 1);
            let slot_reused = unsafe {
                pool.upload(
                    w.as_ptr() as *const c_void,
                    weight_slot_bytes,
                    s.as_ptr() as *const c_void,
                    scale_slot_bytes,
                )
                .unwrap()
            };
            assert_eq!(slot_reused, slot1);
        }
    }
}
