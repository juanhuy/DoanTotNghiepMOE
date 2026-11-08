//! Cooperative Linux memory guard. This is not an allocator or a kernel RSS cap.
use std::{fs, io};

const MIB: usize = 1024 * 1024;

#[derive(Clone, Copy)]
pub(crate) struct RamGuard {
    pub(crate) rss_limit: usize,
    pub(crate) system_headroom: usize,
}

#[derive(Default)]
struct Snapshot {
    rss: Option<usize>,
    available: Option<usize>,
    total: Option<usize>,
}

fn kib_field(text: &str, key: &str) -> Option<usize> {
    text.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        (name == key)
            .then(|| {
                value
                    .split_whitespace()
                    .next()?
                    .parse::<usize>()
                    .ok()?
                    .checked_mul(1024)
            })
            .flatten()
    })
}

fn snapshot() -> Snapshot {
    let info = fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
    let mut result = Snapshot {
        rss: kib_field(&status, "VmRSS"),
        available: kib_field(&info, "MemAvailable"),
        total: kib_field(&info, "MemTotal"),
    };
    // Honor the visible cgroup-v2 limit as well as host memory. Ancestor limits
    // outside this namespace may still be tighter; no hard-cap claim is made.
    let limit = fs::read_to_string("/sys/fs/cgroup/memory.max")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok());
    let used = fs::read_to_string("/sys/fs/cgroup/memory.current")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok());
    if let Some(limit) = limit {
        result.total = Some(result.total.map_or(limit, |total| total.min(limit)));
        if let Some(used) = used {
            let remaining = limit.saturating_sub(used);
            result.available = Some(
                result
                    .available
                    .map_or(remaining, |available| available.min(remaining)),
            );
        }
    }
    result
}

pub(crate) fn process_memory_bytes() -> (Option<usize>, Option<usize>) {
    let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
    (kib_field(&status, "VmRSS"), kib_field(&status, "VmSwap"))
}

pub(crate) fn current_thread_id() -> Option<u32> {
    #[cfg(target_os = "linux")]
    {
        fs::read_link("/proc/thread-self")
            .ok()?
            .file_name()?
            .to_str()?
            .parse()
            .ok()
    }
    #[cfg(not(target_os = "linux"))]
    None
}

/// Allocator bookkeeping, not exact live Rust payload or resident memory.
/// Only called from opt-in memory tracing. No allocator policy is changed.
pub(crate) fn allocator_memory() -> crate::generation::AllocatorMemorySample {
    #[allow(unused_mut)]
    let mut sample = crate::generation::AllocatorMemorySample {
        status: "unsupported_platform",
        libc_version: None,
        preload_env_present: std::env::var_os("LD_PRELOAD").is_some(),
        arena_bytes: None,
        in_use_bytes: None,
        free_bytes: None,
        mmap_bytes: None,
        query_ns: None,
    };
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    glibc_memory::fill(&mut sample);
    sample
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod glibc_memory {
    use std::{
        ffi::{c_char, c_void, CStr},
        sync::OnceLock,
        time::Instant,
    };

    // Exact glibc malloc.h layout: ten size_t fields, returned by value.
    // Do not reduce this struct to the four fields exported in the trace.
    #[repr(C)]
    struct Mallinfo2 {
        arena: usize,
        ordblks: usize,
        smblks: usize,
        hblks: usize,
        hblkhd: usize,
        usmblks: usize,
        fsmblks: usize,
        uordblks: usize,
        fordblks: usize,
        keepcost: usize,
    }
    type Query = unsafe extern "C" fn() -> Mallinfo2;
    struct Probe {
        query: Option<Query>,
        version: String,
    }
    #[link(name = "dl")]
    extern "C" {
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn gnu_get_libc_version() -> *const c_char;
    }
    static PROBE: OnceLock<Probe> = OnceLock::new();

    pub(super) fn fill(sample: &mut crate::generation::AllocatorMemorySample) {
        let probe = PROBE.get_or_init(|| {
            // SAFETY: GNU libc exports a process-lifetime NUL-terminated static
            // version string. RTLD_DEFAULT (NULL on GNU/Linux) resolves the
            // optional mallinfo2 symbol without imposing GLIBC_2.33 at link time.
            // A resolved function has malloc.h's ABI/layout above; the symbol
            // remains loaded with libc for this process. No pointer is retained
            // into an allocator arena and no allocation is freed by this code.
            unsafe {
                let version = CStr::from_ptr(gnu_get_libc_version())
                    .to_string_lossy()
                    .into_owned();
                let ptr = dlsym(std::ptr::null_mut(), c"mallinfo2".as_ptr());
                Probe {
                    query: if ptr.is_null() {
                        None
                    } else {
                        Some(std::mem::transmute::<*mut c_void, Query>(ptr))
                    },
                    version,
                }
            }
        });
        sample.libc_version = Some(probe.version.clone());
        sample.status = "symbol_unavailable";
        if let Some(query) = probe.query {
            let started = Instant::now();
            // SAFETY: signature/layout and process lifetime established above.
            // glibc synchronizes arena reads internally. Results across arenas
            // are sequential, not an atomic snapshot; tcache/metadata and other
            // allocators/mappings prevent interpreting this as exact live heap.
            // The engine never changes mallopt settings concurrently.
            let info = unsafe { query() };
            sample.query_ns = Some(started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
            sample.status = "glibc_mallinfo2";
            sample.arena_bytes = Some(info.arena);
            sample.in_use_bytes = Some(info.uordblks);
            sample.free_bytes = Some(info.fordblks);
            sample.mmap_bytes = Some(info.hblkhd);
        }
    }
}

fn setting(name: &str, default: usize) -> io::Result<usize> {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("{name} must be positive bytes"),
                )
            }),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(_) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid {name}"),
        )),
    }
}

impl RamGuard {
    pub(crate) fn from_environment() -> io::Result<Self> {
        let initial = snapshot();
        let total = initial.total;
        if initial.rss.is_none() || initial.available.is_none() {
            tracing::warn!("RAM telemetry unavailable; cooperative guard cannot observe every limit on this platform");
        }
        let guard = Self {
            rss_limit: setting(
                "MOE_RAM_LIMIT_BYTES",
                total.map_or(6144 * MIB, |total| (total / 2).min(6144 * MIB)),
            )?,
            system_headroom: setting(
                "MOE_RAM_HEADROOM_BYTES",
                total.map_or(1024 * MIB, |total| (total / 8).min(1024 * MIB)),
            )?,
        };
        tracing::info!(rss_limit_bytes=guard.rss_limit, system_headroom_bytes=guard.system_headroom,
            "Cooperative RAM guard; limit includes dense/cache/request/allocator RSS, not a kernel cap");
        Ok(guard)
    }

    pub(crate) fn check(&self, reserve: usize) -> io::Result<()> {
        self.check_snapshot(&snapshot(), reserve)
    }

    fn check_snapshot(&self, current: &Snapshot, reserve: usize) -> io::Result<()> {
        let rss_over = current
            .rss
            .is_some_and(|rss| rss.saturating_add(reserve) > self.rss_limit);
        let available_low = current
            .available
            .is_some_and(|available| available < self.system_headroom.saturating_add(reserve));
        if rss_over || available_low {
            return Err(io::Error::new(io::ErrorKind::OutOfMemory, format!(
                "RAM guard: rss={:?}, limit={}, available={:?}, headroom={}, reserve={} bytes; reduce cache/context or free memory",
                current.rss, self.rss_limit, current.available, self.system_headroom, reserve)));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_probe_reports_availability_without_inventing_live_bytes() {
        let sample = allocator_memory();
        match sample.status {
            "glibc_mallinfo2" => {
                assert!(sample.libc_version.as_ref().is_some_and(|s| !s.is_empty()));
                assert_eq!(
                    sample
                        .in_use_bytes
                        .unwrap()
                        .checked_add(sample.free_bytes.unwrap()),
                    sample.arena_bytes
                );
                assert!(sample.mmap_bytes.is_some());
                assert!(sample.query_ns.is_some());
            }
            "symbol_unavailable" | "unsupported_platform" => {
                assert!(sample.arena_bytes.is_none());
                assert!(sample.in_use_bytes.is_none());
                assert!(sample.free_bytes.is_none());
                assert!(sample.mmap_bytes.is_none());
                assert!(sample.query_ns.is_none());
            }
            status => panic!("unexpected allocator status: {status}"),
        }
        #[cfg(target_os = "linux")]
        {
            let tid = current_thread_id().expect("Linux proc thread-self");
            assert!(std::path::Path::new(&format!("/proc/self/task/{tid}")).exists());
        }
    }

    #[test]
    fn headroom_and_rss_reserve_reject_before_allocation() {
        let guard = RamGuard {
            rss_limit: 1000,
            system_headroom: 200,
        };
        let current = Snapshot {
            rss: Some(700),
            available: Some(500),
            total: None,
        };
        assert!(guard.check_snapshot(&current, 300).is_ok());
        assert_eq!(
            guard.check_snapshot(&current, 301).unwrap_err().kind(),
            io::ErrorKind::OutOfMemory
        );
        let pressure = Snapshot {
            rss: Some(400),
            available: Some(199),
            total: None,
        };
        assert!(guard.check_snapshot(&pressure, 0).is_err());
        assert!(guard.check_snapshot(&current, usize::MAX).is_err());
    }

    #[test]
    fn proc_fields_are_kib_and_absent_fields_are_unknown() {
        assert_eq!(
            kib_field("VmRSS: 123 kB\nVmSwap: 20 kB", "VmRSS"),
            Some(123 * 1024)
        );
        assert_eq!(kib_field("MemFree: 100 kB", "MemAvailable"), None);
    }
}
