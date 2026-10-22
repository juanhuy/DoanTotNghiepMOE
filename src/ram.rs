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
