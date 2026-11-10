use std::{
    collections::{HashMap, HashSet, VecDeque},
    io,
    path::PathBuf,
    sync::Arc,
};
use tokio::{fs, sync::Mutex, task::JoinSet};

use crate::router::ExpertId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    Vram,
    Ram,
    Nvme,
}

#[derive(Debug, Clone)]
pub struct ExpertStore {
    root: PathBuf,
}

impl ExpertStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub async fn read(&self, expert_id: ExpertId) -> io::Result<Vec<u8>> {
        fs::read(self.root.join(format!("expert-{expert_id}.bin"))).await
    }

    pub async fn write(&self, expert_id: ExpertId, bytes: &[u8]) -> io::Result<()> {
        fs::create_dir_all(&self.root).await?;
        fs::write(self.root.join(format!("expert-{expert_id}.bin")), bytes).await
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheStats {
    pub ram_hits: u64,
    pub nvme_reads: u64,
    pub evictions: u64,
}

struct State {
    vram: HashMap<ExpertId, Arc<Vec<u8>>>,
    ram: HashMap<ExpertId, Arc<Vec<u8>>>,
    lru: VecDeque<ExpertId>,
    ram_bytes: usize,
    stats: CacheStats,
}

#[derive(Clone)]
pub struct TieredMemoryManager {
    store: ExpertStore,
    ram_capacity: usize,
    ram_byte_capacity: Option<usize>,
    state: Arc<Mutex<State>>,
}

impl TieredMemoryManager {
    pub fn new(store: ExpertStore, ram_capacity: usize) -> Self {
        Self {
            store,
            ram_capacity,
            ram_byte_capacity: None,
            state: Arc::new(Mutex::new(State {
                vram: HashMap::new(),
                ram: HashMap::new(),
                lru: VecDeque::new(),
                ram_bytes: 0,
                stats: CacheStats {
                    ram_hits: 0,
                    nvme_reads: 0,
                    evictions: 0,
                },
            })),
        }
    }

    pub fn with_ram_byte_capacity(store: ExpertStore, capacity_bytes: usize) -> Self {
        let mut manager = Self::new(store, usize::MAX);
        manager.ram_byte_capacity = Some(capacity_bytes);
        manager
    }

    pub async fn pin_vram(&self, expert_id: ExpertId, bytes: Vec<u8>) {
        self.state
            .lock()
            .await
            .vram
            .insert(expert_id, Arc::new(bytes));
    }

    pub async fn get(&self, expert_id: ExpertId) -> io::Result<(MemoryTier, Arc<Vec<u8>>)> {
        let mut state = self.state.lock().await;
        if let Some(bytes) = state.vram.get(&expert_id) {
            return Ok((MemoryTier::Vram, Arc::clone(bytes)));
        }
        if let Some(bytes) = state.ram.get(&expert_id).cloned() {
            state.stats.ram_hits += 1;
            state.lru.retain(|id| *id != expert_id);
            state.lru.push_back(expert_id);
            return Ok((MemoryTier::Ram, bytes));
        }
        drop(state);

        let bytes = Arc::new(self.store.read(expert_id).await?);
        let mut state = self.state.lock().await;
        state.stats.nvme_reads += 1;
        // Another request may have populated this entry while the read was in flight.
        // Do not account for the same bytes twice or evict unrelated experts.
        if let Some(existing) = state.ram.get(&expert_id).cloned() {
            state.lru.retain(|id| *id != expert_id);
            state.lru.push_back(expert_id);
            return Ok((MemoryTier::Ram, existing));
        }
        let fits = self
            .ram_byte_capacity
            .is_none_or(|capacity| bytes.len() <= capacity);
        if self.ram_capacity > 0 && fits {
            while state.ram.len() >= self.ram_capacity
                || self
                    .ram_byte_capacity
                    .is_some_and(|capacity| state.ram_bytes > capacity - bytes.len())
            {
                if let Some(oldest) = state.lru.pop_front() {
                    if let Some(removed) = state.ram.remove(&oldest) {
                        state.ram_bytes -= removed.len();
                    }
                    state.stats.evictions += 1;
                } else {
                    break;
                }
            }
            state.ram_bytes += bytes.len();
            state.ram.insert(expert_id, Arc::clone(&bytes));
            state.lru.push_back(expert_id);
        }
        Ok((MemoryTier::Nvme, bytes))
    }

    pub async fn stats(&self) -> CacheStats {
        self.state.lock().await.stats.clone()
    }

    pub async fn cached_ram_bytes(&self) -> usize {
        self.state.lock().await.ram_bytes
    }

    /// Warm distinct experts concurrently before computation begins.
    pub async fn prefetch(&self, expert_ids: &[ExpertId]) -> io::Result<()> {
        let mut tasks = JoinSet::new();
        let mut seen = HashSet::new();
        for &id in expert_ids {
            if seen.insert(id) {
                let manager = self.clone();
                tasks.spawn(async move { manager.get(id).await.map(|_| ()) });
            }
        }
        while let Some(result) = tasks.join_next().await {
            result.map_err(|error| io::Error::other(error.to_string()))??;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn promotes_nvme_reads_into_lru_cache() {
        let dir = tempfile::tempdir().unwrap();
        let store = ExpertStore::new(dir.path());
        store.write(1, b"one").await.unwrap();
        store.write(2, b"two").await.unwrap();
        let manager = TieredMemoryManager::new(store, 1);

        assert_eq!(manager.get(1).await.unwrap().0, MemoryTier::Nvme);
        assert_eq!(manager.get(1).await.unwrap().0, MemoryTier::Ram);
        manager.get(2).await.unwrap();
        let stats = manager.stats().await;
        assert_eq!(stats.ram_hits, 1);
        assert_eq!(stats.nvme_reads, 2);
        assert_eq!(stats.evictions, 1);
    }

    #[tokio::test]
    async fn byte_capacity_evicts_and_skips_oversized_experts() {
        let dir = tempfile::tempdir().unwrap();
        let store = ExpertStore::new(dir.path());
        store.write(1, b"one").await.unwrap();
        store.write(2, b"four").await.unwrap();
        store.write(3, b"oversized").await.unwrap();
        let manager = TieredMemoryManager::with_ram_byte_capacity(store, 4);
        manager.get(1).await.unwrap();
        assert_eq!(manager.cached_ram_bytes().await, 3);
        manager.get(2).await.unwrap();
        assert_eq!(manager.cached_ram_bytes().await, 4);
        manager.get(3).await.unwrap();
        assert_eq!(manager.cached_ram_bytes().await, 4);
        assert_eq!(manager.get(2).await.unwrap().0, MemoryTier::Ram);
    }

    #[tokio::test]
    async fn concurrent_reads_account_for_each_cached_expert_once() {
        let dir = tempfile::tempdir().unwrap();
        let store = ExpertStore::new(dir.path());
        store.write(0, &vec![7; 4096]).await.unwrap();
        let manager = TieredMemoryManager::with_ram_byte_capacity(store, 8192);
        let mut tasks = JoinSet::new();
        let barrier = Arc::new(tokio::sync::Barrier::new(16));
        for _ in 0..16 {
            let manager = manager.clone();
            let barrier = barrier.clone();
            tasks.spawn(async move {
                barrier.wait().await;
                let (_, bytes) = manager.get(0).await.unwrap();
                assert_eq!(bytes.len(), 4096);
            });
        }
        while let Some(result) = tasks.join_next().await {
            result.unwrap();
        }
        assert_eq!(manager.cached_ram_bytes().await, 4096);
        assert_eq!(manager.stats().await.evictions, 0);
    }

    #[tokio::test]
    async fn prefetch_deduplicates_ids() {
        let dir = tempfile::tempdir().unwrap();
        let store = ExpertStore::new(dir.path());
        store.write(1, b"one").await.unwrap();
        let manager = TieredMemoryManager::new(store, 2);
        manager.prefetch(&[1, 1]).await.unwrap();
        assert_eq!(manager.stats().await.nvme_reads, 1);
        assert_eq!(manager.get(1).await.unwrap().0, MemoryTier::Ram);
    }
}
