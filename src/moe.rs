use std::{io, sync::Arc};

use crate::{
    memory::TieredMemoryManager,
    router::{ExpertId, Router},
};

const MAGIC: &[u8; 4] = b"MEX1";

/// One row-wise INT8 matrix. File layout: magic, rows, columns (little endian
/// u32), then for each row one little endian f32 scale and `columns` i8 values.
#[derive(Debug, Clone)]
pub struct QuantizedMatrix {
    rows: usize,
    columns: usize,
    scales: Vec<f32>,
    values: Vec<i8>,
}

impl QuantizedMatrix {
    pub fn quantize(rows: usize, columns: usize, values: &[f32]) -> io::Result<Self> {
        if rows == 0
            || columns == 0
            || rows.checked_mul(columns) != Some(values.len())
            || rows > u32::MAX as usize
            || columns > u32::MAX as usize
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid matrix shape or values",
            ));
        }
        let mut scales = Vec::with_capacity(rows);
        let mut quantized = Vec::with_capacity(values.len());
        for row in values.chunks_exact(columns) {
            let max = row.iter().fold(0.0_f32, |acc, value| acc.max(value.abs()));
            let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
            scales.push(scale);
            quantized.extend(
                row.iter()
                    .map(|value| (value / scale).round().clamp(-127.0, 127.0) as i8),
            );
        }
        Ok(Self {
            rows,
            columns,
            scales,
            values: quantized,
        })
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(12 + self.rows * (4 + self.columns));
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&(self.rows as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.columns as u32).to_le_bytes());
        for row in 0..self.rows {
            bytes.extend_from_slice(&self.scales[row].to_le_bytes());
            bytes.extend(
                self.values[row * self.columns..(row + 1) * self.columns]
                    .iter()
                    .map(|value| *value as u8),
            );
        }
        bytes
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        let invalid = || io::Error::new(io::ErrorKind::InvalidData, "invalid MEX1 expert matrix");
        if bytes.len() < 12 || &bytes[..4] != MAGIC {
            return Err(invalid());
        }
        let rows = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let columns = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let expected = rows
            .checked_mul(columns.checked_add(4).ok_or_else(invalid)?)
            .and_then(|size| size.checked_add(12))
            .ok_or_else(invalid)?;
        if rows == 0 || columns == 0 || bytes.len() != expected {
            return Err(invalid());
        }
        let mut scales = Vec::with_capacity(rows);
        let mut values = Vec::with_capacity(rows * columns);
        for row in bytes[12..].chunks_exact(columns + 4) {
            let scale = f32::from_le_bytes(row[..4].try_into().unwrap());
            if !scale.is_finite() || scale <= 0.0 {
                return Err(invalid());
            }
            scales.push(scale);
            values.extend(row[4..].iter().map(|value| *value as i8));
        }
        Ok(Self {
            rows,
            columns,
            scales,
            values,
        })
    }

    pub fn multiply(&self, input: &[f32]) -> io::Result<Vec<f32>> {
        if input.len() != self.columns || input.iter().any(|value| !value.is_finite()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "input dimension or value is invalid",
            ));
        }
        Ok(self
            .values
            .chunks_exact(self.columns)
            .zip(&self.scales)
            .map(|(row, scale)| {
                row.iter()
                    .zip(input)
                    .map(|(weight, value)| *weight as f32 * *value)
                    .sum::<f32>()
                    * scale
            })
            .collect())
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
}

/// Executes a weighted Top-K linear MoE layer. This is a computational building
/// block, not a language model: routing is supplied by the caller's Router.
#[derive(Clone)]
pub struct MoELayer {
    router: Router,
    memory: TieredMemoryManager,
    dimension: usize,
}

impl MoELayer {
    pub fn new(router: Router, memory: TieredMemoryManager, dimension: usize) -> io::Result<Self> {
        if dimension == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "dimension must be positive",
            ));
        }
        Ok(Self {
            router,
            memory,
            dimension,
        })
    }

    pub async fn forward(
        &self,
        token_id: u32,
        hidden: &[f32],
    ) -> io::Result<(Vec<f32>, Vec<ExpertId>)> {
        if hidden.len() != self.dimension {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "hidden dimension mismatch",
            ));
        }
        let decision = self.router.route(token_id);
        self.execute(hidden, decision).await
    }

    pub async fn forward_hidden(&self, hidden: &[f32]) -> io::Result<(Vec<f32>, Vec<ExpertId>)> {
        if hidden.len() != self.dimension {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "hidden dimension mismatch",
            ));
        }
        let decision = self
            .router
            .route_hidden(hidden)
            .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
        self.execute(hidden, decision).await
    }

    async fn execute(
        &self,
        hidden: &[f32],
        decision: crate::router::RoutingDecision,
    ) -> io::Result<(Vec<f32>, Vec<ExpertId>)> {
        let mut output = vec![0.0; self.dimension];
        let mut experts = Vec::with_capacity(decision.experts.len());
        for (id, weight) in decision.experts {
            let (_, bytes): (_, Arc<Vec<u8>>) = self.memory.get(id).await?;
            let matrix = QuantizedMatrix::decode(&bytes)?;
            if matrix.rows() != self.dimension || matrix.columns() != self.dimension {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "expert dimension mismatch",
                ));
            }
            let result = matrix.multiply(hidden)?;
            for (target, value) in output.iter_mut().zip(result) {
                *target += weight * value;
            }
            experts.push(id);
        }
        Ok((output, experts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::ExpertStore;

    #[tokio::test]
    async fn weighted_experts_execute_and_cache() {
        let dir = tempfile::tempdir().unwrap();
        let store = ExpertStore::new(dir.path());
        let identity = QuantizedMatrix::quantize(2, 2, &[1.0, 0.0, 0.0, 1.0]).unwrap();
        for id in 0..2 {
            store.write(id, &identity.encode()).await.unwrap();
        }
        let memory = TieredMemoryManager::new(store, 2);
        let layer = MoELayer::new(Router::new(2, 2).unwrap(), memory.clone(), 2).unwrap();
        let (output, ids) = layer.forward(7, &[2.0, -3.0]).await.unwrap();
        assert_eq!(ids.len(), 2);
        assert!((output[0] - 2.0).abs() < 0.02);
        assert!((output[1] + 3.0).abs() < 0.02);
        layer.forward(7, &[2.0, -3.0]).await.unwrap();
        assert_eq!(memory.stats().await.ram_hits, 2);
    }

    #[tokio::test]
    async fn learned_routing_selects_expert_by_hidden_state() {
        let dir = tempfile::tempdir().unwrap();
        let store = ExpertStore::new(dir.path());
        for (id, gain) in [(0, 1.0), (1, 2.0)] {
            let matrix = QuantizedMatrix::quantize(2, 2, &[gain, 0.0, 0.0, gain]).unwrap();
            store.write(id, &matrix.encode()).await.unwrap();
        }
        let router = Router::from_weights(2, 1, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let layer = MoELayer::new(router, TieredMemoryManager::new(store, 1), 2).unwrap();
        let (output, ids) = layer.forward_hidden(&[1.0, 3.0]).await.unwrap();
        assert_eq!(ids, vec![1]);
        assert!((output[0] - 2.0).abs() < 0.02);
        assert!((output[1] - 6.0).abs() < 0.02);
    }

    #[test]
    fn malformed_matrix_is_rejected() {
        let matrix = QuantizedMatrix::quantize(1, 2, &[1.0, -1.0]).unwrap();
        let mut bytes = matrix.encode();
        bytes.pop();
        assert!(QuantizedMatrix::decode(&bytes).is_err());
    }
}
