use std::cmp::Ordering;

pub type ExpertId = u32;

#[derive(Debug, Clone, PartialEq)]
pub struct RoutingDecision {
    pub experts: Vec<(ExpertId, f32)>,
}

#[derive(Debug, Clone)]
pub struct Router {
    expert_count: u32,
    top_k: usize,
    weights: Option<Vec<f32>>,
    hidden_size: usize,
}

impl Router {
    pub fn new(expert_count: u32, top_k: usize) -> Result<Self, String> {
        if expert_count == 0 || top_k == 0 || top_k as u32 > expert_count {
            return Err("top_k must be between 1 and expert_count".into());
        }
        Ok(Self {
            expert_count,
            top_k,
            weights: None,
            hidden_size: 0,
        })
    }

    /// `weights` is a row-major [expert_count, hidden_size] routing matrix.
    pub fn from_weights(
        expert_count: u32,
        top_k: usize,
        hidden_size: usize,
        weights: Vec<f32>,
    ) -> Result<Self, String> {
        let mut router = Self::new(expert_count, top_k)?;
        if hidden_size == 0
            || (expert_count as usize).checked_mul(hidden_size) != Some(weights.len())
            || weights.iter().any(|value| !value.is_finite())
        {
            return Err("invalid router weight shape or value".into());
        }
        router.hidden_size = hidden_size;
        router.weights = Some(weights);
        Ok(router)
    }

    pub fn route_hidden(&self, hidden: &[f32]) -> Result<RoutingDecision, String> {
        let weights = self
            .weights
            .as_ref()
            .ok_or("router has no learned weights")?;
        if hidden.len() != self.hidden_size || hidden.iter().any(|value| !value.is_finite()) {
            return Err("invalid hidden state".into());
        }
        let scores = weights
            .chunks_exact(self.hidden_size)
            .enumerate()
            .map(|(id, row)| {
                (
                    id as ExpertId,
                    row.iter().zip(hidden).map(|(a, b)| a * b).sum::<f32>(),
                )
            })
            .collect();
        Ok(self.select_top_k(scores))
    }

    pub fn route(&self, token_id: u32) -> RoutingDecision {
        let scores: Vec<_> = (0..self.expert_count)
            .map(|expert_id| {
                let mixed = token_id
                    .wrapping_mul(1_664_525)
                    .wrapping_add(expert_id.wrapping_mul(1_013_904_223));
                let score = (mixed % 10_000) as f32 / 10_000.0;
                (expert_id, score)
            })
            .collect();
        self.select_top_k(scores)
    }

    fn select_top_k(&self, mut scores: Vec<(ExpertId, f32)>) -> RoutingDecision {
        scores.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scores.truncate(self.top_k);
        let max = scores[0].1;
        let total: f32 = scores
            .iter_mut()
            .map(|(_, score)| {
                *score = (*score - max).exp();
                *score
            })
            .sum();
        for (_, score) in &mut scores {
            *score /= total;
        }
        RoutingDecision { experts: scores }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_returns_normalized_top_k() {
        let decision = Router::new(8, 2).unwrap().route(42);
        assert_eq!(decision.experts.len(), 2);
        let total: f32 = decision.experts.iter().map(|(_, weight)| weight).sum();
        assert!((total - 1.0).abs() < 0.0001);
    }

    #[test]
    fn learned_router_uses_hidden_state() {
        let router = Router::from_weights(3, 2, 2, vec![1.0, 0.0, 0.0, 1.0, -1.0, -1.0]).unwrap();
        let decision = router.route_hidden(&[2.0, 1.0]).unwrap();
        assert_eq!(
            decision
                .experts
                .iter()
                .map(|(id, _)| *id)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(
            (decision
                .experts
                .iter()
                .map(|(_, weight)| weight)
                .sum::<f32>()
                - 1.0)
                .abs()
                < 1e-6
        );
    }
}
