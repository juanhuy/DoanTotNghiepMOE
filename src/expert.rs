//! Full SwiGLU expert FFN. JSON f32 weights for the reference checkpoint format.
use crate::{backend::Matrix, model::ModelConfig};
use serde::{Deserialize, Serialize};
use std::io;

#[derive(Clone, Serialize, Deserialize)]
pub struct ExpertWeights {
    pub gate: Matrix,
    pub up: Matrix,
    pub down: Matrix,
}

impl ExpertWeights {
    pub fn validate(&self, config: &ModelConfig) -> io::Result<()> {
        self.gate
            .validate(config.intermediate_size, config.hidden_size)?;
        self.up
            .validate(config.intermediate_size, config.hidden_size)?;
        self.down
            .validate(config.hidden_size, config.intermediate_size)
    }

    pub fn forward(&self, hidden: &[f32]) -> io::Result<Vec<f32>> {
        let gate = self.gate.multiply(hidden)?;
        let up = self.up.multiply(hidden)?;
        if gate.len() != up.len() {
            return Err(crate::backend::invalid("FFN gate/up shape mismatch"));
        }
        let activated: Vec<_> = gate
            .iter()
            .zip(up)
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        self.down.multiply(&activated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn swiglu_matches_scalar_calculation() {
        let matrix = |x| Matrix {
            rows: 1,
            columns: 1,
            values: vec![x],
        };
        let expert = ExpertWeights {
            gate: matrix(2.),
            up: matrix(3.),
            down: matrix(4.),
        };
        let result = expert.forward(&[0.5]).unwrap();
        assert!((result[0] - 6. / (1. + (-1f32).exp())).abs() < 1e-6);
    }
}
