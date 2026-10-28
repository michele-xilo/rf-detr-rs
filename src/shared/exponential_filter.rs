pub struct EmaFilter {
    alpha: f32,
    state: Option<f32>,
}

impl EmaFilter {
    pub fn new(alpha: f32) -> Self {
        Self { alpha, state: None }
    }

    pub fn apply(&mut self, value: f32) -> f32 {
        let smoothed = match self.state {
            Some(prev) => self.alpha * value + (1.0 - self.alpha) * prev,
            None => value,
        };
        self.state = Some(smoothed);
        smoothed
    }

    pub fn reset(&mut self) {
        self.state = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_value_passes_through() {
        // Arrange
        let mut filter = EmaFilter::new(0.3);

        // Act
        let result = filter.apply(0.8);

        // Assert
        assert!((result - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_smoothing_blends_values() {
        // Arrange
        let mut filter = EmaFilter::new(0.3);
        filter.apply(0.9);

        // Act
        let smoothed = filter.apply(0.5);

        // Assert — 0.3 * 0.5 + 0.7 * 0.9 = 0.78
        assert!((smoothed - 0.78).abs() < 1e-5);
    }

    #[test]
    fn test_reset_clears_state() {
        // Arrange
        let mut filter = EmaFilter::new(0.3);
        filter.apply(0.9);
        filter.reset();

        // Act
        let result = filter.apply(0.5);

        // Assert — after reset, value should pass through raw
        assert!((result - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_convergence_over_many_frames() {
        // Arrange
        let mut filter = EmaFilter::new(0.2);
        filter.apply(1.0);

        // Act — feed constant 0.0 for many frames
        let mut val = 0.0;
        for _ in 0..50 {
            val = filter.apply(0.0);
        }

        // Assert — should converge toward 0
        assert!(val < 0.01);
    }
}
