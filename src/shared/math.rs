pub fn sigmoid(x: f32) -> f32 {
    if x > 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sigmoid() {
        let sigmoid_value = sigmoid(0.0);
        assert_abs_diff_eq!(sigmoid_value, 0.5, epsilon = 1e-6);
        let sigmoid_value = sigmoid(1.0);
        assert_abs_diff_eq!(sigmoid_value, 0.731_058_6, epsilon = 1e-6);
        let sigmoid_value = sigmoid(-1.0);
        assert_abs_diff_eq!(sigmoid_value, 0.268_941_43, epsilon = 1e-6);
    }
}
