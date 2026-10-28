use super::types::MeanStdNormalizationConfig;
use anyhow::{bail, Result};
use image::DynamicImage;
use ndarray::Array4;

pub const RGB_CHANNELS: usize = 3;

pub fn load_image_from_path(path: &str) -> Result<DynamicImage> {
    let image = image::open(path)?;
    Ok(image)
}

// Non-aspect-preserving (stretch) resize. Box averaging is ~3x faster than
// a Triangle filter on 1080p sources with comparable antialiasing quality for
// downscaling — the win comes from integer averaging instead of float
// convolution.
pub fn resize_image_exact_fast(image: &DynamicImage, dim: (u32, u32)) -> DynamicImage {
    DynamicImage::ImageRgb8(image::imageops::thumbnail(&image.to_rgb8(), dim.0, dim.1))
}

pub fn raw_rgb_to_normalized_tensor(
    rgb_data: &[u8],
    width: u32,
    height: u32,
    config: &MeanStdNormalizationConfig,
    target_shape: &(usize, usize, usize, usize),
) -> Result<Array4<f32>> {
    if rgb_data.len() != (width * height * 3) as usize {
        bail!("RGB data length doesn't match width * height * 3");
    }
    if !matches!(*target_shape, (1, RGB_CHANNELS, h, w) if h == height as usize && w == width as usize)
    {
        bail!(
            "Expected NCHW (1,3,{height},{width}) tensor shape, got {:?}",
            target_shape
        );
    }

    let mut tensor = Array4::<f32>::zeros(*target_shape);
    for y in 0..height as usize {
        for x in 0..width as usize {
            let pixel_idx = (y * width as usize + x) * 3;
            tensor[[0, 0, y, x]] = (rgb_data[pixel_idx] as f32 - config.mean[0]) / config.std[0];
            tensor[[0, 1, y, x]] =
                (rgb_data[pixel_idx + 1] as f32 - config.mean[1]) / config.std[1];
            tensor[[0, 2, y, x]] =
                (rgb_data[pixel_idx + 2] as f32 - config.mean[2]) / config.std[2];
        }
    }

    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_load_image_from_path_with_invalid_path() {
        // Act
        let result = load_image_from_path("invalid_path.jpg");

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_resize_image_exact_fast_stretches_to_dims() {
        // Arrange — 1920x1080 stretched to a square, aspect NOT preserved
        let image = DynamicImage::new_rgb8(1920, 1080);

        // Act
        let resized = resize_image_exact_fast(&image, (384, 384));

        // Assert
        assert_eq!((resized.width(), resized.height()), (384, 384));
    }

    #[test]
    fn test_mean_std_normalization_is_channel_first() {
        // Arrange — 1x2 image: one red pixel, one blue pixel
        let rgb = [255, 0, 0, 0, 0, 255];
        let config = MeanStdNormalizationConfig {
            mean: [0.0, 0.0, 0.0],
            std: [255.0, 255.0, 255.0],
        };

        // Act
        let tensor = raw_rgb_to_normalized_tensor(&rgb, 2, 1, &config, &(1, 3, 1, 2)).unwrap();

        // Assert
        assert_abs_diff_eq!(tensor[[0, 0, 0, 0]], 1.0);
        assert_abs_diff_eq!(tensor[[0, 2, 0, 0]], 0.0);
        assert_abs_diff_eq!(tensor[[0, 0, 0, 1]], 0.0);
        assert_abs_diff_eq!(tensor[[0, 2, 0, 1]], 1.0);
    }

    #[test]
    fn test_mean_std_normalization_subtracts_mean_and_divides_std() {
        // Arrange
        let rgb = [100, 150, 200];
        let config = MeanStdNormalizationConfig {
            mean: [50.0, 50.0, 50.0],
            std: [10.0, 20.0, 50.0],
        };

        // Act
        let tensor = raw_rgb_to_normalized_tensor(&rgb, 1, 1, &config, &(1, 3, 1, 1)).unwrap();

        // Assert
        assert_abs_diff_eq!(tensor[[0, 0, 0, 0]], 5.0);
        assert_abs_diff_eq!(tensor[[0, 1, 0, 0]], 5.0);
        assert_abs_diff_eq!(tensor[[0, 2, 0, 0]], 3.0);
    }

    #[test]
    fn test_normalization_rejects_mismatched_buffer() {
        // Arrange
        let config = MeanStdNormalizationConfig {
            mean: [0.0; 3],
            std: [1.0; 3],
        };

        // Act
        let result = raw_rgb_to_normalized_tensor(&[0; 5], 2, 1, &config, &(1, 3, 1, 2));

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_normalization_rejects_channel_last_shape() {
        // Arrange
        let config = MeanStdNormalizationConfig {
            mean: [0.0; 3],
            std: [1.0; 3],
        };

        // Act
        let result = raw_rgb_to_normalized_tensor(&[0; 6], 2, 1, &config, &(1, 1, 2, 3));

        // Assert
        assert!(result.is_err());
    }
}
