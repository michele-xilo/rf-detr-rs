use super::config::{
    with_default_execution_providers, RfDetrSegVariant, RF_DETR_INPUT_NAME, RF_DETR_MEAN_0_255,
    RF_DETR_STD_0_255,
};
#[cfg(not(target_os = "linux"))]
use super::config::{RF_DETR_DETS_NAME, RF_DETR_LABELS_NAME, RF_DETR_MASKS_NAME};
use super::{SegmentationResult, SegmentorTrait};
use crate::postprocess::process_rf_detr_seg_outputs;
use crate::shared::image::{raw_rgb_to_normalized_tensor, resize_image_exact_fast};
use crate::shared::types::{MeanStdNormalizationConfig, ModelSize};
use anyhow::Result;
use image::DynamicImage;
#[cfg(not(target_os = "linux"))]
use ort::memory::Allocator;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::PathBuf;

pub struct RfDetrSegmentor {
    session: Session,
    variant: RfDetrSegVariant,
    #[cfg(not(target_os = "linux"))]
    io_binding: ort::session::IoBinding,
    #[cfg(not(target_os = "linux"))]
    input_tensor: Tensor<f32>,
}

impl RfDetrSegmentor {
    pub fn with_model_size(
        model_size: ModelSize,
        segment: bool,
        model_dir: impl Into<PathBuf>,
    ) -> Result<Self> {
        let variant = RfDetrSegVariant::from_model_size(&model_size, segment);
        let model_path = model_dir.into().join(variant.model_filename);
        Self::build(model_path, variant)
    }

    // For callers that ship the weights under their own filename, so the
    // variant cannot be inferred from the stem the way `new` does.
    pub fn from_path_with_size(
        model_path: impl Into<PathBuf>,
        model_size: ModelSize,
        segment: bool,
    ) -> Result<Self> {
        let variant = RfDetrSegVariant::from_model_size(&model_size, segment);
        Self::build(model_path.into(), variant)
    }

    fn build(model_path: PathBuf, variant: RfDetrSegVariant) -> Result<Self> {
        let session = with_default_execution_providers(Session::builder()?)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e: ort::Error<_>| ort::Error::<()>::from(e))?
            .commit_from_file(&model_path)?;

        #[cfg(not(target_os = "linux"))]
        let (io_binding, input_tensor) = {
            let mut io_binding = session.create_binding()?;

            let allocator = Allocator::default();
            let shape = variant.input_tensor_shape();
            let input_shape = [
                shape.0 as i64,
                shape.1 as i64,
                shape.2 as i64,
                shape.3 as i64,
            ];
            let input_tensor = Tensor::<f32>::new(&allocator, input_shape)?;
            io_binding.bind_input(RF_DETR_INPUT_NAME, &input_tensor)?;

            let memory_info = allocator.memory_info();
            io_binding.bind_output_to_device(RF_DETR_DETS_NAME, memory_info)?;
            io_binding.bind_output_to_device(RF_DETR_LABELS_NAME, memory_info)?;
            if variant.segment {
                io_binding.bind_output_to_device(RF_DETR_MASKS_NAME, memory_info)?;
            }

            (io_binding, input_tensor)
        };

        Ok(Self {
            session,
            variant,
            #[cfg(not(target_os = "linux"))]
            io_binding,
            #[cfg(not(target_os = "linux"))]
            input_tensor,
        })
    }
}

impl SegmentorTrait for RfDetrSegmentor {
    fn new(model_path: impl Into<PathBuf>) -> Result<Self> {
        let path: PathBuf = model_path.into();
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();

        let model_size = if stem.contains("large") {
            ModelSize::Heavy
        } else if stem.contains("medium") {
            ModelSize::Full
        } else if stem.contains("small") {
            ModelSize::Lite
        } else if stem.contains("nano") {
            ModelSize::Nano
        } else {
            anyhow::bail!(
                "cannot infer RF-DETR-Seg variant from '{}'; use `with_model_size` instead",
                path.display()
            );
        };

        let segment = stem.contains("seg");
        Self::build(
            path,
            RfDetrSegVariant::from_model_size(&model_size, segment),
        )
    }

    fn infer(&mut self, frame: &DynamicImage) -> Result<Vec<SegmentationResult>> {
        let original_width = frame.width();
        let original_height = frame.height();

        // RF-DETR uses non-aspect-preserving resize to (R, R), no padding. Box
        // averaging (~3x faster than Triangle on 1080p) matches training-time
        // bilinear closely enough that detections are unchanged.
        let r = self.variant.resolution;
        let resized = resize_image_exact_fast(frame, (r, r));

        let mean_std = MeanStdNormalizationConfig {
            mean: RF_DETR_MEAN_0_255,
            std: RF_DETR_STD_0_255,
        };
        let input_data = raw_rgb_to_normalized_tensor(
            resized.to_rgb8().as_raw(),
            r,
            r,
            &mean_std,
            &self.variant.input_tensor_shape(),
        )?;

        #[cfg(not(target_os = "linux"))]
        let outputs = {
            let (_, tensor_data) = self.input_tensor.extract_tensor_mut();
            let contiguous;
            let slice = match input_data.as_slice() {
                Some(s) => s,
                None => {
                    contiguous = input_data.as_standard_layout().into_owned();
                    contiguous.as_slice().ok_or_else(|| {
                        anyhow::anyhow!("Failed to get contiguous slice from input tensor")
                    })?
                }
            };
            tensor_data.copy_from_slice(slice);
            self.io_binding.synchronize_inputs()?;

            crate::time_it!(
                "Ort run rf_detr_seg with IO Binding",
                self.session.run_binding(&self.io_binding)?
            )
        };

        #[cfg(target_os = "linux")]
        let outputs = {
            let input_tensor = Tensor::from_array(input_data)?;
            crate::time_it!(
                "Ort run rf_detr_seg",
                self.session
                    .run(ort::inputs![RF_DETR_INPUT_NAME => input_tensor])
            )?
        };

        process_rf_detr_seg_outputs(&outputs, self.variant, original_width, original_height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Detection tests need a real photo with COCO objects in it; none ships
    // with the repo, so point RF_DETR_TEST_IMAGE at one.
    fn test_image_path() -> String {
        std::env::var("RF_DETR_TEST_IMAGE").unwrap_or_else(|_| "assets/test_image.jpg".into())
    }

    #[test]
    #[ignore]
    fn test_rf_detr_segmentor_creation() {
        // Act
        let segmentor = RfDetrSegmentor::with_model_size(ModelSize::Nano, true, "./models");

        // Assert
        assert!(segmentor.is_ok(), "Segmentor creation failed");
    }

    #[test]
    #[ignore]
    fn test_rf_detr_segmentor_infer() {
        // Arrange
        let mut segmentor =
            RfDetrSegmentor::with_model_size(ModelSize::Nano, true, "./models").unwrap();
        let image = crate::shared::image::load_image_from_path(&test_image_path()).unwrap();

        // Act
        let results = segmentor.infer(&image);

        // Assert
        assert!(results.is_ok(), "Segmentor inference failed");
        let results = results.unwrap();
        assert!(!results.is_empty(), "No segmentation results");
        for result in &results {
            assert!(result.confidence > 0.0);
            assert!(!result.class_name.is_empty());
            let mask = result.mask.as_ref().expect("seg mode yields a mask");
            assert!(mask.width() > 0);
            assert!(mask.height() > 0);
        }
    }

    #[test]
    #[ignore]
    fn test_rf_detr_detector_infer_has_no_mask() {
        // Arrange
        let mut detector =
            RfDetrSegmentor::with_model_size(ModelSize::Nano, false, "./models").unwrap();
        let image = crate::shared::image::load_image_from_path(&test_image_path()).unwrap();

        // Act
        let results = detector.infer(&image).unwrap();

        // Assert — detection mode returns boxes but never masks/contours
        assert!(!results.is_empty(), "No detection results");
        for result in &results {
            assert!(result.mask.is_none());
            assert!(result.contour.is_empty());
        }
    }
}
