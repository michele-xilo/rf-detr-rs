use crate::shared::execution_providers::register;
use crate::shared::types::ModelSize;
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    target_os = "windows"
))]
use ort::execution_providers::CUDAExecutionProvider;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use ort::execution_providers::CoreMLExecutionProvider;
#[cfg(target_os = "windows")]
use ort::execution_providers::DirectMLExecutionProvider;
use ort::session::builder::SessionBuilder;

pub const RF_DETR_INPUT_NAME: &str = "input";
pub const RF_DETR_DETS_NAME: &str = "dets";
pub const RF_DETR_LABELS_NAME: &str = "labels";
pub const RF_DETR_MASKS_NAME: &str = "masks";

// Pre-scaled to 0-255 because `raw_rgb_to_normalized_tensor` subtracts mean
// from raw u8 pixel values rather than scaling first.
pub const RF_DETR_MEAN_0_255: [f32; 3] = [0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0];
pub const RF_DETR_STD_0_255: [f32; 3] = [0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0];

// The classification head emits `num_classes + 1` slots — the extra slot is the
// DETR-convention "no-object" class (lwdetr.py:404). Slots 0..89 map to COCO ids
// 1..90 via COCO_CLASS_NAMES; slot 90 is out-of-bounds in that table and is
// naturally filtered by the empty-name check in postprocess.
pub const RF_DETR_NUM_CLASSES: usize = 91;
pub const RF_DETR_MASK_DOWNSAMPLE_RATIO: u32 = 4;

pub const RF_DETR_CONFIDENCE_THRESHOLD: f32 = 0.5;
// Logit-space threshold equivalent to sigmoid > 0.5 — saves a sigmoid per pixel
// in the mask upsample loop.
pub const RF_DETR_MASK_LOGIT_THRESHOLD: f32 = 0.0;

#[derive(Debug, Clone, Copy)]
pub struct RfDetrSegVariant {
    pub model_filename: &'static str,
    pub resolution: u32,
    pub num_queries: usize,
    // Detection and seg exports share the same backbone/inputs/dets/labels; the
    // seg export just adds a `masks` output. When false we run the lighter
    // detection-only model and skip all mask work.
    pub segment: bool,
}

impl RfDetrSegVariant {
    pub const fn from_model_size(model_size: &ModelSize, segment: bool) -> Self {
        // Detection and seg exports have their own native resolutions and query
        // counts (see rfdetr/config.py); they are not interchangeable.
        let (model_filename, resolution, num_queries) = match (model_size, segment) {
            (ModelSize::Nano, true) => ("rf-detr-seg-nano.onnx", 312, 100),
            (ModelSize::Nano, false) => ("rf-detr-nano.onnx", 384, 300),
            (ModelSize::Lite, true) => ("rf-detr-seg-small.onnx", 384, 100),
            (ModelSize::Lite, false) => ("rf-detr-small.onnx", 512, 300),
            (ModelSize::Medium | ModelSize::Full, true) => ("rf-detr-seg-medium.onnx", 432, 200),
            (ModelSize::Medium | ModelSize::Full, false) => ("rf-detr-medium.onnx", 576, 300),
            (ModelSize::Large | ModelSize::Heavy, true) => ("rf-detr-seg-large.onnx", 504, 200),
            (ModelSize::Large | ModelSize::Heavy, false) => ("rf-detr-large.onnx", 704, 300),
        };
        Self {
            model_filename,
            resolution,
            num_queries,
            segment,
        }
    }

    pub const fn input_tensor_shape(&self) -> (usize, usize, usize, usize) {
        (1, 3, self.resolution as usize, self.resolution as usize)
    }

    pub const fn mask_resolution(&self) -> usize {
        (self.resolution / RF_DETR_MASK_DOWNSAMPLE_RATIO) as usize
    }
}

pub fn with_default_execution_providers(builder: SessionBuilder) -> ort::Result<SessionBuilder> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let coreml_provider = {
        // RF-DETR's DINOv2 backbone fragments into ~110 CPU↔GPU partitions under
        // CoreML's legacy NeuralNetwork model format (~6× slower than necessary).
        // MLProgram collapses that to ~16 partitions and supports more ops, so
        // we always opt in here. ANE is excluded: it slows RF-DETR down.
        let mut provider = CoreMLExecutionProvider::default()
            .with_compute_units(ort::execution_providers::coreml::ComputeUnits::CPUAndGPU)
            .with_static_input_shapes(true)
            .with_model_format(ort::execution_providers::coreml::ModelFormat::MLProgram);

        // Compiling RF-DETR's MLProgram takes ~1 min; cache the compiled model
        // so only the first launch pays it.
        if let Some(home) = std::env::var_os("HOME") {
            let cache_dir = std::path::Path::new(&home).join("Library/Caches/rf-detr/coreml");
            if std::fs::create_dir_all(&cache_dir).is_ok() {
                provider = provider.with_model_cache_dir(cache_dir.to_string_lossy());
            }
        }

        provider
    };

    register(
        builder,
        vec![
            #[cfg(any(
                all(target_os = "linux", target_arch = "x86_64"),
                target_os = "windows"
            ))]
            Box::new(CUDAExecutionProvider::default()),
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            Box::new(coreml_provider),
            #[cfg(target_os = "windows")]
            Box::new(DirectMLExecutionProvider::default()),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use parameterized::parameterized;

    #[parameterized(
        size = {ModelSize::Nano, ModelSize::Lite, ModelSize::Medium, ModelSize::Full, ModelSize::Large, ModelSize::Heavy},
        expected_res = {312_u32, 384, 432, 432, 504, 504},
        expected_queries = {100_usize, 100, 200, 200, 200, 200},
    )]
    fn test_variant_from_model_size(size: ModelSize, expected_res: u32, expected_queries: usize) {
        // Act
        let variant = RfDetrSegVariant::from_model_size(&size, true);

        // Assert
        assert_eq!(variant.resolution, expected_res);
        assert_eq!(variant.num_queries, expected_queries);
        assert_eq!(variant.input_tensor_shape().2, expected_res as usize);
        assert_eq!(variant.input_tensor_shape().3, expected_res as usize);
    }

    #[test]
    fn test_detection_variant_drops_seg_prefix() {
        // Act
        let seg = RfDetrSegVariant::from_model_size(&ModelSize::Nano, true);
        let det = RfDetrSegVariant::from_model_size(&ModelSize::Nano, false);

        // Assert — different file and native geometry, seg flag reflects the choice
        assert!(seg.model_filename.contains("rf-detr-seg-"));
        assert!(!det.model_filename.contains("rf-detr-seg-"));
        assert_eq!((seg.resolution, seg.num_queries), (312, 100));
        assert_eq!((det.resolution, det.num_queries), (384, 300));
        assert!(seg.segment && !det.segment);
    }

    #[test]
    fn test_mask_resolution_is_input_over_four() {
        // Arrange
        let variant = RfDetrSegVariant::from_model_size(&ModelSize::Heavy, true);

        // Act
        let mask_res = variant.mask_resolution();

        // Assert — 504 / 4 = 126
        assert_eq!(mask_res, 126);
    }
}
