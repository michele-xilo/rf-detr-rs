pub mod coco_labels;
pub mod config;
pub mod postprocess;
pub mod segmentor;
pub mod shared;
pub mod tracker;
pub mod visualization;

pub use postprocess::process_rf_detr_seg_outputs;
pub use segmentor::RfDetrSegmentor;
pub use tracker::{SegmentationTracker, TrackedSegmentation};
pub use visualization::draw_segmentation_on_image;

use crate::shared::types::{Bbox, Landmark2D};
use anyhow::Result;
use image::DynamicImage;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct SegmentationMask {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

impl SegmentationMask {
    pub fn try_new(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        if data.len() != width as usize * height as usize {
            return Err(anyhow::anyhow!(
                "SegmentationMask: buffer length {} != width*height ({}*{}={})",
                data.len(),
                width,
                height,
                width as usize * height as usize,
            ));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        assert_eq!(
            data.len(),
            width as usize * height as usize,
            "SegmentationMask buffer length mismatch"
        );
        Self {
            width,
            height,
            data,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct SegmentationResult {
    pub class_id: u32,
    pub class_name: String,
    pub confidence: f32,
    pub bbox: Bbox,
    // `None` in detection-only mode; `Some` when the seg model ran.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub mask: Option<SegmentationMask>,
    pub contour: Vec<Landmark2D>,
}

pub trait SegmentorTrait {
    fn new(model_path: impl Into<PathBuf>) -> Result<Self>
    where
        Self: Sized;
    fn infer(&mut self, frame: &DynamicImage) -> Result<Vec<SegmentationResult>>;
}
