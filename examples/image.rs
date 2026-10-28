use anyhow::{anyhow, Result};
use rf_detr::shared::image::load_image_from_path;
use rf_detr::shared::types::ModelSize;
use rf_detr::{draw_segmentation_on_image, RfDetrSegmentor, SegmentorTrait};
use tracing::info;

pub const OUTPUT_PATH: &str = "assets/outputs";

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let image_input_path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("usage: image <path/to/image.jpg> [--segment]"))?;
    // Detection only by default; `--segment` runs RF-DETR-Seg for instance masks.
    let segment = std::env::args().any(|a| a == "--segment");

    let mut segmentor = RfDetrSegmentor::with_model_size(ModelSize::Heavy, segment, "./models")?;
    info!("Successfully initialized RF-DETR (segment={segment})");

    let frame = load_image_from_path(&image_input_path)?;
    info!("Loaded image from {}", image_input_path);

    let results = segmentor.infer(&frame)?;
    info!("Found {} objects", results.len());

    for result in &results {
        info!(
            "  {} (class {}) — confidence: {:.2}, bbox: {:?}, contour points: {}",
            result.class_name,
            result.class_id,
            result.confidence,
            result.bbox.to_xyxy(),
            result.contour.len()
        );
    }

    let output_image = draw_segmentation_on_image(&frame, &results)?;
    std::fs::create_dir_all(OUTPUT_PATH)?;
    let stem = std::path::Path::new(&image_input_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image");
    let output_path = format!("{OUTPUT_PATH}/rf_detr_{stem}.png");
    output_image.save(&output_path)?;
    info!("Saved segmentation output to {}", output_path);

    Ok(())
}
