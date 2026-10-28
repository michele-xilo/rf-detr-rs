use super::coco_labels::COCO_CLASS_NAMES;
use crate::config::{
    RfDetrSegVariant, RF_DETR_CONFIDENCE_THRESHOLD, RF_DETR_DETS_NAME, RF_DETR_LABELS_NAME,
    RF_DETR_MASKS_NAME, RF_DETR_MASK_LOGIT_THRESHOLD, RF_DETR_NUM_CLASSES,
};
use crate::shared::math::sigmoid;
use crate::shared::types::{Bbox, BboxXYXY, Landmark2D};
use crate::{SegmentationMask, SegmentationResult};
use anyhow::{anyhow, Result};
use ort::session::SessionOutputs;

pub fn process_rf_detr_seg_outputs(
    outputs: &SessionOutputs,
    variant: RfDetrSegVariant,
    original_width: u32,
    original_height: u32,
) -> Result<Vec<SegmentationResult>> {
    let (_, dets) = outputs[RF_DETR_DETS_NAME].try_extract_tensor::<f32>()?;
    let (_, labels) = outputs[RF_DETR_LABELS_NAME].try_extract_tensor::<f32>()?;

    let q = variant.num_queries;
    let c = RF_DETR_NUM_CLASSES;
    let mask_h = variant.mask_resolution();
    let mask_w = variant.mask_resolution();

    if dets.len() != q * 4 {
        return Err(anyhow!(
            "dets length {} does not match num_queries*4 = {}",
            dets.len(),
            q * 4
        ));
    }
    if labels.len() != q * c {
        return Err(anyhow!(
            "labels length {} does not match num_queries*num_classes = {}",
            labels.len(),
            q * c
        ));
    }

    // Only the seg export carries a masks output; detection mode never reads it.
    let masks: Option<&[f32]> = if variant.segment {
        let (_, masks) = outputs[RF_DETR_MASKS_NAME].try_extract_tensor::<f32>()?;
        if masks.len() != q * mask_h * mask_w {
            return Err(anyhow!(
                "masks length {} does not match num_queries*mask_h*mask_w = {}",
                masks.len(),
                q * mask_h * mask_w
            ));
        }
        Some(masks)
    } else {
        None
    };

    let scored = top_k_indices(labels, q, c, q);

    let orig_w_f = original_width as f32;
    let orig_h_f = original_height as f32;

    let mut results = Vec::with_capacity(scored.len());
    for cand in scored {
        if cand.score < RF_DETR_CONFIDENCE_THRESHOLD {
            continue;
        }

        let class_name = COCO_CLASS_NAMES
            .get(cand.class_idx)
            .copied()
            .unwrap_or_default();
        if class_name.is_empty() {
            continue;
        }
        let det_base = cand.query_idx * 4;
        let cx = dets[det_base];
        let cy = dets[det_base + 1];
        let w = dets[det_base + 2];
        let h = dets[det_base + 3];

        let x1 = ((cx - w / 2.0) * orig_w_f).max(0.0);
        let y1 = ((cy - h / 2.0) * orig_h_f).max(0.0);
        let x2 = ((cx + w / 2.0) * orig_w_f).min(orig_w_f);
        let y2 = ((cy + h / 2.0) * orig_h_f).min(orig_h_f);

        if x2 <= x1 || y2 <= y1 {
            continue;
        }

        let bbox = Bbox::BboxXYXY(BboxXYXY { x1, y1, x2, y2 });

        let (mask, contour) = match masks {
            Some(masks) => {
                let mask_offset = cand.query_idx * mask_h * mask_w;
                let query_mask = &masks[mask_offset..mask_offset + mask_h * mask_w];
                let cropped_mask = upsample_mask_to_bbox(
                    query_mask,
                    mask_w,
                    mask_h,
                    x1,
                    y1,
                    x2,
                    y2,
                    original_width,
                    original_height,
                );
                let contour = extract_contour(&cropped_mask, x1, y1);
                (Some(cropped_mask), contour)
            }
            None => (None, Vec::new()),
        };

        results.push(SegmentationResult {
            class_id: cand.class_idx as u32,
            class_name: class_name.to_string(),
            confidence: cand.score,
            bbox,
            mask,
            contour,
        });
    }

    results.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    Ok(results)
}

#[derive(Debug, Clone, Copy)]
struct ScoredQuery {
    score: f32,
    query_idx: usize,
    class_idx: usize,
}

fn top_k_indices(
    labels: &[f32],
    num_queries: usize,
    num_classes: usize,
    k: usize,
) -> Vec<ScoredQuery> {
    // For each query pick its best-scoring class, so every query contributes
    // at most one candidate to the final ranked list.
    let mut per_query: Vec<ScoredQuery> = (0..num_queries)
        .filter_map(|qi| {
            let base = qi * num_classes;
            (0..num_classes)
                .map(|ci| ScoredQuery {
                    score: sigmoid(labels[base + ci]),
                    query_idx: qi,
                    class_idx: ci,
                })
                .max_by(|a, b| a.score.total_cmp(&b.score))
        })
        .collect();

    let k = k.min(per_query.len());
    if k == 0 {
        return Vec::new();
    }
    let pivot = (k - 1).min(per_query.len() - 1);
    per_query.select_nth_unstable_by(pivot, |a, b| b.score.total_cmp(&a.score));
    per_query.truncate(k);
    per_query.sort_by(|a, b| b.score.total_cmp(&a.score));
    per_query
}

#[allow(clippy::too_many_arguments)]
pub fn upsample_mask_to_bbox(
    mask: &[f32],
    mask_w: usize,
    mask_h: usize,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    original_width: u32,
    original_height: u32,
) -> SegmentationMask {
    let bbox_w = ((x2 - x1).round() as u32).max(1);
    let bbox_h = ((y2 - y1).round() as u32).max(1);

    let x_scale = mask_w as f32 / original_width as f32;
    let y_scale = mask_h as f32 / original_height as f32;
    let max_sx = (mask_w - 1) as f32;
    let max_sy = (mask_h - 1) as f32;

    let mut data = vec![0u8; (bbox_w * bbox_h) as usize];

    for py in 0..bbox_h {
        let orig_y = y1 + py as f32;
        let src_y = (orig_y * y_scale).clamp(0.0, max_sy);
        let sy0 = src_y as usize;
        let sy1 = (sy0 + 1).min(mask_h - 1);
        let fy = src_y - sy0 as f32;

        for px in 0..bbox_w {
            let orig_x = x1 + px as f32;
            let src_x = (orig_x * x_scale).clamp(0.0, max_sx);
            let sx0 = src_x as usize;
            let sx1 = (sx0 + 1).min(mask_w - 1);
            let fx = src_x - sx0 as f32;

            let v00 = mask[sy0 * mask_w + sx0];
            let v10 = mask[sy0 * mask_w + sx1];
            let v01 = mask[sy1 * mask_w + sx0];
            let v11 = mask[sy1 * mask_w + sx1];

            let val = v00 * (1.0 - fx) * (1.0 - fy)
                + v10 * fx * (1.0 - fy)
                + v01 * (1.0 - fx) * fy
                + v11 * fx * fy;

            if val > RF_DETR_MASK_LOGIT_THRESHOLD {
                data[(py * bbox_w + px) as usize] = 255;
            }
        }
    }

    SegmentationMask::new(bbox_w, bbox_h, data)
}

pub fn extract_contour(mask: &SegmentationMask, offset_x: f32, offset_y: f32) -> Vec<Landmark2D> {
    let w = mask.width as usize;
    let h = mask.height as usize;
    let mut contour = Vec::new();

    for y in 0..h {
        for x in 0..w {
            if mask.data[y * w + x] == 0 {
                continue;
            }
            let is_boundary = x == 0
                || y == 0
                || x == w - 1
                || y == h - 1
                || mask.data[y * w + (x - 1)] == 0
                || mask.data[y * w + (x + 1)] == 0
                || mask.data[(y - 1) * w + x] == 0
                || mask.data[(y + 1) * w + x] == 0;

            if is_boundary {
                contour.push(Landmark2D {
                    x: x as f32 + offset_x,
                    y: y as f32 + offset_y,
                });
            }
        }
    }

    contour
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_variant() -> RfDetrSegVariant {
        RfDetrSegVariant::from_model_size(&crate::shared::types::ModelSize::Nano, true)
    }

    #[test]
    fn test_top_k_each_query_at_most_once() {
        // Arrange — 2 queries × 2 classes; query 0 has two high-scoring classes
        let labels = vec![
            5.0, 4.0, // query 0: both classes score high
            -10.0, -10.0, // query 1: both classes score low
        ];

        // Act — request all possible (query, class) pairs
        let top = top_k_indices(&labels, 2, 2, 4);

        // Assert — each query contributes at most one entry
        let unique_queries: rustc_hash::FxHashSet<usize> =
            top.iter().map(|s| s.query_idx).collect();
        assert_eq!(
            top.len(),
            unique_queries.len(),
            "duplicate query_idx in top_k result"
        );
    }

    #[test]
    fn test_top_k_picks_highest_scored_slots() {
        // Arrange — 2 queries × 3 classes; highest logit at (q=1, c=2)
        let labels = vec![
            -10.0, -10.0, -10.0, // query 0
            -10.0, -10.0, 5.0, // query 1: class 2 wins
        ];

        // Act
        let top = top_k_indices(&labels, 2, 3, 2);

        // Assert
        assert_eq!(top[0].query_idx, 1);
        assert_eq!(top[0].class_idx, 2);
        assert!(top[0].score > 0.9);
        assert!(top[1].score < 0.1);
    }

    #[test]
    fn test_upsample_mask_identity_threshold() {
        // Arrange — 4×4 logits all positive → entire bbox should be foreground
        let mask = vec![1.0_f32; 16];

        // Act
        let upsampled = upsample_mask_to_bbox(&mask, 4, 4, 0.0, 0.0, 4.0, 4.0, 4, 4);

        // Assert
        assert_eq!(upsampled.width, 4);
        assert_eq!(upsampled.height, 4);
        assert!(upsampled.data.iter().all(|&v| v == 255));
    }

    #[test]
    fn test_upsample_mask_negative_logits_are_background() {
        // Arrange
        let mask = vec![-1.0_f32; 16];

        // Act
        let upsampled = upsample_mask_to_bbox(&mask, 4, 4, 0.0, 0.0, 4.0, 4.0, 4, 4);

        // Assert
        assert!(upsampled.data.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_extract_contour_single_pixel() {
        // Arrange
        let mask = SegmentationMask::new(3, 3, vec![0, 0, 0, 0, 255, 0, 0, 0, 0]);

        // Act
        let contour = extract_contour(&mask, 10.0, 20.0);

        // Assert
        assert_eq!(contour.len(), 1);
        assert_eq!(contour[0].x, 11.0);
        assert_eq!(contour[0].y, 21.0);
    }

    #[test]
    fn test_extract_contour_filled_square() {
        // Arrange
        let mask = SegmentationMask::new(4, 4, vec![255; 16]);

        // Act
        let contour = extract_contour(&mask, 0.0, 0.0);

        // Assert — 4x4 grid has 12 boundary pixels (4 interior pixels are NOT boundary)
        assert_eq!(contour.len(), 12);
    }

    #[test]
    fn test_variant_dummy_shapes() {
        // Arrange
        let v = dummy_variant();

        // Assert — sanity: nano = 312x312 / 4 = 78
        assert_eq!(v.mask_resolution(), 78);
    }
}
