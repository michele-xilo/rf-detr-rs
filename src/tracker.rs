use super::{SegmentationResult, SegmentorTrait};
use crate::shared::exponential_filter::EmaFilter;
use anyhow::Result;
use image::DynamicImage;
use rustc_hash::FxHashMap;

const IOU_MATCH_THRESHOLD: f32 = 0.3;
const CONFIDENCE_ALPHA: f32 = 0.2;

#[derive(Debug, Clone)]
pub struct TrackedSegmentation {
    pub instance_id: u32,
    pub result: SegmentationResult,
}

pub struct SegmentationTracker<S: SegmentorTrait> {
    segmentor: S,
    next_instance_id: u32,
    previous_results: Vec<TrackedSegmentation>,
    confidence_filters: FxHashMap<u32, EmaFilter>,
}

impl<S: SegmentorTrait> SegmentationTracker<S> {
    pub fn new(segmentor: S) -> Self {
        Self {
            segmentor,
            next_instance_id: 0,
            previous_results: Vec::new(),
            confidence_filters: FxHashMap::default(),
        }
    }

    pub fn track(&mut self, frame: &DynamicImage) -> Result<Vec<TrackedSegmentation>> {
        let new_results = self.segmentor.infer(frame)?;

        if self.previous_results.is_empty() {
            // First frame: assign fresh IDs to all detections
            let tracked: Vec<TrackedSegmentation> = new_results
                .into_iter()
                .map(|result| {
                    let id = self.next_instance_id;
                    self.next_instance_id += 1;
                    let mut filter = EmaFilter::new(CONFIDENCE_ALPHA);
                    filter.apply(result.confidence);
                    self.confidence_filters.insert(id, filter);
                    TrackedSegmentation {
                        instance_id: id,
                        result,
                    }
                })
                .collect();
            self.previous_results = tracked.clone();
            return Ok(tracked);
        }

        // Greedy IoU matching: match new detections to previous tracked instances
        let mut used_prev = vec![false; self.previous_results.len()];
        let mut tracked = Vec::with_capacity(new_results.len());

        // Build match candidates sorted by IoU descending
        let mut candidates: Vec<(usize, usize, f32)> = Vec::new();
        for (new_idx, new_result) in new_results.iter().enumerate() {
            for (prev_idx, prev_tracked) in self.previous_results.iter().enumerate() {
                // Only match same class
                if new_result.class_id != prev_tracked.result.class_id {
                    continue;
                }
                let iou = new_result.bbox.calculate_iou(&prev_tracked.result.bbox);
                if iou > IOU_MATCH_THRESHOLD {
                    candidates.push((new_idx, prev_idx, iou));
                }
            }
        }
        candidates.sort_by(|a, b| b.2.total_cmp(&a.2));

        let mut assigned_new = vec![false; new_results.len()];
        let mut assignments: Vec<Option<u32>> = vec![None; new_results.len()];

        for (new_idx, prev_idx, _iou) in &candidates {
            if assigned_new[*new_idx] || used_prev[*prev_idx] {
                continue;
            }
            assignments[*new_idx] = Some(self.previous_results[*prev_idx].instance_id);
            assigned_new[*new_idx] = true;
            used_prev[*prev_idx] = true;
        }

        // Build final tracked results with EMA-smoothed confidence
        let mut new_filters = FxHashMap::default();
        for (idx, mut result) in new_results.into_iter().enumerate() {
            let instance_id = match assignments[idx] {
                Some(id) => id,
                None => {
                    let id = self.next_instance_id;
                    self.next_instance_id += 1;
                    id
                }
            };

            // Apply EMA: reuse existing filter for matched instances, create new for unmatched
            let mut filter = self
                .confidence_filters
                .remove(&instance_id)
                .unwrap_or_else(|| EmaFilter::new(CONFIDENCE_ALPHA));
            result.confidence = filter.apply(result.confidence);
            new_filters.insert(instance_id, filter);

            tracked.push(TrackedSegmentation {
                instance_id,
                result,
            });
        }

        self.confidence_filters = new_filters;
        self.previous_results = tracked.clone();
        Ok(tracked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::types::{Bbox, BboxXYXY, Landmark2D};
    use crate::SegmentationMask;

    struct MockSegmentor {
        results: Vec<Vec<SegmentationResult>>,
        call_idx: usize,
    }

    impl SegmentorTrait for MockSegmentor {
        fn new(_model_path: impl Into<std::path::PathBuf>) -> Result<Self> {
            Ok(Self {
                results: vec![],
                call_idx: 0,
            })
        }

        fn infer(&mut self, _frame: &DynamicImage) -> Result<Vec<SegmentationResult>> {
            let idx = self.call_idx.min(self.results.len() - 1);
            self.call_idx += 1;
            Ok(self.results[idx].clone())
        }
    }

    fn make_result(class_id: u32, x1: f32, y1: f32, x2: f32, y2: f32) -> SegmentationResult {
        make_result_with_confidence(class_id, x1, y1, x2, y2, 0.9)
    }

    fn make_result_with_confidence(
        class_id: u32,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        confidence: f32,
    ) -> SegmentationResult {
        SegmentationResult {
            class_id,
            class_name: "test".to_string(),
            confidence,
            bbox: Bbox::BboxXYXY(BboxXYXY { x1, y1, x2, y2 }),
            mask: Some(SegmentationMask::new(1, 1, vec![255])),
            contour: vec![Landmark2D { x: x1, y: y1 }],
        }
    }

    #[test]
    fn test_tracker_assigns_ids_on_first_frame() {
        // Arrange
        let segmentor = MockSegmentor {
            results: vec![vec![
                make_result(0, 0.0, 0.0, 100.0, 100.0),
                make_result(1, 200.0, 200.0, 300.0, 300.0),
            ]],
            call_idx: 0,
        };
        let mut tracker = SegmentationTracker::new(segmentor);
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(640, 480));

        // Act
        let tracked = tracker.track(&image).unwrap();

        // Assert
        assert_eq!(tracked.len(), 2);
        assert_eq!(tracked[0].instance_id, 0);
        assert_eq!(tracked[1].instance_id, 1);
    }

    #[test]
    fn test_tracker_maintains_ids_across_frames() {
        // Arrange
        let segmentor = MockSegmentor {
            results: vec![
                vec![make_result(0, 100.0, 100.0, 200.0, 200.0)],
                // Slightly moved object
                vec![make_result(0, 110.0, 110.0, 210.0, 210.0)],
            ],
            call_idx: 0,
        };
        let mut tracker = SegmentationTracker::new(segmentor);
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(640, 480));

        // Act
        let first = tracker.track(&image).unwrap();
        let second = tracker.track(&image).unwrap();

        // Assert
        assert_eq!(first[0].instance_id, 0);
        assert_eq!(second[0].instance_id, 0); // Same ID preserved
    }

    #[test]
    fn test_tracker_assigns_new_id_for_new_object() {
        // Arrange
        let segmentor = MockSegmentor {
            results: vec![
                vec![make_result(0, 100.0, 100.0, 200.0, 200.0)],
                // Original object + new far-away object
                vec![
                    make_result(0, 110.0, 110.0, 210.0, 210.0),
                    make_result(1, 500.0, 500.0, 600.0, 600.0),
                ],
            ],
            call_idx: 0,
        };
        let mut tracker = SegmentationTracker::new(segmentor);
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(640, 480));

        // Act
        tracker.track(&image).unwrap();
        let second = tracker.track(&image).unwrap();

        // Assert
        assert_eq!(second[0].instance_id, 0); // Matched
        assert_eq!(second[1].instance_id, 1); // New ID
    }

    #[test]
    fn test_confidence_smoothing_across_frames() {
        // Arrange: same object with confidence 0.9 then 0.5
        let segmentor = MockSegmentor {
            results: vec![
                vec![make_result_with_confidence(
                    0, 100.0, 100.0, 200.0, 200.0, 0.9,
                )],
                vec![make_result_with_confidence(
                    0, 110.0, 110.0, 210.0, 210.0, 0.5,
                )],
            ],
            call_idx: 0,
        };
        let mut tracker = SegmentationTracker::new(segmentor);
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(640, 480));

        // Act
        let first = tracker.track(&image).unwrap();
        let second = tracker.track(&image).unwrap();

        // Assert: first frame uses raw confidence
        assert!((first[0].result.confidence - 0.9).abs() < 1e-5);
        // Second frame: EMA = 0.2 * 0.5 + 0.8 * 0.9 = 0.82
        let expected = CONFIDENCE_ALPHA * 0.5 + (1.0 - CONFIDENCE_ALPHA) * 0.9;
        assert!(
            (second[0].result.confidence - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            second[0].result.confidence
        );
    }

    #[test]
    fn test_new_object_gets_raw_confidence() {
        // Arrange: first frame has one object, second frame adds a new one
        let segmentor = MockSegmentor {
            results: vec![
                vec![make_result_with_confidence(
                    0, 100.0, 100.0, 200.0, 200.0, 0.9,
                )],
                vec![
                    make_result_with_confidence(0, 110.0, 110.0, 210.0, 210.0, 0.8),
                    make_result_with_confidence(1, 500.0, 500.0, 600.0, 600.0, 0.7),
                ],
            ],
            call_idx: 0,
        };
        let mut tracker = SegmentationTracker::new(segmentor);
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(640, 480));

        // Act
        tracker.track(&image).unwrap();
        let second = tracker.track(&image).unwrap();

        // Assert: new object (id=1) should have raw confidence, no smoothing
        let new_obj = second.iter().find(|t| t.instance_id == 1).unwrap();
        assert!((new_obj.result.confidence - 0.7).abs() < 1e-5);
    }
}
