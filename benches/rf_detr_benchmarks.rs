use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::DynamicImage;
use rf_detr::config::RfDetrSegVariant;
use rf_detr::postprocess::{extract_contour, upsample_mask_to_bbox};
use rf_detr::shared::types::{Bbox, BboxXYXY, ModelSize};
use rf_detr::{draw_segmentation_on_image, SegmentationMask, SegmentationResult};

fn synthetic_query_mask(mask_w: usize, mask_h: usize) -> Vec<f32> {
    (0..mask_w * mask_h)
        .map(|i| {
            let y = (i / mask_w) as f32 / mask_h as f32;
            let x = (i % mask_w) as f32 / mask_w as f32;
            (x - 0.5) * 2.0 + (y - 0.5) * 2.0
        })
        .collect()
}

fn make_result(x1: f32, y1: f32, x2: f32, y2: f32, class_id: u32) -> SegmentationResult {
    let w = (x2 - x1) as u32;
    let h = (y2 - y1) as u32;
    let mut mask_data = vec![0u8; (w * h) as usize];
    for py in h / 4..3 * h / 4 {
        for px in w / 4..3 * w / 4 {
            mask_data[(py * w + px) as usize] = 255;
        }
    }
    let mask = SegmentationMask::new(w, h, mask_data);
    let contour = extract_contour(&mask, x1, y1);
    let class_names = ["person", "bicycle", "car", "motorcycle", "bus"];
    SegmentationResult {
        class_id,
        class_name: class_names[class_id as usize % class_names.len()].to_string(),
        confidence: 0.85,
        bbox: Bbox::BboxXYXY(BboxXYXY { x1, y1, x2, y2 }),
        mask: Some(mask),
        contour,
    }
}

fn bench_upsample_mask_full_frame(c: &mut Criterion) {
    let variant = RfDetrSegVariant::from_model_size(&ModelSize::Heavy, true);
    let mw = variant.mask_resolution();
    let mh = variant.mask_resolution();
    let mask = synthetic_query_mask(mw, mh);
    c.bench_function("upsample_mask_to_bbox_full_frame_1280x720", |b| {
        b.iter(|| {
            black_box(upsample_mask_to_bbox(
                black_box(&mask),
                mw,
                mh,
                0.0,
                0.0,
                1280.0,
                720.0,
                1280,
                720,
            ))
        })
    });
}

fn bench_upsample_mask_small_bbox(c: &mut Criterion) {
    let variant = RfDetrSegVariant::from_model_size(&ModelSize::Heavy, true);
    let mw = variant.mask_resolution();
    let mh = variant.mask_resolution();
    let mask = synthetic_query_mask(mw, mh);
    c.bench_function("upsample_mask_to_bbox_small_100x100", |b| {
        b.iter(|| {
            black_box(upsample_mask_to_bbox(
                black_box(&mask),
                mw,
                mh,
                400.0,
                300.0,
                500.0,
                400.0,
                1280,
                720,
            ))
        })
    });
}

fn bench_extract_contour(c: &mut Criterion) {
    let mask_data: Vec<u8> = (0..720usize * 1280)
        .map(|i| {
            let y = i / 1280;
            let x = i % 1280;
            if x > 320 && x < 960 && y > 180 && y < 540 {
                255
            } else {
                0
            }
        })
        .collect();
    let mask = SegmentationMask::new(1280, 720, mask_data);
    c.bench_function("extract_contour_1280x720", |b| {
        b.iter(|| {
            black_box(extract_contour(
                black_box(&mask),
                black_box(0.0),
                black_box(0.0),
            ))
        })
    });
}

fn bench_draw_segmentation_1_result(c: &mut Criterion) {
    let image = DynamicImage::new_rgb8(1280, 720);
    let results = vec![make_result(100.0, 100.0, 500.0, 400.0, 2)];
    c.bench_function("draw_segmentation_1_result_1280x720", |b| {
        b.iter(|| {
            black_box(draw_segmentation_on_image(black_box(&image), black_box(&results)).unwrap())
        })
    });
}

fn bench_draw_segmentation_5_results(c: &mut Criterion) {
    let image = DynamicImage::new_rgb8(1280, 720);
    let results: Vec<_> = (0..5)
        .map(|i| {
            let x1 = 50.0 + i as f32 * 220.0;
            make_result(x1, 100.0, x1 + 200.0, 350.0, i)
        })
        .collect();
    c.bench_function("draw_segmentation_5_results_1280x720", |b| {
        b.iter(|| {
            black_box(draw_segmentation_on_image(black_box(&image), black_box(&results)).unwrap())
        })
    });
}

criterion_group!(
    benches,
    bench_upsample_mask_full_frame,
    bench_upsample_mask_small_bbox,
    bench_extract_contour,
    bench_draw_segmentation_1_result,
    bench_draw_segmentation_5_results,
);
criterion_main!(benches);
