use crate::SegmentationResult;
use ab_glyph::{FontArc, PxScale};
use anyhow::{anyhow, Result};
use image::{DynamicImage, GenericImageView, Rgb as ImageRgb, RgbImage};
use std::sync::OnceLock;

use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;

const MASK_ALPHA: f32 = 0.4;
const BBOX_THICKNESS: i32 = 2;
const FONT_SIZE: f32 = 36.0;
const LABEL_PADDING: i32 = 3;

const WHITE: [u8; 3] = [255, 255, 255];
const BLACK: [u8; 3] = [0, 0, 0];

const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Roboto.ttf");

static FONT: OnceLock<FontArc> = OnceLock::new();

fn cached_font() -> Result<&'static FontArc> {
    if let Some(font) = FONT.get() {
        return Ok(font);
    }
    let font = FontArc::try_from_slice(FONT_BYTES)
        .map_err(|e| anyhow!("Failed to parse embedded font: {e}"))?;
    Ok(FONT.get_or_init(|| font))
}

pub fn draw_segmentation_on_image(
    frame: &DynamicImage,
    results: &[SegmentationResult],
) -> Result<RgbImage> {
    let mut img = frame.to_rgb8();
    let (width, height) = frame.dimensions();

    let font = cached_font()?;
    let scale = PxScale::from(FONT_SIZE);

    for result in results.iter() {
        let bbox = result.bbox.to_xyxy();
        let x1 = bbox.x1.max(0.0) as u32;
        let y1 = bbox.y1.max(0.0) as u32;
        let x2 = bbox.x2.min(width as f32 - 1.0).max(0.0) as u32;
        let y2 = bbox.y2.min(height as f32 - 1.0).max(0.0) as u32;
        let bbox_w = x2.saturating_sub(x1);
        let bbox_h = y2.saturating_sub(y1);

        let drawable_mask = result
            .mask
            .as_ref()
            .filter(|m| bbox_w > 0 && bbox_h > 0 && m.width() > 0 && m.height() > 0);
        if let Some(mask) = drawable_mask {
            for py in y1..y2 {
                for px in x1..x2 {
                    let mx = ((px - x1) as f32 / bbox_w as f32 * mask.width() as f32) as u32;
                    let my = ((py - y1) as f32 / bbox_h as f32 * mask.height() as f32) as u32;
                    let mx = mx.min(mask.width() - 1);
                    let my = my.min(mask.height() - 1);
                    let mask_val = mask.data()[(my * mask.width() + mx) as usize];
                    if mask_val > 0 {
                        let pixel = img.get_pixel_mut(px, py);
                        pixel.0[0] = ((1.0 - MASK_ALPHA) * pixel.0[0] as f32
                            + MASK_ALPHA * WHITE[0] as f32)
                            as u8;
                        pixel.0[1] = ((1.0 - MASK_ALPHA) * pixel.0[1] as f32
                            + MASK_ALPHA * WHITE[1] as f32)
                            as u8;
                        pixel.0[2] = ((1.0 - MASK_ALPHA) * pixel.0[2] as f32
                            + MASK_ALPHA * WHITE[2] as f32)
                            as u8;
                    }
                }
            }
        }

        if x2 > x1 && y2 > y1 {
            let rect = Rect::at(x1 as i32, y1 as i32).of_size(x2 - x1, y2 - y1);
            let box_color = ImageRgb(WHITE);
            for thickness in 0..BBOX_THICKNESS {
                if let Some(thick_rect) = Rect::at(rect.left() - thickness, rect.top() - thickness)
                    .of_size(
                        rect.width() + 2 * thickness as u32,
                        rect.height() + 2 * thickness as u32,
                    )
                    .intersect(Rect::at(0, 0).of_size(width, height))
                {
                    draw_hollow_rect_mut(&mut img, thick_rect, box_color);
                }
            }

            // Draw label background + text above the bounding box
            let label = format!("{} {:.0}%", result.class_name, result.confidence * 100.0);
            let (text_w, text_h) = imageproc::drawing::text_size(scale, &font, &label);
            let label_h = text_h as i32 + LABEL_PADDING * 2;
            let label_w = text_w as i32 + LABEL_PADDING * 2;
            let label_y = (y1 as i32 - label_h).max(0);
            let label_x = x1 as i32;

            // Draw filled background rectangle for the label
            if let Some(bg_rect) = Rect::at(label_x, label_y)
                .of_size(label_w as u32, label_h as u32)
                .intersect(Rect::at(0, 0).of_size(width, height))
            {
                draw_filled_rect_mut(&mut img, bg_rect, box_color);
            }

            let text_x = label_x + LABEL_PADDING;
            let text_y = label_y + LABEL_PADDING;
            draw_text_mut(
                &mut img,
                ImageRgb(BLACK),
                text_x,
                text_y,
                scale,
                &font,
                &label,
            );
        }
    }

    Ok(img)
}
