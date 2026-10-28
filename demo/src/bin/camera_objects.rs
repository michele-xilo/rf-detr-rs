use anyhow::{Context, Result};
use demo::select::select_camera;
use demo::theme;
use eframe::egui;
use image::{DynamicImage, RgbImage};
use nokhwa::utils::{CameraFormat, FrameFormat, Resolution};
use nokhwa::{
    nokhwa_initialize,
    pixel_format::RgbFormat,
    threaded::CallbackCamera,
    utils::{RequestedFormat, RequestedFormatType},
};
use rf_detr::shared::types::ModelSize;
use rf_detr::{
    RfDetrSegmentor, SegmentationTracker, TrackedSegmentation, draw_segmentation_on_image,
};
use std::sync::{Arc, Mutex};
use tracing::{error, info};

const CAMERA_FRAME_RATE: u32 = 30;
const CAMERA_RESOLUTION: (u32, u32) = (1920, 1080);
// Detection only by default; flip to true to run the heavier RF-DETR-Seg model
// and get instance masks.
const SEGMENT: bool = false;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let mut seg_tracker = SegmentationTracker::new(RfDetrSegmentor::with_model_size(
        ModelSize::Nano,
        SEGMENT,
        "./models",
    )?);

    let (seg_frame_tx, seg_frame_rx) = flume::bounded::<DynamicImage>(1);

    let seg_results: Arc<Mutex<Vec<TrackedSegmentation>>> = Arc::new(Mutex::new(Vec::new()));
    let seg_results_for_thread = Arc::clone(&seg_results);
    let seg_results_for_callback = Arc::clone(&seg_results);

    std::thread::spawn(move || {
        let min_interval = std::time::Duration::from_millis(50);
        let mut last_inference = std::time::Instant::now() - min_interval;
        while let Ok(frame) = seg_frame_rx.recv() {
            if last_inference.elapsed() < min_interval {
                continue;
            }
            last_inference = std::time::Instant::now();
            let start = std::time::Instant::now();
            if let Ok(tracked) = seg_tracker.track(&frame) {
                let elapsed = start.elapsed();
                info!("Segmentation time {elapsed:?}");
                if let Ok(mut data) = seg_results_for_thread.lock() {
                    *data = tracked;
                }
            }
        }
    });

    let image_data: Arc<Mutex<Option<egui::ColorImage>>> = Arc::new(Mutex::new(None));
    let image_data_clone = Arc::clone(&image_data);

    nokhwa_initialize(|granted| {
        info!("Camera permission: {}", granted);
    });
    let format =
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(CameraFormat::new(
            Resolution::new(CAMERA_RESOLUTION.0, CAMERA_RESOLUTION.1),
            FrameFormat::YUYV,
            CAMERA_FRAME_RATE,
        )));
    let selected = select_camera(&format)?;
    let mut camera = CallbackCamera::new(selected.index().clone(), format, move |buffer| {
        let resolution = buffer.resolution();
        let width = resolution.width() as usize;
        let height = resolution.height() as usize;

        let rgb_buffer = match buffer.decode_image::<RgbFormat>() {
            Ok(buf) => buf,
            Err(e) => {
                error!("Failed to decode camera frame: {e}");
                return;
            }
        };
        let raw_pixels = rgb_buffer.as_raw();

        let Some(rgb_image) = RgbImage::from_raw(width as u32, height as u32, raw_pixels.to_vec())
        else {
            error!("Failed to create RgbImage from raw buffer");
            return;
        };
        let dynamic_image = DynamicImage::ImageRgb8(rgb_image);

        let _ = seg_frame_tx.try_send(dynamic_image.clone());

        let results: Vec<_> = seg_results_for_callback
            .lock()
            .map(|tracked| tracked.iter().map(|t| t.result.clone()).collect())
            .unwrap_or_default();

        let mut final_image = dynamic_image;
        if !results.is_empty()
            && let Ok(seg_image) = draw_segmentation_on_image(&final_image, &results)
        {
            final_image = DynamicImage::ImageRgb8(seg_image);
        }

        let Some(color_image) = demo::frame::dynamic_image_to_color_image(&final_image) else {
            error!("Camera frame size mismatch, dropping frame");
            return;
        };

        if let Ok(mut data) = image_data_clone.lock() {
            *data = Some(color_image);
        }
    })?;

    camera
        .open_stream()
        .context("Failed to open camera stream")?;

    let camera_resolution = camera
        .camera_format()
        .map(|f| f.resolution())
        .context("Failed to get camera format")?;

    let camera_width = camera_resolution.width() as f32;
    let camera_height = camera_resolution.height() as f32;

    info!("Camera streaming started with resolution {camera_width}x{camera_height}");

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([camera_width * 1.5, camera_height])
            .with_resizable(true)
            .with_title("RF-DETR — Object Segmentation"),
        ..Default::default()
    };

    let app = MyApp {
        image_data,
        seg_results,
        texture_handle: None,
    };

    eframe::run_native(
        "RF-DETR — Object Segmentation",
        options,
        Box::new(|cc| {
            theme::apply(&cc.egui_ctx);
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe::run_native failed: {e}"))?;

    info!("Stopping camera stream");
    Ok(())
}

struct MyApp {
    image_data: Arc<Mutex<Option<egui::ColorImage>>>,
    seg_results: Arc<Mutex<Vec<TrackedSegmentation>>>,
    texture_handle: Option<egui::TextureHandle>,
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Ok(mut data) = self.image_data.lock()
            && let Some(color_image) = data.take()
        {
            self.texture_handle = Some(ctx.load_texture(
                "camera_frame",
                color_image,
                egui::TextureOptions::default(),
            ));
        }

        egui::SidePanel::right("objects_panel")
            .min_width(280.0)
            .default_width(320.0)
            .resizable(true)
            .frame(
                egui::Frame::new()
                    .fill(theme::INK_900)
                    .inner_margin(egui::Margin::same(24))
                    .stroke(egui::Stroke::new(1.0_f32, theme::BORDER)),
            )
            .show(ctx, |ui| {
                ui.label(theme::eyebrow("DETECTIONS"));
                ui.add_space(4.0);
                ui.label(theme::heading("Tracked Objects"));
                ui.add_space(4.0);

                let tracked: Vec<TrackedSegmentation> = self
                    .seg_results
                    .lock()
                    .map(|d| d.clone())
                    .unwrap_or_default();
                let count = tracked.len();

                ui.horizontal(|ui| {
                    theme::status_dot(ui, theme::TURQUOISE);
                    ui.label(theme::body(format!(
                        "{count} active instance{}",
                        if count == 1 { "" } else { "s" }
                    )));
                });

                ui.add_space(16.0);
                ui.separator();
                ui.add_space(12.0);

                if tracked.is_empty() {
                    theme::card_frame().show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        ui.vertical(|ui| {
                            ui.label(theme::eyebrow("STATUS"));
                            ui.add_space(4.0);
                            ui.label(theme::body("No objects detected."));
                            ui.label(theme::body("Point the camera at a scene."));
                        });
                    });
                } else {
                    egui::ScrollArea::vertical()
                        .max_height(ui.available_height() - 60.0)
                        .show(ui, |ui| {
                            for t in tracked.iter() {
                                detection_card(ui, t);
                                ui.add_space(10.0);
                            }
                        });
                }
            });

        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(theme::BACKGROUND)
                    .inner_margin(egui::Margin::same(24)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.label(theme::eyebrow("LIVE CAMERA"));
                        ui.add_space(4.0);
                        ui.label(theme::heading("Object Segmentation"));
                    });
                    ui.add_space(16.0);
                    ui.with_layout(egui::Layout::top_down(egui::Align::LEFT), |ui| {
                        ui.add_space(10.0);
                        ui.horizontal(|ui| {
                            theme::status_dot(ui, theme::TURQUOISE);
                            ui.label(theme::body("LIVE · 30 FPS"));
                        });
                    });
                });
                ui.add_space(12.0);

                theme::card_frame().show(ui, |ui| {
                    if let Some(texture) = &self.texture_handle {
                        let panel_width = ui.available_width();
                        let aspect = texture.size_vec2().y / texture.size_vec2().x;
                        let img_height = panel_width * aspect;
                        ui.add(
                            egui::Image::from_texture(texture)
                                .fit_to_exact_size(egui::vec2(panel_width, img_height))
                                .corner_radius(egui::CornerRadius::same(12)),
                        );
                    } else {
                        ui.centered_and_justified(|ui| {
                            ui.label(theme::body("Waiting for camera frame…"));
                        });
                    }
                });
            });

        ctx.request_repaint();
    }
}

fn detection_card(ui: &mut egui::Ui, t: &TrackedSegmentation) {
    theme::card_frame()
        .inner_margin(egui::Margin::same(14))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(theme::eyebrow(format!("ID {:02}", t.instance_id)));
                    });
                    ui.add_space(2.0);
                    ui.label(
                        egui::RichText::new(&t.result.class_name)
                            .color(theme::TEXT_PRIMARY)
                            .text_style(egui::TextStyle::Name("H4".into())),
                    );
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let confidence = t.result.confidence.clamp(0.0, 1.0);
                    let accent = if confidence >= 0.75 {
                        theme::TURQUOISE
                    } else {
                        theme::RUST
                    };
                    ui.label(
                        egui::RichText::new(format!("{:.0}%", confidence * 100.0))
                            .color(accent)
                            .text_style(egui::TextStyle::Name("Stat".into())),
                    );
                });
            });
        });
}
