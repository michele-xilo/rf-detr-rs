// Diagnostic benchmark for RF-DETR-Seg under different ORT execution provider
// configurations. Useful when triaging inference latency regressions on a new
// model export or a new ORT release.
//
//   cargo run --release --example ep_bench -- nano   default  <image>
//   cargo run --release --example ep_bench -- heavy  cpu      <image>
//   cargo run --release --example ep_bench -- nano   coreml-neuralnetwork <image>
//
// EP options:
//   default               — production config (CoreML MLProgram on macOS, CPU otherwise)
//   cpu                   — force CPU EP only
//   coreml-neuralnetwork  — legacy CoreML model format (heavy fragmentation)
//   coreml-ane            — CoreML with ANE enabled (CPUAndGPUAndNeuralEngine)
use anyhow::{anyhow, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use rf_detr::config::{
    with_default_execution_providers, RfDetrSegVariant, RF_DETR_INPUT_NAME, RF_DETR_MEAN_0_255,
    RF_DETR_STD_0_255,
};
use rf_detr::process_rf_detr_seg_outputs;
use rf_detr::shared::image::{
    load_image_from_path, raw_rgb_to_normalized_tensor, resize_image_exact_fast,
};
use rf_detr::shared::types::{MeanStdNormalizationConfig, ModelSize};
use std::time::Instant;

fn build_session(model_path: &str, ep: &str) -> Result<Session> {
    let builder = Session::builder().map_err(|e| anyhow!("{e:?}"))?;
    let builder = match ep {
        "default" => with_default_execution_providers(builder).map_err(|e| anyhow!("{e:?}"))?,
        "cpu" => builder,
        #[cfg(target_os = "macos")]
        "coreml-neuralnetwork" => {
            use ort::execution_providers::coreml::ComputeUnits;
            use ort::execution_providers::CoreMLExecutionProvider;
            builder
                .with_execution_providers([CoreMLExecutionProvider::default()
                    .with_compute_units(ComputeUnits::CPUAndGPU)
                    .with_static_input_shapes(true)
                    .build()])
                .map_err(|e| anyhow!("{e:?}"))?
        }
        #[cfg(target_os = "macos")]
        "coreml-ane" => {
            use ort::execution_providers::coreml::{ComputeUnits, ModelFormat};
            use ort::execution_providers::CoreMLExecutionProvider;
            builder
                .with_execution_providers([CoreMLExecutionProvider::default()
                    .with_compute_units(ComputeUnits::All)
                    .with_static_input_shapes(true)
                    .with_model_format(ModelFormat::MLProgram)
                    .build()])
                .map_err(|e| anyhow!("{e:?}"))?
        }
        other => return Err(anyhow!("unknown ep '{other}'")),
    };
    builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("{e:?}"))?
        .commit_from_file(model_path)
        .map_err(|e| anyhow!("{e:?}"))
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let size = std::env::args().nth(1).unwrap_or_else(|| "nano".into());
    let ep = std::env::args().nth(2).unwrap_or_else(|| "default".into());

    let model_size = match size.as_str() {
        "nano" => ModelSize::Nano,
        "small" | "lite" => ModelSize::Lite,
        "medium" | "full" => ModelSize::Full,
        "large" | "heavy" => ModelSize::Heavy,
        _ => return Err(anyhow!("unknown size '{size}'")),
    };
    let variant = RfDetrSegVariant::from_model_size(&model_size, false);
    let model_path = format!("./models/{}", variant.model_filename);

    println!("variant={size} ep={ep} model={model_path}");
    let mut session = build_session(&model_path, &ep)?;

    let img_path = std::env::args()
        .nth(3)
        .ok_or_else(|| anyhow!("usage: ep_bench <size> <ep> <path/to/image.jpg>"))?;
    let image = load_image_from_path(&img_path)?;
    let (orig_w, orig_h) = (image.width(), image.height());
    let r = variant.resolution;
    let resized = resize_image_exact_fast(&image, (r, r));
    let mean_std = MeanStdNormalizationConfig {
        mean: RF_DETR_MEAN_0_255,
        std: RF_DETR_STD_0_255,
    };
    let input = raw_rgb_to_normalized_tensor(
        resized.to_rgb8().as_raw(),
        r,
        r,
        &mean_std,
        &variant.input_tensor_shape(),
    )?;

    // warmup
    {
        let t = Tensor::from_array(input.clone())?;
        let _ = session.run(ort::inputs![RF_DETR_INPUT_NAME => t])?;
    }

    let n = 5;
    let inf_start = Instant::now();
    for _ in 0..n {
        let t = Tensor::from_array(input.clone())?;
        let _ = session.run(ort::inputs![RF_DETR_INPUT_NAME => t])?;
    }
    let inf_avg = inf_start.elapsed() / n;

    let t = Tensor::from_array(input.clone())?;
    let outs = session.run(ort::inputs![RF_DETR_INPUT_NAME => t])?;
    let pp_start = Instant::now();
    let dets = process_rf_detr_seg_outputs(&outs, variant, orig_w, orig_h)?;
    let pp_dur = pp_start.elapsed();

    println!(
        "  inference avg over {n}: {inf_avg:?}   postprocess: {pp_dur:?}   detections: {}",
        dets.len()
    );
    Ok(())
}
