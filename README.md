# rf-detr-rs

[RF-DETR](https://github.com/roboflow/rf-detr) object detection and instance
segmentation in Rust. Inference runs on ONNX Runtime through
[`ort`](https://github.com/pykeio/ort), with hardware acceleration where it is
available (CUDA, CoreML, DirectML).

- 80 COCO classes, boxes, confidences and class names.
- Optional instance masks and contours from the RF-DETR-Seg models.
- A small IoU tracker that keeps instance ids stable across frames and
  smooths confidences.
- An egui live camera demo.

## Usage

```toml
[dependencies]
rf-detr = { git = "https://github.com/michele-xilo/rf-detr-rs" }
```

```rust
use rf_detr::shared::types::ModelSize;
use rf_detr::{RfDetrSegmentor, SegmentorTrait};

let mut detector = RfDetrSegmentor::with_model_size(ModelSize::Nano, false, "./models")?;
let frame = image::open("street.jpg")?;
for det in detector.infer(&frame)? {
    println!("{} {:.2} {:?}", det.class_name, det.confidence, det.bbox.to_xyxy());
}
```

Pass `segment = true` to load the RF-DETR-Seg model instead; each result then
carries a `mask` cropped to its box and a `contour`. Wrap the detector in
`SegmentationTracker` to get stable `instance_id`s over a video stream, and use
`draw_segmentation_on_image` to render results. The `serde` feature derives
`Serialize` on the result types.

## Models

No weights ship with this repository. Pull them from
[michele-xilo/rf-detr-rs](https://huggingface.co/michele-xilo/rf-detr-rs) into `models/` with the
[`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli). The
Hugging Face repo is private for now, so run `hf auth login` first:

```bash
just get-models
```

To build them from upstream instead, export them from the upstream
checkpoints with the bundled script (needs [uv](https://docs.astral.sh/uv/);
it installs `rfdetr` and PyTorch into a throwaway environment):

```bash
just export-model nano            # models/rf-detr-nano.onnx
just export-model nano --seg      # models/rf-detr-seg-nano.onnx
```

`ModelSize` picks the file under `models/`:

| `ModelSize` | Detection | Segmentation |
|---|---|---|
| `Nano` | `rf-detr-nano.onnx` (384 px, 300 queries) | `rf-detr-seg-nano.onnx` (312 px, 100 queries) |
| `Lite` | `rf-detr-small.onnx` (512 px) | `rf-detr-seg-small.onnx` (384 px) |
| `Medium`, `Full` | `rf-detr-medium.onnx` (576 px) | `rf-detr-seg-medium.onnx` (432 px) |
| `Large`, `Heavy` | `rf-detr-large.onnx` (704 px) | `rf-detr-seg-large.onnx` (504 px) |

Weights are © Roboflow and released under the
[Apache License 2.0](https://github.com/roboflow/rf-detr/blob/develop/LICENSE).

## Examples

```bash
just example-image path/to/photo.jpg              # writes assets/outputs/rf_detr_<name>.png
just example-image path/to/photo.jpg --segment    # with instance masks
just camera-objects                               # live egui camera demo (nano, detection)
```

## Development

```bash
just build        # build the library
just test         # unit tests (no weights needed)
just test-models  # ignored tests: need models/ and RF_DETR_TEST_IMAGE=<photo>
just lint         # clippy -D warnings, rustfmt, cargo-machete
just licenses     # cargo-deny license check
just bench        # criterion benchmarks for postprocessing and drawing
just bench-ep path/to/photo.jpg nano cpu   # inference latency per execution provider
```

Rust 1.91 or newer. On Linux the demo also needs `libfontconfig-dev` and
`libudev-dev`.

## Platforms

| Platform | Execution provider |
|---|---|
| macOS (Apple Silicon) | CoreML (MLProgram, CPU + GPU). The first launch compiles the model (about a minute) and caches it in `~/Library/Caches/rf-detr/coreml`. |
| Linux x86_64 | CUDA (needs CUDA 12 and cuDNN 9), CPU fallback |
| Windows | CUDA, then DirectML, CPU fallback |
| Anything else | CPU |

Execution providers that fail to register are logged and skipped, so inference
always falls back to the CPU.

## License

The code is dual-licensed under [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option. The bundled Roboto font
(`assets/fonts/Roboto.ttf`), used to draw labels, is distributed under the
Apache License 2.0.
