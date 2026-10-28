# Recipes are tagged with [group(...)]; `just --list` prints them by section.
# Requires just >= 1.27.

default:
    @just --list

# ---------------------------------------------------------------------------
# Checks: lint, format, test, audit
# ---------------------------------------------------------------------------

[group('check')]
lint:
    cargo clippy --workspace --all-targets --all-features -- -D warnings && \
    cargo fmt --all -- --check && \
    cargo machete

[group('check')]
format:
    cargo fmt --all && \
    cargo fix --allow-dirty --workspace

[group('check')]
test:
    cargo nextest run --workspace --all-features

# Runs the tests that need ONNX weights in ./models and a photo in
# RF_DETR_TEST_IMAGE (defaults to assets/test_image.jpg).
[group('check')]
test-models:
    cargo nextest run --all-features --run-ignored only

[group('check')]
audit:
    cargo audit

[group('check')]
licenses:
    cargo deny check licenses

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

[group('build')]
build:
    cargo build

[group('build')]
release:
    cargo build --release

[group('build')]
clean:
    cargo clean

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

[group('bench')]
bench:
    cargo bench --bench rf_detr_benchmarks

# Inference latency per execution provider; see examples/ep_bench.rs for EPs.
[group('bench')]
bench-ep IMAGE SIZE="nano" EP="default":
    RUST_LOG=info,ort=error cargo run --release --example ep_bench -- {{ SIZE }} {{ EP }} {{ IMAGE }}

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

[group('example')]
example-image IMAGE *ARGS:
    RUST_LOG=info,ort=error cargo run --release --example image -- {{ IMAGE }} {{ ARGS }}

[group('example')]
camera-objects:
    RUST_LOG=demo=info,rf_detr=debug,ort=error cargo run --release -p demo --bin camera-objects

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Exports one RF-DETR checkpoint to models/. SIZE is nano, small, medium or
# large; pass SEG="--seg" for the instance segmentation variant.
# Pulls this repo's ONNX models from Hugging Face into models/.
[group('models')]
get-models:
    hf download michele-xilo/rf-detr-rs --local-dir models --include "*.onnx"

[group('models')]
export-model SIZE="nano" SEG="":
    uv run scripts/export_rf_detr.py --size {{ SIZE }} {{ SEG }} --out-dir models
