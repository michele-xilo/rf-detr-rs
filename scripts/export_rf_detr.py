# /// script
# requires-python = ">=3.11"
# dependencies = ["rfdetr[onnx]==1.6.5"]
# ///
"""Export an RF-DETR checkpoint to the ONNX file name the crate expects.

Upstream's exporter already emits the `input` / `dets` / `labels` (/ `masks`)
tensors the crate reads, at each variant's native resolution; this script only
picks the class and renames `inference_model.onnx`.

    uv run scripts/export_rf_detr.py --size nano --out-dir models
    uv run scripts/export_rf_detr.py --size large --seg --out-dir models
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import rfdetr

DETECTION = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}
SEGMENTATION = {
    "nano": "RFDETRSegNano",
    "small": "RFDETRSegSmall",
    "medium": "RFDETRSegMedium",
    "large": "RFDETRSegLarge",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=sorted(DETECTION), default="nano")
    parser.add_argument("--seg", action="store_true", help="export the segmentation variant")
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    class_name = (SEGMENTATION if args.seg else DETECTION)[args.size]
    model = getattr(rfdetr, class_name)()

    file_name = f"rf-detr-{'seg-' if args.seg else ''}{args.size}.onnx"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        model.export(output_dir=tmp)
        shutil.move(Path(tmp) / "inference_model.onnx", args.out_dir / file_name)
    print(f"wrote {args.out_dir / file_name}")


if __name__ == "__main__":
    main()
