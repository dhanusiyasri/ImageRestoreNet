"""
Standalone evaluation script — this is the file KLA's benchmarking team runs AS-IS.

Usage (exactly as required by the submission spec):
    python infer.py --input_dir /path/to/test/NoisyLR --output_dir /path/to/outputs \
                     --weights ./weights/model_best.pt

Behavior:
    - Loads the trained model from --weights (falls back to ./weights/model_best.pt).
    - Reads every .npy file in --input_dir (the degraded 128x128 test inputs, matching
      the exact format KLA delivered: float32, single-channel, range may exceed [0,1]).
    - Runs a single fast forward pass per image (no test-time ensembling by default —
      this keeps the reported inference-time number honest and representative).
    - Clamps the output to [0,1] (valid image intensity range) and writes it to
      --output_dir with the SAME filename as the input (e.g. 000000.npy -> 000000.npy).
    - Also saves a PNG preview alongside each .npy output for quick visual QA
      (does not affect scoring, purely a convenience for reviewers).
    - Prints per-image and average inference time, excluding model load / GPU warmup.

No manual edits required. Runs on CPU or GPU automatically (uses CUDA if available).
"""

import argparse
import os
import glob
import time

import numpy as np
import torch
from PIL import Image

from models.rrdb_lite import RestorationNet


def run_ensemble(model, tensor):
    """
    Test-time flip ensembling: run the model on the original input plus horizontal,
    vertical, and 180-degree-rotated versions, un-flip each output, and average.
    Typically adds a small but real quality boost (+0.1-0.3 dB PSNR range) at the
    cost of ~4x inference time. Optional -- off by default so the benchmarked speed
    number stays representative of the fast single-pass path.
    """
    variants = [
        (tensor, lambda x: x),                                    # identity
        (torch.flip(tensor, dims=[3]), lambda x: torch.flip(x, dims=[3])),   # horizontal flip
        (torch.flip(tensor, dims=[2]), lambda x: torch.flip(x, dims=[2])),   # vertical flip
        (torch.flip(tensor, dims=[2, 3]), lambda x: torch.flip(x, dims=[2, 3])),  # 180 rotation
    ]
    outputs = []
    for inp, undo in variants:
        pred = model(inp)
        outputs.append(undo(pred))
    return torch.stack(outputs, dim=0).mean(dim=0)


def load_model(weights_path, device):
    model = RestorationNet(in_channels=1, channels=64, growth=32, num_blocks=8, scale=2)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                         help="Directory containing degraded .npy test images")
    parser.add_argument("--output_dir", required=True,
                         help="Directory to write restored .npy outputs")
    parser.add_argument("--weights", default="./weights/model_best.pt",
                         help="Path to trained model checkpoint")
    parser.add_argument("--save_png_preview", action="store_true", default=True,
                         help="Also save a PNG preview of each output (default: on)")
    parser.add_argument("--fp16", action="store_true",
                         help="Use FP16 inference (faster on GPU, requires CUDA)")
    parser.add_argument("--ensemble", action="store_true",
                         help="Enable test-time flip ensembling for a small quality boost "
                              "(~4x slower per image -- off by default so the reported "
                              "benchmark timing reflects the fast single-pass path)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.weights, device)
    if args.fp16 and device.type == "cuda":
        model = model.half()

    input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    if not input_files:
        raise FileNotFoundError(f"No .npy files found in {args.input_dir}")
    print(f"Found {len(input_files)} input images.")
    if args.ensemble:
        print("Ensemble mode ON (4x flip-averaged) -- timing below reflects this, "
              "NOT the fast single-pass default used for benchmarking.")

    # Warm-up pass (excluded from reported timing) — important for fair GPU benchmarking
    dummy = torch.zeros(1, 1, 128, 128).to(device)
    if args.fp16 and device.type == "cuda":
        dummy = dummy.half()
    with torch.no_grad():
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_times = []
    with torch.no_grad():
        for fpath in input_files:
            fname = os.path.basename(fpath)
            arr = np.load(fpath).astype(np.float32)
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
            if args.fp16 and device.type == "cuda":
                tensor = tensor.half()

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            pred = run_ensemble(model, tensor) if args.ensemble else model(tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            inference_times.append(dt)

            pred = pred.float().clamp(0.0, 1.0).squeeze().cpu().numpy()

            out_path = os.path.join(args.output_dir, fname)
            np.save(out_path, pred.astype(np.float32))

            if args.save_png_preview:
                png_path = os.path.join(args.output_dir, fname.replace(".npy", ".png"))
                Image.fromarray((pred * 255).astype(np.uint8)).save(png_path)

    inference_times = np.array(inference_times)
    print(f"\nProcessed {len(input_files)} images.")
    print(f"Avg inference time/image: {inference_times.mean()*1000:.2f} ms")
    print(f"Median inference time/image: {np.median(inference_times)*1000:.2f} ms")
    print(f"Total inference time: {inference_times.sum():.2f} s")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
