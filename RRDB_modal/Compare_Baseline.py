"""
Compares your trained model against a plain bicubic-upscale baseline (no AI at all)
on the SAME held-out validation split used during training, to prove concretely
whether the model adds real value over doing nothing.

Uses the exact same make_train_val_split() function with the same default
val_fraction/seed as train.py, so this reproduces the identical validation set
your model was scored on during training -- an apples-to-apples comparison.

Usage:
    python compare_baseline.py --lr_dir ..\\train_data\\train\\train\\NoisyLR \\
                                --gt_dir ..\\train_data\\train\\train\\GT \\
                                --weights .\\weights\\model_best.pt \\
                                --out_dir .\\baseline_comparison

Outputs:
    - Printed summary table: avg PSNR/SSIM for bicubic-only vs your model, plus the
      improvement (delta) your model provides.
    - baseline_comparison/per_image_results.csv: full per-image breakdown.
    - baseline_comparison/panel_XXXXXX.png: a handful of side-by-side comparison
      images (degraded input | bicubic-only | your model | ground truth) for slides.
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from models.rrdb_lite import RestorationNet
from utils.dataset import make_train_val_split
from utils.metrics import compute_psnr, compute_ssim


def bicubic_upscale(lr_arr, scale=2):
    """Plain bicubic upsample, no AI at all -- this is the 'do nothing' baseline."""
    h, w = lr_arr.shape
    pil_img = Image.fromarray(lr_arr.astype(np.float32), mode="F")
    up = pil_img.resize((w * scale, h * scale), Image.BICUBIC)
    return np.array(up)


def make_comparison_panel(lr, bicubic_out, model_out, gt, save_path):
    """Side-by-side: degraded input | bicubic-only | model output | ground truth."""
    def to_uint8(x, upscale_to=None):
        x = np.clip(x, 0.0, 1.0)
        img = Image.fromarray((x * 255).astype(np.uint8))
        if upscale_to is not None:
            img = img.resize(upscale_to, Image.NEAREST)
        return img

    h, w = gt.shape
    lr_disp = to_uint8(lr, upscale_to=(w, h))
    bicubic_disp = to_uint8(bicubic_out)
    model_disp = to_uint8(model_out)
    gt_disp = to_uint8(gt)

    gap = 8
    panel = Image.new("L", (w * 4 + gap * 3, h), 255)
    panel.paste(lr_disp, (0, 0))
    panel.paste(bicubic_disp, (w + gap, 0))
    panel.paste(model_disp, (2 * (w + gap), 0))
    panel.paste(gt_disp, (3 * (w + gap), 0))
    panel.save(save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out_dir", default="./baseline_comparison")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                         help="Must match the value used during training (default 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Must match the value used during training (default 42)")
    parser.add_argument("--num_panels", type=int, default=6,
                         help="How many visual comparison panels to save")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, val_ids = make_train_val_split(args.lr_dir, args.val_fraction, args.seed)
    print(f"Validation set: {len(val_ids)} images (same split used during training)")

    model = RestorationNet(in_channels=1, channels=64, growth=32, num_blocks=8, scale=2)
    checkpoint = torch.load(args.weights, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    bicubic_psnrs, bicubic_ssims = [], []
    model_psnrs, model_ssims = [], []
    rows = []

    panel_indices = set(np.linspace(0, len(val_ids) - 1, args.num_panels, dtype=int).tolist())

    with torch.no_grad():
        for i, fname in enumerate(val_ids):
            lr = np.load(os.path.join(args.lr_dir, fname)).astype(np.float32)
            gt = np.load(os.path.join(args.gt_dir, fname)).astype(np.float32)

            # Baseline: plain bicubic, no AI
            bicubic_out = bicubic_upscale(lr, scale=2)
            b_psnr = compute_psnr(bicubic_out, gt)
            b_ssim = compute_ssim(bicubic_out, gt)

            # Your trained model
            tensor = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(tensor).clamp(0.0, 1.0).squeeze().cpu().numpy()
            m_psnr = compute_psnr(pred, gt)
            m_ssim = compute_ssim(pred, gt)

            bicubic_psnrs.append(b_psnr)
            bicubic_ssims.append(b_ssim)
            model_psnrs.append(m_psnr)
            model_ssims.append(m_ssim)

            rows.append((fname, b_psnr, b_ssim, m_psnr, m_ssim))

            if i in panel_indices:
                panel_path = os.path.join(args.out_dir, f"panel_{fname.replace('.npy', '.png')}")
                make_comparison_panel(lr, bicubic_out, pred, gt, panel_path)

    # Write per-image CSV
    csv_path = os.path.join(args.out_dir, "per_image_results.csv")
    with open(csv_path, "w") as f:
        f.write("filename,bicubic_psnr,bicubic_ssim,model_psnr,model_ssim\n")
        for fname, bp, bs, mp, ms in rows:
            f.write(f"{fname},{bp:.4f},{bs:.4f},{mp:.4f},{ms:.4f}\n")

    avg_b_psnr, avg_b_ssim = np.mean(bicubic_psnrs), np.mean(bicubic_ssims)
    avg_m_psnr, avg_m_ssim = np.mean(model_psnrs), np.mean(model_ssims)

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON -- bicubic-only vs your trained model")
    print("=" * 60)
    print(f"{'Metric':<12}{'Bicubic-only':>15}{'Your Model':>15}{'Improvement':>15}")
    print(f"{'PSNR (dB)':<12}{avg_b_psnr:>15.2f}{avg_m_psnr:>15.2f}{avg_m_psnr - avg_b_psnr:>+15.2f}")
    print(f"{'SSIM':<12}{avg_b_ssim:>15.4f}{avg_m_ssim:>15.4f}{avg_m_ssim - avg_b_ssim:>+15.4f}")
    print("=" * 60)
    print(f"\nPer-image results saved to: {csv_path}")
    print(f"Comparison panels saved to: {args.out_dir}/panel_*.png")


if __name__ == "__main__":
    main()