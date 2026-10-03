"""
Computes ALL THREE officially required metrics -- PSNR, SSIM, and LPIPS -- for your
trained model AND a plain bicubic baseline, on the same held-out validation split
used during training. This produces the complete results table for Slide 6.

LPIPS requires torchvision + the lpips package (already confirmed working on your
machine). The first run will download a small pretrained AlexNet backbone the lpips
package uses internally -- this needs internet once, then it's cached locally.

Usage:
    python full_metrics_eval.py --lr_dir ..\\train_data\\train\\train\\NoisyLR \\
                                 --gt_dir ..\\train_data\\train\\train\\GT \\
                                 --weights .\\weights\\model_best.pt \\
                                 --out_dir .\\full_metrics

Outputs:
    - Printed summary table: PSNR / SSIM / LPIPS for bicubic-only vs your model.
    - full_metrics/per_image_results.csv: full per-image breakdown, all 3 metrics.
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from models.rrdb_lite import RestorationNet
from utils.dataset import make_train_val_split
from utils.metrics import compute_psnr, compute_ssim, compute_lpips


def bicubic_upscale(lr_arr, scale=2):
    h, w = lr_arr.shape
    pil_img = Image.fromarray(lr_arr.astype(np.float32), mode="F")
    up = pil_img.resize((w * scale, h * scale), Image.BICUBIC)
    return np.array(up)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out_dir", default="./full_metrics")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                         help="Must match the value used during training (default 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Must match the value used during training (default 42)")
    parser.add_argument("--max_samples", type=int, default=None,
                         help="Optional cap on number of validation images to evaluate "
                              "(LPIPS is slower than PSNR/SSIM; use this to test quickly first)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, val_ids = make_train_val_split(args.lr_dir, args.val_fraction, args.seed)
    if args.max_samples is not None:
        val_ids = val_ids[:args.max_samples]
    print(f"Evaluating on {len(val_ids)} validation images")

    model = RestorationNet(in_channels=1, channels=64, growth=32, num_blocks=8, scale=2)
    checkpoint = torch.load(args.weights, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    b_psnrs, b_ssims, b_lpips = [], [], []
    m_psnrs, m_ssims, m_lpips = [], [], []
    rows = []

    print("Running evaluation (LPIPS is the slowest metric -- this may take a few minutes)...")
    with torch.no_grad():
        for idx, fname in enumerate(val_ids):
            lr = np.load(os.path.join(args.lr_dir, fname)).astype(np.float32)
            gt = np.load(os.path.join(args.gt_dir, fname)).astype(np.float32)

            # Bicubic baseline (no AI)
            bicubic_out = bicubic_upscale(lr, scale=2)
            bp = compute_psnr(bicubic_out, gt)
            bs = compute_ssim(bicubic_out, gt)
            bl = compute_lpips(bicubic_out, gt, device=str(device))

            # Trained model
            tensor = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(tensor).clamp(0.0, 1.0).squeeze().cpu().numpy()
            mp = compute_psnr(pred, gt)
            ms = compute_ssim(pred, gt)
            ml = compute_lpips(pred, gt, device=str(device))

            b_psnrs.append(bp); b_ssims.append(bs); b_lpips.append(bl)
            m_psnrs.append(mp); m_ssims.append(ms); m_lpips.append(ml)
            rows.append((fname, bp, bs, bl, mp, ms, ml))

            if (idx + 1) % 50 == 0:
                print(f"  {idx + 1}/{len(val_ids)} done...")

    csv_path = os.path.join(args.out_dir, "per_image_results.csv")
    with open(csv_path, "w") as f:
        f.write("filename,bicubic_psnr,bicubic_ssim,bicubic_lpips,"
                "model_psnr,model_ssim,model_lpips\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},"
                    f"{r[4]:.4f},{r[5]:.4f},{r[6]:.4f}\n")

    avg_bp, avg_bs, avg_bl = np.mean(b_psnrs), np.mean(b_ssims), np.mean(b_lpips)
    avg_mp, avg_ms, avg_ml = np.mean(m_psnrs), np.mean(m_ssims), np.mean(m_lpips)

    print("\n" + "=" * 68)
    print("FULL RESULTS TABLE -- PSNR / SSIM / LPIPS (official metrics)")
    print("=" * 68)
    print(f"{'Metric':<14}{'Bicubic-only':>16}{'Your Model':>16}{'Improvement':>16}")
    print(f"{'PSNR (dB) ^':<14}{avg_bp:>16.2f}{avg_mp:>16.2f}{avg_mp - avg_bp:>+16.2f}")
    print(f"{'SSIM ^':<14}{avg_bs:>16.4f}{avg_ms:>16.4f}{avg_ms - avg_bs:>+16.4f}")
    print(f"{'LPIPS v':<14}{avg_bl:>16.4f}{avg_ml:>16.4f}{avg_ml - avg_bl:>+16.4f}")
    print("=" * 68)
    print("(^ higher is better, v lower is better)")
    print(f"\nPer-image results saved to: {csv_path}")


if __name__ == "__main__":
    main()