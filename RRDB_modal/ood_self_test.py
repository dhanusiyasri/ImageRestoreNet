"""
Out-of-distribution (OOD) self-test.

Directly addresses two gaps called out in the official evaluation criteria:
  1. "Tests generalization and robustness -- whether your model can handle image
     types it has never seen." Your train/val split is all from KLA's own dataset,
     so it's still in-distribution. This script tests on images from a COMPLETELY
     different source (scikit-image's bundled sample images -- photos, textures,
     patterns -- structurally nothing like semiconductor chip images).
  2. Isolates each degradation type separately (speckle-only, Gaussian-only,
     combined) so you can report how the model handles Gaussian noise specifically,
     not just blended in with everything else.

These images have known ground truth (they start clean), so we apply our OWN
synthetic degradation pipeline to create input/GT pairs, matching the recipe
described in the problem statement: speckle (multiplicative, can overshoot [0,1]),
Gaussian noise (additive blur/haze), and 2x downsampling.

Usage:
    python ood_self_test.py --weights .\\weights\\model_best.pt --out_dir .\\ood_results

No internet or extra downloads required -- the sample images are bundled with
scikit-image, which you already have installed.
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
import skimage.data as skdata
from skimage.transform import resize as sk_resize

from models.rrdb_lite import RestorationNet
from utils.metrics import compute_psnr, compute_ssim, compute_lpips

# Diverse, genuinely OOD sample set: natural photos, textures, patterns --
# nothing resembling semiconductor structures.
SAMPLE_NAMES = ["camera", "coins", "checkerboard", "brick", "grass",
                "gravel", "astronaut", "moon", "page"]


def load_clean_gt(name, size=256):
    """Load a scikit-image sample, convert to grayscale float32 [0,1], resize to `size`."""
    img = getattr(skdata, name)()
    if img.ndim == 3:  # color (e.g. astronaut) -> grayscale
        img = np.mean(img, axis=2)
    img = img.astype(np.float32) / 255.0
    img = sk_resize(img, (size, size), anti_aliasing=True).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def degrade(gt, mode, lr_size=128, seed=0):
    """
    Synthetic degradation pipeline, matching the problem statement's description.
    mode: "speckle_only", "gaussian_only", or "combined"
    Returns the degraded LR array (may exceed [0,1], by design, matching real data).
    """
    rng = np.random.RandomState(seed)

    # 1) downsample (bicubic) -- same as KLA's real degraded inputs
    pil_img = Image.fromarray((gt * 255).astype(np.uint8))
    lr = pil_img.resize((lr_size, lr_size), Image.BICUBIC)
    lr = np.array(lr).astype(np.float32) / 255.0

    # 2) add noise depending on mode
    if mode in ("speckle_only", "combined"):
        speckle = rng.randn(*lr.shape).astype(np.float32) * 0.15
        lr = lr + lr * speckle  # multiplicative -- can push values outside [0,1]

    if mode in ("gaussian_only", "combined"):
        gauss = rng.randn(*lr.shape).astype(np.float32) * 0.06
        lr = lr + gauss  # additive, softens/hazes

    return lr.astype(np.float32)


def run_model(model, lr, device):
    tensor = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).clamp(0.0, 1.0).squeeze().cpu().numpy()
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out_dir", default="./ood_results")
    parser.add_argument("--gt_size", type=int, default=256)
    parser.add_argument("--lr_size", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = RestorationNet(in_channels=1, channels=64, growth=32, num_blocks=8, scale=2)
    checkpoint = torch.load(args.weights, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    modes = ["speckle_only", "gaussian_only", "combined"]
    results = {m: {"psnr": [], "ssim": [], "lpips": []} for m in modes}
    rows = []

    for name in SAMPLE_NAMES:
        gt = load_clean_gt(name, size=args.gt_size)
        for mode in modes:
            lr = degrade(gt, mode, lr_size=args.lr_size, seed=hash(name) % 1000)
            pred = run_model(model, lr, device)

            psnr = compute_psnr(pred, gt)
            ssim = compute_ssim(pred, gt)
            lpips_val = compute_lpips(pred, gt, device=str(device))

            results[mode]["psnr"].append(psnr)
            results[mode]["ssim"].append(ssim)
            results[mode]["lpips"].append(lpips_val)
            rows.append((name, mode, psnr, ssim, lpips_val))

            # Save one visual panel per sample/mode: degraded | restored | ground truth
            def to_uint8(x):
                x = np.clip(x, 0.0, 1.0)
                return Image.fromarray((x * 255).astype(np.uint8))
            lr_disp = to_uint8(lr).resize((args.gt_size, args.gt_size), Image.NEAREST)
            pred_disp = to_uint8(pred)
            gt_disp = to_uint8(gt)
            gap = 8
            panel = Image.new("L", (args.gt_size * 3 + gap * 2, args.gt_size), 255)
            panel.paste(lr_disp, (0, 0))
            panel.paste(pred_disp, (args.gt_size + gap, 0))
            panel.paste(gt_disp, (2 * (args.gt_size + gap), 0))
            panel.save(os.path.join(args.out_dir, f"ood_{name}_{mode}.png"))

    # CSV
    csv_path = os.path.join(args.out_dir, "ood_per_image_results.csv")
    with open(csv_path, "w") as f:
        f.write("sample_name,degradation_mode,psnr,ssim,lpips\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f}\n")

    print("\n" + "=" * 72)
    print("OUT-OF-DISTRIBUTION SELF-TEST -- results by degradation type")
    print(f"(evaluated on {len(SAMPLE_NAMES)} images NOT from the KLA training distribution)")
    print("=" * 72)
    print(f"{'Degradation':<16}{'Avg PSNR (dB)':>16}{'Avg SSIM':>14}{'Avg LPIPS':>14}")
    for mode in modes:
        p = np.mean(results[mode]["psnr"])
        s = np.mean(results[mode]["ssim"])
        l = np.mean(results[mode]["lpips"])
        print(f"{mode:<16}{p:>16.2f}{s:>14.4f}{l:>14.4f}")
    print("=" * 72)
    print(f"\nPer-image results: {csv_path}")
    print(f"Visual panels (degraded | restored | ground truth): {args.out_dir}/ood_*.png")


if __name__ == "__main__":
    main()