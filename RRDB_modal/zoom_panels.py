"""
Zoomed-detail comparison panels.

The whole point of this challenge is that fine detail (a single pixel or small
region) can hide a defect. Full-image before/after panels don't make that visible
to a panel from a few meters away. This script adds a second row to each panel:
a tightly zoomed crop on the highest-detail region of the image (found automatically
via local variance -- a proxy for "busy," structurally rich areas like edges/textures,
which is where restoration quality matters most and where defects would hide).

Usage:
    python zoom_panels.py --lr_dir ..\\train_data\\train\\train\\NoisyLR \\
                           --gt_dir ..\\train_data\\train\\train\\GT \\
                           --weights .\\weights\\model_best.pt \\
                           --out_dir .\\zoom_panels --num_panels 4

Outputs:
    zoom_panels/zoompanel_XXXXXX.png -- top row: full degraded | full model output |
    full ground truth. Bottom row: the same three, cropped tightly to the highest-
    detail region and upscaled for visibility.
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from models.rrdb_lite import RestorationNet
from utils.dataset import make_train_val_split


def find_high_detail_region(gt, crop_size=64):
    """
    Finds the crop_size x crop_size window with the highest local variance in the
    ground truth -- a simple, fast proxy for "most structurally busy" region, i.e.
    where fine detail (and therefore potential defects) is concentrated.
    """
    h, w = gt.shape
    # downsample the variance search grid for speed -- coarse local variance map
    step = 8
    best_score, best_pos = -1, (0, 0)
    for y in range(0, h - crop_size, step):
        for x in range(0, w - crop_size, step):
            patch = gt[y:y + crop_size, x:x + crop_size]
            score = patch.var()
            if score > best_score:
                best_score = score
                best_pos = (y, x)
    return best_pos


def to_uint8(x):
    x = np.clip(x, 0.0, 1.0)
    return Image.fromarray((x * 255).astype(np.uint8))


def make_zoom_panel(lr, pred, gt, save_path, crop_size=64, zoom_factor=3):
    h, w = gt.shape  # gt resolution (e.g. 256x256)
    lr_up = np.array(to_uint8(lr).resize((w, h), Image.NEAREST)).astype(np.float32) / 255.0

    top_imgs = [lr_up, pred, gt]

    y, x = find_high_detail_region(gt, crop_size=crop_size)
    zoomed = []
    for img in top_imgs:
        crop = img[y:y + crop_size, x:x + crop_size]
        crop_img = to_uint8(crop).resize(
            (crop_size * zoom_factor, crop_size * zoom_factor), Image.NEAREST)
        zoomed.append(crop_img)

    gap = 8
    top_row = Image.new("L", (w * 3 + gap * 2, h), 255)
    for i, img in enumerate(top_imgs):
        top_row.paste(to_uint8(img), (i * (w + gap), 0))

    zoom_h = crop_size * zoom_factor
    zoom_row = Image.new("L", (w * 3 + gap * 2, zoom_h), 255)
    # scale each zoomed crop's placement to roughly align under its full-image counterpart
    slot_w = w + gap
    for i, img in enumerate(zoomed):
        px = i * slot_w + (w - zoom_h) // 2
        px = max(0, px)
        zoom_row.paste(img, (px, 0))

    panel = Image.new("L", (w * 3 + gap * 2, h + zoom_h + 12), 255)
    panel.paste(top_row, (0, 0))
    panel.paste(zoom_row, (0, h + 12))
    panel.save(save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out_dir", default="./zoom_panels")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_panels", type=int, default=6)
    parser.add_argument("--crop_size", type=int, default=64,
                         help="Size (in GT pixels) of the zoomed detail region")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, val_ids = make_train_val_split(args.lr_dir, args.val_fraction, args.seed)

    model = RestorationNet(in_channels=1, channels=64, growth=32, num_blocks=8, scale=2)
    checkpoint = torch.load(args.weights, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    chosen = val_ids[:args.num_panels]
    print(f"Generating {len(chosen)} zoomed comparison panels...")

    with torch.no_grad():
        for fname in chosen:
            lr = np.load(os.path.join(args.lr_dir, fname)).astype(np.float32)
            gt = np.load(os.path.join(args.gt_dir, fname)).astype(np.float32)

            tensor = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(tensor).clamp(0.0, 1.0).squeeze().cpu().numpy()

            save_path = os.path.join(args.out_dir, f"zoompanel_{fname.replace('.npy', '.png')}")
            make_zoom_panel(lr, pred, gt, save_path, crop_size=args.crop_size)

    print(f"Saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
