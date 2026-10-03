"""
Training script for the joint denoise + 2x super-resolution restoration model.

Usage:
    python train.py --lr_dir /path/to/NoisyLR --gt_dir /path/to/GT \
                     --out_dir ./weights --epochs 100

Two-stage curriculum:
    Stage 1 (denoise-focused pretrain): fewer epochs, Charbonnier-only, gets the
             network calibrated quickly on the core fidelity signal.
    Stage 2 (joint fine-tune): full composite loss (Charbonnier + SSIM + edge + FFT),
             trains the model on the full objective for final quality.

Saves the best checkpoint (by validation PSNR) to out_dir/model_best.pt, and the
final epoch's weights to out_dir/model_final.pt.
"""

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.rrdb_lite import RestorationNet
from utils.losses import CharbonnierLoss, CompositeRestorationLoss
from utils.dataset import RestorationDataset, make_train_val_split
from utils.metrics import compute_psnr, compute_ssim


def evaluate(model, loader, device):
    model.eval()
    psnrs, ssims = [], []
    with torch.no_grad():
        for lr, gt, _ in loader:
            lr, gt = lr.to(device), gt.to(device)
            pred = model(lr).clamp(0.0, 1.0)
            for i in range(pred.shape[0]):
                p = pred[i, 0].cpu().numpy()
                g = gt[i, 0].cpu().numpy()
                psnrs.append(compute_psnr(p, g))
                ssims.append(compute_ssim(p, g))
    model.train()
    return float(np.mean(psnrs)), float(np.mean(ssims))


def run_epoch(model, loader, optimizer, loss_fn, device, is_charbonnier_only=False):
    total_loss = 0.0
    n_batches = 0
    for lr, gt, _ in loader:
        lr, gt = lr.to(device), gt.to(device)
        optimizer.zero_grad()
        pred = model(lr)

        if is_charbonnier_only:
            loss = loss_fn(pred, gt)
        else:
            loss, _ = loss_fn(pred, gt)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_dir", required=True, help="Path to NoisyLR folder")
    parser.add_argument("--gt_dir", required=True, help="Path to GT folder")
    parser.add_argument("--out_dir", default="./weights")
    parser.add_argument("--crop_lr", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--stage1_epochs", type=int, default=15)
    parser.add_argument("--stage2_epochs", type=int, default=85)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ids, val_ids = make_train_val_split(args.lr_dir, args.val_fraction, args.seed)
    print(f"Train samples: {len(train_ids)}  Val samples: {len(val_ids)}")

    train_ds = RestorationDataset(args.lr_dir, args.gt_dir, crop_lr=args.crop_lr,
                                   scale=2, train=True, file_ids=train_ids)
    val_ds = RestorationDataset(args.lr_dir, args.gt_dir, crop_lr=args.crop_lr,
                                 scale=2, train=False, file_ids=val_ids)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = RestorationNet(in_channels=1, channels=64, growth=32, num_blocks=8,
                            scale=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total_epochs = args.stage1_epochs + args.stage2_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    charbonnier = CharbonnierLoss()
    composite_loss = CompositeRestorationLoss()

    best_psnr = -1.0
    log_path = os.path.join(args.out_dir, "training_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,stage,train_loss,val_psnr,val_ssim,lr,epoch_time_sec\n")

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        stage = 1 if epoch <= args.stage1_epochs else 2
        loss_fn = charbonnier if stage == 1 else composite_loss

        train_loss = run_epoch(model, train_loader, optimizer, loss_fn, device,
                                is_charbonnier_only=(stage == 1))
        val_psnr, val_ssim = evaluate(model, val_loader, device)
        scheduler.step()
        dt = time.time() - t0

        print(f"[Stage {stage}] Epoch {epoch}/{total_epochs}  "
              f"train_loss={train_loss:.4f}  val_psnr={val_psnr:.2f}  "
              f"val_ssim={val_ssim:.4f}  time={dt:.1f}s")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{stage},{train_loss:.6f},{val_psnr:.4f},"
                    f"{val_ssim:.4f},{optimizer.param_groups[0]['lr']:.6f},{dt:.1f}\n")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "val_psnr": val_psnr, "val_ssim": val_ssim},
                       os.path.join(args.out_dir, "model_best.pt"))

    torch.save({"model_state_dict": model.state_dict(), "epoch": total_epochs},
               os.path.join(args.out_dir, "model_final.pt"))
    print(f"Done. Best val PSNR: {best_psnr:.2f}. Weights saved to {args.out_dir}")


if __name__ == "__main__":
    main()
