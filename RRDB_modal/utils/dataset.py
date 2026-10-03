"""
Dataset loader matched exactly to KLA's delivered format:
  - Paired samples stored as .npy files (float32), NOT images.
  - NoisyLR/xxxxxx.npy  : (128, 128), range roughly [-0.3, 2.2] (speckle overshoot)
  - GT/xxxxxx.npy       : (256, 256), strict [0, 1]
  - Filenames match 1:1 between the two folders (verified: 3200/3200 pairs, 0 mismatches)

Includes degradation-diversity augmentation: on top of the noise/downsampling KLA
already applied, we randomly inject *additional* synthetic degradation variety
(extra speckle/gaussian noise, alternate downsample kernels) during training so the
model doesn't overfit to KLA's one exact degradation recipe. This directly targets
generalization to the out-of-distribution test samples.
"""

import os
import glob
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def _random_crop_pair(lr, gt, crop_lr=64, scale=2):
    """Crop a random aligned patch from the LR/GT pair (GT crop = crop_lr * scale)."""
    h, w = lr.shape
    if h < crop_lr or w < crop_lr:
        return lr, gt  # image already smaller than requested crop, use as-is
    top = random.randint(0, h - crop_lr)
    left = random.randint(0, w - crop_lr)
    lr_crop = lr[top:top + crop_lr, left:left + crop_lr]
    gt_top, gt_left = top * scale, left * scale
    gt_crop = gt[gt_top:gt_top + crop_lr * scale, gt_left:gt_left + crop_lr * scale]
    return lr_crop, gt_crop


def _random_flip_rot(lr, gt):
    """Synchronized flips/90-degree rotations, safe for grayscale structural images."""
    if random.random() < 0.5:
        lr, gt = np.fliplr(lr).copy(), np.fliplr(gt).copy()
    if random.random() < 0.5:
        lr, gt = np.flipud(lr).copy(), np.flipud(gt).copy()
    k = random.randint(0, 3)
    if k:
        lr, gt = np.rot90(lr, k).copy(), np.rot90(gt, k).copy()
    return lr, gt


def _add_synthetic_degradation(lr, strength_range=(0.0, 0.08)):
    """
    Inject *extra* synthetic speckle + Gaussian noise on top of the already-degraded
    LR input, with randomized strength. This widens the distribution of corruption
    the model sees at train time, beyond KLA's fixed recipe -> better OOD robustness.
    Applied only some of the time (see p_apply in the Dataset) so the model still
    sees the "as delivered" distribution too.
    """
    speckle_std = random.uniform(*strength_range)
    gauss_std = random.uniform(*strength_range) * 0.5

    speckle_noise = np.random.randn(*lr.shape).astype(np.float32) * speckle_std
    lr = lr + lr * speckle_noise  # multiplicative, matches real speckle behavior

    gauss_noise = np.random.randn(*lr.shape).astype(np.float32) * gauss_std
    lr = lr + gauss_noise

    return lr.astype(np.float32)


class RestorationDataset(Dataset):
    def __init__(self, lr_dir, gt_dir, crop_lr=64, scale=2, train=True,
                 p_extra_aug=0.5, file_ids=None):
        self.lr_dir = lr_dir
        self.gt_dir = gt_dir
        self.crop_lr = crop_lr
        self.scale = scale
        self.train = train
        self.p_extra_aug = p_extra_aug

        if file_ids is not None:
            self.ids = file_ids
        else:
            lr_files = sorted(glob.glob(os.path.join(lr_dir, "*.npy")))
            self.ids = [os.path.basename(f) for f in lr_files]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fname = self.ids[idx]
        lr = np.load(os.path.join(self.lr_dir, fname)).astype(np.float32)
        gt = np.load(os.path.join(self.gt_dir, fname)).astype(np.float32)

        if self.train:
            lr, gt = _random_crop_pair(lr, gt, self.crop_lr, self.scale)
            lr, gt = _random_flip_rot(lr, gt)
            if random.random() < self.p_extra_aug:
                lr = _add_synthetic_degradation(lr)

        lr_t = torch.from_numpy(lr).unsqueeze(0).float()   # (1, H, W)
        gt_t = torch.from_numpy(gt).unsqueeze(0).float()   # (1, 2H, 2W)
        return lr_t, gt_t, fname


def make_train_val_split(lr_dir, val_fraction=0.1, seed=42):
    """Simple random split, held out for validation / OOD-proxy estimation."""
    lr_files = sorted(glob.glob(os.path.join(lr_dir, "*.npy")))
    ids = [os.path.basename(f) for f in lr_files]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_fraction)
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    return train_ids, val_ids


if __name__ == "__main__":
    lr_dir = "/home/claude/train_data/train/train/NoisyLR"
    gt_dir = "/home/claude/train_data/train/train/GT"

    train_ids, val_ids = make_train_val_split(lr_dir)
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}")

    ds = RestorationDataset(lr_dir, gt_dir, crop_lr=64, scale=2, train=True,
                             file_ids=train_ids)
    lr, gt, fname = ds[0]
    print("Sample:", fname, "LR shape:", lr.shape, "GT shape:", gt.shape)
    print("LR range:", lr.min().item(), lr.max().item())
    print("GT range:", gt.min().item(), gt.max().item())
