"""
Evaluation metrics: PSNR, SSIM (skimage, ground truth is always [0,1] so data_range=1.0
is fixed and correct), and LPIPS if available (optional heavier perceptual metric).
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_lpips_model = None


def compute_psnr(pred, target):
    """pred, target: numpy arrays (H,W), float, expected range [0,1]."""
    pred = np.clip(pred, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)
    return peak_signal_noise_ratio(target, pred, data_range=1.0)


def compute_ssim(pred, target):
    pred = np.clip(pred, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)
    return structural_similarity(target, pred, data_range=1.0)


def compute_lpips(pred, target, device="cpu"):
    """
    Optional perceptual metric. Lazily imports lpips so the rest of the pipeline
    works even if the package isn't installed. Expects pred/target as (H,W) in [0,1].
    """
    global _lpips_model
    try:
        import torch
        import lpips
    except ImportError:
        return None

    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex").to(device)
        _lpips_model.eval()

    def to_tensor(x):
        x = np.clip(x, 0.0, 1.0)
        t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
        t = t.repeat(1, 3, 1, 1)  # grayscale -> 3-channel for LPIPS's VGG/Alex backbone
        t = t * 2 - 1  # LPIPS expects [-1, 1]
        return t.to(device)

    with torch.no_grad():
        d = _lpips_model(to_tensor(pred), to_tensor(target))
    return d.item()


if __name__ == "__main__":
    a = np.random.rand(256, 256).astype(np.float32)
    b = a + np.random.randn(256, 256).astype(np.float32) * 0.05
    print("PSNR:", compute_psnr(b, a))
    print("SSIM:", compute_ssim(b, a))
    print("LPIPS:", compute_lpips(b, a))
