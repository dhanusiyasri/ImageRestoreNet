"""
Evaluation metrics matching RRDB_modal/utils/metrics.py:
PSNR, SSIM (skimage, data_range=1.0), and optional LPIPS.
"""
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_lpips_model = None


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """pred, target: numpy arrays (H,W), float, expected range [0,1]."""
    pred = np.clip(pred, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)
    return float(peak_signal_noise_ratio(target, pred, data_range=1.0))


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """pred, target: numpy arrays (H,W), float, expected range [0,1]."""
    pred = np.clip(pred, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)
    return float(structural_similarity(target, pred, data_range=1.0))


def compute_lpips(pred: np.ndarray, target: np.ndarray, device: str = "cpu"):
    """Optional perceptual metric matching RRDB_modal/utils/metrics.py."""
    global _lpips_model
    try:
        import torch
        import lpips
    except ImportError:
        return None

    try:
        if _lpips_model is None:
            _lpips_model = lpips.LPIPS(net="alex").to(device)
            _lpips_model.eval()

        def to_tensor(x):
            x = np.clip(x, 0.0, 1.0)
            t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
            t = t.repeat(1, 3, 1, 1)
            t = t * 2 - 1
            return t.to(device)

        with torch.no_grad():
            d = _lpips_model(to_tensor(pred), to_tensor(target))
        return float(d.item())
    except Exception:
        return None


def compute_psnr_ssim(img_a: Image.Image, img_b: Image.Image):
    """
    Computes PSNR, SSIM, and LPIPS between raw input and restored output
    using skimage with data_range=1.0 matching RRDB_modal/utils/metrics.py.
    """
    # Resize img_a to match img_b (2x restored resolution) via bicubic interpolation
    a = np.asarray(img_a.resize(img_b.size, Image.BICUBIC).convert("L"), dtype=np.float32) / 255.0
    b = np.asarray(img_b.convert("L"), dtype=np.float32) / 255.0

    psnr = compute_psnr(b, a)
    ssim = compute_ssim(b, a)
    lpips_val = compute_lpips(b, a)
    return psnr, ssim, lpips_val

