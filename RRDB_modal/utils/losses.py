"""
Composite loss for joint denoise + super-resolution.

Components:
  1. Charbonnier (robust L1)  - pixel fidelity, robust to noisy-pixel outliers
  2. SSIM loss                - structural similarity
  3. Sobel gradient loss      - forces sharp edges (defects live at edges)
  4. FFT high-frequency loss  - explicitly penalizes missing high-freq content
                                lost during downsampling (the "novelty" term)

All losses operate on single-channel (B,1,H,W) tensors in the GT's [0,1] range.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Smooth, robust L1 variant: sqrt((pred-target)^2 + eps^2)."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class SSIMLoss(nn.Module):
    """1 - SSIM, computed with a Gaussian window (single channel)."""

    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer("window", self._create_window(window_size, sigma))

    @staticmethod
    def _gaussian(window_size, sigma):
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _create_window(self, window_size, sigma):
        g1d = self._gaussian(window_size, sigma)
        g2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)
        return g2d.unsqueeze(0).unsqueeze(0)  # (1,1,K,K)

    def forward(self, pred, target, data_range=1.0):
        window = self.window.to(pred.device).type_as(pred)
        pad = self.window_size // 2

        mu_p = F.conv2d(pred, window, padding=pad)
        mu_t = F.conv2d(target, window, padding=pad)

        mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t

        sigma_p2 = F.conv2d(pred * pred, window, padding=pad) - mu_p2
        sigma_t2 = F.conv2d(target * target, window, padding=pad) - mu_t2
        sigma_pt = F.conv2d(pred * target, window, padding=pad) - mu_pt

        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2

        ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / (
            (mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2)
        )
        return 1.0 - ssim_map.mean()


class SobelEdgeLoss(nn.Module):
    """L1 distance between Sobel gradient magnitudes of pred vs target."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def _gradient_mag(self, x):
        gx = F.conv2d(x, self.kx.to(x.device).type_as(x), padding=1)
        gy = F.conv2d(x, self.ky.to(x.device).type_as(x), padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred, target):
        return F.l1_loss(self._gradient_mag(pred), self._gradient_mag(target))


class FFTLoss(nn.Module):
    """
    L1 distance in the frequency domain (magnitude spectrum).
    Directly targets high-frequency detail lost during downsampling —
    this is the loss term that specifically fights SR blur.
    """

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        return F.l1_loss(pred_mag, target_mag)


class CompositeRestorationLoss(nn.Module):
    """
    Weighted sum of all terms. Weights are exposed so they can be tuned/ablated
    (useful directly for the "Innovation & Results" ablation table).
    """

    def __init__(self, w_charbonnier=1.0, w_ssim=0.3, w_edge=0.5, w_fft=0.3):
        super().__init__()
        self.w_charbonnier = w_charbonnier
        self.w_ssim = w_ssim
        self.w_edge = w_edge
        self.w_fft = w_fft

        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.edge = SobelEdgeLoss()
        self.fft = FFTLoss()

    def forward(self, pred, target):
        l_char = self.charbonnier(pred, target)
        l_ssim = self.ssim(pred, target)
        l_edge = self.edge(pred, target)
        l_fft = self.fft(pred, target)

        total = (
            self.w_charbonnier * l_char
            + self.w_ssim * l_ssim
            + self.w_edge * l_edge
            + self.w_fft * l_fft
        )
        parts = {
            "charbonnier": l_char.item(),
            "ssim": l_ssim.item(),
            "edge": l_edge.item(),
            "fft": l_fft.item(),
            "total": total.item(),
        }
        return total, parts


if __name__ == "__main__":
    pred = torch.rand(2, 1, 256, 256, requires_grad=True)
    target = torch.rand(2, 1, 256, 256)
    loss_fn = CompositeRestorationLoss()
    total, parts = loss_fn(pred, target)
    total.backward()
    print("Loss parts:", parts)
    print("Grad exists:", pred.grad is not None)
