"""
Lightweight RRDB-based restoration network for joint denoising + 2x super-resolution.

Design:
  - Fully convolutional, works on any input H,W (must be even).
  - Residual learning on top of a bicubic upsample: the network only has to learn
    the *correction* (denoise + sharpen + hallucinate detail), not reconstruct the
    whole image from scratch. This stabilizes training a lot for small datasets.
  - RDB (Residual Dense Block) -> RRDB (Residual in Residual Dense Block), same
    family as ESRGAN but shrunk down (fewer blocks, fewer channels) for speed.
  - PixelShuffle upsampling head for the 2x factor (learned, not naive).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """5-conv dense block with local residual scaling (standard RDB from ESRGAN)."""

    def __init__(self, channels=64, growth=32, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(channels, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4 * growth, channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        c1 = self.act(self.conv1(x))
        c2 = self.act(self.conv2(torch.cat([x, c1], 1)))
        c3 = self.act(self.conv3(torch.cat([x, c1, c2], 1)))
        c4 = self.act(self.conv4(torch.cat([x, c1, c2, c3], 1)))
        c5 = self.conv5(torch.cat([x, c1, c2, c3, c4], 1))
        return x + c5 * self.res_scale


class RRDB(nn.Module):
    """Residual in Residual Dense Block: 3 RDBs + outer residual."""

    def __init__(self, channels=64, growth=32, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale
        self.rdb1 = ResidualDenseBlock(channels, growth, res_scale)
        self.rdb2 = ResidualDenseBlock(channels, growth, res_scale)
        self.rdb3 = ResidualDenseBlock(channels, growth, res_scale)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * self.res_scale


class ChannelAttention(nn.Module):
    """Lightweight squeeze-excite style channel attention (CBAM-lite, channel-only)."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(channels // reduction, 4), 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 4), channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.avg_pool(x))
        return x * w


class NoiseEstimationHead(nn.Module):
    """
    Small auxiliary branch (CBDNet-style) that predicts a per-pixel noise-level map
    from the input. This map is concatenated back into the main branch so the network
    can condition its correction strength on how corrupted each region looks, letting
    one model handle speckle-heavy and Gaussian-heavy regions differently instead of
    applying one blanket denoising strength everywhere.
    """

    def __init__(self, in_channels=1, mid=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class RestorationNet(nn.Module):
    """
    Full model: noise-map head -> shallow feature extraction -> N x RRDB (+ channel
    attention every few blocks) -> pixel-shuffle 2x upsample head -> residual add onto
    bicubic-upsampled input.

    Input:  (B, 1, H, W)   grayscale, arbitrary even H, W
    Output: (B, 1, 2H, 2W) restored, denoised + super-resolved
    """

    def __init__(self, in_channels=1, channels=64, growth=32, num_blocks=8,
                 scale=2, res_scale=0.2, use_noise_head=True):
        super().__init__()
        self.scale = scale
        self.use_noise_head = use_noise_head

        head_in = in_channels + (1 if use_noise_head else 0)
        if use_noise_head:
            self.noise_head = NoiseEstimationHead(in_channels)

        self.conv_first = nn.Conv2d(head_in, channels, 3, 1, 1)

        blocks = []
        for i in range(num_blocks):
            blocks.append(RRDB(channels, growth, res_scale))
            if (i + 1) % 3 == 0:
                blocks.append(ChannelAttention(channels))
        self.body = nn.Sequential(*blocks)

        self.conv_body = nn.Conv2d(channels, channels, 3, 1, 1)

        # PixelShuffle upsample head for scale=2
        up_layers = []
        n_up = 1 if scale == 2 else 2  # supports 2x or 4x if ever needed
        for _ in range(n_up):
            up_layers += [
                nn.Conv2d(channels, channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.upsample = nn.Sequential(*up_layers)

        self.conv_hr = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv_last = nn.Conv2d(channels, in_channels, 3, 1, 1)

    def forward(self, x):
        # x: (B, 1, H, W), raw range (may exceed [0,1] due to speckle overshoot)
        if self.use_noise_head:
            noise_map = self.noise_head(x)
            feat_in = torch.cat([x, noise_map], dim=1)
        else:
            feat_in = x

        feat = self.conv_first(feat_in)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat  # global residual within feature space

        up_feat = self.upsample(feat)
        hr_feat = self.act(self.conv_hr(up_feat))
        residual = self.conv_last(hr_feat)

        # Bicubic baseline upsample of the raw input, then add learned correction
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        out = base + residual
        return out


if __name__ == "__main__":
    # quick shape/sanity check
    model = RestorationNet(in_channels=1, channels=64, growth=32, num_blocks=8, scale=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params/1e6:.2f}M")
    dummy = torch.randn(2, 1, 128, 128)
    out = model(dummy)
    print("Input:", dummy.shape, "Output:", out.shape)
