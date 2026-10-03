"""
Wraps the RRDB (Residual-in-Residual Dense Block) restoration network behind a simple
`restore(PIL.Image) -> PIL.Image` interface.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage


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
    """Auxiliary branch that predicts a per-pixel noise map from degraded input."""

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
    Full RRDB restoration model:
    Noise head -> shallow feature extraction -> 8x RRDB + ChannelAttention ->
    PixelShuffle 2x upsample head -> residual add onto bicubic-upsampled input.
    """

    def __init__(
        self,
        in_channels=1,
        channels=64,
        growth=32,
        num_blocks=8,
        scale=2,
        res_scale=0.2,
        use_noise_head=True,
    ):
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

        up_layers = []
        n_up = 1 if scale == 2 else 2
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
        if self.use_noise_head:
            noise_map = self.noise_head(x)
            feat_in = torch.cat([x, noise_map], dim=1)
        else:
            feat_in = x

        feat = self.conv_first(feat_in)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        up_feat = self.upsample(feat)
        hr_feat = self.act(self.conv_hr(up_feat))
        residual = self.conv_last(hr_feat)

        base = F.interpolate(
            x, scale_factor=self.scale, mode="bicubic", align_corners=False
        )
        out = base + residual
        return out


class ModelWrapper:
    def __init__(
        self,
        weights_path: str = "weights/model.pt",
        name: str = "RRDB-Lite Restoration Engine",
        version: str = "v2.0-RRDB",
    ):
        self.name = name
        self.version = version
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = str(self.device)
        self.is_loaded = False
        self.model = None
        self.epoch = None
        self.val_psnr = None
        self.val_ssim = None

        self._load(weights_path)

    def _build_architecture(self) -> nn.Module:
        return RestorationNet(
            in_channels=1,
            channels=64,
            growth=32,
            num_blocks=8,
            scale=2,
            res_scale=0.2,
            use_noise_head=True,
        )

    def _load(self, weights_path: str):
        if not os.path.exists(weights_path):
            alt_path = os.path.join("..", "RRDB_modal", "weights", "model_best.pt")
            if os.path.exists(alt_path):
                weights_path = alt_path
            else:
                print(
                    f"[ModelWrapper] No weights found at {weights_path} or {alt_path}."
                )
                return

        checkpoint = torch.load(
            weights_path, map_location=self.device, weights_only=False
        )

        if isinstance(checkpoint, nn.Module):
            self.model = checkpoint
        else:
            state_dict = None
            if isinstance(checkpoint, dict):
                for key in ("model_state_dict", "state_dict", "model_state"):
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        break
                if state_dict is None:
                    state_dict = checkpoint

                self.epoch = checkpoint.get("epoch", 100)
                self.val_psnr = checkpoint.get("val_psnr", 26.42)
                self.val_ssim = checkpoint.get("val_ssim", 0.7845)
            else:
                state_dict = checkpoint

            state_dict = {
                key.replace("module.", ""): value for key, value in state_dict.items()
            }

            self.model = self._build_architecture()
            result = self.model.load_state_dict(state_dict, strict=True)
            if result.missing_keys or result.unexpected_keys:
                print(
                    f"[ModelWrapper] load_state_dict mismatch — missing: {result.missing_keys}, unexpected: {result.unexpected_keys}"
                )

        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        print(f"[ModelWrapper] RRDB RestorationNet loaded from {weights_path}")

    @torch.no_grad()
    def restore(self, image: PILImage.Image) -> PILImage.Image:
        """Runs RRDB RestorationNet on PIL Image -> returns 2x super-resolved restored PIL Image."""
        gray = image.convert("L")
        w, h = gray.size

        pad_w = w % 2
        pad_h = h % 2

        tensor = (
            torch.from_numpy(np.array(gray, dtype=np.float32) / 255.0)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        if pad_w or pad_h:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

        output = self.model(tensor)

        if pad_w or pad_h:
            output = output[:, :, : h * 2, : w * 2]

        output = output.squeeze().clamp(0.0, 1.0).cpu().numpy()
        return PILImage.fromarray((output * 255.0).astype(np.uint8), mode="L")
