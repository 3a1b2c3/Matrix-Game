"""
Tiny AutoEncoder (TAE) for WAN 2.2 — decoder-only fast path.

Architecture from madebyollin/taehv (github.com/madebyollin/taehv).
Weights: taew2_2.pth / taew2_2.safetensors from madebyollin/taehv on HuggingFace.

This module provides the TAE decoder with parameterized latent_channels so it
works for WAN 2.2 (48 channels).  The encoder is not used here; MG3 uses the
full WAN 2.2 encoder for encoding (same as the mg_lightvae approach).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _conv(n_in: int, n_out: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class _Clamp(nn.Module):
    """Soft-clamp via scaled tanh: clips ~Gaussian inputs to (-3, 3)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3) * 3


class _MemBlock(nn.Module):
    """Residual block that fuses current feature with a previous-frame memory."""
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        act = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(
            _conv(n_in * 2, n_out), act,
            _conv(n_out, n_out), act,
            _conv(n_out, n_out),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = act

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class _TPool(nn.Module):
    """Temporal pooling: combine *stride* consecutive frames into one."""
    def __init__(self, n_f: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))


class _TGrow(nn.Module):
    """Temporal growth: expand one frame into *stride* frames."""
    def __init__(self, n_f: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _NT, C, H, W = x.shape
        return self.conv(x).reshape(-1, C, H, W)


# ---------------------------------------------------------------------------
# Batch-mode model application (all frames in parallel)
# ---------------------------------------------------------------------------

def _apply_batch(model: nn.Sequential, x: torch.Tensor, N: int) -> torch.Tensor:
    """Apply a sequential model with MemBlocks to an NTCHW tensor in batch mode.

    MemBlock memory is the shifted-by-one input (causal, zero-padded at t=0).
    This is the standard non-streaming mode used for full-video batch decoding.

    Args:
        model: nn.Sequential containing MemBlocks, TPool, TGrow, etc.
        x: (N*T, C, H, W) flattened video tensor.
        N: batch size.

    Returns:
        (N, T', C', H', W') video tensor.
    """
    for b in model:
        if isinstance(b, _MemBlock):
            NT, C, H, W = x.shape
            T = NT // N
            _x = x.reshape(N, T, C, H, W)
            mem = F.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :T].reshape(x.shape)
            x = b(x, mem)
        else:
            x = b(x)
    NT, C, H, W = x.shape
    return x.view(N, NT // N, C, H, W)


# ---------------------------------------------------------------------------
# TAE model
# ---------------------------------------------------------------------------

class TAEModel(nn.Module):
    """Tiny AutoEncoder model with configurable latent channels.

    Architecture is identical to taehv for all latent channel counts;
    only the first/last conv dimensions change.

    Args:
        checkpoint_path: Path to .pth or .safetensors weights.
        latent_channels: Must match the weights file (48 for WAN 2.2).
        patch_size: Pixel-shuffle patch size. 2 for WAN 2.2 (16× spatial total),
                    1 for WAN 2.1 / HunyuanVideo (8× spatial).
        decoder_time_upscale: (stage0, stage1) — each True adds 2× temporal growth.
        decoder_space_upscale: (s0, s1, s2) — each True adds 2× spatial upsample.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        latent_channels: int = 48,
        patch_size: int = 2,
        decoder_time_upscale: tuple = (True, True),
        decoder_space_upscale: tuple = (True, True, True),
    ):
        super().__init__()
        act = nn.ReLU(inplace=True)
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.frames_to_trim = 2 ** sum(decoder_time_upscale) - 1

        out_channels = 3 * patch_size * patch_size  # 12 for patch_size=2, 3 for patch_size=1
        n_f = [256, 128, 64, 64]
        self.decoder = nn.Sequential(
            _Clamp(),
            _conv(latent_channels, n_f[0]), act,
            _MemBlock(n_f[0], n_f[0]), _MemBlock(n_f[0], n_f[0]), _MemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            _TGrow(n_f[0], 1),
            _conv(n_f[0], n_f[1], bias=False),
            _MemBlock(n_f[1], n_f[1]), _MemBlock(n_f[1], n_f[1]), _MemBlock(n_f[1], n_f[1]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
            _TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
            _conv(n_f[1], n_f[2], bias=False),
            _MemBlock(n_f[2], n_f[2]), _MemBlock(n_f[2], n_f[2]), _MemBlock(n_f[2], n_f[2]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
            _TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
            _conv(n_f[2], n_f[3], bias=False),
            act,
            _conv(n_f[3], out_channels),
        )

        if checkpoint_path is not None:
            self._load_weights(checkpoint_path)

    def _load_weights(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pth":
            sd = torch.load(path, map_location="cpu", weights_only=True)
        elif ext == ".safetensors":
            from safetensors.torch import load_file
            sd = load_file(path, device="cpu")
        else:
            raise ValueError(f"TAEModel: unsupported checkpoint format '{ext}' — use .pth or .safetensors")
        sd = self._patch_tgrow(sd)
        # strict=False: checkpoint includes encoder weights; we only have decoder
        missing, unexpected = self.load_state_dict(sd, strict=False)
        unexpected_dec = [k for k in unexpected if not k.startswith("encoder.")]
        if unexpected_dec:
            raise RuntimeError(f"TAEModel: unexpected decoder keys in checkpoint: {unexpected_dec}")

    def _patch_tgrow(self, sd: dict) -> dict:
        """Slice TGrow weight if checkpoint was saved with larger stride."""
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, _TGrow):
                key = f"decoder.{i}.conv.weight"
                if key in sd and sd[key].shape[0] > new_sd[key].shape[0]:
                    sd[key] = sd[key][-new_sd[key].shape[0]:]
        return sd

    def decode_video(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent frames to pixel frames (batch mode).

        Args:
            x: (N, T_lat, C_lat, H_lat, W_lat) latent tensor with ~Gaussian values.

        Returns:
            (N, T_out, 3, H, W) pixel tensor in [0, 1].
            T_out = 4 * T_lat - frames_to_trim.
        """
        N, T, C, H, W = x.shape
        x_flat = x.reshape(N * T, C, H, W)
        out = _apply_batch(self.decoder, x_flat, N)   # (N, T_out, out_channels, H', W')
        out = out.clamp_(0, 1)
        out = out[:, self.frames_to_trim:]             # trim warm-up frames
        if self.patch_size > 1:
            # pixel_shuffle: (N, T, 3*p^2, H, W) → (N, T, 3, H*p, W*p)
            N2, T2, C2, H2, W2 = out.shape
            out = F.pixel_shuffle(out.reshape(N2 * T2, C2, H2, W2), self.patch_size)
            out = out.reshape(N2, T2, 3, H2 * self.patch_size, W2 * self.patch_size)
        return out
