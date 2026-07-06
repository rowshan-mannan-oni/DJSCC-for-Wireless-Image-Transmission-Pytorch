# -*- coding: utf-8 -*-
"""
PyTorch implementation of "Deep Joint Source-Channel Coding for Wireless
Image Transmission" (Bourtsoulatze et al., IEEE TCCN 2019).

This is a port of the Keras implementation in ../Keras_Implementation.

Notes on the port
-----------------
* Keras uses NHWC tensors; PyTorch uses NCHW. All shapes below are NCHW.
* Keras `padding='valid'`  -> PyTorch `padding=0`
  Keras `padding='same'` (k=5, s=1) -> PyTorch `padding=2`
* Keras `PReLU()` (default) learns one slope *per element* of the feature
  map. Here we use the standard `nn.PReLU(num_channels)` (one slope per
  channel), which is the conventional choice and has far fewer parameters.
* The original `NormalizationNoise` layer casts features to complex by
  literally adding `1j`, then does an element-wise "normalization" using a
  transpose. That is mathematically dubious; here we implement the clean
  average-power normalization described in the paper:
      z = sqrt(k * P) * z_tilde / ||z_tilde||_2
  which guarantees the average-power constraint (1/k)E[z^H z] <= P.
"""

import math

import torch
import torch.nn as nn


def calculate_filters(comp_ratio, F=5, n=3072):
    """Number of filters `c` for the last encoder / first decoder conv,
    given a compression ratio k/n.

    comp_ratio : k/n
    F          : conv kernel height/width (5)
    n          : number of input pixels = channels * H * W = 3 * 32 * 32
    """
    return int((comp_ratio * n) / F ** 2)


class Channel(nn.Module):
    """Average-power normalization followed by an AWGN channel.

    snr_db is stored as a (non-trained) buffer so it moves with the module
    across devices and can be changed at any time (e.g. during evaluation).
    """

    def __init__(self, snr_db=20.0, power=1.0):
        super().__init__()
        self.register_buffer("snr_db", torch.tensor(float(snr_db)))
        self.power = float(power)

    def forward(self, z):
        # z: (N, C, H, W) real-valued encoder output.
        n = z.shape[0]
        k = z[0].numel()  # channel bandwidth = C * H * W per sample

        # ----- average-power normalization (per sample) -----
        # z_norm such that (1/k) * ||z_norm||^2 = power
        norm = torch.sqrt((z.reshape(n, -1) ** 2).sum(dim=1))          # ||z||_2
        norm = norm.clamp_min(1e-12).reshape(n, 1, 1, 1)
        z = z * (math.sqrt(k * self.power) / norm)

        # ----- AWGN -----
        snr = 10.0 ** (self.snr_db / 10.0)                # linear SNR
        sig_pwr = (z.reshape(n, -1) ** 2).sum(dim=1) / k  # ~= power
        noise_pwr = sig_pwr / snr
        # complex channel -> split noise power over the two dimensions
        sigma = torch.sqrt(noise_pwr / 2.0).reshape(n, 1, 1, 1)
        noise = sigma * torch.randn_like(z)
        return z + noise


class DeepJSCC(nn.Module):
    """Deep JSCC autoencoder for 32x32x3 images.

    c : number of channels in the latent (transmitted) representation,
        obtained from `calculate_filters(comp_ratio)`.
    """

    def __init__(self, c, snr_db=20.0):
        super().__init__()

        # ------------------------- Encoder -------------------------
        # Spatial trace (H): 32 -> 14 -> 5 -> 5 -> 5 -> 5
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=0), nn.PReLU(16),   # 32->14
            nn.Conv2d(16, 80, 5, stride=2, padding=0), nn.PReLU(80),  # 14->5
            nn.Conv2d(80, 50, 5, stride=1, padding=2), nn.PReLU(50),  # 5
            nn.Conv2d(50, 40, 5, stride=1, padding=2), nn.PReLU(40),  # 5
            nn.Conv2d(40, c, 5, stride=1, padding=2), nn.PReLU(c),    # 5 -> latent
        )

        self.channel = Channel(snr_db=snr_db)

        # ------------------------- Decoder -------------------------
        # Spatial trace (H): 5 -> 5 -> 5 -> 5 -> 13 -> 29 -> (up)58 -> (crop)32
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c, 40, 5, stride=1, padding=2), nn.PReLU(40),   # 5
            nn.ConvTranspose2d(40, 50, 5, stride=1, padding=2), nn.PReLU(50),  # 5
            nn.ConvTranspose2d(50, 80, 5, stride=1, padding=2), nn.PReLU(80),  # 5
            nn.ConvTranspose2d(80, 16, 5, stride=2, padding=0), nn.PReLU(16),  # 5->13
            nn.ConvTranspose2d(16, 3, 5, stride=2, padding=0),                 # 13->29
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")           # 29->58

        self._init_weights()

    def _init_weights(self):
        # Keras used he_normal for the conv kernels.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.encoder(x)
        z_hat = self.channel(z)
        d = self.decoder(z_hat)
        d = torch.sigmoid(d)          # matches sigmoid on the last deconv
        d = self.upsample(d)          # 29 -> 58
        d = d[:, :, 13:45, 13:45]     # Cropping2D((13,13),(13,13)) -> 32x32
        return d

    def set_snr(self, snr_db):
        self.channel.snr_db.fill_(float(snr_db))


if __name__ == "__main__":
    # quick shape check
    c = calculate_filters(0.06)
    print("latent channels c =", c)
    net = DeepJSCC(c, snr_db=10)
    x = torch.rand(4, 3, 32, 32)
    y = net(x)
    print("input :", tuple(x.shape))
    print("output:", tuple(y.shape))
    assert y.shape == x.shape
    print("OK")
