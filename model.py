# -*- coding: utf-8 -*-
"""
PyTorch implementation of "Deep Joint Source-Channel Coding for Wireless
Image Transmission" (Bourtsoulatze et al., IEEE TCCN 2019).

Architecture (CIFAR-10, 32x32x3) follows Fig. 2 of the paper:

  Encoder                             spatial (H=W)   real values
    Conv 5x5x16 /2  PReLU   3->16     32 -> 16
    Conv 5x5x32 /2  PReLU  16->32     16 ->  8
    Conv 5x5x32 /1  PReLU  32->32      8 ->  8
    Conv 5x5x32 /1  PReLU  32->32      8 ->  8
    Conv 5x5xC  /1  PReLU  32->C       8 ->  8         8*8*C = 2k
  Power-normalization + complex AWGN channel  (k = 8*8*C / 2 complex uses)
  Decoder: mirror of the encoder with transpose convs, sigmoid on the last.

Notes
-----
* All convs use kernel 5 with 'same'-style padding=2. With stride 2 this maps
  32 -> 16 -> 8 (PyTorch: floor((in + 2*2 - 5)/2) + 1); the mirror transpose
  convs use padding=2, output_padding=1 to go 8 -> 16 -> 32 exactly, so no
  cropping is needed.
* The channel is a *complex* AWGN channel, matching the paper: the encoder's
  8*8*C real outputs are read as k = 8*8*C/2 complex symbols. See `Channel`.
* PReLU is per-channel (`nn.PReLU(C)`), the conventional choice, rather than
  Keras' per-element default.
"""

import math

import torch
import torch.nn as nn


LATENT_HW = 8  # encoder output spatial size for 32x32 input (32 -> 16 -> 8)


def calculate_filters(comp_ratio, n=3072, latent_hw=LATENT_HW):
    """Number of latent channels `C` for a target bandwidth compression ratio.

    The compression ratio is k/n, where
      * n is the number of real source samples (3*32*32 = 3072 for RGB CIFAR),
      * k is the number of *complex* channel uses.
    The encoder emits `latent_hw**2 * C` real values = 2k, so
        k = latent_hw**2 * C / 2   ->   C = comp_ratio * n / (latent_hw**2 / 2).
    For latent_hw=8, n=3072 this is C = round(96 * comp_ratio)
    (e.g. k/n = 1/6 -> C = 16, k/n = 1/12 -> C = 8).
    """
    C = comp_ratio * n / (latent_hw ** 2 / 2.0)
    return max(1, int(round(C)))


class Channel(nn.Module):
    """Average-power normalization followed by a complex AWGN channel.

    The real encoder output of shape (N, C, H, W) carries d = C*H*W real
    values, read as k = d/2 complex channel symbols. Normalization enforces
    the average-power constraint (1/k) E[z^H z] <= P (here met with equality
    per sample), and the channel adds circularly-symmetric complex Gaussian
    noise n ~ CN(0, N0) with N0 = P / SNR, i.e. variance N0/2 on each of the
    real and imaginary parts. This makes the per-symbol SNR equal to `snr_db`.

    snr_db is a (non-trained) buffer so it moves across devices and can be
    changed at any time (e.g. per test SNR during evaluation).
    """

    def __init__(self, snr_db=20.0, power=1.0):
        super().__init__()
        self.register_buffer("snr_db", torch.tensor(float(snr_db)))
        self.power = float(power)

    def forward(self, z):
        n = z.shape[0]
        d = z[0].numel()        # real values per sample = 2k
        k = d / 2.0             # complex channel uses
        P = self.power

        # ----- average-power normalization (per sample) -----
        # z = sqrt(k P) * z_tilde / ||z_tilde||_2   ->   ||z||^2 = k P
        # => average power per complex symbol is exactly P.
        norm = torch.sqrt((z.reshape(n, -1) ** 2).sum(dim=1)).clamp_min(1e-12)
        z = z * (math.sqrt(k * P) / norm).reshape(n, 1, 1, 1)

        # ----- complex AWGN: N0 = P / SNR, variance N0/2 per real dimension -----
        snr = 10.0 ** (self.snr_db / 10.0)      # linear SNR (tensor scalar)
        sigma = torch.sqrt((P / snr) / 2.0)
        return z + sigma * torch.randn_like(z)


class DeepJSCC(nn.Module):
    """Deep JSCC autoencoder for 32x32x3 images (paper Fig. 2 architecture).

    c : number of channels in the latent (transmitted) representation,
        from `calculate_filters(comp_ratio)`.
    """

    def __init__(self, c, snr_db=20.0):
        super().__init__()

        # ------------------------- Encoder -------------------------
        # H: 32 -> 16 -> 8 -> 8 -> 8   (kernel 5, padding 2 throughout)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.PReLU(16),   # 32->16
            nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.PReLU(32),  # 16->8
            nn.Conv2d(32, 32, 5, stride=1, padding=2), nn.PReLU(32),  # 8
            nn.Conv2d(32, 32, 5, stride=1, padding=2), nn.PReLU(32),  # 8
            nn.Conv2d(32, c, 5, stride=1, padding=2), nn.PReLU(c),    # 8 -> latent
        )

        self.channel = Channel(snr_db=snr_db)

        # ------------------------- Decoder -------------------------
        # Mirror of the encoder. H: 8 -> 8 -> 8 -> 16 -> 32
        # stride-2 transpose convs use output_padding=1 to double exactly.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c, 32, 5, stride=1, padding=2), nn.PReLU(32),   # 8
            nn.ConvTranspose2d(32, 32, 5, stride=1, padding=2), nn.PReLU(32),  # 8
            nn.ConvTranspose2d(32, 32, 5, stride=1, padding=2), nn.PReLU(32),  # 8
            nn.ConvTranspose2d(32, 16, 5, stride=2, padding=2, output_padding=1),
            nn.PReLU(16),                                                      # 8->16
            nn.ConvTranspose2d(16, 3, 5, stride=2, padding=2, output_padding=1),  # 16->32
        )

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
        return torch.sigmoid(d)       # map to [0, 1]

    def set_snr(self, snr_db):
        self.channel.snr_db.fill_(float(snr_db))


if __name__ == "__main__":
    # quick shape + SNR sanity check
    c = calculate_filters(1 / 6)
    print("latent channels c =", c, "(expected 16 for k/n = 1/6)")
    net = DeepJSCC(c, snr_db=10)
    x = torch.rand(4, 3, 32, 32)
    y = net(x)
    print("input :", tuple(x.shape))
    print("output:", tuple(y.shape))
    assert y.shape == x.shape
    print("OK")
