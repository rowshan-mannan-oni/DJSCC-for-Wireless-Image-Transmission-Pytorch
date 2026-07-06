# -*- coding: utf-8 -*-
"""
Separation-based (digital) baselines for Deep JSCC, following the paper
(Bourtsoulatze et al., 2019, Sec. on comparison schemes):

  * Source coding : JPEG or JPEG2000
  * Channel coding: assumed capacity-achieving (ideal)
  * Channel       : complex AWGN, capacity  C = log2(1 + SNR)  [bits / complex use]

For a bandwidth ratio k/n and channel SNR, the maximum source rate is
      R_max = (k/n) * C            [bits / source sample]
so the per-image bit budget is  B = R_max * n = k * C.

We JPEG/JPEG2000-compress each test image and transmit the highest-quality
version whose file size fits within B (assumed error-free below capacity).
If even the smallest encoding exceeds B, the image is in OUTAGE: following the
paper, each colour channel is reconstructed to the mean of its pixels (the
"cliff effect").

These baselines need no trained network -- they only need the test images.

Convention note
---------------
This repo's encoder emits  comp_ratio * n  *real* channel symbols
(n = 32*32*3 = 3072). Two real symbols form one complex channel use, so the
number of complex uses is  k = comp_ratio * n / 2  and
      B = (comp_ratio * n / 2) * log2(1 + SNR)   bits.
Pass --no-complex to instead treat comp_ratio directly as complex-uses/n.
"""

import io
import math
import argparse

import numpy as np
from PIL import Image

N_SOURCE = 32 * 32 * 3  # CIFAR-10 source samples per image


def capacity_bits(snr_db):
    """Complex-AWGN capacity C = log2(1 + SNR) in bits per complex channel use."""
    snr = 10.0 ** (snr_db / 10.0)
    return math.log2(1.0 + snr)


def bit_budget(comp_ratio, snr_db, n_source=N_SOURCE, complex_uses=True):
    """Per-image bit budget B = k * C."""
    k = comp_ratio * n_source / (2.0 if complex_uses else 1.0)
    return k * capacity_bits(snr_db)


def _encode_bits(img_uint8, fmt, **save_kw):
    """Return (bytes_buffer, size_in_bits) for one encoding."""
    buf = io.BytesIO()
    Image.fromarray(img_uint8).save(buf, format=fmt, **save_kw)
    return buf, len(buf.getvalue()) * 8


def _decode(buf):
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)


def _outage_recon(img_uint8):
    """Each colour channel -> its own mean pixel value (paper's outage rule)."""
    out = np.empty_like(img_uint8)
    for ch in range(img_uint8.shape[2]):
        out[:, :, ch] = int(round(img_uint8[:, :, ch].mean()))
    return out


# candidate settings, ordered high-quality -> low-quality
_JPEG_QUALITIES = list(range(95, 0, -5)) + [1]
_J2K_RATES = [1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 200, 300]


def best_recon_within_budget(img_uint8, budget_bits, fmt="JPEG"):
    """Highest-quality encoding whose size <= budget_bits. Returns
    (reconstruction_uint8, outage_bool)."""
    best = None
    if fmt.upper() == "JPEG":
        for q in _JPEG_QUALITIES:  # descending quality
            buf, bits = _encode_bits(img_uint8, "JPEG", quality=int(q))
            if bits <= budget_bits:
                return _decode(buf), False
    elif fmt.upper() in ("JPEG2000", "J2K"):
        for r in _J2K_RATES:  # ascending compression -> descending quality
            buf, bits = _encode_bits(img_uint8, "JPEG2000",
                                     quality_mode="rates", quality_layers=[float(r)])
            if bits <= budget_bits:
                return _decode(buf), False
    else:
        raise ValueError(f"unknown format {fmt}")
    # nothing fit -> outage
    return _outage_recon(img_uint8), True


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return 99.0
    return 10.0 * math.log10(255.0 ** 2 / mse)


def baseline_curve(images, comp_ratios, snr_db, fmt="JPEG",
                   n_source=N_SOURCE, complex_uses=True, verbose=True):
    """Average PSNR over `images` (N,H,W,3 uint8) for each compression ratio.
    Returns (psnr_list, outage_rate_list)."""
    psnrs, outages = [], []
    for cr in comp_ratios:
        budget = bit_budget(cr, snr_db, n_source, complex_uses)
        ps, n_out = [], 0
        for img in images:
            recon, outage = best_recon_within_budget(img, budget, fmt)
            ps.append(_psnr(img, recon))
            n_out += int(outage)
        psnrs.append(float(np.mean(ps)))
        outages.append(n_out / len(images))
        if verbose:
            print(f"  {fmt:9s} k/n={cr:.3f} SNR={snr_db}dB "
                  f"budget={budget:7.0f} bits  PSNR={psnrs[-1]:6.2f} dB  outage={outages[-1]*100:5.1f}%")
    return psnrs, outages


def jpeg2000_available():
    try:
        _encode_bits(np.zeros((8, 8, 3), np.uint8), "JPEG2000",
                     quality_mode="rates", quality_layers=[10.0])
        return True
    except Exception as e:  # missing OpenJPEG plugin
        print(f"[warn] JPEG2000 unavailable in this Pillow build: {e}")
        return False


# ---------- optional standalone plot (baselines only, no model needed) ----------
def _load_cifar_test(data_root, n_images):
    import torchvision
    ds = torchvision.datasets.CIFAR10(data_root, train=False, download=True)
    data = ds.data  # (10000, 32, 32, 3) uint8 NHWC
    if n_images and n_images < len(data):
        data = data[:n_images]
    return data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--comp-ratios", type=float, nargs="+",
                   default=[0.06, 0.09, 0.17, 0.26, 0.34, 0.43, 0.49])
    p.add_argument("--snr", type=float, default=20.0, help="channel SNR in dB")
    p.add_argument("--formats", type=str, nargs="+", default=["JPEG", "JPEG2000"])
    p.add_argument("--n-images", type=int, default=1000, help="test images to average over")
    p.add_argument("--no-complex", action="store_true",
                   help="treat comp_ratio as complex-uses/n (skip the /2 real->complex)")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--out", type=str, default="baselines.png")
    return p.parse_args()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args()
    imgs = _load_cifar_test(args.data_root, args.n_images)
    print(f"Loaded {len(imgs)} CIFAR-10 test images.")

    fmts = [f for f in args.formats
            if f.upper() not in ("JPEG2000", "J2K") or jpeg2000_available()]

    plt.figure(figsize=(7, 5))
    for f in fmts:
        print(f"--- {f} ---")
        psnrs, _ = baseline_curve(imgs, args.comp_ratios, args.snr, fmt=f,
                                  complex_uses=not args.no_complex)
        plt.plot(args.comp_ratios, psnrs, marker="o", ls="-.", label=f"{f} (SNR={args.snr}dB)")

    plt.title(f"Separation baselines, AWGN channel (SNR={args.snr} dB)")
    plt.xlabel("k/n")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved -> {args.out}")
