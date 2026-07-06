# -*- coding: utf-8 -*-
"""
Visualization for the PyTorch Deep JSCC model.
Port of the plotting parts of ../Keras_Implementation/Autoencoder_Evaluation.py

Two outputs:
  1. PSNR (dB) vs compression ratio k/n, one curve per SNR.
  2. A grid of original vs reconstructed test images for one model.

Checkpoints are expected at:
  <ckpt-root>/CompRatio<comp_ratio>_SNR<train_snr>/autoencoder_best.pt
(the layout produced by train.py).
"""

import os
import argparse

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import matplotlib
import matplotlib.pyplot as plt

from model import DeepJSCC, calculate_filters
from evaluate import reconstruct
import baselines as B


MARKERS = ["*", "s", "o", "X", "d", "v", "<", ">", "^", "P", "H", "|"]
COLORS = ["#800080", "#FF00FF", "#000080", "#008080", "#00FFFF", "#008000", "#00FF00"]


def _ckpt_path(ckpt_root, comp_ratio, train_snr):
    return os.path.join(ckpt_root, f"CompRatio{comp_ratio}_SNR{train_snr}", "autoencoder_best.pt")


def _load_model(ckpt, comp_ratio, snr, device):
    model = DeepJSCC(calculate_filters(comp_ratio), snr_db=snr).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.set_snr(snr)
    return model


def get_test_loader(data_root, batch_size=256):
    test_set = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=T.ToTensor())
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)


def get_test_images(loader, n_images):
    """Collect up to n_images test images as (N,H,W,3) uint8 NHWC (for baselines)."""
    out = []
    got = 0
    for x, _ in loader:
        arr = (x.numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
        out.append(arr)
        got += arr.shape[0]
        if got >= n_images:
            break
    return np.concatenate(out)[:n_images]


def plot_psnr_vs_compratio(comp_ratios, snr_list, ckpt_root, device, loader,
                           train_snr=None, out_path="psnr_vs_compratio.png",
                           baseline_formats=None, baseline_n_images=1000):
    """One Deep JSCC curve per test SNR: PSNR vs compression ratio, with
    optional JPEG / JPEG2000 separation baselines overlaid (one per SNR).

    train_snr : if given, always load the checkpoint trained at this SNR and
                only vary the *test-time* SNR. If None, load the checkpoint
                trained at the same SNR as the test SNR (matched train/test).
    baseline_formats : e.g. ["JPEG", "JPEG2000"]; None disables baselines.
    """
    plt.figure(figsize=(7, 5))
    history = []
    for i, snr in enumerate(snr_list):
        psnrs, xs = [], []
        for comp_ratio in comp_ratios:
            tsnr = train_snr if train_snr is not None else snr
            ckpt = _ckpt_path(ckpt_root, comp_ratio, tsnr)
            if not os.path.exists(ckpt):
                print(f"[skip] missing checkpoint: {ckpt}")
                continue
            model = _load_model(ckpt, comp_ratio, snr, device)
            orig, recon = reconstruct(model, loader, device)
            psnr = peak_signal_noise_ratio(orig, recon)
            ssim = structural_similarity(orig, recon, channel_axis=-1)
            print(f"comp_ratio={comp_ratio}  SNR={snr}dB  PSNR={psnr:.3f}  SSIM={ssim:.4f}")
            xs.append(comp_ratio)
            psnrs.append(psnr)
        if xs:
            plt.plot(xs, psnrs, ls="--",
                     c=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)],
                     label=f"Deep JSCC (SNR={snr}dB)")
            history.append({"snr": snr, "comp_ratios": xs, "psnr": psnrs})

    # ---- separation baselines (need no model, only test images) ----
    if baseline_formats:
        imgs = get_test_images(loader, baseline_n_images)
        print(f"Computing baselines on {len(imgs)} images: {baseline_formats}")
        baseline_ls = {"JPEG": ":", "JPEG2000": "-."}
        for i, snr in enumerate(snr_list):
            for fmt in baseline_formats:
                if fmt.upper() in ("JPEG2000", "J2K") and not B.jpeg2000_available():
                    continue
                ps, _ = B.baseline_curve(imgs, comp_ratios, snr, fmt=fmt, verbose=False)
                plt.plot(comp_ratios, ps, ls=baseline_ls.get(fmt.upper(), ":"),
                         c=COLORS[i % len(COLORS)], marker="x", alpha=0.8,
                         label=f"{fmt} (SNR={snr}dB)")
                history.append({"snr": snr, "scheme": fmt,
                                "comp_ratios": list(comp_ratios), "psnr": ps})

    plt.title("AWGN Channel")
    plt.xlabel("k/n")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    plt.ylim(10, 35)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved PSNR curve -> {out_path}")
    return history


def plot_reconstructions(ckpt, comp_ratio, snr, device, loader, n=8,
                         out_path="reconstructions.png"):
    """Top row: original images. Bottom row: reconstructions."""
    model = _load_model(ckpt, comp_ratio, snr, device)
    orig, recon = reconstruct(model, loader, device)
    orig, recon = orig[:n], recon[:n]

    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.4))
    for j in range(n):
        p = peak_signal_noise_ratio(orig[j], recon[j])
        axes[0, j].imshow(orig[j]);  axes[0, j].axis("off")
        axes[1, j].imshow(recon[j]); axes[1, j].axis("off")
        axes[1, j].set_title(f"{p:.1f} dB", fontsize=8)
    axes[0, 0].set_ylabel("original", fontsize=9)
    axes[1, 0].set_ylabel("recon", fontsize=9)
    fig.suptitle(f"k/n={comp_ratio}, SNR={snr} dB")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved reconstructions -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", type=str, default="./checkpoints")
    p.add_argument("--comp-ratios", type=float, nargs="+",
                   default=[0.06, 0.09, 0.17, 0.26, 0.34, 0.43, 0.49])
    p.add_argument("--snr-list", type=float, nargs="+", default=[0, 10, 20],
                   help="test-time SNRs to plot")
    p.add_argument("--train-snr", type=float, default=None,
                   help="if set, always use checkpoints trained at this SNR")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--recon-ckpt", type=str, default=None,
                   help="checkpoint to visualize reconstructions from")
    p.add_argument("--recon-comp-ratio", type=float, default=0.06)
    p.add_argument("--recon-snr", type=float, default=20.0)
    p.add_argument("--baselines", type=str, nargs="*", default=None,
                   help="overlay separation baselines, e.g. --baselines JPEG JPEG2000")
    p.add_argument("--baseline-n-images", type=int, default=1000)
    return p.parse_args()


if __name__ == "__main__":
    matplotlib.use("Agg")  # save to file, no interactive window needed
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_test_loader(args.data_root)

    plot_psnr_vs_compratio(args.comp_ratios, args.snr_list, args.ckpt_root,
                           device, loader, train_snr=args.train_snr,
                           baseline_formats=args.baselines,
                           baseline_n_images=args.baseline_n_images)

    if args.recon_ckpt:
        plot_reconstructions(args.recon_ckpt, args.recon_comp_ratio, args.recon_snr,
                             device, loader)
