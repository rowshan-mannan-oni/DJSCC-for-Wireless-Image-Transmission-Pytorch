# -*- coding: utf-8 -*-
"""
Evaluation script: loads a trained checkpoint, runs it over CIFAR-10 test
images at a chosen SNR and reports PSNR / SSIM.
Port of ../Keras_Implementation/Autoencoder_Evaluation.py
"""

import argparse

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity

from model import DeepJSCC, calculate_filters


@torch.no_grad()
def reconstruct(model, loader, device):
    """Return (originals, reconstructions) as uint8 NHWC arrays from a single
    channel realization. Intended for the qualitative image grid, not metrics."""
    model.eval()
    orig, recon = [], []
    for x, _ in loader:
        x = x.to(device)
        out = model(x).clamp(0, 1)
        orig.append((x.cpu().numpy() * 255).astype(np.uint8))
        recon.append((out.cpu().numpy() * 255).astype(np.uint8))
    orig = np.concatenate(orig).transpose(0, 2, 3, 1)   # NCHW -> NHWC
    recon = np.concatenate(recon).transpose(0, 2, 3, 1)
    return orig, recon


@torch.no_grad()
def psnr_ssim_over_loader(model, loader, device, n_transmissions=10):
    """Average PSNR / SSIM over a data loader, following the paper's protocol.

    Each image is transmitted `n_transmissions` times (default 10) to average
    out the random channel noise; PSNR and SSIM are computed **per image**,
    then averaged over the set. The per-image loop matters: skimage's SSIM
    must see one image at a time, and the mean of per-image PSNR differs from
    a PSNR computed on the pooled MSE of all images.
    """
    model.eval()
    psnr_vals, ssim_vals = [], []
    for x, _ in loader:
        x = x.to(device)
        # accumulate MSE across independent channel realizations, per image
        mse = torch.zeros(x.size(0), device=device)
        recon_sum = torch.zeros_like(x)
        for _ in range(n_transmissions):
            out = model(x).clamp(0, 1)
            mse += ((out - x) ** 2).flatten(1).mean(dim=1)
            recon_sum += out
        mse /= n_transmissions

        # per-image PSNR from the averaged MSE (MAX = 1.0 in [0,1] space)
        psnr_batch = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
        psnr_vals.append(psnr_batch.cpu().numpy())

        # SSIM on the mean reconstruction, one image at a time (uint8, HWC)
        recon_mean = (recon_sum / n_transmissions).cpu().numpy()
        orig_np = x.cpu().numpy()
        for i in range(x.size(0)):
            o = (orig_np[i].transpose(1, 2, 0) * 255).astype(np.uint8)
            r = (recon_mean[i].transpose(1, 2, 0) * 255).astype(np.uint8)
            ssim_vals.append(
                structural_similarity(o, r, channel_axis=-1, data_range=255)
            )

    return float(np.mean(np.concatenate(psnr_vals))), float(np.mean(ssim_vals))


def evaluate_model(ckpt, comp_ratio, snr, device, data_root="./data",
                   batch_size=256, n_transmissions=10):
    """Load a checkpoint and report average PSNR / SSIM on the CIFAR-10 test set."""
    c = calculate_filters(comp_ratio)
    model = DeepJSCC(c, snr_db=snr).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.set_snr(snr)

    test_set = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=T.ToTensor())
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return psnr_ssim_over_loader(model, loader, device, n_transmissions=n_transmissions)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="path to a .pt checkpoint")
    p.add_argument("--comp-ratio", type=float, default=0.06)
    p.add_argument("--snr", type=float, default=20.0, help="test-time SNR in dB")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--n-transmissions", type=int, default=10,
                   help="channel realizations averaged per image (paper: 10)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psnr, ssim = evaluate_model(args.ckpt, args.comp_ratio, args.snr, device,
                                args.data_root, n_transmissions=args.n_transmissions)
    print(f"Compression ratio {args.comp_ratio} | SNR {args.snr} dB")
    print(f"PSNR: {psnr:.3f} dB | SSIM: {ssim:.4f}")
