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
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from model import DeepJSCC, calculate_filters


@torch.no_grad()
def reconstruct(model, loader, device):
    """Return (originals, reconstructions) as uint8 NHWC arrays."""
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


def evaluate_model(ckpt, comp_ratio, snr, device, data_root="./data", batch_size=256):
    c = calculate_filters(comp_ratio)
    model = DeepJSCC(c, snr_db=snr).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.set_snr(snr)

    test_set = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=T.ToTensor())
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    orig, recon = reconstruct(model, loader, device)
    psnr = peak_signal_noise_ratio(orig, recon)
    ssim = structural_similarity(orig, recon, channel_axis=-1)
    return psnr, ssim


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="path to a .pt checkpoint")
    p.add_argument("--comp-ratio", type=float, default=0.06)
    p.add_argument("--snr", type=float, default=20.0, help="test-time SNR in dB")
    p.add_argument("--data-root", type=str, default="./data")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psnr, ssim = evaluate_model(args.ckpt, args.comp_ratio, args.snr, device, args.data_root)
    print(f"Compression ratio {args.comp_ratio} | SNR {args.snr} dB")
    print(f"PSNR: {psnr:.3f} dB | SSIM: {ssim:.4f}")
