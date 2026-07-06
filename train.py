# -*- coding: utf-8 -*-
"""
Training script for the PyTorch Deep JSCC model on CIFAR-10.
Port of ../Keras_Implementation/AutoencoderTrain.py, with hyperparameters
aligned to the paper (Bourtsoulatze et al., 2019):

  * Adam, batch size 64
  * learning rate 1e-3, dropped to 1e-4 after 500k iterations
  * train "until the test-set performance stops improving" -> optional
    early stopping via --patience.
"""

import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from model import DeepJSCC, calculate_filters


def get_loaders(batch_size, data_root="./data", num_workers=2):
    # ToTensor already scales pixels to [0, 1] and gives NCHW tensors.
    tf = T.ToTensor()
    train_set = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=tf)
    test_set = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        total += criterion(out, x).item() * x.size(0)
        n += x.size(0)
    return total / n


def run_training(comp_ratio, snr, epochs=750, batch_size=64, lr=1e-3,
                 lr_drop_iters=500_000, lr_after=1e-4, patience=0,
                 saver_step=50, data_root="./data", ckpt_root="./checkpoints",
                 num_workers=2, device=None, verbose=True):
    """Train one (comp_ratio, snr) model. Returns a dict with the best
    validation loss and checkpoint paths."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print("Using device:", device)

    c = calculate_filters(comp_ratio)
    if verbose:
        print(f"Compression ratio {comp_ratio} -> latent channels c = {c}")

    model = DeepJSCC(c, snr_db=snr).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader, test_loader = get_loaders(batch_size, data_root, num_workers)

    ckpt_dir = os.path.join(ckpt_root, f"CompRatio{comp_ratio}_SNR{snr}")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "autoencoder_best.pt")

    best_val = float("inf")
    since_improved = 0
    global_step = 0
    lr_dropped = False

    for epoch in range(epochs):
        model.train()
        running, seen = 0.0, 0
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)

            # paper LR schedule: drop to lr_after after lr_drop_iters steps
            if not lr_dropped and global_step >= lr_drop_iters:
                for g in optimizer.param_groups:
                    g["lr"] = lr_after
                lr_dropped = True
                if verbose:
                    print(f"  [lr schedule] dropped lr to {lr_after} at step {global_step}")

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, x)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
            global_step += 1

        train_loss = running / seen
        val_loss = evaluate(model, test_loader, criterion, device)
        if verbose:
            print(f"Epoch {epoch + 1:4d}/{epochs} | train MSE {train_loss:.6f} | "
                  f"val MSE {val_loss:.6f} | step {global_step}")

        # save best (analogue of Keras ModelCheckpoint save_best_only)
        if val_loss < best_val:
            best_val = val_loss
            since_improved = 0
            torch.save(model.state_dict(), best_path)
        else:
            since_improved += 1

        # periodic snapshot (analogue of ModelCheckponitsHandler)
        if saver_step > 0 and epoch % saver_step == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"autoencoder_epoch_{epoch}.pt"))

        # early stopping ("until the test-set performance stops improving")
        if patience > 0 and since_improved >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs).")
            break

    if verbose:
        print(f"Done. Best val MSE: {best_val:.6f}  ->  {best_path}")
    return {"best_val": best_val, "ckpt_dir": ckpt_dir, "best_path": best_path}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--comp-ratio", type=float, default=0.06)
    p.add_argument("--snr", type=float, default=10.0, help="SNR in dB used during training")
    p.add_argument("--epochs", type=int, default=750)
    p.add_argument("--batch-size", type=int, default=64, help="paper uses 64; raise to 128/256 on GPU")
    p.add_argument("--lr", type=float, default=1e-3, help="initial learning rate (paper: 1e-3)")
    p.add_argument("--lr-drop-iters", type=int, default=500_000,
                   help="iteration at which lr drops to --lr-after (paper: 500k)")
    p.add_argument("--lr-after", type=float, default=1e-4, help="lr after the drop (paper: 1e-4)")
    p.add_argument("--patience", type=int, default=0,
                   help="early-stop after N epochs without val improvement (0 = disabled)")
    p.add_argument("--saver-step", type=int, default=50)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--ckpt-root", type=str, default="./checkpoints")
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run_training(a.comp_ratio, a.snr, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                 lr_drop_iters=a.lr_drop_iters, lr_after=a.lr_after, patience=a.patience,
                 saver_step=a.saver_step, data_root=a.data_root, ckpt_root=a.ckpt_root,
                 num_workers=a.num_workers)
