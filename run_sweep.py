# -*- coding: utf-8 -*-
"""
Train Deep JSCC across a set of compression ratios, then generate the
PSNR-vs-compression-ratio plot in one command.

Example:
    python run_sweep.py --comp-ratios 0.06 0.09 0.17 0.26 0.34 0.43 0.49 \
        --snr 10 --epochs 750 --batch-size 128
"""

import argparse

import torch

from train import run_training
from visualization import get_test_loader, plot_psnr_vs_compratio


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # ---- train one model per compression ratio ----
    for i, cr in enumerate(args.comp_ratios, 1):
        print(f"\n===== [{i}/{len(args.comp_ratios)}] training comp_ratio={cr}, SNR={args.snr} dB =====")
        run_training(cr, args.snr, epochs=args.epochs, batch_size=args.batch_size,
                     lr=args.lr, lr_after=args.lr_after, lr_drop_iters=args.lr_drop_iters,
                     patience=args.patience, saver_step=args.saver_step,
                     data_root=args.data_root, ckpt_root=args.ckpt_root,
                     num_workers=args.num_workers, device=device)

    # ---- plot PSNR vs compression ratio for the requested test SNRs ----
    if not args.no_plot:
        print("\n===== generating PSNR plot =====")
        loader = get_test_loader(args.data_root)
        plot_psnr_vs_compratio(args.comp_ratios, args.snr_list, args.ckpt_root,
                               device, loader, train_snr=args.snr,
                               out_path=args.plot_out,
                               baseline_formats=args.baselines)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--comp-ratios", type=float, nargs="+",
                   default=[0.06, 0.09, 0.17, 0.26, 0.34, 0.43, 0.49])
    p.add_argument("--snr", type=float, default=10.0, help="training SNR (dB)")
    p.add_argument("--snr-list", type=float, nargs="+", default=None,
                   help="test-time SNRs to plot (default: just the training SNR)")
    p.add_argument("--epochs", type=int, default=750)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-after", type=float, default=1e-4)
    p.add_argument("--lr-drop-iters", type=int, default=500_000)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--saver-step", type=int, default=50)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--ckpt-root", type=str, default="./checkpoints")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--plot-out", type=str, default="psnr_vs_compratio.png")
    p.add_argument("--baselines", type=str, nargs="*", default=["JPEG", "JPEG2000"],
                   help="separation baselines to overlay (empty to disable)")
    p.add_argument("--no-plot", action="store_true", help="skip the plot, only train")
    args = p.parse_args()
    if args.snr_list is None:
        args.snr_list = [args.snr]
    return args


if __name__ == "__main__":
    main(parse_args())
