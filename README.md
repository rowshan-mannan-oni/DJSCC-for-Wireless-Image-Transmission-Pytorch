# Deep JSCC — PyTorch Implementation

A PyTorch implementation of *Deep Joint Source-Channel Coding for Wireless Image
Transmission* (Bourtsoulatze et al., IEEE TCCN 2019), following the paper and
adapting the reference Keras/TensorFlow implementation by
[irdanish11](https://github.com/irdanish11/DJSCC-for-Wireless-Image-Transmission)
into PyTorch. See [Differences from the Keras version](#differences-from-the-keras-version)
for how this port diverges from that reference, and
[References](#references) for full citations.

## Files
- `model.py` — the `DeepJSCC` autoencoder, the `Channel` (power-normalization + AWGN) layer, and `calculate_filters`.
- `train.py` — trains on CIFAR-10 with MSE loss (paper hyperparameters).
- `evaluate.py` — loads a checkpoint and reports PSNR / SSIM.
- `visualization.py` — plots PSNR vs compression ratio (one curve per SNR), optional JPEG/JPEG2000 baselines, and an original-vs-reconstructed image grid.
- `baselines.py` — separation-based JPEG / JPEG2000 + capacity baselines (no trained model needed).
- `run_sweep.py` — trains all compression ratios then auto-generates the PSNR plot in one command.

## Install
```bash
pip install -r requirements.txt
```
> **GPU:** the plain `pip install torch` above resolves to a **CPU** build. For an NVIDIA GPU install the CUDA wheel instead:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
> ```
> Verify with `python -c "import torch; print(torch.cuda.is_available())"`. Training auto-uses the GPU when available.

## Dataset
CIFAR-10 (50k train / 10k test, 32×32) — the small-image dataset from the paper. It is **downloaded automatically** by torchvision on first run to `--data-root` (default `./data`, ~170 MB). Use a shared location to avoid re-downloading across runs, e.g. `--data-root D:/datasets/cifar10`. (The paper's high-resolution experiments use ImageNet for training and Kodak for testing; those are a separate fully-convolutional model this repo does not implement.)

## Train
```bash
python train.py --comp-ratio 0.06 --snr 10 --epochs 750 --batch-size 256
```
Paper hyperparameters are the defaults: Adam, batch 64, lr 1e-3 dropped to 1e-4 after 500k iterations. On GPU raise `--batch-size` to 128/256 (the model uses <0.1 GB VRAM). Add `--patience 40` for early stopping ("until the test set stops improving"). Checkpoints are written to `./checkpoints/CompRatio<r>_SNR<snr>/` (best model + periodic snapshots).

## Evaluate
```bash
python evaluate.py --ckpt ./checkpoints/CompRatio0.06_SNR10/autoencoder_best.pt --comp-ratio 0.06 --snr 20
```

## Visualize
Sweep compression ratios × SNRs and plot PSNR curves (needs the matching checkpoints under `--ckpt-root`):
```bash
python visualization.py --ckpt-root ./checkpoints \
  --comp-ratios 0.06 0.09 0.17 --snr-list 0 10 20 --train-snr 10
```
Add an original-vs-reconstructed image grid:
```bash
python visualization.py --recon-ckpt ./checkpoints/CompRatio0.06_SNR10/autoencoder_best.pt \
  --recon-comp-ratio 0.06 --recon-snr 20
```
Missing checkpoints are skipped with a warning. Plots are saved as PNGs (`psnr_vs_compratio.png`, `reconstructions.png`).

## Full sweep + baselines (closest to the paper's figure)
Train every compression ratio and overlay JPEG/JPEG2000 separation baselines in one command:
```bash
python run_sweep.py --snr 10 --batch-size 256 --baselines JPEG JPEG2000
```
Overlay baselines on an existing set of trained checkpoints:
```bash
python visualization.py --snr-list 0 10 20 --train-snr 10 --baselines JPEG JPEG2000
```
Baselines alone (no model, quick sanity check of the digital scheme):
```bash
python baselines.py --snr 20 --n-images 1000
```

**Baseline method** (from the paper): source-code each image with JPEG/JPEG2000, transmit over a complex-AWGN channel with a *capacity-achieving* code. The per-image bit budget is `B = (k/n)·C·n = k·log₂(1+SNR)`. The highest-quality encoding that fits `B` is sent error-free; if even the smallest encoding exceeds `B` the image is in **outage** and each colour channel is set to its mean pixel value (the "cliff effect"). On tiny 32×32 images JPEG's header overhead alone usually exceeds the budget, so it spends most of the curve in outage — which is precisely why Deep JSCC wins on CIFAR-10.

## Architecture (32×32×3 input)
| Stage | Layers | Spatial size |
|-------|--------|--------------|
| Encoder | 5× Conv2d + PReLU | 32 → 14 → 5 → 5 → 5 → 5 |
| Channel | power normalize + AWGN | 5×5×c |
| Decoder | 5× ConvTranspose2d (+PReLU), sigmoid, upsample×2, center-crop | 5 → 13 → 29 → 58 → 32 |

`c = calculate_filters(comp_ratio)` sets the number of transmitted channels (the compression ratio k/n).

## Differences from the Keras version
- **NCHW** tensors instead of Keras NHWC.
- **PReLU** is per-channel (`nn.PReLU(C)`) rather than Keras' per-element default — fewer parameters, conventional choice.
- The **channel layer** implements the paper's clean average-power normalization
  `z = sqrt(k·P)·z̃ / ‖z̃‖₂` followed by AWGN, instead of the original's element-wise
  complex-cast/transpose normalization (which was mathematically questionable).
- Checkpoints are `state_dict` `.pt` files rather than Keras `.h5`.

## References
- E. Bourtsoulatze, D. Burth Kurka, and D. Gündüz, *"Deep Joint Source-Channel
  Coding for Wireless Image Transmission,"* IEEE Transactions on Cognitive
  Communications and Networking, vol. 5, no. 3, pp. 567–579, 2019.
  [arXiv:1809.01733](https://arxiv.org/abs/1809.01733)
- Reference Keras/TensorFlow implementation this port is adapted from:
  [irdanish11/DJSCC-for-Wireless-Image-Transmission](https://github.com/irdanish11/DJSCC-for-Wireless-Image-Transmission)
