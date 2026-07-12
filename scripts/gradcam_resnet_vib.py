"""Grad-CAM explanation for a trained ResNetVibCNN on vib_spec."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grinding_physic_fusion.visualization import PublicationPlotter
from scripts.train_and_evaluate import smart_load_data, scale_data_dict, model_factory

PublicationPlotter.set_style()

CONFIG = "vib_spec"
MODEL_NAME = "ResNetVibCNN"
CKPT_PATH = Path("checkpoints/ResNetVibCNN_vib_spec_fold0_repeat0.pt")
OUT_DIR = Path("reports/evidence/xai")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cpu")


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.features = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_features)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_grads)

    def _save_features(self, module, input, output):
        self.features = output.detach()

    def _save_grads(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor) -> np.ndarray:
        self.model.zero_grad()
        out = self.model(x)
        out.backward(gradient=torch.ones_like(out))
        grads = self.gradients  # (B, C, H, W)
        feats = self.features   # (B, C, H, W)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * feats).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1).cpu().numpy()  # (B, H, W)
        # Normalize per sample
        for i in range(cam.shape[0]):
            cmin, cmax = cam[i].min(), cam[i].max()
            if cmax > cmin:
                cam[i] = (cam[i] - cmin) / (cmax - cmin)
        return cam

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def main():
    print("Loading data ...", flush=True)
    full_data = smart_load_data([MODEL_NAME], [CONFIG])
    # Use unscaled spectrograms to match the original checkpoint training regime
    X = torch.from_numpy(full_data["vib_spec"]).float().to(DEVICE)
    y = full_data["targets"]
    print(f"X shape: {X.shape}", flush=True)

    print("Loading model ...", flush=True)
    model = model_factory(MODEL_NAME)
    state = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    target_layer = model.res2
    gradcam = GradCAM(model, target_layer)

    # Explain a diverse subset: worst predicted, best predicted, and random
    with torch.no_grad():
        preds = model(X).squeeze().cpu().numpy()
    errors = np.abs(preds - y)
    n = len(X)
    idxs = [int(np.argmax(errors)), int(np.argmin(errors)), int(n // 2)]

    fig, axes = plt.subplots(1, 3, figsize=PublicationPlotter.fig_size(width_mm=183, ratio=0.28))
    mean_cam = []
    for ax, idx in zip(axes, idxs):
        x = X[idx : idx + 1]
        cam = gradcam.generate(x)[0]  # (H, W)
        mean_cam.append(cam)
        spec = x[0, 0].cpu().numpy()  # first channel for visualization
        ax.imshow(spec, aspect="auto", origin="lower", cmap="viridis")
        ax.imshow(cam, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
        ax.set_title(f"idx={idx} pred={preds[idx]:.3f} y={y[idx]:.3f} err={errors[idx]:.3f}")
        ax.set_xlabel("time bin")
        ax.set_ylabel("freq bin")
    gradcam.remove_hooks()
    fig.tight_layout()
    PublicationPlotter.savefig(fig, "gradcam_resnetvib_samples.png", out_dir=OUT_DIR, close=True)
    print(f"Saved Grad-CAM sample plot", flush=True)

    # Aggregate mean activation across all samples
    print("Computing aggregate Grad-CAM ...", flush=True)
    gradcam = GradCAM(model, target_layer)
    cams = []
    batch_size = 16
    for i in range(0, n, batch_size):
        batch = X[i : i + batch_size]
        cam = gradcam.generate(batch)
        cams.append(cam)
    gradcam.remove_hooks()
    all_cams = np.concatenate(cams, axis=0)  # (N, H, W)
    mean_cam = np.abs(all_cams).mean(axis=0)  # (H, W)
    freq_importance = mean_cam.mean(axis=1)  # (H,)
    time_importance = mean_cam.mean(axis=0)  # (W,)

    fig, axes = plt.subplots(1, 3, figsize=PublicationPlotter.fig_size(width_mm=183, ratio=0.28))
    axes[0].imshow(mean_cam, aspect="auto", origin="lower", cmap="hot")
    axes[0].set_title("Mean |Grad-CAM|")
    axes[0].set_xlabel("time bin")
    axes[0].set_ylabel("freq bin")
    axes[1].plot(freq_importance)
    axes[1].set_title("Frequency importance")
    axes[1].set_xlabel("freq bin")
    axes[2].plot(time_importance)
    axes[2].set_title("Time importance")
    axes[2].set_xlabel("time bin")
    fig.tight_layout()
    PublicationPlotter.savefig(fig, "gradcam_resnetvib_aggregate.png", out_dir=OUT_DIR, close=True)
    print(f"Saved Grad-CAM aggregate plot", flush=True)

    # Save CSV
    df = pd.DataFrame({
        "freq_bin": np.arange(len(freq_importance)),
        "mean_importance": freq_importance,
    })
    csv_path = OUT_DIR / "gradcam_resnetvib_freq_importance.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}", flush=True)

if __name__ == "__main__":
    main()
