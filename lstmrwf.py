# train_rwf_improved_batches_lcm_lstm.py
# ------------------------------------------------------------
# Fine-tune R3D-18 on RWF-2000 + LCM + LSTM
# + pandas CSV logging + TensorBoard + poster-ready plots
#
# Outputs:
#   checkpoints/r3d18_best_lcm_lstm.pth
#   checkpoints/r3d18_last_lcm_lstm.pth
#   outputs/metrics/train_log_lcm_lstm.csv
#   outputs/plots/loss_curve.png
#   outputs/plots/accuracy_curve.png
#   outputs/plots/comparison.png
#   outputs/plots/architecture_diagram.png
#   runs/lcm_lstm/  ← TensorBoard logs
# ------------------------------------------------------------

import os
import time
import warnings
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.models.video import r3d_18, R3D_18_Weights
import torchvision.transforms.functional as TF

from sklearn.metrics import accuracy_score, f1_score

# MPS fallback for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

warnings.filterwarnings(
    "once",
    message="The video decoding and encoding capabilities of torchvision"
)

# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    BASE_DIR:  Path = Path(__file__).resolve().parent
    DATA_ROOT: Path = BASE_DIR / "data" / "datasets" / "rawf-2000" / "RWF-2000"

    CLASSES:     tuple = ("NonFight", "Fight")
    NUM_CLASSES: int   = 2

    EPOCHS:      int = 20
    BATCH_SIZE:  int = 2
    NUM_WORKERS: int = 0

    NUM_FRAMES: int = 32
    IMG_SIZE:   int = 112

    DEVICE: str = "cpu"  # MPS does not support adaptive_avg_pool3d — use CPU on Mac

    # Save + TB (unchanged)
    SAVE_DIR:  Path = Path("checkpoints")
    SAVE_NAME: str  = "r3d18_best_RWF_lcm_lstm.pth"
    LOG_DIR:   Path = Path("runs") / "lcm_lstm"   # TensorBoard logs → runs/lcm_lstm/

    PRINT_EVERY: int = 50

    # LCM (unchanged)
    USE_LCM:   bool = True
    LCM_AFTER: str  = "layer4"

    # LSTM (new)
    LSTM_HIDDEN:  int   = 256
    LSTM_LAYERS:  int   = 1
    LSTM_DROPOUT: float = 0.3

    # Output dirs for CSV + plots (new)
    OUT_DIR:     Path = Path("outputs")
    METRICS_DIR: Path = Path("outputs") / "metrics"
    PLOTS_DIR:   Path = Path("outputs") / "plots"


# -----------------------------
# Dataset  (100% unchanged)
# -----------------------------
class RWFVideoDataset(Dataset):
    def __init__(self, root: Path, split: str, classes, num_frames: int, img_size: int, mean, std):
        self.paths  = []
        self.labels = []
        split_dir   = root / split

        exts = {".avi", ".mp4", ".mov", ".mkv"}
        for i, cls in enumerate(classes):
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                raise FileNotFoundError(f"Missing folder: {cls_dir}")
            for p in cls_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    self.paths.append(p)
                    self.labels.append(i)

        if len(self.paths) == 0:
            raise RuntimeError(f"No videos found in {split_dir}")

        self.num_frames = num_frames
        self.img_size   = img_size
        self.mean       = mean
        self.std        = std

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path  = str(self.paths[idx])
        label = int(self.labels[idx])

        try:
            video, _, _ = torchvision.io.read_video(path, pts_unit="sec")
        except Exception:
            clip = torch.zeros(3, self.num_frames, self.img_size, self.img_size, dtype=torch.float32)
            return clip, torch.tensor(label, dtype=torch.long)

        if video is None or video.shape[0] == 0:
            clip = torch.zeros(3, self.num_frames, self.img_size, self.img_size, dtype=torch.float32)
            return clip, torch.tensor(label, dtype=torch.long)

        T    = int(video.shape[0])
        idxs = torch.linspace(0, T - 1, steps=self.num_frames).long()
        clip = video[idxs].float() / 255.0
        clip = clip.permute(0, 3, 1, 2)

        processed = []
        for frame in clip:
            frame = TF.resize(frame, [self.img_size, self.img_size], antialias=True)
            frame = TF.normalize(frame, mean=self.mean, std=self.std)
            processed.append(frame)

        clip = torch.stack(processed).permute(1, 0, 2, 3).contiguous()
        return clip, torch.tensor(label, dtype=torch.long)


# -----------------------------
# LCM module  (100% unchanged)
# -----------------------------
class LCM3D(nn.Module):
    """
    Local Consistency / Local Motion emphasis block:
    depthwise 3D conv -> pointwise 1x1x1 -> BN+ReLU -> gate -> residual add
    """
    def __init__(self, channels: int, k_t: int = 3, k_s: int = 3):
        super().__init__()
        pad_t = k_t // 2
        pad_s = k_s // 2
        self.dw  = nn.Conv3d(channels, channels,
                             kernel_size=(k_t, k_s, k_s),
                             padding=(pad_t, pad_s, pad_s),
                             groups=channels, bias=False)
        self.pw  = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.bn  = nn.BatchNorm3d(channels)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.act(self.bn(self.pw(self.dw(x))))
        return x + y * self.gate(y)


# -----------------------------
# LSTM head  (new)
# -----------------------------
class LSTMHead(nn.Module):
    """
    LCM  → refines WHERE the fight is (local spatial attention)
    LSTM → reasons about WHEN / how it evolves (temporal)
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.drop(out[:, -1, :])


# -----------------------------
# Model: R3D-18 + LCM + LSTM
# -----------------------------
class R3D18WithLCM_LSTM(nn.Module):
    def __init__(self, weights, num_classes=2, lcm_after="layer4",
                 dropout_p=0.4, lstm_hidden=256, lstm_layers=1, lstm_dropout=0.3):
        super().__init__()
        base = r3d_18(weights=weights)

        # backbone (same as your original)
        self.stem   = base.stem
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        # avgpool + base.fc replaced by LSTM

        self.lcm_after = lcm_after
        self.lcm = LCM3D(channels=256 if lcm_after == "layer3" else 512)

        # keep T dim alive for LSTM
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.lstm_head    = LSTMHead(512, lstm_hidden, lstm_layers, lstm_dropout)

        # same dropout_p as your original fc head
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(lstm_hidden, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        if self.lcm_after == "layer3":
            x = self.lcm(x)
        x = self.layer4(x)
        if self.lcm_after == "layer4":
            x = self.lcm(x)
        # (B,512,T',H',W') → (B,T',512)
        x = self.spatial_pool(x).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        return self.fc(self.lstm_head(x))




# -----------------------------
# LSTM-only model (no LCM)
# -----------------------------
class R3D18_LSTM_Only(nn.Module):
    """
    R3D-18 + LSTM only — no LCM.
    Use this when USE_LCM = False in CFG.

    Forward pass:
      Input (B,3,T,H,W)
        → R3D-18 stem + layer1-4   (spatial + temporal features)
        → Spatial Pool             (collapse H & W, keep T)
        → LSTM                     (temporal reasoning)
        → Dropout → FC → logits
    """
    def __init__(self, weights, num_classes=2, dropout_p=0.4,
                 lstm_hidden=256, lstm_layers=1, lstm_dropout=0.3):
        super().__init__()
        base = r3d_18(weights=weights)

        self.stem   = base.stem
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        # no LCM — goes straight to spatial pool

        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.lstm_head    = LSTMHead(512, lstm_hidden, lstm_layers, lstm_dropout)
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(lstm_hidden, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # (B,512,T'H',W') → (B,T',512)
        x = self.spatial_pool(x).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        return self.fc(self.lstm_head(x))

# -----------------------------
# Train / Eval  (unchanged + grad clip)
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, print_every):
    model.train()
    losses = []
    preds_all, targets_all = [], []

    for i, (x, y) in enumerate(loader):
        if i == 0:
            print(f"[TRAIN] First batch ✅ batch_size={x.size(0)}", flush=True)
        if i % print_every == 0:
            print(f"[TRAIN] batch {i}/{len(loader)}", flush=True)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # LSTM stability
        optimizer.step()

        losses.append(loss.item())
        preds_all.append(logits.argmax(1).detach().cpu().numpy())
        targets_all.append(y.detach().cpu().numpy())

    preds_all   = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    return (float(np.mean(losses)),
            float(accuracy_score(targets_all, preds_all)),
            float(f1_score(targets_all, preds_all)))


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, print_every):
    model.eval()
    losses = []
    preds_all, targets_all = [], []

    for i, (x, y) in enumerate(loader):
        if i == 0:
            print(f"[VAL] First batch ✅ batch_size={x.size(0)}", flush=True)
        if i % print_every == 0:
            print(f"[VAL] batch {i}/{len(loader)}", flush=True)

        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)

        losses.append(loss.item())
        preds_all.append(logits.argmax(1).cpu().numpy())
        targets_all.append(y.cpu().numpy())

    preds_all   = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    return (float(np.mean(losses)),
            float(accuracy_score(targets_all, preds_all)),
            float(f1_score(targets_all, preds_all)))


# ══════════════════════════════════════════════
# PLOTS  (new)
# ══════════════════════════════════════════════

def save_loss_curve(csv_path: Path, plots_dir: Path):
    df = pd.read_csv(csv_path)
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train")
    plt.plot(df["epoch"], df["val_loss"],   label="val", linestyle="--")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Train vs Val Loss  (R3D-18 + LCM + LSTM)")
    plt.legend(); plt.tight_layout()
    out = plots_dir / "loss_curve.png"
    plt.savefig(out, dpi=300); plt.close()
    print(f"[✓] Saved: {out}")


def save_accuracy_curve(csv_path: Path, plots_dir: Path):
    df = pd.read_csv(csv_path)
    plt.figure()
    plt.plot(df["epoch"], df["train_acc"], label="train")
    plt.plot(df["epoch"], df["val_acc"],   label="val", linestyle="--")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Train vs Val Accuracy  (R3D-18 + LCM + LSTM)")
    plt.legend(); plt.tight_layout()
    out = plots_dir / "accuracy_curve.png"
    plt.savefig(out, dpi=300); plt.close()
    print(f"[✓] Saved: {out}")


def save_comparison_plot(best_lcm_lstm_acc: float, plots_dir: Path):
    """
    !! Replace baseline_acc and lcm_only_acc with your real numbers !!
    """
    baseline_acc = 0.88   # R3D-18 alone        ← replace
    lcm_only_acc = 0.91   # R3D-18 + LCM only   ← replace

    models = ["R3D-18\nBaseline", "R3D-18\n+ LCM", "R3D-18\n+ LCM + LSTM"]
    accs   = [baseline_acc, lcm_only_acc, best_lcm_lstm_acc]
    colors = ["#6366f1", "#ec4899", "#06b6d4"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(models, accs, color=colors, width=0.45,
                  edgecolor="white", linewidth=0.6)
    ax.set_ylabel("Best Val Accuracy", fontsize=11)
    ax.set_title("Model Comparison: Baseline vs +LCM vs +LCM+LSTM", fontsize=12, pad=12)
    ax.set_ylim(max(0, min(accs) - 0.08), 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.1%}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = plots_dir / "comparison.png"
    plt.savefig(out, dpi=300); plt.close()
    print(f"[✓] Saved: {out}")


def save_architecture_diagram(plots_dir: Path):
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.set_axis_off()

    blocks = [
        ("Input\nClip\n(B,3,T,H,W)",  "#1e3a5f"),
        ("R3D-18\nBackbone\nlayer1→4", "#312e81"),
        ("LCM\nLocal Context\nModule",  "#831843"),
        ("Spatial\nPool\n(B,T',512)",   "#064e3b"),
        ("LSTM\nTemporal\nh=256",       "#0c4a6e"),
        ("FC\n256→2",                   "#431407"),
        ("Fight /\nNonFight",           "#450a0a"),
    ]
    fg_colors = ["#93c5fd", "#a5b4fc", "#f9a8d4",
                 "#6ee7b7", "#7dd3fc", "#fdba74", "#fca5a5"]

    n  = len(blocks); w = 0.114; gap = 0.018
    y  = 0.20;        h = 0.60;  x0  = 0.02

    for i, ((label, bg), fg) in enumerate(zip(blocks, fg_colors)):
        xi   = x0 + i * (w + gap)
        rect = FancyBboxPatch((xi, y), w, h,
                              boxstyle="round,pad=0.01,rounding_size=0.015",
                              facecolor=bg, edgecolor=fg, linewidth=1.6,
                              alpha=0.95, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(xi + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=8.5, color=fg, fontweight="bold", linespacing=1.5,
                transform=ax.transAxes)
        if i < n - 1:
            xarr = xi + w + gap / 2
            ax.annotate("",
                        xy=(xarr + 0.001, y + h / 2),
                        xytext=(xarr - 0.001, y + h / 2),
                        xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color="#475569",
                                        lw=2.0, mutation_scale=16))

    for idx, note in {2: "WHERE the\nfight is", 4: "WHEN / how\nit evolves"}.items():
        xi = x0 + idx * (w + gap) + w / 2
        ax.text(xi, y - 0.18, note, ha="center", va="top", fontsize=8,
                color="#94a3b8", linespacing=1.35, transform=ax.transAxes)

    ax.set_xlim(0, 1); ax.set_ylim(-0.1, 1)
    ax.set_title("Violence Detection Architecture: R3D-18 → LCM → LSTM",
                 fontsize=12, fontweight="bold", pad=10, loc="left")
    plt.tight_layout()
    out = plots_dir / "architecture_diagram.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[✓] Saved: {out}")


# -----------------------------
# Main
# -----------------------------
def main():
    cfg    = CFG()
    device = torch.device(cfg.DEVICE)

    cfg.SAVE_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    weights = R3D_18_Weights.DEFAULT
    mean    = weights.transforms().mean
    std     = weights.transforms().std

    train_ds = RWFVideoDataset(cfg.DATA_ROOT, "train", cfg.CLASSES, cfg.NUM_FRAMES, cfg.IMG_SIZE, mean, std)
    val_ds   = RWFVideoDataset(cfg.DATA_ROOT, "val",   cfg.CLASSES, cfg.NUM_FRAMES, cfg.IMG_SIZE, mean, std)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=cfg.NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    if cfg.USE_LCM:
        model = R3D18WithLCM_LSTM(
            weights=weights, num_classes=cfg.NUM_CLASSES,
            lcm_after=cfg.LCM_AFTER, dropout_p=0.4,
            lstm_hidden=cfg.LSTM_HIDDEN, lstm_layers=cfg.LSTM_LAYERS,
            lstm_dropout=cfg.LSTM_DROPOUT,
        )
    else:
        # LSTM only — no LCM
        model = R3D18_LSTM_Only(
            weights      = weights,
            num_classes  = cfg.NUM_CLASSES,
            dropout_p    = 0.4,
            lstm_hidden  = cfg.LSTM_HIDDEN,
            lstm_layers  = cfg.LSTM_LAYERS,
            lstm_dropout = cfg.LSTM_DROPOUT,
        )

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion    = nn.CrossEntropyLoss(label_smoothing=0.1)
    writer       = SummaryWriter(log_dir=str(cfg.LOG_DIR))
    metrics_path = cfg.METRICS_DIR / "train_log_lcm_lstm.csv"

    print("========================================",  flush=True)
    print(f"RWF-2000  R3D-18 + {"LCM + " if cfg.USE_LCM else ""}LSTM",  flush=True)
    print(f"DATA_ROOT   : {cfg.DATA_ROOT}",            flush=True)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}", flush=True)
    print(f"Device      : {device}",                   flush=True)
    print(f"Params      : {total_params:,}",           flush=True)
    print(f"TensorBoard : {cfg.LOG_DIR}",              flush=True)
    print(f"LCM_AFTER={cfg.LCM_AFTER} | LSTM_HIDDEN={cfg.LSTM_HIDDEN}", flush=True)
    print("========================================",  flush=True)

    best_val_acc = 0.0
    best_epoch   = 0
    log_rows     = []

    for epoch in range(1, cfg.EPOCHS + 1):

        lr = 1e-4 if epoch <= 5 else (3e-5 if epoch <= 10 else 1e-5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        t0 = time.time()
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device, cfg.PRINT_EVERY)
        val_loss,   val_acc,   val_f1   = eval_one_epoch(model, val_loader, criterion, device, cfg.PRINT_EVERY)
        dt = time.time() - t0

        # TensorBoard (unchanged)
        writer.add_scalar("Loss/train",     train_loss, epoch)
        writer.add_scalar("Loss/val",       val_loss,   epoch)
        writer.add_scalar("Accuracy/train", train_acc,  epoch)
        writer.add_scalar("Accuracy/val",   val_acc,    epoch)
        writer.add_scalar("F1/train",       train_f1,   epoch)
        writer.add_scalar("F1/val",         val_f1,     epoch)
        writer.add_scalar("LR",             lr,         epoch)

        # CSV logging (new)
        log_rows.append({
            "epoch":          epoch,
            "train_loss":     float(train_loss),
            "train_acc":      float(train_acc),
            "train_f1":       float(train_f1),
            "val_loss":       float(val_loss),
            "val_acc":        float(val_acc),
            "val_f1":         float(val_f1),
            "lr":             lr,
            "epoch_time_sec": float(dt),
            "lcm_after":      cfg.LCM_AFTER,
            "lstm_hidden":    cfg.LSTM_HIDDEN,
            "lstm_layers":    cfg.LSTM_LAYERS,
        })
        pd.DataFrame(log_rows).to_csv(metrics_path, index=False)

        # Save last checkpoint every epoch (new)
        torch.save({
            "epoch":        epoch,
            "model_state":  model.state_dict(),
            "val_acc":      val_acc,
            "classes":      cfg.CLASSES,
            "num_frames":   cfg.NUM_FRAMES,
            "img_size":     cfg.IMG_SIZE,
            "use_lcm":      cfg.USE_LCM,
            "lcm_after":    cfg.LCM_AFTER if cfg.USE_LCM else "",
            "lstm_hidden":  cfg.LSTM_HIDDEN,
            "lstm_layers":  cfg.LSTM_LAYERS,
        }, cfg.SAVE_DIR / "r3d18_last_lcm_lstm.pth")

        # Save best (unchanged + new fields)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "best_val_acc": best_val_acc,
                "classes":      cfg.CLASSES,
                "num_frames":   cfg.NUM_FRAMES,
                "img_size":     cfg.IMG_SIZE,
                "use_lcm":      cfg.USE_LCM,
                "lcm_after":    cfg.LCM_AFTER if cfg.USE_LCM else "",
                "lstm_hidden":  cfg.LSTM_HIDDEN,
                "lstm_layers":  cfg.LSTM_LAYERS,
            }, cfg.SAVE_DIR / cfg.SAVE_NAME)
            print(f"[SAVE] Best updated ✅ epoch={epoch} val_acc={best_val_acc:.4f}", flush=True)

        print(
            f"Epoch {epoch:02d}/{cfg.EPOCHS} | LR={lr:.0e} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
            f"best_val_acc={best_val_acc:.4f} | time {dt:.1f}s",
            flush=True
        )

    writer.close()

    # Save all plots
    save_loss_curve(metrics_path,           cfg.PLOTS_DIR)
    save_accuracy_curve(metrics_path,       cfg.PLOTS_DIR)
    save_comparison_plot(best_val_acc,      cfg.PLOTS_DIR)
    save_architecture_diagram(             cfg.PLOTS_DIR)

    print("\n======================================",  flush=True)
    print("Training Finished ✅",                     flush=True)
    print(f"Final Val Accuracy : {val_acc:.4f}",      flush=True)
    print(f"Final Val F1       : {val_f1:.4f}",       flush=True)
    print(f"Best  Val Accuracy : {best_val_acc:.4f} (epoch {best_epoch})", flush=True)
    print(f"Checkpoint : {cfg.SAVE_DIR / cfg.SAVE_NAME}", flush=True)
    print(f"CSV        : {metrics_path}",             flush=True)
    print(f"Plots      : {cfg.PLOTS_DIR}/",           flush=True)
    print(f"TensorBoard: tensorboard --logdir runs",  flush=True)
    print("======================================\n", flush=True)


if __name__ == "__main__":
    main()