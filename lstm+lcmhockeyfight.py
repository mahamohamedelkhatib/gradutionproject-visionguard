# train_r3d_finetune_lcm_lstm_pandas_plots_15ep.py
# ------------------------------------------------------------
# Fine-tune R3D-18 on HockeyFight (MP4 splits) + LCM + LSTM
# + pandas CSV logging + TensorBoard + poster-ready plots
#
# Folder:
# data/splits_mp4/
#   train/{fight,nonfight}/*.mp4
#   val/{fight,nonfight}/*.mp4
#   test/{fight,nonfight}/*.mp4
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

from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter   # ← NEW

import torchvision
import torchvision.transforms.functional as TF

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# -----------------------------
# Settings  (unchanged from your original)
# -----------------------------
BASE_DIR  = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data" / "splits_mp4"

CLASSES     = ["nonfight", "fight"]   # 0, 1
NUM_CLASSES = 2

# For Mac MPS uncomment the line below:
# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DEVICE = torch.device("cpu")

EPOCHS       = 15
BATCH_SIZE   = 2
LR           = 1e-4
WEIGHT_DECAY = 1e-4

# clip settings
NUM_FRAMES = 16
IMG_SIZE   = 112

# Kinetics-400 style normalization
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
STD  = torch.tensor([0.22803, 0.22145,  0.216989]).view(3, 1, 1, 1)

# ── NEW: LSTM settings ──
LSTM_HIDDEN  = 256   # hidden state size
LSTM_LAYERS  = 1     # stacked LSTM layers (keep 1 for stability on CPU)
LSTM_DROPOUT = 0.3   # dropout between layers (only active if LSTM_LAYERS > 1)

# LCM settings
USE_LCM   = True
LCM_AFTER = "layer4"   # "layer3" (256ch) or "layer4" (512ch)

# logging / outputs
OUT_DIR     = BASE_DIR / "outputs"
METRICS_DIR = OUT_DIR / "metrics"
PLOTS_DIR   = OUT_DIR / "plots"
TB_DIR      = BASE_DIR / "runs" / "lcm_lstm"   # ← TensorBoard logs here
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TB_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Dataset  (100% unchanged)
# -----------------------------
class HockeyMP4Dataset(Dataset):
    def __init__(self, split: str):
        self.split   = split
        self.samples = []

        split_dir = DATA_ROOT / split
        for label, cls in enumerate(CLASSES):
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.glob("*.mp4")):
                self.samples.append((p, label))

        print(f"{split.capitalize()} samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _uniform_indices(self, T: int, n: int):
        if T <= 0:
            return [0] * n
        if T >= n:
            return [int(round(i * (T - 1) / (n - 1))) for i in range(n)]
        idx = list(range(T))
        while len(idx) < n:
            idx.append(T - 1)
        return idx

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            frames, _, _ = torchvision.io.read_video(str(path), pts_unit="sec")
        except Exception:
            clip = torch.zeros(3, NUM_FRAMES, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
            return clip, torch.tensor(label, dtype=torch.long)

        T       = frames.shape[0]
        use_idx = self._uniform_indices(T, NUM_FRAMES)
        frames  = frames[use_idx]                              # (T, H, W, C)
        clip    = frames.permute(3, 0, 1, 2).float() / 255.0  # (C, T, H, W)

        resized = []
        for t in range(clip.shape[1]):
            frame = clip[:, t, :, :]
            frame = TF.resize(frame, [IMG_SIZE, IMG_SIZE], antialias=True)
            resized.append(frame)
        clip = torch.stack(resized, dim=1)   # (C, T, H, W)
        clip = (clip - MEAN) / STD

        return clip, torch.tensor(label, dtype=torch.long)

# -----------------------------
# LCM module  (100% unchanged)
# -----------------------------
class LCM3D(nn.Module):
    """
    Local Context Module:
    depthwise 3D conv → pointwise 1×1×1 → sigmoid gate → residual add
    Focuses on local spatio-temporal motion patterns (WHERE the fight is).
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
        y = self.dw(x)
        y = self.pw(y)
        y = self.bn(y)
        y = self.act(y)
        g = self.gate(y)
        y = y * g
        return x + y   # residual


# ── NEW: LSTM temporal head ──────────────────────────────────
class LSTMHead(nn.Module):
    """
    Temporal reasoning module placed AFTER LCM.

    Takes the spatial feature map from layer4, pools H & W to get one
    feature vector per time-step, then runs an LSTM across time.
    The last hidden state is passed to the classifier.

    Why after LCM?
      LCM refines WHAT each frame looks like (local spatial attention).
      LSTM then reasons about HOW those refined features change over time.
    """
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x):
        # x: (B, T, input_size)
        out, _ = self.lstm(x)       # (B, T, hidden_size)
        last   = out[:, -1, :]      # take last time-step
        return self.drop(last)      # (B, hidden_size)


# -----------------------------
# Model: R3D-18 + LCM + LSTM  (replaces R3D18WithLCM)
# -----------------------------
class R3D18WithLCM_LSTM(nn.Module):
    """
    Forward pass:
      Input (B,3,T,H,W)
        → R3D-18 stem + layer1 + layer2 + layer3
        → [LCM if lcm_after=="layer3"]
        → layer4
        → [LCM if lcm_after=="layer4"]
        → Spatial pool: (B,512,T',H',W') → (B,T',512)
        → LSTM → last hidden (B,256)
        → FC → logits (B,2)
    """
    def __init__(self, num_classes: int = 2, lcm_after: str = "layer4",
                 lstm_hidden: int = LSTM_HIDDEN,
                 lstm_layers: int = LSTM_LAYERS,
                 lstm_dropout: float = LSTM_DROPOUT):
        super().__init__()

        base = torchvision.models.video.r3d_18(
            weights=torchvision.models.video.R3D_18_Weights.DEFAULT
        )

        # backbone (same as your original)
        self.stem   = base.stem
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        # NOTE: avgpool and base.fc are NOT used — LSTM replaces them

        # LCM (same logic as your original)
        self.lcm_after = lcm_after
        if lcm_after == "layer3":
            self.lcm = LCM3D(channels=256)
        elif lcm_after == "layer4":
            self.lcm = LCM3D(channels=512)
        else:
            raise ValueError("lcm_after must be 'layer3' or 'layer4'")

        # NEW: pool spatial dims but keep temporal dim for LSTM
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # (B,512,T',1,1)

        # NEW: LSTM head
        self.lstm_head = LSTMHead(
            input_size  = 512,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            dropout     = lstm_dropout,
        )

        # NEW: classifier takes LSTM hidden size (not 512)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.lcm_after == "layer3":
            x = self.lcm(x)

        x = self.layer4(x)

        if self.lcm_after == "layer4":
            x = self.lcm(x)

        # x: (B, 512, T', H', W')
        x = self.spatial_pool(x)          # (B, 512, T', 1, 1)
        x = x.squeeze(-1).squeeze(-1)     # (B, 512, T')
        x = x.permute(0, 2, 1)            # (B, T', 512) ← sequence for LSTM

        x = self.lstm_head(x)             # (B, lstm_hidden)
        x = self.fc(x)                    # (B, num_classes)
        return x


def build_model(use_lcm: bool = True, lcm_after: str = "layer4"):
    if use_lcm:
        model = R3D18WithLCM_LSTM(num_classes=NUM_CLASSES, lcm_after=lcm_after)
    else:
        # baseline without LCM or LSTM
        model = torchvision.models.video.r3d_18(
            weights=torchvision.models.video.R3D_18_Weights.DEFAULT
        )
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

# -----------------------------
# Train / Eval  (unchanged except grad clip for LSTM stability)
# -----------------------------
@torch.no_grad()
def run_eval(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y   = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss   = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def run_train(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    t0 = time.time()

    for i, (x, y) in enumerate(loader):
        if i % 20 == 0:
            print(f"  batch {i}/{len(loader)}")

        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()

        # ← NEW: gradient clipping (important for LSTM stability)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += x.size(0)

    dt = time.time() - t0
    return total_loss / max(total, 1), correct / max(total, 1), dt

# -----------------------------
# Plotting  (your originals kept + 2 new ones)
# -----------------------------
def save_curves(csv_path: Path):
    """Your original loss + accuracy curves — unchanged."""
    df = pd.read_csv(csv_path)

    # Loss curve
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"])
    plt.plot(df["epoch"], df["val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss  (R3D-18 + LCM + LSTM)")
    plt.legend(["train", "val"])
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "loss_curve.png", dpi=300)
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(df["epoch"], df["train_acc"])
    plt.plot(df["epoch"], df["val_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Val Accuracy  (R3D-18 + LCM + LSTM)")
    plt.legend(["train", "val"])
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_curve.png", dpi=300)
    plt.close()


def save_comparison_plot(csv_path: Path):
    """
    NEW: bar chart comparing baseline vs +LCM vs +LCM+LSTM
    Uses the best val_acc from the current training CSV for +LCM+LSTM.
    The baseline & +LCM values below are placeholders — replace with
    your real numbers after running each variant.
    """
    df = pd.read_csv(csv_path)
    best_lcm_lstm = float(df["val_acc"].max())

    # ── Replace these with your real results ──
    baseline_acc = 0.88   # R3D-18 alone
    lcm_only_acc = 0.91   # R3D-18 + LCM  (from your previous run)
    # ──────────────────────────────────────────

    models  = ["R3D-18\nBaseline", "R3D-18\n+ LCM", "R3D-18\n+ LCM + LSTM"]
    accs    = [baseline_acc, lcm_only_acc, best_lcm_lstm]
    colors  = ["#6366f1", "#ec4899", "#06b6d4"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(models, accs, color=colors, width=0.45, edgecolor="white", linewidth=0.6)
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
    plt.savefig(PLOTS_DIR / "comparison.png", dpi=300)
    plt.close()


def save_architecture_diagram(out_path: Path):
    """
    NEW: full pipeline block diagram for your poster/report.
    Shows: Input → R3D-18 → LCM → Spatial Pool → LSTM → FC → Output
    """
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.set_axis_off()

    blocks = [
        ("Input\nClip\n(B,3,T,H,W)",    "#1e3a5f"),
        ("R3D-18\nBackbone\nlayer1→4",   "#312e81"),
        ("LCM\nLocal Context\nModule",   "#831843"),
        ("Spatial\nPool\n(B,T',512)",    "#064e3b"),
        ("LSTM\nTemporal\nh=256",        "#0c4a6e"),
        ("FC\n256→2",                    "#431407"),
        ("Fight /\nNonfight",            "#450a0a"),
    ]
    fg_colors = ["#93c5fd", "#a5b4fc", "#f9a8d4",
                 "#6ee7b7", "#7dd3fc", "#fdba74", "#fca5a5"]

    n   = len(blocks)
    w   = 0.114
    gap = 0.018
    y   = 0.20
    h   = 0.60
    x0  = 0.02

    for i, ((label, bg), fg) in enumerate(zip(blocks, fg_colors)):
        xi   = x0 + i * (w + gap)
        rect = FancyBboxPatch((xi, y), w, h,
                              boxstyle="round,pad=0.01,rounding_size=0.015",
                              facecolor=bg, edgecolor=fg,
                              linewidth=1.6, alpha=0.95,
                              transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(xi + w / 2, y + h / 2, label,
                ha="center", va="center",
                fontsize=8.5, color=fg, fontweight="bold",
                linespacing=1.5, transform=ax.transAxes)

        if i < n - 1:
            xarr = xi + w + gap / 2
            ax.annotate("",
                        xy=(xarr + 0.001, y + h / 2),
                        xytext=(xarr - 0.001, y + h / 2),
                        xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color="#475569",
                                        lw=2.0, mutation_scale=16))

    # annotations below each key block
    notes = {2: "WHERE the\nfight is",
             4: "WHEN / how\nit evolves"}
    for idx, note in notes.items():
        xi = x0 + idx * (w + gap) + w / 2
        ax.text(xi, y - 0.18, note,
                ha="center", va="top", fontsize=8,
                color="#94a3b8", linespacing=1.35,
                transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1)
    ax.set_title("Violence Detection Architecture: R3D-18 → LCM → LSTM",
                 fontsize=12, fontweight="bold", pad=10, loc="left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_lcm_block_diagram(out_path: Path):
    """Your original LCM block diagram — unchanged."""
    plt.figure(figsize=(10, 2.4))
    ax = plt.gca()
    ax.set_axis_off()

    def box(x, y, w, h, text):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02")
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    y = 0.35
    h = 0.35
    w = 0.16
    gap = 0.03

    x0 = 0.02
    box(x0,              y, w, h, "Input\nFeature (3D)")
    x1 = x0 + w + gap;  box(x1, y, w, h, "Depthwise\n3D Conv")
    x2 = x1 + w + gap;  box(x2, y, w, h, "Pointwise\n1×1×1")
    x3 = x2 + w + gap;  box(x3, y, w, h, "Gate\n(AvgPool+σ)")
    x4 = x3 + w + gap;  box(x4, y, w, h, "Residual\nAdd")

    def arrow(xa, xb):
        ax.annotate("", xy=(xb, y + h / 2), xytext=(xa, y + h / 2),
                    arrowprops=dict(arrowstyle="->", lw=1.5))

    arrow(x0 + w, x1); arrow(x1 + w, x2)
    arrow(x2 + w, x3); arrow(x3 + w, x4)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# -----------------------------
# Main
# -----------------------------
def main():
    print("Device    :", DEVICE)
    print("DATA_ROOT :", DATA_ROOT)
    print("Exists    :", DATA_ROOT.exists())
    print(f"Architecture: R3D-18 + LCM (after {LCM_AFTER}) + LSTM (hidden={LSTM_HIDDEN})")

    train_ds = HockeyMP4Dataset("train")
    val_ds   = HockeyMP4Dataset("val")

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Dataset empty. Check data/splits_mp4/train|val/{fight,nonfight}/*.mp4")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model(use_lcm=USE_LCM, lcm_after=LCM_AFTER).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    ckpt_dir  = BASE_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path    = ckpt_dir / "r3d18_best_lcm_lstm.pth"
    last_path    = ckpt_dir / "r3d18_last_lcm_lstm.pth"
    metrics_path = METRICS_DIR / "train_log_lcm_lstm.csv"

    # ── NEW: TensorBoard writer ──
    writer = SummaryWriter(log_dir=str(TB_DIR))
    print(f"TensorBoard logs → {TB_DIR}")
    print(f"  View with:  tensorboard --logdir {TB_DIR}")

    best_val_acc = -1.0
    log_rows     = []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_time = run_train(model, train_loader, optimizer, criterion)
        va_loss, va_acc          = run_eval(model, val_loader, criterion)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f} | "
            f"time={tr_time:.1f}s"
        )

        # ── NEW: log to TensorBoard ──
        writer.add_scalars("Loss",     {"train": tr_loss, "val": va_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": tr_acc,  "val": va_acc},  epoch)
        writer.add_scalar("Epoch_time_sec", tr_time, epoch)

        # CSV logging (your original)
        log_rows.append({
            "epoch":          epoch,
            "train_loss":     float(tr_loss),
            "train_acc":      float(tr_acc),
            "val_loss":       float(va_loss),
            "val_acc":        float(va_acc),
            "epoch_time_sec": float(tr_time),
            "use_lcm":        int(USE_LCM),
            "lcm_after":      LCM_AFTER if USE_LCM else "",
            "lstm_hidden":    LSTM_HIDDEN,
            "lstm_layers":    LSTM_LAYERS,
        })
        pd.DataFrame(log_rows).to_csv(metrics_path, index=False)

        # save last checkpoint (your original)
        torch.save({
            "model_state":  model.state_dict(),
            "epoch":        epoch,
            "val_acc":      va_acc,
            "use_lcm":      USE_LCM,
            "lcm_after":    LCM_AFTER,
            "lstm_hidden":  LSTM_HIDDEN,
            "lstm_layers":  LSTM_LAYERS,
        }, last_path)

        # save best checkpoint (your original)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "model_state":  model.state_dict(),
                "epoch":        epoch,
                "val_acc":      va_acc,
                "use_lcm":      USE_LCM,
                "lcm_after":    LCM_AFTER,
                "lstm_hidden":  LSTM_HIDDEN,
                "lstm_layers":  LSTM_LAYERS,
            }, best_path)
            print(f"  ↑ New best saved (val_acc={va_acc:.4f})")

    writer.close()   # ← flush TensorBoard

    print("\nSaved best    :", best_path)
    print("Saved last    :", last_path)
    print("Best val_acc  :", best_val_acc)
    print("Metrics CSV   :", metrics_path)

    # ── All plots ──
    save_curves(metrics_path)                                          # loss_curve.png + accuracy_curve.png
    save_comparison_plot(metrics_path)                                 # comparison.png
    save_architecture_diagram(PLOTS_DIR / "architecture_diagram.png") # architecture_diagram.png
    save_lcm_block_diagram(PLOTS_DIR / "lcm_block_diagram.png")       # your original

    print("\nSaved plots:")
    for p in sorted(PLOTS_DIR.glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()