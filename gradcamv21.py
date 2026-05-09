"""
run_gradcam_v2.py  —  GradCAM Pipeline V2
==========================================
R3D-18 + LCM + LSTM  |  Onset-aware heatmap  |  Research-grade pred.txt

Per-video output folder:
    <vid>_gradcam_onset.mp4       GradCAM overlay (heatmap from fight onset only)
    <vid>_gradcampp_onset.mp4     GradCAM++ variant
    <vid>_original.mp4            Clean copy
    gradcam_grid.png              2×8 frame grid (GradCAM)
    gradcampp_grid.png            2×8 frame grid (GradCAM++)
    raw_grid.png                  2×8 frame grid (raw)
    pred.txt                      Full research-grade metadata
    timeline.png                  Violence confidence curve over time

Usage:
    python run_gradcam_v2.py --dataset hockeyfight
    python run_gradcam_v2.py --dataset rwf
    python run_gradcam_v2.py --dataset both
    python run_gradcam_v2.py --dataset both --thresh 0.65 --epoch 12 --val_acc 0.98
"""

import os, cv2, torch, argparse, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.models.video import r3d_18, R3D_18_Weights
import torchvision.transforms as T

# ──────────────────────────────────────────────
#  CONFIG  — edit paths here if needed
# ──────────────────────────────────────────────
PROJECT_ROOT      = Path.home() / "Desktop" / "gradproject3"
HOCKEYFIGHT_CKPT  = PROJECT_ROOT / "checkpoints" / "r3d18_best_lcm_lstm.pth"
RWF_CKPT          = PROJECT_ROOT / "checkpoints" / "r3d18_best_RWF_lcm_lstm.pth"
HOCKEYFIGHT_TEST  = PROJECT_ROOT / "data" / "splits_mp4" / "test"
RWF_TEST          = PROJECT_ROOT / "datasets" / "rawf_2000" / "RWF-2000" / "test"
OUTPUT_ROOT       = PROJECT_ROOT / "outputsgradv222"

CLIP_LEN          = 16
CLIP_STRIDE       = 4
FRAME_SIZE        = (112, 112)
VIOLENCE_THRESH   = 0.65
SPIKE_DELTA       = 0.15
TARGET_LAYER_STR  = "layer4[-1].conv2"
GRID_ROWS, GRID_COLS = 2, 8
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {
    "hockeyfight": {"fight": 1, "nofight": 0, "Fight": 1, "NoFight": 0},
    "rwf":         {"Fight": 1, "NoFight": 0,
                    "Violence": 1, "NonViolence": 0, "NonFight": 0},
}
FIGHT_LABELS = {"fight", "Fight", "Violence", "violence"}


# ──────────────────────────────────────────────
#  LCM MODULE
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
#  LCM3D  — exact copy from your training code
#  Keys: lcm.dw / lcm.pw / lcm.bn / lcm.act / lcm.gate.0 / lcm.gate.1
# ──────────────────────────────────────────────
class LCM3D(nn.Module):
    """
    Exact match to training code LCM3D:
      dw   = depthwise Conv3d (k_t=3, k_s=3, groups=C)
      pw   = pointwise Conv3d 1x1x1
      bn   = BatchNorm3d
      act  = ReLU(inplace=True)
      gate = Sequential(AdaptiveAvgPool3d(1), Conv3d(C,C,1), Sigmoid)
    forward: y = act(bn(pw(dw(x)))); return x + y * gate(y)
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


# ──────────────────────────────────────────────
#  LSTMHead — exact copy from your training code
#  Keys: lstm_head.lstm.* / lstm_head.drop.*
#  NOTE: fc lives on the MAIN model, not inside LSTMHead
# ──────────────────────────────────────────────
class LSTMHead(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.drop(out[:, -1, :])


# ──────────────────────────────────────────────
#  HF MODEL  — R3D18WithLCM_LSTM (HockeyFight)
#  fc = nn.Linear(lstm_hidden, num_classes)
# ──────────────────────────────────────────────
class R3D18WithLCM_LSTM_HF(nn.Module):
    def __init__(self, num_classes=2, lcm_after="layer4",
                 lstm_hidden=256, lstm_layers=1, lstm_dropout=0.3):
        super().__init__()
        base = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.stem   = base.stem
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4          # GradCAM target: layer4[-1].conv2
        self.lcm_after    = lcm_after
        self.lcm          = LCM3D(channels=256 if lcm_after == "layer3" else 512)
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.lstm_head    = LSTMHead(512, lstm_hidden, lstm_layers, lstm_dropout)
        self.fc           = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        x = self.stem(x);   x = self.layer1(x)
        x = self.layer2(x); x = self.layer3(x)
        if self.lcm_after == "layer3":
            x = self.lcm(x)
        x = self.layer4(x)
        if self.lcm_after == "layer4":
            x = self.lcm(x)
        x = self.spatial_pool(x).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        return self.fc(self.lstm_head(x))


# ──────────────────────────────────────────────
#  RWF MODEL  — R3D18WithLCM_LSTM (RWF-2000)
#  fc = nn.Sequential(Dropout(0.4), Linear(lstm_hidden, num_classes))
# ──────────────────────────────────────────────
class R3D18WithLCM_LSTM_RWF(nn.Module):
    def __init__(self, num_classes=2, lcm_after="layer4", dropout_p=0.4,
                 lstm_hidden=256, lstm_layers=1, lstm_dropout=0.3):
        super().__init__()
        base = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.stem   = base.stem
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4          # GradCAM target: layer4[-1].conv2
        self.lcm_after    = lcm_after
        self.lcm          = LCM3D(channels=256 if lcm_after == "layer3" else 512)
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.lstm_head    = LSTMHead(512, lstm_hidden, lstm_layers, lstm_dropout)
        self.fc           = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(lstm_hidden, num_classes)
        )

    def forward(self, x):
        x = self.stem(x);   x = self.layer1(x)
        x = self.layer2(x); x = self.layer3(x)
        if self.lcm_after == "layer3":
            x = self.lcm(x)
        x = self.layer4(x)
        if self.lcm_after == "layer4":
            x = self.lcm(x)
        x = self.spatial_pool(x).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        return self.fc(self.lstm_head(x))


def load_model(ckpt_path, num_classes=2, dataset_name="hockeyfight"):
    raw = torch.load(ckpt_path, map_location=DEVICE)

    # ── Read config saved in checkpoint ───────────────────────
    lstm_hidden  = raw.get("lstm_hidden", 256)     if isinstance(raw, dict) else 256
    lstm_layers  = raw.get("lstm_layers", 1)       if isinstance(raw, dict) else 1
    lcm_after    = raw.get("lcm_after",  "layer4") if isinstance(raw, dict) else "layer4"
    epoch        = raw.get("epoch",       None)    if isinstance(raw, dict) else None
    val_acc      = raw.get("val_acc", raw.get("best_val_acc", None)) if isinstance(raw, dict) else None

    print(f"  [CKPT] lcm_after={lcm_after}  lstm_hidden={lstm_hidden}  lstm_layers={lstm_layers}")

    # ── Build exact architecture per dataset ──────────────────
    if dataset_name == "hockeyfight":
        model = R3D18WithLCM_LSTM_HF(
            num_classes=num_classes, lcm_after=lcm_after,
            lstm_hidden=lstm_hidden, lstm_layers=lstm_layers
        ).to(DEVICE)
    else:  # rwf
        model = R3D18WithLCM_LSTM_RWF(
            num_classes=num_classes, lcm_after=lcm_after,
            lstm_hidden=lstm_hidden, lstm_layers=lstm_layers
        ).to(DEVICE)

    # ── Extract weights ────────────────────────────────────────
    if isinstance(raw, dict):
        sd = raw.get("model_state", raw.get("model_state_dict", raw.get("state_dict", None)))
        if sd is None:
            raise KeyError(f"No weights found. Checkpoint keys: {list(raw.keys())}")
    else:
        sd = raw

    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    print(f"  [CKPT] {len(sd)} weight tensors  |  first key: {list(sd.keys())[0]}")

    missing, unexpected = model.load_state_dict(sd, strict=True)
    if not missing and not unexpected:
        print(f"  [OK] All {len(sd)} weights loaded perfectly!")
    else:
        if missing:   print(f"  [WARN] {len(missing)} missing:    {missing[:3]}")
        if unexpected: print(f"  [WARN] {len(unexpected)} unexpected: {unexpected[:3]}")

    # ── Sanity check ──────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, CLIP_LEN, *FRAME_SIZE).to(DEVICE)
        probs = torch.softmax(model(dummy), dim=1)[0]
        print(f"  [SANITY] probs={probs.cpu().numpy()}  max={probs.max().item():.4f}")
        if probs.max().item() < 0.6:
            print(f"  [WARN] Still low confidence — check architecture!")
        else:
            print(f"  [OK] Model is predicting confidently")

    return model, epoch, val_acc



# ──────────────────────────────────────────────
#  GRADCAM 3D
# ──────────────────────────────────────────────
class GradCAM3D:
    def __init__(self, model, use_pp=False):
        self.model  = model
        self.use_pp = use_pp
        self.acts   = None
        self.grads  = None
        # Hook into layer4[-1].conv2
        target = model.layer4[-1].conv2
        self._fh = target.register_forward_hook(self._save_acts)
        self._bh = target.register_full_backward_hook(self._save_grads)

    def _save_acts(self, m, inp, out):  self.acts  = out.clone().detach()
    def _save_grads(self, m, gi, go):   self.grads = go[0].detach()

    def __call__(self, x, class_idx=1):
        self.model.zero_grad()
        logits = self.model(x)
        probs  = torch.softmax(logits, dim=1)[0]
        logits[0, class_idx].backward()

        acts  = self.acts[0]   # (C, T, H, W)
        grads = self.grads[0]

        if self.use_pp:
            a2 = grads ** 2
            a3 = grads ** 3
            denom = 2*a2 + (acts * a3).sum(dim=(1,2,3), keepdim=True)
            denom = torch.where(denom != 0, denom, torch.ones_like(denom))
            alpha   = a2 / denom
            weights = (alpha * F.relu(grads)).sum(dim=(1,2,3))
        else:
            weights = grads.mean(dim=(1,2,3))

        cam = F.relu((weights[:,None,None,None] * acts).sum(dim=0))  # (T,H,W)

        cam_frames = []
        for t in range(cam.shape[0]):
            f = cam[t].cpu().numpy().astype(np.float32)
            mn, mx = f.min(), f.max()
            cam_frames.append((f - mn) / (mx - mn + 1e-8))

        return cam_frames, probs.detach().cpu().numpy()

    def remove(self):
        self._fh.remove()
        self._bh.remove()


# ──────────────────────────────────────────────
#  PREPROCESSING
# ──────────────────────────────────────────────
_norm = T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                    std=[0.22803,  0.22145,  0.216989])

def preprocess(frames_bgr):
    tensors = []
    for f in frames_bgr:
        rgb = cv2.cvtColor(cv2.resize(f, FRAME_SIZE), cv2.COLOR_BGR2RGB)
        t   = torch.from_numpy(rgb).float() / 255.0  # (H, W, C)
        t   = t.permute(2, 0, 1)                      # (C, H, W)
        t   = _norm(t)                                # normalize per frame
        tensors.append(t)
    clip = torch.stack(tensors, dim=1)                # (C, T, H, W)
    return clip.unsqueeze(0).to(DEVICE)               # (1, C, T, H, W)


# ──────────────────────────────────────────────
#  OVERLAY + GRID
# ──────────────────────────────────────────────
def overlay(frame, cam, alpha=0.45):
    h, w = frame.shape[:2]
    c = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (w,h))), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1-alpha, c, alpha, 0)

def make_grid(frames, rows=GRID_ROWS, cols=GRID_COLS, thumb=(160,90)):
    n    = rows * cols
    step = max(1, len(frames) // n)
    sel  = [cv2.resize(frames[min(i*step, len(frames)-1)], thumb) for i in range(n)]
    while len(sel) < n:
        sel.append(np.zeros((thumb[1], thumb[0], 3), dtype=np.uint8))
    return np.vstack([np.hstack(sel[r*cols:(r+1)*cols]) for r in range(rows)])


# ──────────────────────────────────────────────
#  TIMELINE PLOT
# ──────────────────────────────────────────────
def save_timeline(frame_scores, onset_frame, fps, out_path, vid_name):
    times  = [i / fps for i in range(len(frame_scores))]
    scores = frame_scores

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(times, scores, color="#2196F3", linewidth=1.5, label="Violence confidence")
    ax.axhline(VIOLENCE_THRESH, color="red",    linewidth=1.2, linestyle="--", label=f"Threshold ({VIOLENCE_THRESH})")
    ax.axhline(VIOLENCE_THRESH - SPIKE_DELTA, color="orange", linewidth=1.0,
               linestyle=":", label=f"Spike baseline (Δ={SPIKE_DELTA})")

    if onset_frame is not None:
        onset_t = onset_frame / fps
        ax.axvline(onset_t, color="green", linewidth=1.8, linestyle="-",
                   label=f"Fight onset  @ {onset_t:.2f}s (frame {onset_frame})")
        ax.fill_between(times, 0, scores,
                        where=[t >= onset_t for t in times],
                        alpha=0.15, color="green", label="Violence region")

    ax.set_xlim(0, max(times) if times else 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("P(violence)", fontsize=11)
    ax.set_title(f"Violence Confidence Timeline — {vid_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120)
    plt.close()


# ──────────────────────────────────────────────
#  PRED.TXT WRITER
# ──────────────────────────────────────────────
def write_pred_txt(path, **k):
    lines = [
        f"dataset:           {k['dataset']}",
        f"video:             {k['video']}",
        f"true_label:        {k['true_label']}",
        f"pred_class:        {k['pred_class']}",
        f"pred_label:        {k['pred_label']}",
        f"correct:           {k['correct']}",
        f"confidence:        {k['confidence']:.4f}",
        f"probs:             {k['probs']}",
        f"model_path:        {k['model_path']}",
        f"model_epoch:       {k['model_epoch']}",
        f"model_val_acc:     {k['model_val_acc']}",
        f"target_layer:      {TARGET_LAYER_STR}",
        f"onset_frame:       {k['onset_frame']}",
        f"onset_time:        {k['onset_time']}",
        f"onset_threshold:   {VIOLENCE_THRESH}",
        f"spike_delta:       {SPIKE_DELTA}",
        f"total_frames:      {k['total_frames']}",
        f"img_size:          {FRAME_SIZE[0]}",
        f"window_size:       {CLIP_LEN}",
        f"window_stride:     {CLIP_STRIDE}",
        f"cam_target_class:  1 (fight)",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ──────────────────────────────────────────────
#  PROCESS ONE VIDEO
# ──────────────────────────────────────────────
def process_video(video_path: Path, model, out_dir: Path,
                  label_name: str, dataset_name: str,
                  ckpt_path: Path, model_epoch, model_val_acc):

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        all_frames.append(frame)
    cap.release()

    if len(all_frames) < CLIP_LEN:
        print(f"  [SKIP] Too short: {video_path.name} ({len(all_frames)} frames)")
        return

    total_frames  = len(all_frames)
    VIOLENCE_CLS  = 1

    # ── Sliding window inference + GradCAM ──
    gc   = GradCAM3D(model, use_pp=False)
    gcpp = GradCAM3D(model, use_pp=True)

    frame_score  = np.zeros(total_frames)
    frame_pred   = np.zeros(total_frames, dtype=int)
    frame_cam    = [None] * total_frames
    frame_campp  = [None] * total_frames

    clip_starts = list(range(0, total_frames - CLIP_LEN + 1, CLIP_STRIDE))
    if not clip_starts:
        clip_starts = [0]

    for start in clip_starts:
        clip = all_frames[start:start + CLIP_LEN]
        if len(clip) < CLIP_LEN:
            clip = clip + [clip[-1]] * (CLIP_LEN - len(clip))

        x = preprocess(clip)
        try:
            cams,   probs  = gc(x,   class_idx=VIOLENCE_CLS)
            campps, _      = gcpp(x, class_idx=VIOLENCE_CLS)
        except Exception as e:
            print(f"  [WARN] clip@{start}: {e}")
            continue

        score = probs[VIOLENCE_CLS]
        pred  = int(np.argmax(probs))
        n_cam = len(cams)

        for i, fi in enumerate(range(start, start + CLIP_LEN)):
            if fi >= total_frames: break
            cam_idx = min(int(i * n_cam / CLIP_LEN), n_cam - 1)
            if score > frame_score[fi]:
                frame_score[fi] = score
                frame_pred[fi]  = pred
                frame_cam[fi]   = cams[cam_idx]
                frame_campp[fi] = campps[cam_idx]

    gc.remove();  gcpp.remove()

    # ── Onset detection ──
    # Onset = first frame where score >= VIOLENCE_THRESH AND
    #         score > (baseline before it) + SPIKE_DELTA
    onset_frame   = None
    baseline_window = max(1, CLIP_LEN)
    for i in range(total_frames):
        s = frame_score[i]
        if s < VIOLENCE_THRESH:
            continue
        baseline = np.mean(frame_score[max(0, i - baseline_window):i]) if i > 0 else 0.0
        if s - baseline >= SPIKE_DELTA:
            onset_frame = i
            break

    # Fallback: if model detected violence but no spike, use first frame >= thresh
    if onset_frame is None:
        for i, s in enumerate(frame_score):
            if s >= VIOLENCE_THRESH:
                onset_frame = i
                break

    overall_probs  = np.array([1.0, 0.0])  # default
    # Recompute final probs on best clip (centered around onset or middle)
    best_start     = onset_frame if onset_frame is not None else total_frames // 2
    best_start     = max(0, min(best_start, total_frames - CLIP_LEN))
    best_clip      = all_frames[best_start:best_start + CLIP_LEN]
    if len(best_clip) < CLIP_LEN:
        best_clip += [best_clip[-1]] * (CLIP_LEN - len(best_clip))
    with torch.no_grad():
        logits_best = model(preprocess(best_clip))
        overall_probs = torch.softmax(logits_best, dim=1)[0].cpu().numpy()

    pred_class   = int(np.argmax(overall_probs))
    confidence   = float(overall_probs[pred_class])
    is_fight     = pred_class == VIOLENCE_CLS
    true_is_fight = label_name in FIGHT_LABELS

    # Label strings
    pred_label_str = "Fight" if is_fight else "NoFight"
    true_label_str = "Fight" if true_is_fight else "NoFight"
    correct        = (pred_class == (1 if true_is_fight else 0))

    onset_time_str = f"{onset_frame / fps:.2f}s" if onset_frame is not None else "N/A"
    onset_frame_val = onset_frame if onset_frame is not None else "N/A"

    # ── Write videos ──
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w_orig = cv2.VideoWriter(str(out_dir / f"{stem}_original.mp4"),       fourcc, fps, (W, H))
    w_gc   = cv2.VideoWriter(str(out_dir / f"{stem}_gradcam_onset.mp4"),  fourcc, fps, (W, H))
    w_gcpp = cv2.VideoWriter(str(out_dir / f"{stem}_gradcampp_onset.mp4"),fourcc, fps, (W, H))

    out_gc, out_gcpp = [], []

    for i, frame in enumerate(all_frames):
        w_orig.write(frame)
        activate = (
            onset_frame is not None and
            i >= onset_frame and
            frame_score[i] >= VIOLENCE_THRESH and
            frame_cam[i] is not None
        )
        fg   = overlay(frame, frame_cam[i])   if activate else frame.copy()
        fgpp = overlay(frame, frame_campp[i]) if activate else frame.copy()
        w_gc.write(fg);    out_gc.append(fg)
        w_gcpp.write(fgpp); out_gcpp.append(fgpp)

    w_orig.release(); w_gc.release(); w_gcpp.release()

    # ── Grids ──
    cv2.imwrite(str(out_dir / "raw_grid.png"),       make_grid(all_frames))
    cv2.imwrite(str(out_dir / "gradcam_grid.png"),   make_grid(out_gc))
    cv2.imwrite(str(out_dir / "gradcampp_grid.png"), make_grid(out_gcpp))

    # ── Timeline plot ──
    save_timeline(list(frame_score), onset_frame, fps,
                  out_dir / "timeline.png", video_path.name)

    # ── pred.txt ──
    prob_str = f"[{overall_probs[0]:.8f} {overall_probs[1]:.8f}]"
    write_pred_txt(
        out_dir / "pred.txt",
        dataset      = dataset_name,
        video        = str(video_path.relative_to(PROJECT_ROOT)),
        true_label   = true_label_str,
        pred_class   = pred_class,
        pred_label   = pred_label_str,
        correct      = correct,
        confidence   = confidence,
        probs        = prob_str,
        model_path   = str(ckpt_path.relative_to(PROJECT_ROOT)),
        model_epoch  = model_epoch  if model_epoch  is not None else "N/A",
        model_val_acc= f"{model_val_acc:.2f}" if model_val_acc is not None else "N/A",
        onset_frame  = onset_frame_val,
        onset_time   = onset_time_str,
        total_frames = total_frames,
    )

    status = "✓" if correct else "✗"
    print(f"  {status} {video_path.name}  pred={pred_label_str}  "
          f"true={true_label_str}  conf={confidence:.4f}  onset={onset_frame_val}")


# ──────────────────────────────────────────────
#  PROCESS DATASET
# ──────────────────────────────────────────────
def process_dataset(name, test_root, ckpt, out_root, cli_epoch, cli_val_acc):
    print(f"\n{'='*62}")
    print(f"  Dataset : {name}")
    print(f"  Test    : {test_root}")
    print(f"  Ckpt    : {ckpt}")
    print(f"{'='*62}")

    if not ckpt.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt}"); return
    if not test_root.exists():
        print(f"[ERROR] Test dir not found: {test_root}"); return

    model, ckpt_epoch, ckpt_val_acc = load_model(ckpt, dataset_name=name)
    epoch   = cli_epoch   if cli_epoch   is not None else ckpt_epoch
    val_acc = cli_val_acc if cli_val_acc is not None else ckpt_val_acc
    print(f"  Device  : {DEVICE}  |  Epoch: {epoch}  |  Val acc: {val_acc}")

    for class_dir in sorted(test_root.iterdir()):
        if not class_dir.is_dir(): continue
        label = class_dir.name
        vids  = (sorted(class_dir.glob("*.mp4")) +
                 sorted(class_dir.glob("*.avi")) +
                 sorted(class_dir.glob("*.mov")))
        print(f"\n  [{label}]  {len(vids)} videos")
        for vid in vids:
            out_dir = out_root / name / label / vid.stem
            process_video(vid, model, out_dir, label, name,
                          ckpt, epoch, val_acc)


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────
def main():
    global VIOLENCE_THRESH, CLIP_STRIDE

    parser = argparse.ArgumentParser(description="GradCAM V2 — R3D-18 + LCM + LSTM")
    parser.add_argument("--dataset",  choices=["hockeyfight","rwf","both"], default="both")
    parser.add_argument("--thresh",   type=float, default=VIOLENCE_THRESH,
                        help="Violence confidence threshold (default 0.65)")
    parser.add_argument("--stride",   type=int,   default=CLIP_STRIDE,
                        help="Sliding window stride (default 4)")
    parser.add_argument("--epoch",    type=int,   default=None,
                        help="Model epoch to show in pred.txt (e.g. 12)")
    parser.add_argument("--val_acc",  type=float, default=None,
                        help="Model val accuracy to show in pred.txt (e.g. 0.98)")
    parser.add_argument("--output",   type=str,   default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    VIOLENCE_THRESH = args.thresh
    CLIP_STRIDE     = args.stride
    out_root        = Path(args.output)

    if args.dataset in ("hockeyfight", "both"):
        process_dataset("hockeyfight", HOCKEYFIGHT_TEST, HOCKEYFIGHT_CKPT,
                        out_root, args.epoch, args.val_acc)
    if args.dataset in ("rwf", "both"):
        process_dataset("rwf", RWF_TEST, RWF_CKPT,
                        out_root, args.epoch, args.val_acc)

    print(f"\n✅  All done!  Outputs → {out_root}")


if __name__ == "__main__":
    main()