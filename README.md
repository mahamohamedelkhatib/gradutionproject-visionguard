# VisionGuard 🛡️ — Explainable Real-Time Violence Detection

**Graduation Project · BSc Computer Science (AI), University of Hertfordshire · 2026**

VisionGuard is a real-time violence/fight detection system that reaches **96–98% accuracy** on the RWF-2000 and Hockey Fight benchmark datasets, while showing *why* it flagged a clip through visual explainability — deployed live as a Streamlit app on Hugging Face Spaces.

## How it works

- **Backbone:** R3D-18 (3D ResNet-18) for spatio-temporal video features
- **LCM (Local Context Module):** a custom depthwise 3D-conv + gating block that sharpens *where* motion is happening in the frame
- **BiLSTM head:** reasons over the sequence of frames to judge *when/how* an action unfolds into a fight
- **Combined CAM Fusion:** GradCAM, GradCAM++, SmoothGradCAM++ and LayerCAM overlays fused together, so you can see which regions/frames drove the prediction
- **ByteTrack:** multi-object tracking used to assign aggressor/victim roles across frames

## Training

Two training scripts (`lstmrwf.py`, `lstm+lcmhockeyfight.py`) cover the two benchmark datasets:
- **RWF-2000** — Real World Fights dataset
- **Hockey Fight** dataset

Each logs metrics to CSV/TensorBoard and generates loss/accuracy curves plus an architecture diagram. `gradcamv21.py` implements the Combined CAM Fusion explainability layer used at inference.

## App

`visionguard.py` is the Streamlit dashboard: upload or stream raw video, get a live Fight/Non-Fight verdict with confidence, explainability overlays, a fight-face detector, and per-user login/history. A packaged, deploy-ready copy of this dashboard lives in [VIOLENCE-DETCETION-DASHBOARD3](https://github.com/mahamohamedelkhatib/VIOLENCE-DETCETION-DASHBOARD3).

## Tech stack

Python · PyTorch · torchvision · OpenCV · Streamlit · pandas · scikit-learn · TensorBoard

## Files

| File | Purpose |
|---|---|
| `visionguard.py` | Streamlit app (VisionGuard v9) |
| `lstmrwf.py` | Train R3D-18 + LCM + LSTM on RWF-2000 |
| `lstm+lcmhockeyfight.py` | Train R3D-18 + LCM + LSTM on Hockey Fight dataset |
| `gradcamv21.py` | Combined CAM Fusion explainability |
