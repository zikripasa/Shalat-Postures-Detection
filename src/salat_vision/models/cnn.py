"""CNN transfer learning with torchvision backbones (ImageNet weights).

Two uses:

* **Frozen embeddings** ("linear probe"): pooled backbone features + a regularised
  linear classifier. The right tool for tens of training images.
* **Fine-tuning**: backbone + new 4-way head trained end-to-end with augmentation,
  a group-aware validation split, early stopping and checkpointing.

Preprocessing is identical everywhere (training, evaluation, ONNX export, API):
person crop (box + padding) -> aspect-preserving letterbox to ``size`` -> ImageNet
normalisation. Letterboxing keeps the body's aspect ratio, a strong posture cue.
"""

from __future__ import annotations

import copy
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LETTERBOX_FILL = (124, 116, 104)  # ImageNet mean colour (uint8)

BACKBONES = {
    "mobilenet_v3_large": {"weights": "MobileNet_V3_Large_Weights.IMAGENET1K_V2", "dim": 960},
    "resnet18": {"weights": "ResNet18_Weights.IMAGENET1K_V1", "dim": 512},
}


def letterbox(rgb: np.ndarray, size: int, fill=LETTERBOX_FILL) -> np.ndarray:
    """Resize keeping aspect ratio and pad to ``size x size``."""
    h, w = rgb.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (nw, nh), interpolation=interp)
    out = np.empty((size, size, 3), dtype=np.uint8)
    out[:] = fill
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def to_tensor_array(rgb_sq: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> float32 3xHxW, ImageNet-normalised."""
    x = rgb_sq.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def preprocess_batch(crops: list[np.ndarray], size: int, flip: bool = False) -> np.ndarray:
    arrs = []
    for c in crops:
        sq = letterbox(c, size)
        if flip:
            sq = sq[:, ::-1]
        arrs.append(to_tensor_array(sq))
    return np.stack(arrs).astype(np.float32)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _torch_home(cfg) -> None:
    if cfg is not None and "TORCH_HOME" not in os.environ:
        os.environ["TORCH_HOME"] = str(cfg.path("pretrained_dir") / "torch")


def build_backbone(name: str, *, pretrained: bool = True, cfg=None):
    """Return ``(module mapping Bx3xHxW -> BxD, D)``."""
    import torch.nn as nn
    import torchvision.models as tvm

    _torch_home(cfg)
    if name == "mobilenet_v3_large":
        m = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        body = nn.Sequential(m.features, m.avgpool, nn.Flatten(1))
        return body, 960
    if name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        body = nn.Sequential(*list(m.children())[:-1], nn.Flatten(1))
        return body, 512
    raise ValueError(f"unknown backbone {name}")


def extract_embeddings(backbone, crops: list[np.ndarray], *, size: int = 224, flip: bool = False, batch_size: int = 16) -> np.ndarray:
    import torch

    backbone.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(crops), batch_size):
            x = torch.from_numpy(preprocess_batch(crops[i:i + batch_size], size, flip=flip))
            out.append(backbone(x).numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


# ------------------------------------------------------------------------------ fine-tuning
@dataclass
class FinetuneResult:
    model: object
    history: list[dict] = field(default_factory=list)
    best_epoch: int = -1
    train_seconds: float = 0.0


def _make_classifier(backbone_name: str, n_classes: int, cfg=None, dropout: float = 0.2):
    import torch.nn as nn

    body, dim = build_backbone(backbone_name, pretrained=True, cfg=cfg)

    class PostureCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = body
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_classes))

        def forward(self, x):
            return self.head(self.backbone(x))

    return PostureCNN()


def _freeze_bn(module) -> None:
    import torch.nn as nn

    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


def _augment(rgb_sq: np.ndarray, rng: np.random.Generator, size: int) -> np.ndarray:
    """Light, label-preserving augmentation on a letterboxed square image."""
    img = rgb_sq
    if rng.random() < 0.5:
        img = img[:, ::-1]
    # random scale/translate (zoom 0.85-1.0 crop, then resize back)
    s = rng.uniform(0.85, 1.0)
    cs = int(size * s)
    x0 = rng.integers(0, size - cs + 1)
    y0 = rng.integers(0, size - cs + 1)
    img = cv2.resize(np.ascontiguousarray(img[y0:y0 + cs, x0:x0 + cs]), (size, size), interpolation=cv2.INTER_LINEAR)
    # small rotation
    ang = rng.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((size / 2, size / 2), ang, 1.0)
    img = cv2.warpAffine(img, M, (size, size), borderValue=LETTERBOX_FILL)
    # colour jitter (brightness/contrast/saturation)
    f = img.astype(np.float32)
    f = f * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
    gray = f.mean(axis=2, keepdims=True)
    f = gray + (f - gray) * rng.uniform(0.7, 1.3)
    return np.clip(f, 0, 255).astype(np.uint8)


def finetune(backbone_name: str, train_crops: list[np.ndarray], y_train: np.ndarray, val_crops: list[np.ndarray],
             y_val: np.ndarray, *, n_classes: int, cfg, seed: int, checkpoint_path: str | Path | None = None,
             log_every: int = 0) -> FinetuneResult:
    """Fine-tune a backbone + head with early stopping on validation loss."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score

    c = cfg["cnn"]
    size, epochs, bs, patience = int(c["image_size"]), int(c["epochs"]), int(c["batch_size"]), int(c["patience"])
    seed_everything(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    model = _make_classifier(backbone_name, n_classes, cfg)
    _freeze_bn(model)
    head_params = list(model.head.parameters())
    body_params = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": body_params, "lr": float(c["lr_backbone"])},
        {"params": head_params, "lr": float(c["lr_head"])},
    ], weight_decay=float(c["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1) / n_classes
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights), label_smoothing=float(c["label_smoothing"]))
    eval_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights))

    train_sq = [letterbox(x, size) for x in train_crops]
    x_val = torch.from_numpy(preprocess_batch(val_crops, size)) if len(val_crops) else None
    yv = torch.from_numpy(np.asarray(y_val, dtype=np.int64)) if len(val_crops) else None
    rng = np.random.default_rng(seed)

    best = (float("inf"), -1, None)
    history: list[dict] = []
    t_start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        _freeze_bn(model)
        order = rng.permutation(len(train_sq))
        tot, correct, n = 0.0, 0, 0
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            xb = torch.from_numpy(np.stack([to_tensor_array(_augment(train_sq[j], rng, size)) for j in idx]))
            yb = torch.from_numpy(np.asarray(y_train[idx], dtype=np.int64))
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            correct += int((logits.argmax(1) == yb).sum())
            n += len(idx)
        sched.step()
        rec = {"epoch": epoch + 1, "train_loss": tot / max(n, 1), "train_acc": correct / max(n, 1)}
        if x_val is not None:
            model.eval()
            with torch.inference_mode():
                lv = model(x_val)
                rec["val_loss"] = float(eval_loss_fn(lv, yv))
                pv = lv.argmax(1).numpy()
                rec["val_acc"] = float((pv == yv.numpy()).mean())
                rec["val_macro_f1"] = float(f1_score(yv.numpy(), pv, average="macro", labels=list(range(n_classes)), zero_division=0))
            if rec["val_loss"] < best[0] - 1e-4:
                best = (rec["val_loss"], epoch + 1, copy.deepcopy(model.state_dict()))
        history.append(rec)
        if log_every and (epoch + 1) % log_every == 0:
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()})
        if x_val is not None and best[1] > 0 and epoch + 1 - best[1] >= patience:
            break
    if best[2] is not None:
        model.load_state_dict(best[2])
    model.eval()
    if checkpoint_path:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "backbone": backbone_name, "n_classes": n_classes,
                    "best_epoch": best[1], "history": history}, checkpoint_path)
    return FinetuneResult(model, history, best[1] if best[1] > 0 else len(history), time.perf_counter() - t_start)


def predict_proba_torch(model, crops: list[np.ndarray], *, size: int = 224, batch_size: int = 16) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(crops), batch_size):
            x = torch.from_numpy(preprocess_batch(crops[i:i + batch_size], size))
            out.append(torch.softmax(model(x), dim=1).numpy())
    return np.concatenate(out, axis=0)
