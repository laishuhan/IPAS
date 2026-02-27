# encoder.py
# -*- coding: utf-8 -*-
"""
encoder.py
----------
职责：
1) ReportEncoder：将 x32 -> embedding z（推理时用）
2) SSL 预训练：对比学习（InfoNCE），增强模拟 OCR 噪声（mask + jitter）
3) 提供 load/save/encode 的稳定接口，供 train.py / main_eval.py 调用

说明：
- 输入特征默认来自 preprocess.preprocess_raw_sample，shape=(1,32)
- encoder 输出 embedding（默认 latent_dim=128）
- 对比学习训练时使用 projector 输出（默认 output_dim=64）
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 本项目内依赖
from preprocess import preprocess_raw_sample, read_jsonl


# =========================================================
# 1) Model: Encoder + Projector
# =========================================================

class ReportEncoder(nn.Module):
    """
    Encoder 用于推理输出 embedding（h）
    Projector 仅用于对比学习训练输出 z（更适配 InfoNCE）
    """
    def __init__(self, input_dim: int = 32, latent_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(),
        )

        self.projector = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor, return_embedding: bool = True) -> torch.Tensor:
        """
        return_embedding=True: 返回 encoder embedding (h)
        return_embedding=False: 返回 projector 输出 (z) 用于对比学习
        """
        h = self.encoder(x)
        if return_embedding:
            return h
        return self.projector(h)


# =========================================================
# 2) Augmentation: mask + jitter
# =========================================================

@dataclass
class SSLAugmentConfig:
    mask_prob: float = 0.15       # 随机掩码比例
    jitter_std: float = 0.02      # 数值抖动标准差
    jitter_positive_only: bool = True  # 仅对 >0 的值加抖动（更贴近数值特征）


def augment_features(x: torch.Tensor, cfg: SSLAugmentConfig) -> torch.Tensor:
    """
    x: (d,) or (B,d)
    返回：增强后的 x_aug（同 shape）
    """
    x_aug = x.clone()

    # mask
    mask = (torch.rand_like(x_aug) < float(cfg.mask_prob))
    x_aug[mask] = -1.0

    # jitter (small noise)
    if float(cfg.jitter_std) > 0:
        noise = torch.randn_like(x_aug) * float(cfg.jitter_std)
        if cfg.jitter_positive_only:
            x_aug = x_aug + noise * (x_aug > 0).float()
        else:
            x_aug = x_aug + noise
    return x_aug


# =========================================================
# 3) Contrastive loss: InfoNCE
# =========================================================

def info_nce_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    SimCLR-style InfoNCE with cosine similarity.
    z_i, z_j: (B, d)
    """
    assert z_i.ndim == 2 and z_j.ndim == 2, "z_i/z_j must be (B,d)"
    B = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)  # (2B, d)

    # cosine similarity matrix
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # (2B,2B)

    # positive pairs positions
    sim_ij = torch.diag(sim, B)
    sim_ji = torch.diag(sim, -B)
    positives = torch.cat([sim_ij, sim_ji], dim=0)  # (2B,)

    # mask out self-similarity
    eye = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    negatives = sim[~eye].view(2 * B, -1)  # (2B, 2B-1)

    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1) / float(temperature)
    labels = torch.zeros(2 * B, device=z.device, dtype=torch.long)
    return F.cross_entropy(logits, labels)


# =========================================================
# 4) SSL Dataset: jsonl -> x32, then augmentation
# =========================================================

class MedicalSSLDataset(Dataset):
    """
    读取 jsonl，每行一个样本 dict
    预处理为 x32（d=32）
    __getitem__ 返回两份增强视图 (x_aug1, x_aug2): (d,)
    """
    def __init__(
        self,
        jsonl_path: str,
        max_indicators: int = 15,
        max_values: int = 15,
        augment_cfg: Optional[SSLAugmentConfig] = None,
    ):
        self.jsonl_path = jsonl_path
        self.max_indicators = int(max_indicators)
        self.max_values = int(max_values)
        self.augment_cfg = augment_cfg or SSLAugmentConfig()

        data = read_jsonl(jsonl_path)
        if not data:
            raise ValueError(f"Empty dataset or cannot read jsonl: {jsonl_path}")

        xs: List[torch.Tensor] = []
        for obj in data:
            x = preprocess_raw_sample(obj, max_indicators=self.max_indicators, max_values=self.max_values)  # (1,32)
            xs.append(x.squeeze(0))  # (32,)
        self.data_tensor = torch.stack(xs, dim=0)  # (N,32)

    def __len__(self) -> int:
        return int(self.data_tensor.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data_tensor[idx]  # (32,)
        x1 = augment_features(x, self.augment_cfg)
        x2 = augment_features(x, self.augment_cfg)
        return x1, x2


# =========================================================
# 5) Train SSL Encoder
# =========================================================

@dataclass
class SSLTrainConfig:
    epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    temperature: float = 0.1
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    seed: int = 42
    num_workers: int = 0
    latent_dim: int = 128
    output_dim: int = 64
    input_dim: int = 32


def _pick_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_ssl_encoder(
    jsonl_path: str,
    out_encoder_path: str,
    out_meta_path: Optional[str] = None,
    augment_cfg: Optional[SSLAugmentConfig] = None,
    train_cfg: Optional[SSLTrainConfig] = None,
    max_indicators: int = 15,
    max_values: int = 15,
) -> Dict[str, Any]:
    """
    训练对比学习 encoder，并保存权重。
    返回训练 meta（loss 曲线等）。
    """
    cfg = train_cfg or SSLTrainConfig()
    aug = augment_cfg or SSLAugmentConfig()
    set_seed(cfg.seed)

    device = _pick_device(cfg.device)
    ds = MedicalSSLDataset(
        jsonl_path=jsonl_path,
        max_indicators=max_indicators,
        max_values=max_values,
        augment_cfg=aug,
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        drop_last=True if len(ds) >= int(cfg.batch_size) else False,
    )

    model = ReportEncoder(
        input_dim=int(cfg.input_dim),
        latent_dim=int(cfg.latent_dim),
        output_dim=int(cfg.output_dim),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))

    losses: List[float] = []
    for epoch in range(int(cfg.epochs)):
        model.train()
        total = 0.0
        steps = 0

        for x_i, x_j in loader:
            x_i = x_i.to(device)
            x_j = x_j.to(device)

            opt.zero_grad()
            z_i = model(x_i, return_embedding=False)
            z_j = model(x_j, return_embedding=False)

            loss = info_nce_loss(z_i, z_j, temperature=float(cfg.temperature))
            loss.backward()
            opt.step()

            total += float(loss.item())
            steps += 1

        avg = total / max(steps, 1)
        losses.append(avg)

        # 简单日志（每5轮）
        if epoch % 5 == 0:
            print(f"[SSL] Epoch {epoch:03d} | loss={avg:.4f} | device={device.type}")

    # save
    os.makedirs(os.path.dirname(out_encoder_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), out_encoder_path)

    meta = {
        "jsonl_path": jsonl_path,
        "epochs": int(cfg.epochs),
        "batch_size": int(cfg.batch_size),
        "lr": float(cfg.lr),
        "temperature": float(cfg.temperature),
        "seed": int(cfg.seed),
        "device": str(device),
        "input_dim": int(cfg.input_dim),
        "latent_dim": int(cfg.latent_dim),
        "output_dim": int(cfg.output_dim),
        "augment": {
            "mask_prob": float(aug.mask_prob),
            "jitter_std": float(aug.jitter_std),
            "jitter_positive_only": bool(aug.jitter_positive_only),
        },
        "loss_curve": losses,
        "n_samples": int(len(ds)),
    }

    if out_meta_path is not None:
        os.makedirs(os.path.dirname(out_meta_path) or ".", exist_ok=True)
        with open(out_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


# =========================================================
# 6) Load/Encode (inference helpers)
# =========================================================

def load_encoder(encoder_path: str, input_dim: int = 32, latent_dim: int = 128, output_dim: int = 64) -> ReportEncoder:
    """
    加载 encoder 权重（推理用：只需要 encoder 部分输出 embedding）。
    """
    model = ReportEncoder(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim)
    sd = torch.load(encoder_path, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model


@torch.no_grad()
def encode(encoder: ReportEncoder, x32: torch.Tensor) -> np.ndarray:
    """
    x32: torch.Tensor shape=(1,32) or (B,32)
    返回 embedding: np.ndarray shape=(B, latent_dim)
    """
    if x32.ndim == 1:
        x32 = x32.unsqueeze(0)
    h = encoder(x32, return_embedding=True)  # (B, latent_dim)
    return h.detach().cpu().numpy()
