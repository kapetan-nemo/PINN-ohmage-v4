"""Базовые модели прогноза SOH для сравнения с механистической моделью.

* A — эмпирическая: по первым K измеренным циклам подгоняется кривая
  ``SOH(k) = a + b·√(k−k0) + c·(k−k0)`` методом наименьших квадратов
  и экстраполируется;
* B — чисто нейросетевая (data-driven): MLP, сопоставляющий вектору
  признаков элемента и нормированному номеру цикла значение SOH;
  обучается по всем циклам элементов train-сплита;
* C — чисто механистическая: рекурсивный прогон с литературными
  константами OKane2022 без кодирующей сети (z = 0), начальное состояние
  — из идентификации этапа 1.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def fit_empirical(cycles: np.ndarray, soh: np.ndarray) -> np.ndarray:
    """Коэффициенты [a, b, c] эмпирической модели по первым циклам."""
    k0 = cycles[0]
    x = cycles - k0
    X = np.stack([np.ones_like(x), np.sqrt(np.clip(x, 0, None)), x], axis=1)
    coef, *_ = np.linalg.lstsq(X, soh, rcond=None)
    return coef


def predict_empirical(coef: np.ndarray, cycles: np.ndarray,
                      k0: float) -> np.ndarray:
    x = np.asarray(cycles, float) - k0
    return coef[0] + coef[1] * np.sqrt(np.clip(x, 0, None)) + coef[2] * x


class SOHNet(nn.Module):
    """Модель B: (признаки элемента, норм. цикл) → SOH."""

    def __init__(self, feature_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats: torch.Tensor, k_norm: torch.Tensor) -> torch.Tensor:
        if feats.ndim == 1:
            feats = feats.expand(len(k_norm), -1)
        x = torch.cat([feats, k_norm.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)


def train_sohnet(records: list[dict], feature_dim: int, epochs: int = 400,
                 lr: float = 3e-3, verbose: bool = False) -> SOHNet:
    """Обучение модели B.

    ``records`` — список ``{"feats", "cycles", "soh", "k_max"}``;
    цикл нормируется делением на максимальный цикл элемента.
    """
    net = SOHNet(feature_dim).double()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        tot = torch.zeros(())
        for r in records:
            kn = torch.tensor(r["cycles"] / r["k_max"], dtype=torch.float64)
            y = torch.tensor(r["soh"], dtype=torch.float64)
            tot = tot + ((net(r["feats"], kn) - y) ** 2).mean()
        (tot / len(records)).backward()
        opt.step()
        if verbose and ep % 50 == 0:
            print(f"ep {ep}: {float(tot):.5f}")
    return net


def predict_sohnet(net: SOHNet, feats: torch.Tensor, cycles: np.ndarray,
                   k_max: float) -> np.ndarray:
    with torch.no_grad():
        kn = torch.tensor(cycles / k_max, dtype=torch.float64)
        f = feats.expand(len(cycles), -1) if feats.ndim == 1 else feats
        return net(f, kn).numpy()
