"""Переходная модель медленного состояния: Δs за цикл как функция s и
признаков элемента.

Механический канал (SEI + осаждение) уже даёт приращение Δq_li, но
при избытке запаса лития не объясняет плавный распад ёмкости —
деградация активной массы (LAM) в механике не представлена. Сеть
доучивает недостающие приращения по извлечённым траекториям ψ(k):

    выходы (на цикл, неотрицательные или знакопостоянные):
      dq_resid  — дополнительная потеря лития, доля q_li0;
      dlam_n    — убыль относит. ёмкости анода;
      dlam_p    — убыль относит. ёмкости катода;
      dlog_r    — прирост log10(R/R0) (R растёт на ~4 декады за жизнь,
                  поэтому состояние — логарифм).

Все выходы — «мagnitude» через softplus с масштабом; λ и q_li могут
только убывать, R только расти — шум идентификации фильтруется знаком.

Вход: динамическая часть (нормированное состояние + механическое
приращение + индекс цикла) и статические признаки ``build_features``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pinn_soh.models.encoder import FEATURE_DIM

DYN_NAMES = (
    "q_rel",       # q_li / q_li0
    "lam_n",       # текущая отн. ёмкость анода
    "lam_p",       # текущая отн. ёмкость катода
    "r_rel",       # log10(r / r0)
    "dq_mech_rel", # механическое приращение Δq_li за цикл / q_li0
    "log_k",       # log10(цикл + 1) / 4
    "gap",         # разрыв до следующей точки / 20
    "dq_prev",     # приращение q_li на прошлом шаге (инерция режима)
)
DYN_DIM = len(DYN_NAMES)

# масштабы выходов (за цикл): безразм. множитель механики, доля q_li0,
# доли λ, доля r0
OUT_SCALE = torch.tensor([1.0, 3e-4, 3e-3, 3e-3, 2e-3], dtype=torch.float64)
OUT_NAMES = ("g_mech", "dq_resid_rel", "dlam_n", "dlam_p", "dr_rel")


class TransitionNet(nn.Module):
    """MLP: (динамика ‖ признаки) → приращения за цикл (grey-box).

    Выходы:
      ``g_mech`` ∈ (0,3) — множитель механического Δq_li (подавление
        завышенной механики на здоровых элементах);
      остальные — неотрицательные приращения через softplus·scale.
    """

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DYN_DIM + feature_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 5),
        )
        # нулевая инициализация выходного слоя: приращения стартуют
        # малыми (softplus(0)·scale), развёртка не уходит в насыщение
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # структурное замедление: Δs_eff = Δs · (k+1)^β, β обучаемые
        # (β<0 — ранняя деградация быстрее стационарной, как в данных)
        self.beta = nn.Parameter(torch.zeros(4, dtype=torch.float64))
        self.register_buffer("scale", OUT_SCALE.clone())

    def forward(self, dyn: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """``dyn`` (..., DYN_DIM), ``feats`` (..., F) → (..., 5)."""
        x = torch.cat([dyn, feats.expand(*dyn.shape[:-1], -1)], dim=-1)
        o = self.net(x.double())
        g = 3.0 * torch.sigmoid(o[..., 0])
        k1 = 10.0 ** (dyn[..., 5:6] * 4.0)      # восстановление k+1
        decay = k1 ** self.beta.clamp(-1.5, 0.5)
        mag = self.scale[1:] * F.softplus(o[..., 1:]) * decay
        return torch.cat([g.unsqueeze(-1), mag], dim=-1)
