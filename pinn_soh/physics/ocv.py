"""Параметрические кривые равновесного потенциала электродов.

Кривая равновесного потенциала каждого электрода задаётся монотонно
убывающей по степени литирования ``θ`` нейросетевой функцией:
``U(θ; c) = U0(c) − ∫₀^θ softplus(f(θ', c)) dθ'``, где ``f`` — малая
полносвязная сеть, ``c`` — латентный код материала (без явной метки
электрохимической системы). Интеграл вычисляется на фиксированной сетке
методом трапеций, значение в произвольной точке — линейной интерполяцией,
что сохраняет монотонность и дифференцируемость по ``θ``, ``c`` и весам.

Убывание по θ соответствует обоим электродам при принятом соглашении
(θ — доля литирования): у графита U падает от ~1,2 В при θ→0 до ~0,05 В
при θ→1; у катодных материалов (NMC, LFP, NCA, LCO) U растёт при
делитировании, т. е. тоже убывает по θ.

Этап 0 (предварительное обучение): для литературных полуэлементных кривых
(``data/params/ocv_curves.json``, выгрузка из PyBaMM) подбирается латентный
код ``c``, воспроизводящий кривую с погрешностью < 5 мВ; найденные коды
служат якорями латентного пространства.

``calibrate_windows`` — калибровка стехиометрических окон: выравнивание
положений пиков dQ/dV полного элемента с пиками |dU/dθ| кривой анода
(стадийные переходы графита ≈ 0,09/0,13/0,21 В относительно Li/Li⁺
обнаруживаются автоматически по производной идентифицированной кривой).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

# Число узлов сетки по θ для интегрирования и интерполяции.
GRID_POINTS = 257


class MonotoneOCV(nn.Module):
    """Монотонно убывающая по θ кривая равновесного потенциала, обусловленная
    латентным кодом ``c``.

    Параметры: сеть наклона ``f(θ, c)`` и смещение ``u_anchor(c)`` —
    значение потенциала в точке θ = ``theta_anchor``.
    """

    def __init__(self, latent_dim: int = 3, hidden: int = 48, n_layers: int = 2,
                 theta_anchor: float = 0.5):
        super().__init__()
        self.latent_dim = latent_dim
        self.theta_anchor = theta_anchor
        layers: list[nn.Module] = []
        d_in = 1 + latent_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d_in, hidden), nn.Tanh()]
            d_in = hidden
        layers.append(nn.Linear(d_in, 1))
        self.slope_net = nn.Sequential(*layers)
        self.anchor_net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.slope_net.double()
        self.anchor_net.double()
        grid = torch.linspace(0.0, 1.0, GRID_POINTS, dtype=torch.float64)
        self.register_buffer("theta_grid", grid)

    def _slope(self, theta: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Положительный наклон |dU/dθ| в узлах ``theta``."""
        if c.dim() == 1:
            c = c.unsqueeze(0)
        th = theta.reshape(-1, 1)
        cc = c.expand(th.shape[0], -1) if c.shape[0] == 1 else c
        raw = self.slope_net(torch.cat([th, cc], dim=-1)).squeeze(-1)
        return nn.functional.softplus(raw).to(torch.float64) + 1e-3

    def grid_values(self, c: torch.Tensor) -> torch.Tensor:
        """Значения U на сетке ``theta_grid`` для кода ``c`` (float64)."""
        c = c.to(torch.float64)
        slopes = self._slope(self.theta_grid, c)
        dth = self.theta_grid[1] - self.theta_grid[0]
        # интеграл трапеций от 0 до θ
        cs = torch.cumsum((slopes[:-1] + slopes[1:]) * 0.5 * dth, dim=0)
        integral = torch.cat([torch.zeros(1, dtype=torch.float64, device=cs.device), cs])
        u_anchor = self.anchor_net(c).reshape(-1)[0].to(torch.float64)
        idx = torch.argmin((self.theta_grid - self.theta_anchor).abs())
        return u_anchor + integral[idx] - integral  # убывающая: U(θ)=A−∫f

    def forward(self, theta: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """U(θ; c) для произвольных θ (линейная интерполяция по сетке)."""
        theta = theta.to(torch.float64)
        grid_u = self.grid_values(c)
        return torchinterp1(self.theta_grid, grid_u, theta)

    def dudt(self, theta: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """|dU/dθ| — наклон кривой (положительный; U убывает по θ)."""
        return self._slope(theta.to(torch.float64), c)


def torchinterp1(xp: torch.Tensor, fp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Линейная интерполяция fp(xp) в точках x (xp строго возрастает)."""
    x = x.clamp(xp[0], xp[-1])
    idx = torch.searchsorted(xp, x.contiguous()) - 1
    idx = idx.clamp(0, xp.numel() - 2)
    x0, x1 = xp[idx], xp[idx + 1]
    f0, f1 = fp[idx], fp[idx + 1]
    w = (x - x0) / (x1 - x0).clamp_min(1e-12)
    return f0 + w * (f1 - f0)


def load_literature_curves(path: str | Path) -> dict[str, dict]:
    """Читает выгруженные литературные кривые (формат export_pybamm_params.py)."""
    data = json.loads(Path(path).read_text())
    return {k: v for k, v in data.items() if "theta" in v and "u" in v}


def fit_latent_code(
    ocv: MonotoneOCV,
    theta: torch.Tensor,
    u_target: torch.Tensor,
    latent_dim: int | None = None,
    steps: int = 2000,
    lr: float = 2e-2,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[torch.Tensor, float]:
    """Подбирает латентный код ``c`` под литературную кривую U(θ).

    Оптимизируются только ``c`` и (при необходимости) аффинное смещение —
    веса сети заморожены. Возвращает ``(c, rmse_mV)``.
    """
    torch.manual_seed(seed)
    d = latent_dim or ocv.latent_dim
    c = torch.zeros(d, dtype=torch.float64, requires_grad=True)
    shift = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    scale = torch.ones(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([{"params": [c]}, {"params": [shift, scale], "lr": lr * 0.5}], lr=lr)
    th = theta.to(torch.float64)
    ut = u_target.to(torch.float64)
    for i in range(steps):
        opt.zero_grad()
        pred = ocv(th, c) * scale + shift
        loss = torch.mean((pred - ut) ** 2)
        loss.backward()
        opt.step()
        if verbose and (i % 500 == 0):
            print(f"  iter {i}: rmse {loss.item() ** 0.5 * 1000:.2f} мВ")
    with torch.no_grad():
        rmse = float(torch.mean((ocv(th, c) * scale + shift - ut) ** 2).sqrt())
    return c.detach(), rmse * 1000.0


@dataclass
class WindowCalibration:
    """Результат калибровки стехиометрических окон."""

    theta_n0: float  # доля литирования анода в начале цикла (конец разряда)
    theta_p0: float  # доля литирования катода в начале цикла
    peak_error_mv: float  # расхождение положений пиков dQ/dV, мВ
    details: dict = field(default_factory=dict)


def calibrate_windows(
    ocv_n: MonotoneOCV,
    ocv_p: MonotoneOCV,
    c: torch.Tensor,
    theta_n0_init: float,
    theta_p0_init: float,
    rho: float,
    peak_voltages_data: list[float],
    search_halfwidth: float = 0.12,
    steps: int = 41,
) -> WindowCalibration:
    """Уточняет ``θ_n0`` по положениям пиков dQ/dV.

    Пики полного элемента при заряде соответствуют стадийным переходам
    анода: максимумам |dU_n/dθ|. Для каждого обнаруженного пика анода
    ``θ_pk`` его положение на шкале напряжения полного элемента равно
    ``U_p(θ_p(θ_pk)) − U_n(θ_pk)``. Сдвиг ``θ_n0`` подбирается так, чтобы
    предсказанные положения пиков совпали с измеренными
    ``peak_voltages_data`` (минимизация среднеквадратичного расхождения по
    сетке сдвигов). Возвращает :class:`WindowCalibration`.
    """
    with torch.no_grad():
        # пики анодной кривой: локальные максимумы |dU_n/dθ|
        th = ocv_n.theta_grid
        slope = ocv_n.dudt(th, c)
        pk_idx = []
        for i in range(1, len(slope) - 1):
            if slope[i] >= slope[i - 1] and slope[i] >= slope[i + 1] and slope[i] > 1.5 * slope.median():
                pk_idx.append(i)
        theta_peaks = th[pk_idx]
        u_peaks = ocv_n(theta_peaks, c)

        best = (None, torch.inf)
        shifts = torch.linspace(-search_halfwidth, search_halfwidth, steps, dtype=torch.float64)
        data = torch.tensor(sorted(peak_voltages_data), dtype=torch.float64)
        for s in shifts:
            tn0 = (theta_n0_init + s).item()
            tp0 = theta_p0_init - rho * s
            # напряжение пика на шкале полного элемента:
            # V_pk = U_p(θ_p(pk)) − U_n(pk), θ_p из баланса по току:
            # в точке пика θ_p = θ_p0 − ρ(θ_pk − θ_n0) при движении от θ_n0
            tp_at_pk = tp0 - rho * (theta_peaks - tn0)
            v_model = ocv_p(tp_at_pk, c) - ocv_n(theta_peaks, c)
            v_model, _ = torch.sort(v_model)
            k = min(len(v_model), len(data))
            if k == 0:
                continue
            err = torch.mean((v_model[:k] - data[:k]) ** 2)
            if err < best[1]:
                best = (s, err)
    s = best[0] if best[0] is not None else torch.zeros((), dtype=torch.float64)
    err_mv = float(best[1].sqrt() * 1000) if best[1] != torch.inf else float("nan")
    return WindowCalibration(
        theta_n0=float(theta_n0_init + s),
        theta_p0=float(theta_p0_init - rho * s),
        peak_error_mv=err_mv,
        details={"n_peaks_model": len(theta_peaks), "theta_peaks": theta_peaks.tolist(),
                 "u_peaks_v": u_peaks.tolist()},
    )
