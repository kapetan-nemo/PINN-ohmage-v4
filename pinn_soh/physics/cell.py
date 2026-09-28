"""Редуцированная модель одиночной частицы — быстрый масштаб.

Соглашения и обозначения (соответствуют плану, раздел 2.1):

* ток ``i`` приводится к направлению «заряд положителен» вызывающей
  стороной (``i_chg = sign · i_a``);
* ``θ_n``, ``θ_p`` — доли литирования отрицательного и положительного
  электродов; кривые ``U_n(θ; c)``, ``U_p(θ; c)`` монотонно убывают по θ
  (см. ``ocv.py``);
* при заряде θ_n растёт, θ_p убывает; баланс лития между электродами
  задаётся соотношением ёмкостей ``ρ = Q_n/Q_p``:

    ``θ_p = θ_p0 − ρ·(θ_n − θ_n0)``;

* перенапряжения обоих электродов складываются в направлении тока:
    ``η_k = (2RT/F)·arcsinh( i / (2·A·j0_k(θ_k)) )``,
    ``φ_n = U_n − η_n`` — потенциал анода относительно Li/Li⁺;
* омическое падение и пассивирующий слой объединены в ``r_total``;
* диффузионная поляризация — 1–2 экспоненциальные моды с точным
  дискретным обновлением;
* напряжение элемента:
    ``V = U_p(θ_p) − U_n(θ_n) + η_p + η_n + i·R_total + Σ V_diff,m``.

Все вычисления — torch, float64, центральный процессор; обратное
распространение по времени поддерживается (переходы — дифференцируемые
операции над тензорами).
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from pinn_soh.physics.ocv import MonotoneOCV

F_CONST = 96485.33212  # постоянная Фарадея, Кл/моль
R_CONST = 8.314462618  # газовая постоянная, Дж/(моль·К)
T_REF = 298.15         # опорная температура, К


@dataclass
class CellGeometry:
    """Конструктивные параметры элемента, постоянные на всём сроке службы."""

    q_n_ah: float          # ёмкость анода при полном окне θ_n∈[0,1], А·ч
    rho: float             # Q_n/Q_p
    area_m2: float         # активная площадь электрода (для плотностей тока), м²
    j0_n_a_m2: float       # эталонная плотность обменного тока анода, А/м²
    j0_p_a_m2: float       # эталонная плотность обменного тока катода, А/м²
    c: torch.Tensor = None  # латентный код материалов (общий для пары OCV)

    @property
    def q_p_ah(self) -> float:
        return self.q_n_ah / self.rho


@dataclass
class CycleInit:
    """Идентифицируемые параметры одного цикла."""

    theta_n0: float                 # θ_n в начале цикла
    theta_p0: float | None = None   # θ_p в начале цикла (None → из баланса n_li)
    r_total_ohm: float = 0.05       # R_Ω + R_SEI, Ом
    tau_d_s: tuple = (60.0, 600.0)  # постоянные времени диффузионных мод, с
    r_d_ohm: tuple = (0.005, 0.005) # амплитуды поляризации мод, Ом
    theta_n0_from_inventory: float | None = None


def exchange_current(j0_ref_a_m2: float, theta: torch.Tensor,
                     c_e_frac: float = 1.0) -> torch.Tensor:
    """Локальная плотность обменного тока ``j0·√(θ(1−θ))·√c_e``."""
    return j0_ref_a_m2 * torch.sqrt((theta.clamp(1e-4, 1 - 1e-4)
                                     * (1 - theta.clamp(1e-4, 1 - 1e-4)))
                                    ) * c_e_frac ** 0.5


def bv_overpotential(i_a: torch.Tensor, area_m2: float, j0_a_m2: torch.Tensor,
                     temp_k: float = T_REF) -> torch.Tensor:
    """Перенапряжение Батлера–Фольмера электрода (знаковое, в направлении i).

    ``η = (2RT/F)·arcsinh( i / (2·A·j0) )`` — при α_a = α_c = 0,5.
    """
    j_lim = (2.0 * area_m2 * j0_a_m2).clamp_min(1e-12)
    return (2.0 * R_CONST * temp_k / F_CONST) * torch.asinh(i_a / j_lim)


def simulate_cycle(
    t_s: torch.Tensor,
    i_a: torch.Tensor,
    ocv_n: MonotoneOCV,
    ocv_p: MonotoneOCV,
    geom: CellGeometry,
    init: CycleInit,
    temp_k: float = T_REF,
    residual=None,
) -> dict[str, torch.Tensor]:
    """Пошаговое моделирование одного цикла по измеренному профилю тока.

    Параметры
    ----------
    t_s, i_a : тензоры одинаковой длины; время в секундах (возрастающее),
        ток в амперах, положительный при заряде.
    init.theta_p0 : начальная стехиометрия катода; если ``None`` — не
        используется (для режима идентификации указывается явно).

    Возвращает словарь тензоров: ``v_hat, phi_n, theta_n, theta_p, eta_n,
    eta_p, v_diff`` и вектор дискретных времён ``dt``.
    """
    n = t_s.numel()
    dt = torch.diff(t_s, prepend=t_s[:1]).clamp_min(0.0)
    c = geom.c.to(torch.float64)
    x = torch.tensor(float(init.theta_n0), dtype=torch.float64)
    tp0 = torch.tensor(float(init.theta_p0), dtype=torch.float64)
    modes = len(init.tau_d_s)
    vd = torch.zeros(modes, dtype=torch.float64)

    v_hat = torch.empty(n, dtype=torch.float64)
    phi_n = torch.empty(n, dtype=torch.float64)
    th_n = torch.empty(n, dtype=torch.float64)
    th_p = torch.empty(n, dtype=torch.float64)
    eta_n = torch.empty(n, dtype=torch.float64)
    eta_p = torch.empty(n, dtype=torch.float64)
    v_diff = torch.empty(n, dtype=torch.float64)

    for k in range(n):
        i_k = i_a[k]
        tp = tp0 - geom.rho * (x - init.theta_n0)
        j0n = exchange_current(geom.j0_n_a_m2, x)
        j0p = exchange_current(geom.j0_p_a_m2, tp)
        en = bv_overpotential(i_k, geom.area_m2, j0n, temp_k)
        ep = bv_overpotential(i_k, geom.area_m2, j0p, temp_k)
        vds = vd.sum()
        v = (ocv_p(tp.unsqueeze(0), c) - ocv_n(x.unsqueeze(0), c)
             + ep + en + i_k * init.r_total_ohm + vds)
        if residual is not None:
            v = v + residual(x, i_k)
        v_hat[k] = v
        phi_n[k] = ocv_n(x.unsqueeze(0), c) - en
        th_n[k] = x
        th_p[k] = tp
        eta_n[k] = en
        eta_p[k] = ep
        v_diff[k] = vds
        # обновление состояния на интервале [t_k, t_{k+1}] током I_k
        if k + 1 < n:
            step = t_s[k + 1] - t_s[k]
            x = x + i_k * step / (3600.0 * geom.q_n_ah)
            al = torch.exp(torch.tensor([-step / tau for tau in init.tau_d_s],
                                        dtype=torch.float64))
            rd = torch.tensor(list(init.r_d_ohm), dtype=torch.float64)
            vd = vd * al + rd * (1 - al) * i_k
    return {
        "v_hat": v_hat, "phi_n": phi_n, "theta_n": th_n, "theta_p": th_p,
        "eta_n": eta_n, "eta_p": eta_p, "v_diff": v_diff, "dt": dt,
    }


def simulate_cycle_vectorized(
    t_s: torch.Tensor,
    i_a: torch.Tensor,
    ocv_n: MonotoneOCV,
    ocv_p: MonotoneOCV,
    geom: CellGeometry,
    init: CycleInit,
    temp_k: float = T_REF,
    residual=None,
) -> dict[str, torch.Tensor]:
    """Векторизованный вариант: то же обновление состояния без петли по
    времени для необратимых операций; последовательность θ_n вычисляется
    кумулятивным интегралом тока. Расчёт идентичен пошаговому с точностью
    до порядка вычислений; используется в обучении для ускорения.

    Примечание: V_diff — линейная фильтрация I; здесь реализована через
    кумулятивную свёртку экспоненциального ядра с переменным шагом, что
    эквивалентно точному обновлению при усреднении I на шаге.
    """
    n = t_s.numel()
    dt = torch.diff(t_s, prepend=t_s[:1]).clamp_min(0.0)
    c = geom.c.to(torch.float64)

    # θ_n по закону Фарадея: θ_n[k] = θ_n0 + Σ_{j<k} I_j·(t_{j+1}−t_j)/(3600·Q_n)
    fwd = torch.diff(t_s)  # длина n−1: интервалы [t_j, t_{j+1}]
    dq = torch.cat(
        [torch.zeros(1, dtype=torch.float64, device=t_s.device),
         torch.cumsum(i_a[:-1] * fwd, dim=0)]
    ) / (3600.0 * geom.q_n_ah)
    theta_n = init.theta_n0 + dq
    theta_p = init.theta_p0 - geom.rho * (theta_n - init.theta_n0)

    j0n = exchange_current(geom.j0_n_a_m2, theta_n)
    j0p = exchange_current(geom.j0_p_a_m2, theta_p)
    eta_n = bv_overpotential(i_a, geom.area_m2, j0n, temp_k)
    eta_p = bv_overpotential(i_a, geom.area_m2, j0p, temp_k)

    # V_diff: те же точные переходы RC-мод, что и в пошаговой версии;
    # петля только по модам и точкам, без вызова сети.
    v_diff = torch.zeros(n, dtype=torch.float64)
    for tau, r_d in zip(init.tau_d_s, init.r_d_ohm):
        vd = torch.zeros((), dtype=torch.float64)
        acc = torch.empty(n, dtype=torch.float64)
        for k in range(n):
            if k > 0:
                step = t_s[k] - t_s[k - 1]
                al = torch.exp(-step / tau)
                vd = vd * al + r_d * (1 - al) * i_a[k - 1]
            acc[k] = vd
        v_diff = v_diff + acc

    v_hat = (ocv_p(theta_p, c) - ocv_n(theta_n, c)
             + eta_p + eta_n + i_a * init.r_total_ohm + v_diff)
    if residual is not None:
        v_hat = v_hat + residual(theta_n, i_a)
    phi_n = ocv_n(theta_n, c) - eta_n
    return {
        "v_hat": v_hat, "phi_n": phi_n, "theta_n": theta_n, "theta_p": theta_p,
        "eta_n": eta_n, "eta_p": eta_p, "v_diff": v_diff, "dt": dt,
    }

