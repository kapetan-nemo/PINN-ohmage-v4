"""Замкнутое моделирование протокола циклирования.

Для будущих циклов ток задаётся не измерением, а регулятором протокола:
постоянный ток до отсечки по напряжению, постоянное напряжение до отсечки
по току, паузы заданной длительности. Реализована вложенность
``workflow`` (повторяющиеся группы шагов) из метаданных Aurora.

Класс :class:`CellSimulator` — обёртка состояния редуцированной модели
одиночной частицы с атомарными операциями ``voltage(i)`` (оценка
напряжения без продвижения) и ``advance(i, dt)`` (обновление θ_n и
диффузионных мод). Используется и регулятором протокола, и, при
необходимости, идентификацией.

Режимы шагов (поле ``mode`` :class:`~pinn_soh.data.metadata.ProtocolStep`):
``cc_charge, cv_charge, cc_discharge, rest, workflow``.
"""

import math
from dataclasses import dataclass, field

import torch

from pinn_soh.data.metadata import ProtocolStep
from pinn_soh.physics.cell import (
    F_CONST,
    R_CONST,
    T_REF,
    bv_overpotential,
    exchange_current,
)


class CellSimulator:
    """Состояние быстрой модели и атомарные операции над ним (float64)."""

    def __init__(self, ocv_n, ocv_p, geom, r_total_ohm: float,
                 tau_d_s=(60.0, 600.0), r_d_ohm=(0.005, 0.005),
                 temp_k: float = T_REF):
        self.ocv_n = ocv_n
        self.ocv_p = ocv_p
        self.geom = geom
        self.r_total = r_total_ohm
        self.tau = torch.tensor(list(tau_d_s), dtype=torch.float64)
        self.r_d = torch.tensor(list(r_d_ohm), dtype=torch.float64)
        self.temp_k = temp_k
        # быстрое состояние
        self.x = 0.0            # θ_n
        self.tp0 = 0.0          # θ_p в начале цикла
        self.x0 = 0.0           # θ_n в начале цикла
        self.vd = torch.zeros(len(self.tau), dtype=torch.float64)

    def begin_cycle(self, theta_n0: float, theta_p0: float) -> None:
        """Инициализация быстрого состояния в начале цикла (после паузы
        диффузионные моды сброшены — жёсткое ограничение V_diff(0)=0)."""
        self.x = float(theta_n0)
        self.x0 = float(theta_n0)
        self.tp0 = float(theta_p0)
        self.vd.zero_()

    def theta_p(self) -> float:
        return self.tp0 - self.geom.rho * (self.x - self.x0)

    def voltage(self, i: float) -> tuple[float, float]:
        """Напряжение и потенциал анода при токе ``i`` (заряд положителен),
        без продвижения состояния."""
        c = self.geom.c
        tp = self.theta_p()
        xt = torch.tensor(self.x, dtype=torch.float64)
        tpt = torch.tensor(tp, dtype=torch.float64)
        i_t = torch.tensor(i, dtype=torch.float64)
        en = bv_overpotential(i_t, self.geom.area_m2,
                              exchange_current(self.geom.j0_n_a_m2, xt), self.temp_k)
        ep = bv_overpotential(i_t, self.geom.area_m2,
                              exchange_current(self.geom.j0_p_a_m2, tpt), self.temp_k)
        v = (self.ocv_p(tpt.unsqueeze(0), c) - self.ocv_n(xt.unsqueeze(0), c)
             + ep + en + i_t * self.r_total + self.vd.sum())
        phi_n = self.ocv_n(xt.unsqueeze(0), c) - en
        return float(v), float(phi_n)

    def advance(self, i: float, dt: float) -> None:
        """Продвижение состояния на ``dt`` секунд при токе ``i``."""
        self.x += i * dt / (3600.0 * self.geom.q_n_ah)
        al = torch.exp(-dt / self.tau)
        self.vd = self.vd * al + self.r_d * (1 - al) * i


@dataclass
class SimPoint:
    t: float
    i: float
    v: float
    phi_n: float


@dataclass
class CycleTrace:
    """Результат моделирования одного цикла."""

    t: list = field(default_factory=list)
    i: list = field(default_factory=list)
    v: list = field(default_factory=list)
    phi_n: list = field(default_factory=list)

    def append(self, t, i, v, phi_n):
        self.t.append(t); self.i.append(i); self.v.append(v); self.phi_n.append(phi_n)

    def discharge_capacity_ah(self) -> float:
        q = 0.0
        for k in range(1, len(self.t)):
            iavg = 0.5 * (self.i[k] + self.i[k - 1])
            q += max(-iavg, 0.0) * (self.t[k] - self.t[k - 1])
        return q / 3600.0

    def charge_capacity_ah(self) -> float:
        q = 0.0
        for k in range(1, len(self.t)):
            iavg = 0.5 * (self.i[k] + self.i[k - 1])
            q += max(iavg, 0.0) * (self.t[k] - self.t[k - 1])
        return q / 3600.0


def _solve_current_for_voltage(sim: CellSimulator, v_target: float,
                               i_lo: float, i_hi: float, tol: float = 1e-7,
                               max_iter: int = 60) -> float:
    """Ток, при котором модель даёт напряжение ``v_target`` (бисекция;
    V(i) монотонно возрастает по i)."""
    v_lo, _ = sim.voltage(i_lo)
    v_hi, _ = sim.voltage(i_hi)
    if v_target <= v_lo:
        return i_lo
    if v_target >= v_hi:
        return i_hi
    for _ in range(max_iter):
        mid = 0.5 * (i_lo + i_hi)
        v_mid, _ = sim.voltage(mid)
        if abs(v_mid - v_target) < tol:
            return mid
        if v_mid < v_target:
            i_lo = mid
        else:
            i_hi = mid
    return 0.5 * (i_lo + i_hi)


def simulate_protocol(
    sim: CellSimulator,
    steps: list[ProtocolStep],
    t0: float = 0.0,
    dt_max: float = 30.0,
    max_step_s: float = 40 * 3600.0,
) -> CycleTrace:
    """Моделирует один цикл по списку шагов протокола.

    Возвращает :class:`CycleTrace` с моментами времени, током, напряжением
    и потенциалом анода. Шаг ``workflow`` разворачивается в повторы
    вложенных шагов (учитывается в пределах одного вызова как один проход;
    для моделирования нескольких циклов функция вызывается по циклу).
    """
    trace = CycleTrace()
    ctx = {"i_cc": 0.0}  # ток последнего шага постоянного тока (предел для CV)

    def emit(t, i):
        v, pn = sim.voltage(i)
        trace.append(t, i, v, pn)

    def run_step(step: ProtocolStep, t: float) -> float:
        mode = step.mode
        if mode == "workflow":
            for _ in range(step.repeat or 1):
                for s in step.steps:
                    t = run_step(s, t)
            return t
        if mode == "rest":
            t_end = t + (step.duration_s or 0.0)
            while t < t_end - 1e-9:
                dt = min(dt_max, t_end - t)
                emit(t, 0.0)
                sim.advance(0.0, dt)
                t += dt
            emit(t, 0.0)
            return t
        if mode in ("cc_charge", "cc_discharge"):
            sign = 1.0 if mode == "cc_charge" else -1.0
            i_cc = sign * abs(step.current_a or 0.0)
            ctx["i_cc"] = i_cc
            v_lim = step.voltage_limit_v
            dur = step.duration_s or max_step_s
            t_end = t + dur
            emit(t, i_cc)
            while t < t_end - 1e-9:
                dt = min(dt_max, t_end - t)
                # прогноз напряжения в конце шага для уточнения dt у отсечки
                v_next, _ = _voltage_after(sim, i_cc, dt)
                if v_lim is not None and (
                    (i_cc > 0 and v_next >= v_lim) or (i_cc < 0 and v_next <= v_lim)
                ):
                    # дробление шага до точного пересечения отсечки
                    lo, hi = 0.0, dt
                    for _ in range(50):
                        mid = 0.5 * (lo + hi)
                        v_mid, _ = _voltage_after(sim, i_cc, mid)
                        hit = (i_cc > 0 and v_mid >= v_lim) or (i_cc < 0 and v_mid <= v_lim)
                        if hit:
                            hi = mid
                        else:
                            lo = mid
                    sim.advance(i_cc, hi)
                    t += hi
                    emit(t, i_cc)
                    break
                sim.advance(i_cc, dt)
                t += dt
                emit(t, i_cc)
            return t
        if mode == "cv_charge":
            v_hold = step.voltage_limit_v
            i_cut = abs(step.current_cutoff_a or 0.0)
            dur = step.duration_s or max_step_s
            t_end = t + dur
            while t < t_end - 1e-9:
                i_cv = _solve_current_for_voltage(
                    sim, v_hold, 0.0, step.current_a or abs(ctx["i_cc"]) or 0.05
                )
                emit(t, i_cv)
                if i_cv <= i_cut or i_cv <= 1e-9:
                    break
                dt = min(dt_max, t_end - t)
                sim.advance(i_cv, dt)
                t += dt
            emit(t, 0.0)
            return t
        # неизвестный режим — пропуск без продвижения
        return t

    for st in steps:
        t0 = run_step(st, t0)
    return trace


def _voltage_after(sim: CellSimulator, i: float, dt: float) -> tuple[float, float]:
    """Напряжение после продвижения на dt током i без изменения состояния."""
    x_new = sim.x + i * dt / (3600.0 * sim.geom.q_n_ah)
    al = torch.exp(-dt / sim.tau)
    vd_new = sim.vd * al + sim.r_d * (1 - al) * i
    saved = (sim.x, sim.vd.clone())
    sim.x, sim.vd = x_new, vd_new
    v, pn = sim.voltage(i)
    sim.x, sim.vd = saved
    return v, pn


def main_loop_steps(protocol: list[ProtocolStep]) -> list[ProtocolStep]:
    """Шаги основного цикла: последняя группа ``workflow`` протокола."""
    workflows = [s for s in protocol if s.mode == "workflow"]
    if not workflows:
        return [s for s in protocol if s.mode != "workflow"]
    main = workflows[-1]
    return list(main.steps)
