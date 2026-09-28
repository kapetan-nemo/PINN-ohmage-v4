"""Export literature half-electrode OCV curves and degradation-kinetics
constants from PyBaMM parameter sets.

Outputs
-------
data/params/ocv_curves.json
    {"<label>": {"electrode": "negative"|"positive", "material": str,
                 "param_set": str, "ocp_function": str,
                 "c_max_mol_m3": float|None, "theta": [...], "u": [...]}}
    theta is a fraction of lithiation (stoichiometry). Only the physically
    meaningful window of theta is saved (finite U within electrode-specific
    voltage bounds and, for tabulated fits, inside the interpolant's data
    range -- cubic extrapolation outside the data range is nonphysical).

data/params/kinetics_<paramset>.json
    {"param_set": str,
     "constants": {key: {"value": float|str, "unit": str}},
     "unavailable_keys": [...]}

Usage
-----
    python scripts/export_pybamm_params.py            # export everything
    python scripts/export_pybamm_params.py --check    # verify ocv_curves.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pybamm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "params"

# Stoichiometry grid required by the specification
THETA = np.linspace(0.01, 0.99, 201)

# Physical voltage windows used to clip the literature fits to their
# meaningful stoichiometry range (extreme fits diverge outside them).
U_BOUNDS = {"negative": (0.0, 3.0), "positive": (2.5, 5.0)}

# (label, parameter set, electrode, material description)
OCV_SPECS = [
    # --- negative electrodes (graphite) ---
    ("graphite_Chen2020", "Chen2020", "negative", "graphite (LG M50, analytic fit)"),
    ("graphite_OKane2022", "OKane2022", "negative", "graphite (LG M50, tabulated)"),
    ("graphite_Ecker2015", "Ecker2015", "negative", "graphite (Ecker2015 / Kokam)"),
    ("graphite_Mohtat2020", "Mohtat2020", "negative", "graphite (Peyman MPM fit)"),
    ("graphite_Kim2011", "NCA_Kim2011", "negative", "graphite (Kim2011)"),
    ("graphite_Marquis2019", "Marquis2019", "negative", "graphite MCMB2528 (Dualfoil1998)"),
    ("graphite_Ramadass2004", "Ramadass2004", "negative", "graphite (Ramadass2004)"),
    ("graphite_Ai2020", "Ai2020", "negative", "graphite (Enertech, tabulated)"),
    # --- positive electrodes ---
    ("nmc811_Chen2020", "Chen2020", "positive", "NMC811 (LG M50)"),
    ("nmc811_OKane2022", "OKane2022", "positive", "NMC811 (LG M50)"),
    ("nmc811_ORegan2022", "ORegan2022", "positive", "NMC811 (LG M50)"),
    ("nco_Ecker2015", "Ecker2015", "positive", "NCO (Ecker2015 / Kokam)"),
    ("lfp_Prada2013", "Prada2013", "positive", "LFP (Afshar2017 fit)"),
    ("nmc_Mohtat2020", "Mohtat2020", "positive", "NMC (Peyman MPM fit)"),
    ("nca_Kim2011", "NCA_Kim2011", "positive", "NCA (Kim2011)"),
    ("lco_Ai2020", "Ai2020", "positive", "LCO (Ai2020 / Enertech, tabulated)"),
    ("lco_Marquis2019", "Marquis2019", "positive", "LCO (Dualfoil1998)"),
    ("lco_Ramadass2004", "Ramadass2004", "positive", "LCO (Ramadass2004)"),
    ("nmc_Xu2019", "Xu2019", "positive", "NMC (Xu2019 fit)"),
]

# Degradation-kinetics keys to extract (scalar values preferred; function
# entries are recorded by name). Missing keys are reported as unavailable.
KINETIC_KEYS = [
    # --- SEI ---
    "SEI growth rate constant [m.s-1]",
    "SEI kinetic rate constant [m.s-1]",
    "SEI resistivity [Ohm.m]",
    "SEI reaction exchange current density [A.m-2]",
    "SEI partial molar volume [m3.mol-1]",
    "SEI molar volume [m3.mol-1]",
    "Outer SEI solvent diffusivity [m2.s-1]",
    "Inner SEI solvent diffusivity [m2.s-1]",
    "SEI solvent diffusivity [m2.s-1]",
    "Bulk solvent concentration [mol.m-3]",
    "SEI electron conductivity [S.m-1]",
    "SEI lithium interstitial diffusivity [m2.s-1]",
    "Lithium interstitial reference concentration [mol.m-3]",
    "SEI open-circuit potential [V]",
    "Inner SEI open-circuit potential [V]",
    "Outer SEI open-circuit potential [V]",
    "Initial SEI thickness [m]",
    "Initial SEI on cracks thickness [m]",
    "Ratio of lithium moles to SEI moles",
    "SEI growth activation energy [J.mol-1]",
    # --- lithium plating / dead lithium ---
    "Lithium plating kinetic rate constant [m.s-1]",
    "Lithium plating transfer coefficient",
    "Lithium plating partial molar volume [m3.mol-1]",
    "Lithium metal partial molar volume [m3.mol-1]",
    "Dead lithium decay constant [s-1]",
    "Dead lithium decay rate [s-1]",
    "Exchange-current density for plating [A.m-2]",
    "Exchange-current density for stripping [A.m-2]",
    "Initial plated lithium concentration [mol.m-3]",
    "Typical plated lithium concentration [mol.m-3]",
    # --- reference ---
    "Reference temperature [K]",
]

KINETIC_PARAM_SETS = ["OKane2022"]

# Catch-all for degradation-related keys not listed in KINETIC_KEYS
_DEGRADATION_RE = re.compile(
    r"SEI|plating|plated|dead lithium|lithium metal|interstitial|"
    r"solvent|crack|LAM|loss of active",
    re.IGNORECASE,
)


def _eval_ocp(func, theta):
    """Evaluate a PyBaMM OCP entry on a numpy grid of stoichiometries.

    Plain functions return a numpy array directly; some parameter sets store
    OCP as a function returning a pybamm.Interpolant symbol, which must be
    evaluated after being built on a pybamm.Array node.
    """
    try:
        out = func(theta)
        return np.asarray(out, dtype=float).ravel(), None
    except Exception:
        interp = func(pybamm.Array(theta))
        u = np.asarray(interp.evaluate(), dtype=float).ravel()
        # data range of the underlying table (no extrapolation)
        xs = getattr(interp, "x", None)
        x_range = None
        if xs:
            x_arr = np.asarray(xs[0] if isinstance(xs, (list, tuple)) else xs, dtype=float)
            x_range = (float(np.nanmin(x_arr)), float(np.nanmax(x_arr)))
        return u, x_range


def _valid_mask(u, electrode, x_range):
    lo, hi = U_BOUNDS[electrode]
    mask = np.isfinite(u) & (u >= lo) & (u <= hi)
    if x_range is not None:
        eps = 1e-12
        mask &= (THETA >= x_range[0] - eps) & (THETA <= x_range[1] + eps)
    return mask


def _longest_run(mask):
    """Return slice of the longest contiguous run of True in a bool mask."""
    best_len, best = 0, None
    start = None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        if (not ok or i == len(mask) - 1) and start is not None:
            end = i if not ok else i + 1
            if end - start > best_len:
                best_len, best = end - start, slice(start, end)
            start = None
    return best


def export_ocv_curves():
    curves, skipped = {}, []
    for label, param_set, electrode, material in OCV_SPECS:
        key = f"{electrode.capitalize()} electrode OCP [V]"
        try:
            pv = pybamm.ParameterValues(param_set)
        except Exception as exc:  # pragma: no cover
            skipped.append((label, f"parameter set failed to load: {exc}"))
            continue
        func = pv.get(key)
        if not callable(func):
            skipped.append((label, f"'{key}' is not callable ({func!r})"))
            continue
        try:
            u, x_range = _eval_ocp(func, THETA)
        except Exception as exc:
            skipped.append((label, f"evaluation failed: {exc}"))
            continue
        mask = _valid_mask(u, electrode, x_range)
        run = _longest_run(mask)
        if run is None:
            skipped.append((label, "no physically valid theta window"))
            continue
        cmax_key = f"Maximum concentration in {electrode} electrode [mol.m-3]"
        c_max = pv.get(cmax_key)
        curves[label] = {
            "electrode": electrode,
            "material": material,
            "param_set": param_set,
            "ocp_function": getattr(func, "__name__", str(type(func).__name__)),
            "c_max_mol_m3": float(c_max) if np.isscalar(c_max) else None,
            "theta": THETA[run].tolist(),
            "u": u[run].tolist(),
        }
    return curves, skipped


def export_kinetics(param_set):
    pv = pybamm.ParameterValues(param_set)
    constants, unavailable = {}, []
    seen = set()

    def _record(key):
        seen.add(key)
        if key not in pv:
            unavailable.append(key)
            return
        value = pv[key]
        m = re.search(r"\[([^\]]+)\]\s*$", key)
        unit = m.group(1) if m else "dimensionless"
        if np.isscalar(value):
            entry = {"value": float(value), "unit": unit}
        elif callable(value):
            entry = {
                "value": f"<function {getattr(value, '__name__', 'anonymous')}>",
                "unit": unit,
            }
        else:
            entry = {"value": str(value), "unit": unit}
        constants[key] = entry

    for key in KINETIC_KEYS:
        _record(key)
    # pick up any other degradation-related keys present in the set
    for key in pv.keys():
        if key not in seen and _DEGRADATION_RE.search(key):
            _record(key)

    return {
        "param_set": param_set,
        "constants": constants,
        "unavailable_keys": unavailable,
    }


def check_ocv_curves(path):
    with open(path) as f:
        curves = json.load(f)
    print(f"{len(curves)} curves in {path}\n")
    print(
        f"{'label':24s} {'theta range':17s} {'n':>4s} {'U(0.05)':>8s} "
        f"{'U(0.95)':>8s}  monotonicity"
    )
    for label, c in sorted(curves.items()):
        th = np.asarray(c["theta"])
        u = np.asarray(c["u"])
        u05 = float(np.interp(0.05, th, u)) if th[0] <= 0.05 <= th[-1] else float("nan")
        u95 = float(np.interp(0.95, th, u)) if th[0] <= 0.95 <= th[-1] else float("nan")
        du = np.diff(u)
        signs = np.sign(du)
        n_changes = int(np.sum(signs[1:] * signs[:-1] < 0))
        if n_changes == 0:
            direction = "decreasing" if du[0] < 0 else "increasing"
            mono = f"monotone {direction}"
        else:
            mono = f"NON-monotone ({n_changes} sign changes of dU/dtheta)"
        print(
            f"{label:24s} [{th.min():.3f}, {th.max():.3f}] {len(th):4d} "
            f"{u05:8.4f} {u95:8.4f}  {mono}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify data/params/ocv_curves.json"
    )
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ocv_path = OUT_DIR / "ocv_curves.json"
    if args.check:
        check_ocv_curves(ocv_path)
        return

    curves, skipped = export_ocv_curves()
    with open(ocv_path, "w") as f:
        json.dump(curves, f, indent=2)
    print(f"wrote {ocv_path} ({len(curves)} curves)")
    for label, reason in skipped:
        print(f"  SKIPPED {label}: {reason}")

    for ps in KINETIC_PARAM_SETS:
        kin = export_kinetics(ps)
        path = OUT_DIR / f"kinetics_{ps}.json"
        with open(path, "w") as f:
            json.dump(kin, f, indent=2)
        print(
            f"wrote {path} ({len(kin['constants'])} constants, "
            f"{len(kin['unavailable_keys'])} unavailable keys)"
        )
        for key in kin["unavailable_keys"]:
            print(f"  unavailable: {key}")

    check_ocv_curves(ocv_path)


if __name__ == "__main__":
    sys.exit(main())
