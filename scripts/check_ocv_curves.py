"""Sanity check for data/params/ocv_curves.json.

For every exported half-electrode OCV curve prints the saved theta range,
U at theta=0.05 and theta=0.95 (linear interpolation of stored samples),
and whether the sign of dU/dtheta is constant over the stored window.
Local non-monotonicity is reported as a fact; data are not modified.

Usage: python scripts/check_ocv_curves.py
"""

import json
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "data" / "params" / "ocv_curves.json"


def main():
    curves = json.loads(PATH.read_text())
    print(f"{len(curves)} curves in {PATH}\n")
    hdr = (
        f"{'label':24s} {'electrode':9s} {'theta range':17s} {'n':>4s} "
        f"{'U(0.05)':>8s} {'U(0.95)':>8s}  derivative"
    )
    print(hdr)
    print("-" * len(hdr))
    for label, c in sorted(curves.items()):
        th = np.asarray(c["theta"], dtype=float)
        u = np.asarray(c["u"], dtype=float)
        u05 = float(np.interp(0.05, th, u)) if th[0] <= 0.05 <= th[-1] else float("nan")
        u95 = float(np.interp(0.95, th, u)) if th[0] <= 0.95 <= th[-1] else float("nan")
        du = np.diff(u)
        signs = np.sign(du[du != 0])
        n_changes = int(np.sum(signs[1:] * signs[:-1] < 0)) if signs.size else 0
        if n_changes == 0:
            direction = "decreasing" if signs[0] < 0 else "increasing"
            mono = f"monotone {direction}"
        else:
            mono = f"NON-monotone ({n_changes} sign changes)"
        print(
            f"{label:24s} {c['electrode']:9s} [{th.min():.3f}, {th.max():.3f}]"
            f" {len(th):4d} {u05:8.4f} {u95:8.4f}  {mono}"
        )


if __name__ == "__main__":
    main()
