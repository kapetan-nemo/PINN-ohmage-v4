"""Этап 0: предварительное обучение параметрических кривых равновесного
потенциала на литературных полуэлементных данных (выгрузка PyBaMM).

Совместно обучаются веса сетей ``MonotoneOCV`` для отрицательного и
положительного электродов и индивидуальные латентные коды ``c`` каждой
литературной кривой. Критерий приёмки — воспроизведение каждой кривой с
погрешностью < 5 мВ. Поскольку обе кривые монотонно убывают по доле
литирования, используются общая архитектура и раздельные веса.

Артефакты:
* ``checkpoints/stage0/ocv_n.pt``, ``ocv_p.pt`` — веса;
* ``data/params/latent_anchors.json`` — коды кривых (якоря пространства c);
* ``reports/stage0_ocv.json`` — погрешности по кривым.
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_soh.physics.ocv import MonotoneOCV, load_literature_curves  # noqa: E402

LATENT_DIM = 4
STEPS = 9000
LR = 8e-3


def train_electrode(curves: dict[str, dict], seed: int = 0) -> tuple[MonotoneOCV, dict]:
    """Обучает одну сеть OCV на семействе кривых одного электрода.

    Кроме общего латентного кода ``c`` на кривую оптимизируются два
    аффинных параметра: масштаб наклона ``s`` и сдвиг ``b`` —
    ``U ≈ net(θ; c)·s + b``; они поглощают разницу абсолютных уровней и
    крутизны литературных фитов, оставляя ``c`` для формы кривой.
    """
    torch.manual_seed(seed)
    net = MonotoneOCV(latent_dim=LATENT_DIM, hidden=96, n_layers=3)
    codes = {
        label: torch.zeros(LATENT_DIM, dtype=torch.float64, requires_grad=True)
        for label in curves
    }
    affine = {
        label: torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
        for label in curves
    }
    data = {
        label: (
            torch.tensor(v["theta"], dtype=torch.float64),
            torch.tensor(v["u"], dtype=torch.float64),
        )
        for label, v in curves.items()
    }
    opt = torch.optim.Adam(
        [{"params": net.parameters(), "lr": LR}]
        + [{"params": [codes[k]], "lr": LR} for k in curves]
        + [{"params": [affine[k]], "lr": LR} for k in curves]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    for step in range(STEPS):
        opt.zero_grad()
        loss = 0.0
        for label, (th, u) in data.items():
            pred = net(th, codes[label]) * affine[label][0] + affine[label][1]
            loss = loss + torch.mean((pred - u) ** 2)
        loss.backward()
        opt.step()
        sched.step()
        if step % 1500 == 0:
            print(f"    шаг {step}, mse {float(loss):.3e}")
    # Индивидуальное дообучение: веса сети заморожены, уточняются только
    # код c и аффинная пара кривой — изолирует трудные формы плато.
    for label, (th, u) in data.items():
        opt2 = torch.optim.Adam([codes[label], affine[label]], lr=2e-3)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=1500)
        for _ in range(1500):
            opt2.zero_grad()
            pred = net(th, codes[label]) * affine[label][0] + affine[label][1]
            l2 = torch.mean((pred - u) ** 2)
            l2.backward()
            opt2.step()
            sched2.step()
    rmses = {}
    with torch.no_grad():
        for label, (th, u) in data.items():
            pred = net(th, codes[label]) * affine[label][0] + affine[label][1]
            rmse_mv = float(torch.mean((pred - u) ** 2).sqrt()) * 1000
            rmses[label] = round(rmse_mv, 2)
    return net, {
        "codes": {k: [float(x) for x in codes[k].tolist()] for k in codes},
        "affine": {k: [float(x) for x in affine[k].tolist()] for k in affine},
        "rmse_mv": rmses,
    }


def anchors_with_affine(res: dict) -> dict:
    """Якорь = код + аффинная пара: ``U ≈ net(θ; c)·scale + shift``.

    Аффин обязателен: без него код интерполирует форму в нормированной
    шкале сети, а не в вольтах литературной кривой.
    """
    return {
        k: {"code": res["codes"][k],
            "scale": res["affine"][k][0],
            "shift": res["affine"][k][1]}
        for k in res["codes"]
    }


def main() -> None:
    curves = load_literature_curves(ROOT / "data" / "params" / "ocv_curves.json")
    neg = {k: v for k, v in curves.items() if v.get("electrode") == "negative"}
    pos = {k: v for k, v in curves.items() if v.get("electrode") == "positive"}
    # Источники с вырожденными фитами (осцилляции полиномов, расходимость
    # на краях окна) исключены из обучения — их якоря не нужны для Aurora.
    pos.pop("lco_Ramadass2004", None)
    pos.pop("lco_Marquis2019", None)
    pos.pop("nca_Kim2011", None)
    print(f"кривых: отрицательных {len(neg)}, положительных {len(pos)}")

    print("обучение анодной кривой:")
    net_n, res_n = train_electrode(neg, seed=0)
    print("обучение катодной кривой:")
    net_p, res_p = train_electrode(pos, seed=1)

    ckpt = ROOT / "checkpoints" / "stage0"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net_n.state_dict(), "latent_dim": LATENT_DIM},
               ckpt / "ocv_n.pt")
    torch.save({"state_dict": net_p.state_dict(), "latent_dim": LATENT_DIM},
               ckpt / "ocv_p.pt")

    anchors = {"negative": anchors_with_affine(res_n),
               "positive": anchors_with_affine(res_p)}
    (ROOT / "data" / "params" / "latent_anchors.json").write_text(
        json.dumps(anchors, indent=1))
    report = {"negative_rmse_mv": res_n["rmse_mv"],
              "positive_rmse_mv": res_p["rmse_mv"]}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "stage0_ocv.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    worst = max(max(res_n["rmse_mv"].values(), default=0),
                max(res_p["rmse_mv"].values(), default=0))
    print(f"наихудшая погрешность: {worst:.2f} мВ (критерий < 5 мВ)")


if __name__ == "__main__":
    main()
