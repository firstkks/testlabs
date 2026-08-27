# labs/ch08-gait-analysis/02_gait_diagram.py
"""8.2 내 로봇의 gait diagram 그리기.

01이 저장한 접촉 로그를 읽어 3장의 도구(quadbook.gait.plot_gait_diagram)로 실측 gait diagram을 그리고, 생성기가 명령한 도해와 나란히 놓는다.
가로축은 명령 주기 T = 1/f 로 정규화한 '주기' 단위. 두 주기를 그린다.
그림: out/ch08-fig03-trot-diagram.png, out/ch08-fig04-walk-diagram.png
실행: uv run python ch08-gait-analysis/02_gait_diagram.py   (--view --speed 0.5: walk 1 Hz 재생)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook import model_path
from quadbook.cpg import GaitGenerator
from quadbook.gait import LEG_ORDER, contacts_to_intervals, plot_gait_diagram
from quadbook.render import play, want_view
from quadbook.robot import load
from quadbook.sim import WALK_KP, standard_pd

OUT = Path(__file__).resolve().parent / "out"
L = np.load(OUT / "logs.npz")


def measured_intervals(name, t0, n_cycles=2):
    """t0부터 n_cycles 주기 동안의 실측 접촉 구간을 주기 단위로."""
    t, c, f = L[f"{name}_t"], L[f"{name}_contact"], float(L[f"{name}_freq"])
    T = 1.0 / f
    m = (t >= t0) & (t <= t0 + n_cycles * T)
    return {leg: [((s - t0) / T, (e - t0) / T) for s, e in contacts_to_intervals(t[m], c[m][:, i])] for i, leg in enumerate(LEG_ORDER)}


def commanded_intervals(name, t0, n_cycles=2):
    t, fc, f = L[f"{name}_t"], L[f"{name}_foot_cmd"], float(L[f"{name}_freq"])
    T = 1.0 / f
    m = (t >= t0) & (t <= t0 + n_cycles * T)
    return {leg: [((s - t0) / T, (e - t0) / T) for s, e in contacts_to_intervals(t[m], fc[m][:, i] < 1e-6)] for i, leg in enumerate(LEG_ORDER)}


def lf_touchdown_after(name, t_from):
    """t_from 이후 첫 LF 접지 시각 (도해의 0점)."""
    t, c = L[f"{name}_t"], L[f"{name}_contact"][:, 0]
    starts = [s for s, _ in contacts_to_intervals(t, c) if s > t_from]
    return starts[0]


for name, title, fig_name in (("trot2", "trot 2 Hz", "ch08-fig03-trot-diagram.png"), ("walk1", "walk 1 Hz", "ch08-fig04-walk-diagram.png")):
    t0 = lf_touchdown_after(name, 3.0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    plot_gait_diagram(axes[0], commanded_intervals(name, t0), x_max=2, color="#999999", title=f"(a) {title}: commanded")
    plot_gait_diagram(axes[1], measured_intervals(name, t0), x_max=2, color="#333333", title=f"(b) {title}: measured (contact list)")
    fig.tight_layout(); fig.savefig(OUT / fig_name, dpi=200); plt.close(fig)
    meas = measured_intervals(name, t0); cmd = commanded_intervals(name, t0)
    print(f"{title}: 도해 0점 = LF 접지 {t0:.3f} s")
    for leg in LEG_ORDER:
        print(f"    {leg}: 명령 지지 구간 " + ", ".join(f"[{s:.2f}, {e:.2f}]" for s, e in cmd[leg]) + "  | 실측 " + ", ".join(f"[{s:.2f}, {e:.2f}]" for s, e in meas[leg]))

# trot 1 Hz: 지지기 끝에 발이 뜨는 모습
t0 = lf_touchdown_after("trot1", 3.0)
meas = measured_intervals("trot1", t0)
print("trot 1 Hz: LF 실측 지지 구간 " + ", ".join(f"[{s:.2f}, {e:.2f}]" for s, e in meas["LF"]) + "  (명령은 [0.00, 0.50], [1.00, 1.50])")
print("saved", OUT / "ch08-fig03-trot-diagram.png", OUT / "ch08-fig04-walk-diagram.png")

if want_view():
    model, data = load(model_path()); pd = standard_pd(model, data, kp=WALK_KP)
    gen = GaitGenerator(gait="walk", freq=1.0, duty=0.75, step_length=0.08, step_height=0.04)
    state = {"i": 0, "q": gen.targets_at(gen.osc.theta), "qd": np.zeros(12)}
    def _step():
        if data.time >= 0.5 and state["i"] % 5 == 0: state["q"], state["qd"] = gen.targets_with_velocity(model.opt.timestep * 5)
        data.ctrl[:] = pd.torque(data, state["q"], state["qd"]); mujoco.mj_step(model, data); state["i"] += 1
    play(model, data, step_fn=_step, duration=8.0, title="walk 1 Hz: LF → RH → RF → LH 순서로 한 발씩")
