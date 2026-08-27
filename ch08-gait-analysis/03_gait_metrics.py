# labs/ch08-gait-analysis/03_gait_metrics.py
"""8.3 duty factor와 위상 측정: 3장의 이론과 재회.

01의 로그에서 β(duty factor), T(주기), φ(LF 기준 상대 위상), λ(보폭), v(속도)를 3.2절의 정의대로 계산하고 명령값과 비교한다.
그리고 trot의 주파수를 1~3 Hz로 바꾸며(각 6초 걷기) β와 φ가 어떻게 변하는지 본다. 3장: 동물은 빨라질수록 β가 줄어든다.
그림: out/ch08-fig05-metrics-vs-freq.png, out/ch08-fig06-stride.png
실행: uv run python ch08-gait-analysis/03_gait_metrics.py   (--view --speed 0.5: trot 3 Hz 재생)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook import model_path
from quadbook.analysis import gait_metrics
from quadbook.cpg import GaitGenerator
from quadbook.gait import GAIT_PHASES, LEG_ORDER
from quadbook.render import play, want_view
from quadbook.robot import load
from quadbook.sim import WALK_KP, rollout, standard_pd

OUT = Path(__file__).resolve().parent / "out"
L = np.load(OUT / "logs.npz")


def report(name, log_t, log_c, log_x, freq, duty, gait):
    mtr = gait_metrics(log_t, log_c, t_from=2.0, x=log_x)
    cmd_phase = [GAIT_PHASES[gait][l] for l in LEG_ORDER]
    print(f"{name}: 주기 T 실측 {mtr['period']:.3f} s (명령 {1/freq:.3f}; LF 접지 간격으로는 {mtr['period_td']:.3f}), 주기 수 {mtr['n_cycles']}")
    print(f"    β 실측 {np.round(mtr['duty'], 2)}  명령 {duty}  | 주기당 접촉 조각 수 {np.round(mtr['fragments'], 1)} (1이면 깨끗)")
    print(f"    φ 실측 {np.round(mtr['phase'], 2)}  명령 {cmd_phase}")
    print(f"    보폭 λ 실측 {mtr['stride']:.3f} m (명령 L/β = {0.08/duty:.3f}), 속도 {mtr['speed']:.3f} m/s (= λ·f: {mtr['stride']*freq:.3f})")
    return mtr


print("=== 01의 로그에서 (3.2절 정의) ===")
for name, gait in (("trot2", "trot"), ("trot1", "trot"), ("walk1", "walk")):
    report(name, L[f"{name}_t"], L[f"{name}_contact"], L[f"{name}_x"], float(L[f"{name}_freq"]), float(L[f"{name}_duty"]), gait)

print("=== trot 주파수 스윕 (보폭 0.08, 명령 β 0.5) ===")
freqs = (1.0, 1.5, 2.0, 2.5, 3.0)
rows = []
for f in freqs:
    model, data = load(model_path()); pd = standard_pd(model, data, kp=WALK_KP)
    gen = GaitGenerator(gait="trot", freq=f, duty=0.5, step_length=0.08, step_height=0.04)
    log = rollout(model, data, gen, pd, duration=6.0, control_every=5)
    mtr = gait_metrics(log["t"], log["contact"], t_from=2.0, x=log["x"])
    rows.append((f, mtr))
    print(f"    f = {f:.1f} Hz: β {np.round(mtr['duty'], 2)}  φ(RF LH RH) {np.round(mtr['phase'][1:], 2)}  조각/주기 {np.round(mtr['fragments'], 1)}  T {mtr['period']:.3f} s  λ {mtr['stride']:.3f} m  v {mtr['speed']:.3f} m/s")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for i, leg in enumerate(LEG_ORDER):
    axes[0].plot(freqs, [m["duty"][i] for _, m in rows], "o-", label=leg)
axes[0].axhline(0.5, color="#999999", ls=":", label="commanded 0.5")
axes[0].set_xlabel("frequency [Hz]"); axes[0].set_ylabel("duty factor (measured)"); axes[0].set_title("(a) duty factor vs frequency", fontsize=10); axes[0].legend(fontsize=8)
for i, leg in enumerate(LEG_ORDER[1:], start=1):
    axes[1].plot(freqs, [(m["phase"][i] + 0.25) % 1.0 - 0.25 for _, m in rows], "o-", label=f"{leg} (cmd {GAIT_PHASES['trot'][leg]})")
for v in (0.0, 0.5): axes[1].axhline(v, color="#999999", ls=":", lw=1)
axes[1].set_xlabel("frequency [Hz]"); axes[1].set_ylabel("relative phase to LF [cycle]"); axes[1].set_ylim(-0.27, 0.77); axes[1].set_title("(b) relative phase vs frequency", fontsize=10); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch08-fig05-metrics-vs-freq.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.4))
v = [m["speed"] for _, m in rows]
ax.plot(v, [m["stride"] for _, m in rows], "o-", label="measured stride")
ax.axhline(0.16, color="#999999", ls=":", label="commanded L / beta = 0.16 m")
for (f, m) in rows: ax.annotate(f"{f:g} Hz", (m["speed"], m["stride"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.set_xlabel("speed [m/s]"); ax.set_ylabel("stride length [m]"); ax.set_title("stride length vs speed (trot, step length 0.08)", fontsize=10); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch08-fig06-stride.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch08-fig05-metrics-vs-freq.png", OUT / "ch08-fig06-stride.png")

if want_view():
    model, data = load(model_path()); pd = standard_pd(model, data, kp=WALK_KP)
    gen = GaitGenerator(gait="trot", freq=3.0, duty=0.5, step_length=0.08, step_height=0.04)
    state = {"i": 0, "q": gen.targets_at(gen.osc.theta), "qd": np.zeros(12)}
    def _step():
        if data.time >= 0.5 and state["i"] % 5 == 0: state["q"], state["qd"] = gen.targets_with_velocity(model.opt.timestep * 5)
        data.ctrl[:] = pd.torque(data, state["q"], state["qd"]); mujoco.mj_step(model, data); state["i"] += 1
    play(model, data, step_fn=_step, duration=6.0, title="trot 3 Hz")
