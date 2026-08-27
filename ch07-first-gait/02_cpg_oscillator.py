# labs/ch07-first-gait/02_cpg_oscillator.py
"""7.2 이것이 CPG다: 결합 위상 진동자.

(a) 다리 4개의 위상을 제멋대로 시작시킨다. 결합 k = 0 이면 위상차가 그대로 남고, k > 0 이면 trot 위상표로 끌려간다.
(b) 3초에 LH 다리의 위상을 0.25 주기 어긋나게 흔든다. 결합이 있으면 돌아온다.
(c) 진동자 → 발 궤적 → IK 파이프라인이 만든 한 주기의 명령: 발 높이(명령한 gait diagram)와 LF 관절 목표.
그림: out/ch07-fig03-phase-coupling.png, out/ch07-fig04-cpg-pipeline.png
실행: uv run python ch07-first-gait/02_cpg_oscillator.py   (--view: 공중에 든 로봇이 진동자에 맞춰 다리를 젓는다)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook import model_path
from quadbook.cpg import GaitGenerator, PhaseOscillators
from quadbook.gait import GAIT_PHASES, LEG_ORDER
from quadbook.render import play, want_view
from quadbook.robot import load

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
DT, F = 0.01, 2.0
rng = np.random.default_rng(7)
init = rng.random(4)
print("초기 위상 (LF RF LH RH):", np.round(init, 2), "| trot 목표 위상:", [GAIT_PHASES["trot"][l] for l in LEG_ORDER])


def run(k, perturb=True, T=6.0):
    osc = PhaseOscillators(F, GAIT_PHASES["trot"], coupling=k, init=init)
    t, rel = [], []
    for i in range(int(T / DT)):
        if perturb and abs(i * DT - 3.0) < DT / 2:
            osc.theta[2] = (osc.theta[2] + 0.25) % 1.0        # LH 다리를 1/4 주기 어긋나게
        th = osc.step(DT)
        t.append(i * DT); rel.append(((th - th[0]) + 0.25) % 1.0 - 0.25)      # LF 기준 상대 위상, [-0.25, 0.75)로 표시
    return np.array(t), np.array(rel)


fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, k in zip(axes, (0.0, 2.0)):
    t, rel = run(k)
    for i, leg in enumerate(LEG_ORDER[1:], start=1):
        ax.plot(t, rel[:, i], label=f"{leg} - LF")
    for v in (0.0, 0.5):
        ax.axhline(v, color="#999999", ls=":", lw=1)
    ax.axvline(3.0, color="#d62728", ls="--", lw=1)
    ax.set_title(f"coupling k = {k:g}" + ("  (no coupling)" if k == 0 else "  (pulled to trot pattern)"), fontsize=10)
    ax.set_xlabel("time [s]"); ax.set_ylim(-0.27, 0.77)
    err = np.abs(((rel[-1] - np.array([GAIT_PHASES["trot"][l] for l in LEG_ORDER])) + 0.5) % 1.0 - 0.5)
    conv = t[np.where(np.abs(((rel - np.array([GAIT_PHASES['trot'][l] for l in LEG_ORDER])) + 0.5) % 1.0 - 0.5).max(axis=1) < 0.02)[0]]
    first = conv[conv < 3.0]; after = conv[conv > 3.0]
    print(f"k = {k:g}: 6초 뒤 목표 위상차와의 오차 최대 {err.max():.3f} 주기 | 수렴(오차 < 0.02) " + (f"{first[0]:.2f} s, 3초 외란 뒤 재수렴 {after[0]:.2f} s" if len(first) and len(after) else "안 함"))
axes[0].set_ylabel("phase relative to LF [cycle]"); axes[1].legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "ch07-fig03-phase-coupling.png", dpi=200); plt.close(fig)

# ---- (c) 파이프라인이 만든 한 주기 ----
gen = GaitGenerator("trot", freq=F, duty=0.5, step_length=0.08, step_height=0.04)
ts = np.arange(0, 1.0 / F, DT / 2)
foot_h, q = [], []
for _ in ts:
    q.append(gen.targets(DT / 2)); foot_h.append(gen.foot_height_cmd(gen.osc.theta))
foot_h, q = np.array(foot_h), np.array(q)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for i, leg in enumerate(LEG_ORDER):
    axes[0].plot(ts * F, 1000 * foot_h[:, i] + 45 * (3 - i), label=leg)
axes[0].set_yticks([45 * (3 - i) for i in range(4)]); axes[0].set_yticklabels(LEG_ORDER)
axes[0].set_xlabel("gait cycle"); axes[0].set_title("(a) commanded foot height per leg (flat = stance)", fontsize=10)
axes[1].plot(ts * F, q[:, 1], label="LF hip target"); axes[1].plot(ts * F, q[:, 2], label="LF knee target")
axes[1].set_xlabel("gait cycle"); axes[1].set_ylabel("joint target [rad]"); axes[1].set_title("(b) joint targets from IK", fontsize=10); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch07-fig04-cpg-pipeline.png", dpi=200); plt.close(fig)
stance = (foot_h < 1e-6).mean(axis=0)
print("명령한 duty factor (다리별):", np.round(stance, 2), "| 관절 목표 범위 엉덩이", np.round([q[:,1].min(), q[:,1].max()], 2), "무릎", np.round([q[:,2].min(), q[:,2].max()], 2))
print("saved", OUT / "ch07-fig03-phase-coupling.png", OUT / "ch07-fig04-cpg-pipeline.png")

if want_view():
    model, data = load(model_path()); model.opt.gravity[:] = 0; data.qpos[2] = 0.45
    from quadbook.sim import standard_pd
    pd = standard_pd(model, data)
    gen = GaitGenerator("trot", freq=1.0, init_phase=init)
    def _step():
        data.ctrl[:] = pd.torque(data, gen.targets(model.opt.timestep)); mujoco.mj_step(model, data)
    play(model, data, step_fn=_step, duration=10.0, title="공중에서: 제멋대로 시작한 위상이 trot으로 맞춰진다 (1 Hz)")
