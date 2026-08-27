# labs/ch09-rl-refinement/01_random_policies.py
"""9.3 보조 실험: 학습 전의 '무작위 정책'은 얼마나 걷는가.

강화학습은 무작위 탐색에서 시작한다. 세 가지 출발점을 20 에피소드(각 4초)씩 돌려 비교한다.
(a) 백지 토크: 매 제어 스텝 12개 토크를 ±12 N·m 균등 난수로.
(b) 백지 관절 목표: 서기 자세 ± 0.3 rad 균등 난수를 PD 목표로.
(c) CPG + 잔차: 7장 trot 생성기의 목표에 ± 0.1 rad 균등 난수를 더한 것을 PD 목표로 (10.3절의 action 설계).
지표: 4초 동안의 전진 거리, 넘어진 에피소드 비율, 평균 전진 속도(보상 신호의 대리).
그림: out/ch09-fig04-random-policies.png
실행: uv run python ch09-rl-refinement/01_random_policies.py   (--view --speed 0.5: (c)를 한 에피소드 본다)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook import model_path
from quadbook.cpg import GaitGenerator
from quadbook.render import play, want_view
from quadbook.robot import STAND_POSE_BALANCED, load
from quadbook.sim import WALK_KP, fallen, standard_pd

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
CE, T, N_EP = 5, 4.0, 20
rng = np.random.default_rng(0)


def episode(kind, seed):
    r = np.random.default_rng(seed)
    model, data = load(model_path()); pd = standard_pd(model, data, kp=WALK_KP)
    gen = GaitGenerator(gait="trot", freq=2.0, duty=0.5, step_length=0.08, step_height=0.04)
    dt_ctrl = model.opt.timestep * CE
    q, qd = STAND_POSE_BALANCED.copy(), np.zeros(12); tau = np.zeros(12); i = 0
    x0 = data.body("torso").xpos[0]; fell = False; t_fall = T
    while data.time < T:
        if i % CE == 0:
            if kind == "random torque":
                tau = r.uniform(-12, 12, 12)
            elif kind == "random targets":
                q = STAND_POSE_BALANCED + r.uniform(-0.3, 0.3, 12); qd = np.zeros(12)
            else:
                q, qd = gen.targets_with_velocity(dt_ctrl); q = q + r.uniform(-0.1, 0.1, 12)
        data.ctrl[:] = tau if kind == "random torque" else pd.torque(data, q, qd)
        mujoco.mj_step(model, data); i += 1
        if not fell and fallen(data):
            fell, t_fall = True, data.time
    return data.body("torso").xpos[0] - x0, fell, t_fall


kinds = ("random torque", "random targets", "CPG + random residual")
res = {}
print(f"무작위 정책 {N_EP} 에피소드 × {T:.0f}초")
for kind in kinds:
    d, f, tf = zip(*[episode(kind, s) for s in range(N_EP)])
    res[kind] = (np.array(d), np.array(f), np.array(tf))
    print(f"  {kind:22s}: 전진 거리 평균 {np.mean(d):+.2f} m (최소 {np.min(d):+.2f}, 최대 {np.max(d):+.2f}) | 넘어짐 {100*np.mean(f):.0f}% | 넘어지기까지 평균 {np.mean(tf):.1f} s | 평균 속도 {np.mean(d)/T:+.3f} m/s")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
colors = ("#d62728", "#ff7f0e", "#2ca02c")
for k, (kind, c) in enumerate(zip(kinds, colors)):
    d = res[kind][0]
    axes[0].scatter(np.full(N_EP, k) + rng.uniform(-0.15, 0.15, N_EP), d, color=c, s=18, alpha=0.8)
    axes[0].plot([k - 0.25, k + 0.25], [d.mean()] * 2, color="#333333", lw=2)
axes[0].axhline(0, color="#999999", ls=":"); axes[0].set_xticks(range(3)); axes[0].set_xticklabels(kinds, fontsize=8)
axes[0].set_ylabel("forward distance in 4 s [m]"); axes[0].set_title("(a) distance per episode (bar = mean)", fontsize=10)
axes[1].bar(kinds, [100 * res[k][1].mean() for k in kinds], color=colors)
axes[1].set_ylabel("episodes that fell [%]"); axes[1].set_ylim(0, 105); axes[1].set_title("(b) fall rate", fontsize=10); axes[1].tick_params(axis="x", labelsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch09-fig04-random-policies.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch09-fig04-random-policies.png")

if want_view():
    model, data = load(model_path()); pd = standard_pd(model, data, kp=WALK_KP)
    gen = GaitGenerator(gait="trot", freq=2.0, duty=0.5, step_length=0.08, step_height=0.04)
    r = np.random.default_rng(1); state = {"i": 0, "q": STAND_POSE_BALANCED.copy(), "qd": np.zeros(12)}
    def _step():
        if state["i"] % CE == 0:
            q, qd = gen.targets_with_velocity(model.opt.timestep * CE); state["q"], state["qd"] = q + r.uniform(-0.1, 0.1, 12), qd
        data.ctrl[:] = pd.torque(data, state["q"], state["qd"]); mujoco.mj_step(model, data); state["i"] += 1
    play(model, data, step_fn=_step, duration=8.0, title="CPG + 무작위 잔차(±0.1 rad): 학습 전에도 걷는다")
