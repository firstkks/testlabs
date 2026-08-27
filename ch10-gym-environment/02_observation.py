# labs/ch10-gym-environment/02_observation.py
"""10.2 observation 설계: 로봇은 무엇을 알아야 하는가.

행동 0으로 한 에피소드를 돌리며 관측 53차원을 기록하고, 묶음별로 무엇이 보이는지와 값의 크기(스케일)를 본다.
스케일이 제각각인 관측은 11.2절의 정규화(VecNormalize)가 필요한 이유다.
그림: out/ch10-fig03-observation-traces.png, out/ch10-fig04-observation-scale.png
실행: uv run python ch10-gym-environment/02_observation.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quadbook.env import QuadrupedEnv

OUT = Path(__file__).resolve().parent / "out"
GROUPS = [("joint offset", 0, 12), ("joint vel x0.05", 12, 24), ("gravity (body)", 24, 27), ("gyro x0.25", 27, 30), ("lin vel (body)", 30, 33), ("CPG phase sin/cos", 33, 41), ("prev action", 41, 53)]

env = QuadrupedEnv(random_init_phase=False)
obs, _ = env.reset(seed=0)
O, T = [obs], [0.0]
while True:
    obs, r, term, trunc, info = env.step(np.zeros(12)); O.append(obs); T.append(info["t"])
    if term or trunc: break
O, T = np.array(O), np.array(T)
print(f"관측 {O.shape[1]}차원, {len(T)} 스텝")
np.set_printoptions(precision=2, suppress=True, linewidth=150)
print("3초 시점의 관측 벡터:"); k = np.argmin(np.abs(T - 3.0))
for name, a, b in GROUPS:
    print(f"  {name:18s} {O[k, a:b]}")
print("묶음별 값의 크기 (에피소드 전체의 표준편차 / 최대 절댓값)")
for name, a, b in GROUPS:
    print(f"  {name:18s} σ {O[:, a:b].std():.3f}  max|·| {np.abs(O[:, a:b]).max():.3f}")

m = (T >= 2.0) & (T <= 3.0)
fig, axes = plt.subplots(2, 2, figsize=(11, 6))
axes[0, 0].plot(T[m], O[m, 24:27]); axes[0, 0].set_title("(a) gravity vector in torso frame (x, y, z)", fontsize=10); axes[0, 0].legend(["x", "y", "z"], fontsize=8)
axes[0, 1].plot(T[m], O[m, 33:37]); axes[0, 1].set_title("(b) sin of CPG phase, four legs", fontsize=10); axes[0, 1].legend(["LF", "RF", "LH", "RH"], fontsize=8)
axes[1, 0].plot(T[m], O[m, 1:3]); axes[1, 0].plot(T[m], O[m, 4:6], ls="--"); axes[1, 0].set_title("(c) joint offsets: LF hip, LF knee (solid), RF hip, RF knee (dashed)", fontsize=10)
axes[1, 1].plot(T[m], O[m, 30:33]); axes[1, 1].set_title("(d) torso linear velocity in body frame (x, y, z)", fontsize=10); axes[1, 1].legend(["vx", "vy", "vz"], fontsize=8)
for ax in axes.flat: ax.set_xlabel("time [s]")
fig.tight_layout(); fig.savefig(OUT / "ch10-fig03-observation-traces.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.4))
names = [g[0] for g in GROUPS]; stds = [O[:, a:b].std() for _, a, b in GROUPS]; mx = [np.abs(O[:, a:b]).max() for _, a, b in GROUPS]
x = np.arange(len(names)); ax.bar(x - 0.2, stds, 0.4, label="std over episode"); ax.bar(x + 0.2, mx, 0.4, label="max |value|")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8, rotation=15); ax.set_yscale("log"); ax.set_ylabel("value (log)"); ax.set_title("observation groups have different scales (after the fixed scaling in env.py)", fontsize=10); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch10-fig04-observation-scale.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch10-fig03-observation-traces.png", OUT / "ch10-fig04-observation-scale.png")
