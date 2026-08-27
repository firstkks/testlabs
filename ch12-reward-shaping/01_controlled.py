# labs/ch12-reward-shaping/01_controlled.py
"""12.1 보상 항 하나씩 추가하기: 통제된 실험의 기술.

같은 환경·같은 seed·같은 스텝(3M)에서 보상 항 하나(방향: heading 1.0, lateral 0.5)만 더한 R1을 11장의 정책과 비교한다.
그림: out/ch12-fig01-heading-paths.png, out/ch12-fig02-controlled-bars.png
실행: uv run python 01_controlled.py   (00_train_all.py 가 먼저 돌아 있어야 한다)
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_lib import OUT, load_policy, run_episode

M = {n: json.load(open(OUT / f"metrics_{n}.json")) for n in ("baseline", "ppo_first", "R1_heading")}
print("5 seed 평균 (10초 에피소드)          속도[m/s]  yaw[°]   y[m]   조각/주기(LF RF LH RH)   충격[N]  CoT   여유[cm]  밀기한계[N]")
for n, m in M.items():
    print(f"  {n:16s} {m['speed']:8.3f} {m['yaw_deg']:+7.0f} {m['y']:+6.2f}   {[round(f, 1) for f in m['fragments']]}   {m['impact']:5.0f}  {m['cot']:.2f}  {100*m['margin']:+6.1f}   {m['push_limit']}")

fig, ax = plt.subplots(figsize=(7.5, 3.8))
for n, c in (("ppo_first", "#ff7f0e"), ("R1_heading", "#2ca02c")):
    model, venv, env = load_policy(n)
    for s in (7, 8, 9):
        lg = run_episode(venv, env, lambda o: model.predict(o, deterministic=True)[0], seed=s)
        ax.plot(lg["x"], lg["y"], color=c, lw=1.2, alpha=0.8, label=(n + " (ch11 reward)" if n == "ppo_first" else n + " (+ heading, lateral)") if s == 7 else None)
    venv.close()
ax.axhline(0, color="#999999", ls=":"); ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title("top view, three seeds each: one reward term fixes the veering", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "ch12-fig01-heading-paths.png", dpi=200); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
names = list(M); cols = ("#999999", "#ff7f0e", "#2ca02c")
axes[0].bar(names, [M[n]["speed"] for n in names], color=cols); axes[0].set_title("speed [m/s]", fontsize=10)
axes[1].bar(names, [abs(M[n]["yaw_deg"]) for n in names], color=cols); axes[1].set_title("|yaw| after 10 s [deg]", fontsize=10)
axes[2].bar(names, [np.mean(M[n]["fragments"][2:]) for n in names], color=cols); axes[2].set_title("hind-leg contact fragments per cycle", fontsize=10)
for ax in axes: ax.tick_params(axis="x", labelsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch12-fig02-controlled-bars.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch12-fig01-heading-paths.png", OUT / "ch12-fig02-controlled-bars.png")
