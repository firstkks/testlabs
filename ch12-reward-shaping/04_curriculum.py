# labs/ch12-reward-shaping/04_curriculum.py
"""12.4 커리큘럼: 쉬운 것부터 어려운 것으로.

밀기 외란(최대 60 N)이 있는 환경에서 (a) 처음부터 60 N으로 6M 스텝 학습(scratch),
(b) R4(평지 3M)에서 이어서 20 → 60 N으로 올리며 3M 학습(첫 시도),
(c) 같은 커리큘럼을 6M으로: 20 → 60 N을 3M에 걸쳐 올리고 60 N에서 3M 더,
(d) 시험보다 센 힘까지: 20 → 80 N을 4M에 걸쳐 올리고 80 N에서 2M,
(e) 비교군: 약한 외란 20 N 고정으로 6M (커리큘럼 없이 외란만),
(f) (c)와 같되 보정 폭 0.15 rad: 잔차 폭이 한계인지 본다. 밀기 생존율과 학습 곡선. 생존율은 out/push_survival.json 에 저장.
그림: out/ch12-fig07-curriculum-curves.png, out/ch12-fig08-push-limits.png
실행: uv run python 04_curriculum.py
"""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_lib import OUT, load_policy, run_episode

names = ("R4_full", "C_scratch_push", "C_const20_6M", "C_curriculum_push", "C_curriculum_6M", "C_curriculum_80", "C_curriculum_6M_wide")
M = {n: json.load(open(OUT / f"metrics_{n}.json")) for n in names}
M["baseline"] = json.load(open(OUT / "metrics_baseline.json")); M["ppo_first"] = json.load(open(OUT / "metrics_ppo_first.json"))
print("5 seed 평균 (평지 평가)           속도[m/s]  넘어짐  yaw[°]  조각/주기                충격[N]  CoT   밀기한계[N]")
for n in ("baseline", "ppo_first") + names:
    m = M[n]; print(f"  {n:18s} {m['speed']:8.3f}  {m['fall_rate']:5.0%}  {m['yaw_deg']:+6.0f}  {[round(f, 1) for f in m['fragments']] if m['fragments'] else 'n/a'}   {m['impact']:5.0f}  {m['cot']:.2f}   {m['push_limit']}")


def curve(name, key="rollout/ep_rew_mean"):
    rows = list(csv.DictReader(open(OUT / f"log_{name}" / "progress.csv")))
    x = np.array([float(r["time/total_timesteps"]) for r in rows if r.get(key)]); y = np.array([float(r[key]) for r in rows if r.get(key)])
    return x, y


fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
COL = {"R4_full": "#999999", "C_scratch_push": "#d62728", "C_const20_6M": "#ff7f0e", "C_curriculum_push": "#2ca02c", "C_curriculum_6M": "#1f77b4", "C_curriculum_80": "#9467bd", "C_curriculum_6M_wide": "#17becf"}
for n in ("R4_full", "C_scratch_push", "C_const20_6M", "C_curriculum_6M", "C_curriculum_6M_wide"):     # 곡선은 다섯 개만 (3M·80 N은 생존율 그림에)
    x, y = curve(n); off = 3e6 if n.startswith("C_c") else 0.0
    axes[0].plot((x + off) / 1e6, y, color=COL[n], lw=1.0, label=n + (" (from R4 at 3M)" if off else ""))
    x, y = curve(n, "rollout/ep_len_mean"); axes[1].plot((x + off) / 1e6, y, color=COL[n], lw=1.0, label=n)
axes[0].axvline(6.0, color="#bbbbbb", ls=":", lw=0.8); axes[1].axvline(6.0, color="#bbbbbb", ls=":", lw=0.8)
axes[0].text(6.05, 250, "ramp ends: 60 N held", fontsize=7, color="#666666")
axes[0].set_xlabel("timesteps [M]"); axes[0].set_ylabel("episode return"); axes[0].set_title("(a) return (pushes make it noisy)", fontsize=10)
axes[1].set_xlabel("timesteps [M]"); axes[1].set_ylabel("episode length"); axes[1].set_title("(b) episode length (below 1000 = falls)", fontsize=10)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=min(4, max(1, len(l))), fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(OUT / "ch12-fig07-curriculum-curves.png", dpi=200); plt.close(fig)

# 세밀한 밀기 시험: 힘 30~90 N, 양방향, seed 3개 → 생존율
forces = (30, 40, 50, 60, 70, 80, 90)
print("밀기 생존율 [%] (힘별, ±y 두 방향 × seed 3)")
surv = {}
for n in names:
    model, venv, env = load_policy(n); pol = lambda o: model.predict(o, deterministic=True)[0]; row = []
    for F in forces:
        ok = [not run_episode(venv, env, pol, seed=s, push=(3.0, sign * F), max_t=6.0)["fell"] for s in (11, 12, 13) for sign in (1, -1)]
        row.append(100 * np.mean(ok))
    surv[n] = row; venv.close()
    print(f"  {n:18s} " + "  ".join(f"{F}N {r:3.0f}" for F, r in zip(forces, row)))
json.dump({"forces": list(forces), **surv}, open(OUT / "push_survival.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(7.5, 3.6))
for n in names:
    ax.plot(forces, surv[n], "o-", color=COL[n], label=n)
ax.set_xlabel("side push [N] for 0.2 s while trotting"); ax.set_ylabel("survived [%] (2 directions x 3 seeds)"); ax.set_ylim(-5, 105); ax.set_title("push robustness", fontsize=10); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch12-fig08-push-limits.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch12-fig07-curriculum-curves.png", OUT / "ch12-fig08-push-limits.png")
