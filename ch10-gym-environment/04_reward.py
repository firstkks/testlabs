# labs/ch10-gym-environment/04_reward.py
"""10.4 reward 설계 첫 단추: 전진, 자세, 에너지.

보상 항의 '날것' 크기를 네 가지 행동에서 잰다: 행동 0(trot), 난수 0.1, 난수 0.4, 제자리 서기(freq 0).
크기를 알아야 가중치를 정할 수 있다. 가중치를 곱한 뒤 항별 기여를 나란히 놓는다.
그림: out/ch10-fig06-reward-terms.png
실행: uv run python ch10-gym-environment/04_reward.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quadbook.env import DEFAULT_WEIGHTS, QuadrupedEnv

OUT = Path(__file__).resolve().parent / "out"
TERMS = ("forward", "attitude", "energy", "alive")


def measure(env, policy, seed=0):
    rng = np.random.default_rng(seed); obs, _ = env.reset(seed=seed)
    acc = {k: [] for k in TERMS}; ret = 0.0; n = 0
    while True:
        obs, r, term, trunc, info = env.step(policy(rng)); ret += r; n += 1
        for k in TERMS: acc[k].append(info["terms"][k])
        if term or trunc: break
    return {k: float(np.mean(v)) for k, v in acc.items()}, ret, n


behaviours = {
    "CPG trot (a = 0)": (QuadrupedEnv(), lambda rng: np.zeros(12)),
    "random 0.1 rad": (QuadrupedEnv(action_scale=0.1), lambda rng: rng.uniform(-1, 1, 12)),
    "random 0.4 rad": (QuadrupedEnv(action_scale=0.4), lambda rng: rng.uniform(-1, 1, 12)),
    "stand still (freq 0)": (QuadrupedEnv(freq=0.0, random_init_phase=False), lambda rng: np.zeros(12)),
}
print("보상 항의 스텝당 평균 (날것)            forward[m/s]  attitude[rad²]  energy[W]   | 가중치 적용 후 스텝당 보상 / 에피소드 누적 / 스텝")
print("가중치:", DEFAULT_WEIGHTS)
res = {}
for name, (env, pol) in behaviours.items():
    terms, ret, n = measure(env, pol); res[name] = (terms, ret, n)
    print(f"  {name:22s} {terms['forward']:+8.3f}     {terms['attitude']:+9.4f}     {terms['energy']:+8.2f}   | {ret/n:+.3f} / {ret:+.1f} / {n}")

fig, ax = plt.subplots(figsize=(8.5, 3.6))
names = list(res); x = np.arange(len(names)); w = 0.2
for i, k in enumerate(("forward", "attitude", "energy")):
    ax.bar(x + (i - 1) * w, [DEFAULT_WEIGHTS[k] * res[n][0][k] for n in names], w, label=f"{k} x {DEFAULT_WEIGHTS[k]}")
ax.plot(x, [res[n][1] / res[n][2] for n in names], "k_", ms=25, mew=2, label="total per step")
ax.axhline(0, color="#999999", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8); ax.set_ylabel("weighted reward per step"); ax.set_title("reward terms after weighting, four behaviours", fontsize=10); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch10-fig06-reward-terms.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch10-fig06-reward-terms.png")
