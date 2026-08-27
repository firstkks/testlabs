# labs/ch10-gym-environment/03_action.py
"""10.3 action 설계: 전체가 아닌 오프셋을 출력하게 하기.

(a) 잔차 모드에서 보정 크기 action_scale을 0.02~0.4 rad로 바꾸며 난수 정책 10 에피소드씩: 전진 거리와 넘어짐.
(b) 비교: 행동이 관절 목표 그 자체인 절대 모드(absolute)에서 난수 정책.
그림: out/ch10-fig05-action-scale.png
실행: uv run python ch10-gym-environment/03_action.py   (--view --speed 0.5: 절대 모드 난수 정책)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quadbook.env import QuadrupedEnv
from quadbook.render import play, want_view

OUT = Path(__file__).resolve().parent / "out"
N = 10


def trial(env, seed):
    rng = np.random.default_rng(seed); obs, _ = env.reset(seed=seed); ret = 0.0
    while True:
        obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, 12)); ret += r
        if term or trunc: return info["x"], term, info["t"], ret


scales = (0.02, 0.05, 0.1, 0.2, 0.4)
rows = []
print(f"(a) 잔차 모드, 난수 정책 {N} 에피소드 × 최대 10초")
for s in scales:
    env = QuadrupedEnv(action_scale=s)
    res = [trial(env, k) for k in range(N)]
    x, fell, tt, ret = map(np.array, zip(*res)); rows.append((s, x, fell, ret))
    print(f"    scale {s:.2f} rad: 전진 {x.mean():+.2f} m (σ {x.std():.2f}), 넘어짐 {100*fell.mean():.0f}%, 평균 생존 {tt.mean():.1f} s, 누적 보상 {ret.mean():.1f}")
env = QuadrupedEnv(action_mode="absolute")
res = [trial(env, k) for k in range(N)]; x, fell, tt, ret = map(np.array, zip(*res))
print(f"(b) 절대 모드(행동 = 관절 목표 자체): 전진 {x.mean():+.2f} m, 넘어짐 {100*fell.mean():.0f}%, 평균 생존 {tt.mean():.1f} s, 누적 보상 {ret.mean():.1f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].errorbar([r[0] for r in rows], [r[1].mean() for r in rows], yerr=[r[1].std() for r in rows], fmt="o-", capsize=3)
axes[0].axhline(x.mean(), color="#d62728", ls=":", label=f"absolute mode: {x.mean():+.2f} m"); axes[0].set_xscale("log")
axes[0].set_xticks(scales); axes[0].set_xticklabels([f"{v:g}" for v in scales]); axes[0].minorticks_off()
axes[0].set_xlabel("action scale [rad]"); axes[0].set_ylabel("forward distance [m]"); axes[0].set_title("(a) random policy: distance vs residual scale", fontsize=10); axes[0].legend(fontsize=8)
axes[1].plot([r[0] for r in rows], [100 * r[2].mean() for r in rows], "o-"); axes[1].axhline(100 * fell.mean(), color="#d62728", ls=":", label="absolute mode"); axes[1].set_xscale("log")
axes[1].set_xticks(scales); axes[1].set_xticklabels([f"{v:g}" for v in scales]); axes[1].minorticks_off()
axes[1].set_xlabel("action scale [rad]"); axes[1].set_ylabel("episodes that fell [%]"); axes[1].set_ylim(-5, 105); axes[1].set_title("(b) fall rate", fontsize=10); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch10-fig05-action-scale.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch10-fig05-action-scale.png")

if want_view():
    env = QuadrupedEnv(action_mode="absolute"); env.reset(seed=0); rng = np.random.default_rng(0)
    play(env.model, env.data, step_fn=lambda: env.step(rng.uniform(-1, 1, 12)), duration=6.0, title="절대 모드: 행동이 관절 목표 자체일 때의 난수 정책")
