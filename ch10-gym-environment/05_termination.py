# labs/ch10-gym-environment/05_termination.py
"""10.5 종료 조건과 리셋: 학습 가능한 환경의 조건.

(a) 난수 정책(0.4 rad) 50 에피소드의 길이 분포: terminated(넘어짐) vs truncated(시간 종료).
(b) 종료 순간의 장면: 높이와 기울기 문턱값이 무엇을 잡는가.
(c) 리셋의 무작위화: 초기 관절 오차와 CPG 위상이 에피소드마다 다르다. 같은 정책으로 20번 리셋해 전진 거리의 분포를 본다.
그림: out/ch10-fig07-episode-lengths.png, out/ch10-fig08-termination-frames.png
실행: uv run python ch10-gym-environment/05_termination.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import quadbook.render as R
from quadbook.control import torso_roll_pitch
from quadbook.env import QuadrupedEnv

OUT = Path(__file__).resolve().parent / "out"

# (a) 길이 분포
env = QuadrupedEnv(action_scale=0.4, render_mode="rgb_array")
lengths, kinds, frames = [], [], []
for seed in range(50):
    rng = np.random.default_rng(seed); obs, _ = env.reset(seed=seed); n = 0
    while True:
        obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, 12)); n += 1
        if term or trunc:
            lengths.append(n); kinds.append("terminated" if term else "truncated")
            if term and len(frames) < 3:
                roll, pitch = torso_roll_pitch(env.data)
                frames.append(R._label(env.render(), f"terminated at t = {info['t']:.2f} s  z = {info['z']:.2f} m  roll {np.degrees(roll):+.0f} pitch {np.degrees(pitch):+.0f}"))
            break
lengths, kinds = np.array(lengths), np.array(kinds)
print(f"(a) 난수 0.4 rad, 50 에피소드: terminated {np.sum(kinds=='terminated')}, truncated {np.sum(kinds=='truncated')} | 길이 평균 {lengths.mean():.0f} 스텝 ({lengths.mean()*env.dt:.1f} s), 최소 {lengths.min()}, 최대 {lengths.max()}")
if frames:
    Image.fromarray(np.concatenate(frames, axis=1)).save(OUT / "ch10-fig08-termination-frames.png")

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(lengths[kinds == "terminated"] * env.dt, bins=20, range=(0, 10), color="#d62728", alpha=0.8, label="terminated (fell)")
ax.hist(lengths[kinds == "truncated"] * env.dt, bins=20, range=(0, 10), color="#4a90d9", alpha=0.8, label="truncated (time limit)")
ax.set_xlabel("episode length [s]"); ax.set_ylabel("episodes"); ax.set_title("random policy (0.4 rad): how episodes end", fontsize=10); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch10-fig07-episode-lengths.png", dpi=200); plt.close(fig)

# (c) 리셋 무작위화
env = QuadrupedEnv()
xs, phases = [], []
for seed in range(20):
    obs, _ = env.reset(seed=seed); phases.append(env.gen.osc.theta.copy())
    q0 = env.data.qpos[7:19].copy()
    while True:
        obs, r, term, trunc, info = env.step(np.zeros(12))
        if term or trunc: xs.append(info["x"]); break
phases = np.array(phases)
print(f"(c) 행동 0, 리셋 20회: 초기 CPG 위상 LF {phases[:,0].min():.2f}~{phases[:,0].max():.2f}, 초기 관절 오차 ±{env.init_noise} rad | 10초 전진 {np.mean(xs):.2f} m (σ {np.std(xs):.2f}, 최소 {np.min(xs):.2f}, 최대 {np.max(xs):.2f})")
env2 = QuadrupedEnv(random_init_phase=False, init_noise=0.0); xs2 = []
for seed in range(5):
    env2.reset(seed=seed)
    while True:
        obs, r, term, trunc, info = env2.step(np.zeros(12))
        if term or trunc: xs2.append(info["x"]); break
print(f"    무작위화 없이 5회: 전진 {np.round(xs2, 3)} (전부 같다: 결정론적 시뮬레이션)")
print("saved", OUT / "ch10-fig07-episode-lengths.png", OUT / "ch10-fig08-termination-frames.png")
