# labs/ch10-gym-environment/01_env_interface.py
"""10.1 gym.Env 인터페이스 이해.

QuadrupedEnv를 만들고 관측·행동 공간을 확인하고 Gymnasium의 env_checker를 통과시킨다.
행동 0(보정 없음)으로 한 에피소드를 돌리면 7장의 trot 그대로여야 한다. 난수 행동과 비교한다.
환경의 처리량(초당 제어 스텝)을 재서 11장의 학습 예산을 가늠한다.
그림: out/ch10-fig01-step-timeline.png, out/ch10-fig02-zero-vs-random.png
실행: uv run python ch10-gym-environment/01_env_interface.py   (--view --speed 0.5: 행동 0 에피소드)
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import quadbook.render as R
from quadbook.env import QuadrupedEnv
from quadbook.render import play, want_view

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

env = QuadrupedEnv(render_mode="rgb_array")
print("observation_space:", env.observation_space)
print("action_space     :", env.action_space)
print(f"제어 주기 dt = {env.dt} s (물리 {env.model.opt.timestep} s × {env.control_every}), 에피소드 최대 {env.max_steps} 스텝 = {env.max_steps*env.dt:.0f} s")
check_env(env, skip_render_check=True)
print("gymnasium env_checker: 통과")


def run(env, policy, seed=0, frames_at=()):
    obs, info = env.reset(seed=seed)
    total, frames, n = 0.0, [], 0
    while True:
        a = policy(obs)
        obs, r, term, trunc, info = env.step(a)
        total += r; n += 1
        if any(abs(info["t"] - ft) < env.dt / 2 for ft in frames_at):
            frames.append(R._label(env.render(), f"t = {info['t']:.2f} s  x = {info['x']:+.2f} m"))
        if term or trunc:
            break
    return {"steps": n, "return": total, "x": info["x"], "terminated": term, "truncated": trunc, "frames": frames}


zero = run(env, lambda o: np.zeros(12), frames_at=(2.0, 6.0))
rng = np.random.default_rng(0)
rand = run(env, lambda o: rng.uniform(-1, 1, 12), frames_at=(2.0, 6.0))
for name, res in (("행동 0 (보정 없음 = 7장 trot)", zero), ("난수 행동 (±0.1 rad)", rand)):
    print(f"{name}: {res['steps']} 스텝, 전진 {res['x']:+.2f} m, 누적 보상 {res['return']:.1f}, terminated {res['terminated']}, truncated {res['truncated']}")
Image.fromarray(np.concatenate(zero["frames"] + rand["frames"], axis=1)).save(OUT / "ch10-fig02-zero-vs-random.png")

# 처리량
obs, _ = env.reset(seed=1); t0 = time.perf_counter(); n = 0
while n < 2000:
    obs, r, term, trunc, _ = env.step(np.zeros(12)); n += 1
    if term or trunc: env.reset()
el = time.perf_counter() - t0
print(f"처리량: 제어 스텝 {n/el:,.0f} /s (시뮬레이션 시간 {n*env.dt/el:.0f}배속). 100만 스텝이면 약 {1e6/(n/el)/60:.0f}분")

# 도해: step() 한 번의 시간축
fig, ax = plt.subplots(figsize=(9, 2.6)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
def box(x, y, w, h, text, fc="#eef3fb", ec="#4a6fa5", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec, lw=1)); ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
box(0.01, 0.55, 0.15, 0.3, "action a\n(12, [-1, 1])")
box(0.19, 0.55, 0.2, 0.3, "CPG target q_cpg(t)\n+ 0.1 * a", fc="#f3f7ee", ec="#5a8a3a")
for k in range(5):
    box(0.42 + k * 0.085, 0.55, 0.075, 0.3, f"PD\nmj_step", fc="#f8f0e8", ec="#a0703a", fs=7)
box(0.86, 0.55, 0.13, 0.3, "obs, reward,\nterminated?")
ax.annotate("", (0.19, 0.7), (0.16, 0.7), arrowprops=dict(arrowstyle="->")); ax.annotate("", (0.42, 0.7), (0.39, 0.7), arrowprops=dict(arrowstyle="->")); ax.annotate("", (0.86, 0.7), (0.845, 0.7), arrowprops=dict(arrowstyle="->"))
ax.text(0.63, 0.42, "5 physics substeps x 0.002 s = one control step of 0.01 s (100 Hz)", ha="center", fontsize=8, color="#555555")
ax.text(0.5, 0.12, "env.step(a): the policy acts at 100 Hz, the joint PD and physics run at 500 Hz", ha="center", fontsize=9)
fig.savefig(OUT / "ch10-fig01-step-timeline.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("saved", OUT / "ch10-fig01-step-timeline.png", OUT / "ch10-fig02-zero-vs-random.png")
env.close()

if want_view():
    env = QuadrupedEnv(); env.reset(seed=0)
    def _step():
        env.step(np.zeros(12))
    play(env.model, env.data, step_fn=lambda: env.step(np.zeros(12)), duration=10.0, title="QuadrupedEnv, 행동 0: 7장의 trot 그대로")
