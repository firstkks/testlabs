# labs/ch07-first-gait/03_trot.py
"""7.3 trot 구현: 대각선 다리의 동기화.

GaitGenerator("trot")로 평지를 8초 걷는다. 리듬 생성기는 100 Hz(control_every 5), PD는 500 Hz.
세 단계: (1) 6장의 서기 제어기 그대로(kp 40, 목표 각속도 0) → (2) 목표 각속도 피드포워드 → (3) kp 80 + 피드포워드.
지표: 평균 속도(이상값 L·f/β와 비교), 관절 추종 오차, 최대 토크, roll·pitch 흔들림, 넘어짐 여부.
그림: out/ch07-fig05-trot-frames.png, out/ch07-fig06-trot-traces.png
실행: uv run python ch07-first-gait/03_trot.py   (--view --speed 0.5 권장)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook import model_path
from quadbook.cpg import GaitGenerator
from quadbook.render import play, snapshots, track_camera, want_view
from quadbook.robot import load
from quadbook.sim import WALK_KP, rollout, standard_pd, summary

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
CONTROL_EVERY = 5


def make(kp=WALK_KP, **kw):
    model, data = load(model_path())
    pd = standard_pd(model, data, kp=kp)
    gen = GaitGenerator("trot", freq=2.0, duty=0.5, step_length=0.08, step_height=0.04, **kw)
    return model, data, pd, gen


print("trot 2 Hz, duty 0.5, 보폭 0.08 m, 발 높이 0.04 m | 이상 속도 L·f/β = 0.320 m/s")
print("  단계                       속도[m/s]  이상 대비  추종 오차  최대 토크  roll σ  pitch σ")
for name, kp, ff in (("(1) kp 40, 각속도 목표 0", 40.0, False), ("(2) kp 40 + 각속도 피드포워드", 40.0, True), ("(3) kp 80 + 각속도 피드포워드", 80.0, True)):
    model, data, pd, gen = make(kp=kp)
    log = rollout(model, data, gen, pd, duration=8.0, control_every=CONTROL_EVERY, feedforward=ff)
    s = summary(log)
    print(f"  {name:26s}  {s['speed']:.3f}     {100*s['speed']/gen.ideal_speed:3.0f}%     {s['track_err']:4.1f}°    {s['tau_max']:5.1f} N·m  {s['roll_std']:.2f}°   {s['pitch_std']:.2f}°" + ("   넘어짐" if s["fell"] else ""))
c = log["contact"][log["t"] >= 2.0]
print(f"  (3)의 이동 거리 {log['x'][-1]-log['x'][0]:.2f} m, 몸통 높이 {s['height']:.3f} m, 실측 발 접촉 비율(LF RF LH RH) {np.round(c.mean(axis=0), 2)} (명령 duty 0.5)")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].plot(log["t"], log["x"] - log["x"][0], label="torso x (measured)")
axes[0].plot(log["t"], np.maximum(0, log["t"] - 0.5) * gen.ideal_speed, ls="--", color="#999999", label=f"ideal L f / beta = {gen.ideal_speed:.2f} m/s")
axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("distance [m]"); axes[0].set_title("(a) forward progress", fontsize=10); axes[0].legend(fontsize=8)
axes[1].plot(log["t"], np.degrees(log["pitch"]), label="pitch"); axes[1].plot(log["t"], np.degrees(log["roll"]), label="roll")
ax2 = axes[1].twinx(); ax2.plot(log["t"], log["z"], color="#2ca02c", lw=0.8, label="height"); ax2.set_ylabel("torso height [m]", color="#2ca02c")
axes[1].set_xlabel("time [s]"); axes[1].set_ylabel("angle [deg]"); axes[1].set_title("(b) torso attitude and height", fontsize=10); axes[1].legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "ch07-fig06-trot-traces.png", dpi=200); plt.close(fig)

# 프레임: 한 주기(0.5 s)를 네 장으로, 몸통을 따라가는 카메라
model, data, pd, gen = make()
cam = track_camera(model, distance=1.3, azimuth=140, elevation=-15)
dt_ctrl = model.opt.timestep * CONTROL_EVERY
state = {"i": 0, "q": gen.targets_at(gen.osc.theta), "qd": np.zeros(12)}
def _step():
    if data.time >= 0.5 and state["i"] % CONTROL_EVERY == 0:
        state["q"], state["qd"] = gen.targets_with_velocity(dt_ctrl)
    data.ctrl[:] = pd.torque(data, state["q"], state["qd"]); mujoco.mj_step(model, data); state["i"] += 1
print("saved", snapshots(model, data, (3.0, 3.125, 3.25, 3.375), OUT / "ch07-fig05-trot-frames.png", camera=cam, step_fn=_step, width=600, height=450))
print("saved", OUT / "ch07-fig06-trot-traces.png")

if want_view():
    model, data, pd, gen = make()
    state = {"i": 0, "q": gen.targets_at(gen.osc.theta), "qd": np.zeros(12)}
    play(model, data, step_fn=_step, duration=10.0, title="trot 2 Hz (개루프 CPG + 관절 PD kp 80 + 속도 피드포워드)")
