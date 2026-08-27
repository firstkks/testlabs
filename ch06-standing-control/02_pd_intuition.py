# labs/ch06-standing-control/02_pd_intuition.py
"""6.2 PD 제어의 원리를 관절 하나로 확인한다.

같은 단진자(4장)를 세 제어기로 60도에 보낸다.
  P만   : 토크 = kp·(목표 − 각도)                → 스프링만 있는 셈이라 계속 출렁인다
  PD    : 토크 = kp·(목표 − 각도) − kd·각속도      → 댐퍼가 출렁임을 흡수한다. 그러나 중력 때문에 목표에 못 미친다
  PD+중력 보상 : 위 토크 + 중력 토크(qfrc_bias)     → 오차가 사라진다
그리고 관절 관성 I에서 고유진동수 ω = sqrt(kp/I), 감쇠비 ζ = kd/(2·sqrt(kp·I))를 계산해 응답과 대조한다.
그림: out/ch06-fig04-pd-terms.png, out/ch06-fig05-pd-frames.png
실행: uv run python ch06-standing-control/02_pd_intuition.py
--view 를 붙이면 PD+중력 보상 제어를 viewer에서 본다.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook.render import play, snapshots, want_view

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

XML = """
<mujoco model="pendulum_pd">
  <option timestep="0.002" integrator="implicitfast"/>
  <asset><texture type="skybox" builtin="gradient" rgb1="0.95 0.95 1" rgb2="0.6 0.7 0.9" width="32" height="512"/></asset>
  <worldbody>
    <light pos="0 -1 3" dir="0 0.3 -1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1"/>
    <camera name="front" pos="0 -2.0 0.7" xyaxes="1 0 0 0 0 1"/>
    <geom type="cylinder" fromto="0 0.15 0 0 0.15 1" size="0.015" rgba="0.35 0.35 0.35 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" fromto="0 0.15 1 0 0 1" size="0.012" rgba="0.35 0.35 0.35 1" contype="0" conaffinity="0"/>
    <geom type="sphere" pos="0 0 1" size="0.03" rgba="0.8 0.2 0.2 1" contype="0" conaffinity="0"/>
    <body pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.0" armature="0.0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" mass="1"/>
    </body>
  </worldbody>
  <actuator><motor name="act" joint="hinge" ctrlrange="-30 30"/></actuator>
</mujoco>
"""
TARGET = np.deg2rad(60)
KP, KD = 20.0, 1.0


def simulate(controller, duration=3.0):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    t, q = [], []
    while data.time < duration:
        data.ctrl[0] = controller(data)
        mujoco.mj_step(model, data)
        t.append(data.time); q.append(data.qpos[0])
    return model, data, np.array(t), np.array(q)


def p_only(data):
    return KP * (TARGET - data.qpos[0])


def pd(data):
    return KP * (TARGET - data.qpos[0]) - KD * data.qvel[0]


def pd_gravity(data):
    return KP * (TARGET - data.qpos[0]) - KD * data.qvel[0] + data.qfrc_bias[0]   # 중력 보상: MuJoCo가 계산한 편향력


runs = {"P only": p_only, "PD": pd, "PD + gravity comp": pd_gravity}
fig, ax = plt.subplots(figsize=(7.5, 3.8))
for name, ctrl in runs.items():
    model, data, t, q = simulate(ctrl)
    ax.plot(t, np.degrees(q), label=name)
    print(f"{name:18s}: 3초 뒤 {np.degrees(q[-1]):6.2f}° (목표 60°), 최대 {np.degrees(q.max()):6.2f}°")
ax.axhline(60, color="#999999", ls=":", lw=1)
ax.set_xlabel("time [s]"); ax.set_ylabel("angle [deg]"); ax.set_title(f"kp = {KP}, kd = {KD}: what each term does"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch06-fig04-pd-terms.png", dpi=200); plt.close(fig)

# ---- 2차계 예측: 관절 관성 I 에서 ω, ζ ----
model = mujoco.MjModel.from_xml_string(XML); data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
M = np.zeros((1, 1)); mujoco.mj_fullM(model, data, M); I = M[0, 0]
omega = np.sqrt(KP / I); zeta = KD / (2 * np.sqrt(KP * I))
period = 2 * np.pi / (omega * np.sqrt(1 - zeta**2))
overshoot = np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2)) * 100
print(f"관절 관성 I = {I:.4f} kg·m² (막대 m l²/3 = {1*0.5**2/3:.4f}) | ω = {omega:.2f} rad/s | ζ = {zeta:.2f} | 감쇠 진동 주기 {period:.2f} s | 예측 overshoot {overshoot:.0f}%")
# 측정: PD+중력 보상 응답의 첫 봉우리
model, data, t, q = simulate(pd_gravity)
peak_i = np.argmax(q); print(f"측정: 첫 봉우리 {np.degrees(q[peak_i]):.1f}° at {t[peak_i]:.2f} s → overshoot {(q[peak_i]-TARGET)/TARGET*100:.0f}%")
kd_crit = 2 * np.sqrt(KP * I)
print(f"임계 감쇠 kd = 2·sqrt(kp·I) = {kd_crit:.2f} (지금 kd {KD}는 ζ {zeta:.2f}의 저감쇠)")

model = mujoco.MjModel.from_xml_string(XML); data = mujoco.MjData(model)
def _step():
    data.ctrl[0] = pd_gravity(data); mujoco.mj_step(model, data)
print("saved", snapshots(model, data, (0.0, 0.2, 0.5, 3.0), OUT / "ch06-fig05-pd-frames.png", camera="front", step_fn=_step, width=600, height=500))

if want_view():
    model = mujoco.MjModel.from_xml_string(XML); data = mujoco.MjData(model)
    def _step2():
        data.ctrl[0] = pd_gravity(data); mujoco.mj_step(model, data)
    play(model, data, step_fn=_step2, duration=5.0, title="PD + 중력 보상 (목표 60도)")
