# labs/ch06-standing-control/01_torque_vs_position.py
"""6.1 토크 제어와 위치 제어.

실험 A) 시험대 다리의 엉덩이 관절에 일정한 토크를 넣는다. 피드백이 없으면 관절은 목표 각도를 '모른다':
        중력 토크와 균형이 맞는 각도에서 멈출 뿐이다. 토크를 끊으면 떨어진다.
실험 B) 같은 관절을 position 액추에이터(내장 PD)로 0.8 rad에 보낸다. 피드백이 있으니 목표에 간다.
실험 C) 완성 로봇을 0.4 m에서 떨어뜨린다. 딱딱한 PD(kp 400)와 순응하는 PD(kp 40)의 착지 충격을 비교한다.
그림: out/ch06-fig01-torque-vs-position.png, out/ch06-fig02-landing-frames.png, out/ch06-fig03-landing-stiffness.png
실행: uv run python ch06-standing-control/01_torque_vs_position.py
--view 를 붙이면 실험 C의 순응 착지를 viewer에서 본다 (macOS: uv run mjpython ... --view --speed 0.5).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook import model_path
from quadbook.control import JointPD, critical_kd, joint_inertia_diag
from quadbook.render import play, snapshots, want_view
from quadbook.robot import STAND_POSE, TOUCH_SENSORS, load

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

LEG_XML = """
<mujoco model="leg_rig">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <asset><texture type="skybox" builtin="gradient" rgb1="0.95 0.95 1" rgb2="0.6 0.7 0.9" width="32" height="512"/></asset>
  <default><joint damping="0.2" armature="0.01"/><default class="leg"><geom type="capsule" rgba="0.55 0.55 0.6 1"/></default></default>
  <worldbody>
    <light pos="1 1 3" dir="-0.3 -0.3 -1"/>
    <camera name="side" pos="0.75 -0.75 0.3" xyaxes="0.7 0.7 0 0 0 1"/>
    <geom type="plane" size="1 1 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="hip_mount" pos="0 0 0.5">
      <geom type="box" size="0.03 0.03 0.03" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>
      <body name="thigh" pos="0 0.04 0">
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.2 2.4"/>
        <geom class="leg" fromto="0 0 0 0 0 -0.2" size="0.025" mass="0.6"/>
        <body name="calf" pos="0 0 -0.2">
          <joint name="knee" type="hinge" axis="0 1 0" range="-2.6 -0.3"/>
          <geom class="leg" fromto="0 0 0 0 0 -0.2" size="0.018" mass="0.3"/>
          <geom name="foot" type="sphere" pos="0 0 -0.2" size="0.02" mass="0.05" rgba="0.15 0.15 0.15 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    {hip_actuator}
    <position name="knee_pos" joint="knee" kp="30" kv="1"/>
  </actuator>
</mujoco>
"""


def run_leg(hip_actuator, hip_cmd, duration=3.0):
    """hip_cmd(t)가 돌려주는 값을 엉덩이 액추에이터에 넣고 무릎은 -1.5 rad로 잡는다."""
    model = mujoco.MjModel.from_xml_string(LEG_XML.format(hip_actuator=hip_actuator))
    data = mujoco.MjData(model)
    t, q = [], []
    while data.time < duration:
        data.ctrl[0] = hip_cmd(data.time)
        data.ctrl[1] = -1.5
        mujoco.mj_step(model, data)
        t.append(data.time)
        q.append(data.qpos[0])
    return np.array(t), np.array(q)


# ---- 실험 A: 토크만 (motor) ----
TORQUE = 1.0
tA, qA = run_leg('<motor name="hip_motor" joint="hip" ctrlrange="-12 12"/>', lambda t: TORQUE if t < 2.0 else 0.0)
# 균형 각도 예측: 각도 theta에서 중력이 엉덩이에 거는 토크를 MuJoCo에 물어(qfrc_bias) 모터 토크와 같아지는 theta를 찾는다
m_rig = mujoco.MjModel.from_xml_string(LEG_XML.format(hip_actuator='<motor name="hip_motor" joint="hip" ctrlrange="-12 12"/>'))
d_rig = mujoco.MjData(m_rig)
thetas = np.linspace(0, np.pi / 2, 900)
grav = []
for th in thetas:
    d_rig.qpos[:] = [th, -1.5]; d_rig.qvel[:] = 0
    mujoco.mj_forward(m_rig, d_rig)
    grav.append(d_rig.qfrc_bias[0])            # 정지 상태의 편향력 = 모터가 이겨야 할 중력 토크
grav = np.array(grav)
theta_pred = thetas[np.argmin(np.abs(grav - TORQUE))]   # 편향력이 모터 토크와 같아지는 각도
theta_meas = qA[(tA > 1.5) & (tA < 2.0)].mean()
print(f"[A] 토크 {TORQUE} N·m: 예측 균형 각도 {np.degrees(theta_pred):.1f}°, 측정 {np.degrees(theta_meas):.1f}° | 토크를 끊은 뒤 3초 각도 {np.degrees(qA[-1]):.1f}°")

# ---- 실험 B: position 액추에이터 ----
tB, qB = run_leg('<position name="hip_pos" joint="hip" kp="20" kv="1" ctrlrange="-3 3"/>', lambda t: 0.8 if t < 2.0 else 0.0)
print(f"[B] position(kp 20): 2초 시점 각도 {np.degrees(qB[(tB > 1.5) & (tB < 2.0)].mean()):.1f}° (목표 45.8°), 목표를 0으로 바꾼 뒤 3초 {np.degrees(qB[-1]):.1f}°")

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(tA, np.degrees(qA), label="(A) motor: constant 1 N.m for 2 s, then 0")
ax.plot(tB, np.degrees(qB), label="(B) position: target 0.8 rad for 2 s, then 0")
ax.axhline(np.degrees(theta_pred), color="#999999", ls=":", lw=1, label="gravity balance angle (predicted)")
ax.axhline(np.degrees(0.8), color="#bbbbbb", ls="--", lw=1)
ax.set_xlabel("time [s]"); ax.set_ylabel("hip angle [deg]"); ax.legend(fontsize=8)
ax.set_title("torque without feedback vs position with built-in feedback")
fig.tight_layout(); fig.savefig(OUT / "ch06-fig01-torque-vs-position.png", dpi=200); plt.close(fig)

# ---- 실험 C: 착지 충격, 딱딱한 PD vs 순응하는 PD ----
def drop(kp, height=0.4, duration=1.5):
    model, data = load(model_path())
    kd = critical_kd(kp, joint_inertia_diag(model, data), zeta=1.0)
    pd = JointPD(kp, kd)
    data.qpos[2] = height
    mujoco.mj_forward(model, data)
    t, z, touch, tau = [], [], [], []
    while data.time < duration:
        data.ctrl[:] = pd.torque(data, STAND_POSE)
        mujoco.mj_step(model, data)
        t.append(data.time); z.append(data.body("torso").xpos[2])
        touch.append(sum(data.sensor(n).data[0] for n in TOUCH_SENSORS))
        tau.append(np.abs(data.ctrl).max())
    return model, data, kd, np.array(t), np.array(z), np.array(touch), np.array(tau)


results = {}
for kp in (40.0, 400.0):
    m, d, kd, t, z, touch, tau = drop(kp)
    results[kp] = (t, z, touch, tau)
    print(f"[C] kp={kp:>5.0f}, kd≈{kd[1]:.2f}: 최대 접촉력 합 {touch.max():6.0f} N (몸무게 84 N), 최대 토크 {tau.max():5.1f} N·m (한계 12), 정지 높이 {z[-1]:.3f} m")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for kp, (t, z, touch, tau) in results.items():
    axes[0].plot(t, touch, label=f"kp = {kp:.0f}")
    axes[1].plot(t, z, label=f"kp = {kp:.0f}")
axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("total foot force [N]"); axes[0].set_title("(a) landing impact"); axes[0].legend(fontsize=8)
axes[1].set_xlabel("time [s]"); axes[1].set_ylabel("torso height [m]"); axes[1].set_title("(b) torso height"); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch06-fig03-landing-stiffness.png", dpi=200); plt.close(fig)

# 착지 프레임 (순응 착지)
m, d = load(model_path()); d.qpos[2] = 0.4
kd = critical_kd(40.0, joint_inertia_diag(m, d)); pd = JointPD(40.0, kd)
def _step():
    d.ctrl[:] = pd.torque(d, STAND_POSE); mujoco.mj_step(m, d)
print("saved", snapshots(m, d, (0.0, 0.15, 0.25, 0.6), OUT / "ch06-fig02-landing-frames.png", camera="side", step_fn=_step, width=700, height=500))
print("saved", OUT / "ch06-fig01-torque-vs-position.png", OUT / "ch06-fig03-landing-stiffness.png")

if want_view():
    m, d = load(model_path()); d.qpos[2] = 0.4
    pd = JointPD(40.0, critical_kd(40.0, joint_inertia_diag(m, d)))
    def _step2():
        d.ctrl[:] = pd.torque(d, STAND_POSE); mujoco.mj_step(m, d)
    play(m, d, step_fn=_step2, duration=5.0, title="순응하는 PD(kp 40)로 착지")
