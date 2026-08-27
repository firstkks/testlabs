# labs/ch05-mjcf-modeling/02_one_leg.py
"""모델 v1: 다리 하나. 세 관절(벌림, 엉덩이, 무릎)을 세계에 고정한 시험대에 달고,
관절 각도를 훑으며 발끝이 닿을 수 있는 영역(작업 공간)을 그린다.

결과 그림: out/ch05-fig03-leg-workspace.png
실행: uv run python ch05-mjcf-modeling/02_one_leg.py
--view 를 붙이면 같은 장면을 viewer 창에서 실시간으로 본다 (macOS: uv run mjpython ... --view).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # 화면 없이 파일로만 그린다 (mjpython 작업 스레드에서도 동작)
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook.render import play, poses, want_view

matplotlib.rcParams["font.family"] = "DejaVu Sans"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

XML = """
<mujoco model="v1_one_leg">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <default>
    <joint damping="0.2" armature="0.01"/>
    <default class="leg"><geom type="capsule" rgba="0.55 0.55 0.6 1"/></default>
  </default>
  <asset><texture type="skybox" builtin="gradient" rgb1="0.95 0.95 1" rgb2="0.6 0.7 0.9" width="32" height="512"/></asset>
  <worldbody>
    <light pos="1 1 3" dir="-0.3 -0.3 -1"/>
    <camera name="side" pos="0.75 -0.75 0.3" xyaxes="0.7 0.7 0 0 0 1"/>
    <geom type="plane" size="1 1 0.1" rgba="0.9 0.9 0.9 1"/>
    <!-- 시험대: 엉덩이를 세계에 고정한다 (freejoint 없음) -->
    <body name="hip_mount" pos="0 0 0.5">
      <geom type="box" size="0.03 0.03 0.03" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>
      <body name="hip" pos="0 0 0">
        <joint name="abduction" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
        <geom class="leg" fromto="0 0 0 0 0.04 0" size="0.03" mass="0.2"/>
        <body name="thigh" pos="0 0.04 0">
          <joint name="hip" type="hinge" axis="0 1 0" range="-1.2 2.4"/>
          <geom class="leg" fromto="0 0 0 0 0 -0.2" size="0.025" mass="0.6"/>
          <body name="calf" pos="0 0 -0.2">
            <joint name="knee" type="hinge" axis="0 1 0" range="-2.6 -0.3"/>
            <geom class="leg" fromto="0 0 0 0 0 -0.2" size="0.018" mass="0.3"/>
            <geom name="foot" type="sphere" pos="0 0 -0.2" size="0.02" mass="0.05" rgba="0.15 0.15 0.15 1"/>
            <site name="foot_site" pos="0 0 -0.2" size="0.01"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
print(f"njnt={model.njnt}, nq={model.nq}, nv={model.nv} (freejoint가 없으니 관절 수와 같다)")
for j in ("abduction", "hip", "knee"):
    print(f"  {j:9s} range = {np.round(model.joint(j).range, 2)} rad")

# 엉덩이·무릎 각도를 훑으며 발끝의 (x, z) 위치를 기록한다 (벌림은 0)
hip_range = np.linspace(*model.joint("hip").range, 60)
knee_range = np.linspace(*model.joint("knee").range, 60)
pts = []
for h in hip_range:
    for k in knee_range:
        data.qpos[:] = [0.0, h, k]
        mujoco.mj_forward(model, data)
        p = data.site("foot_site").xpos - data.body("hip_mount").xpos
        pts.append((p[0], p[2]))
pts = np.array(pts)

# 서 있는 자세(0.8, -1.5)의 발 위치
data.qpos[:] = [0.0, 0.8, -1.5]
mujoco.mj_forward(model, data)
stand = data.site("foot_site").xpos - data.body("hip_mount").xpos
print(f"서기 자세(hip 0.8, knee -1.5)의 발끝: x={stand[0]:+.3f} m, z={stand[2]:+.3f} m (엉덩이 기준)")

# 세 자세를 렌더링: 곧게 편 자세, 서기 자세, 무릎을 깊게 접은 자세
print("saved", poses(model, data, [[0, 0, -0.3], [0, 0.8, -1.5], [0.5, 1.6, -2.4]], OUT / "ch05-fig02-one-leg-poses.png",
                      camera="side", labels=["(0, 0, -0.3)", "standing (0, 0.8, -1.5)", "(0.5, 1.6, -2.4)"], width=600, height=600))

fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.scatter(pts[:, 0], pts[:, 1], s=4, color="#9ecae1", label="reachable foot positions")
ax.plot(0, 0, "ks", ms=8, label="hip joint")
ax.plot(stand[0], stand[2], "ro", ms=9, label="standing pose (0.8, -1.5)")
ax.add_patch(plt.Circle((0, 0), 0.4, fill=False, ls="--", color="#999999"))
ax.set_aspect("equal")
ax.set_xlabel("x forward [m]")
ax.set_ylabel("z up [m]")
ax.set_title("foot workspace, thigh 0.2 m + calf 0.2 m")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "ch05-fig03-leg-workspace.png", dpi=200)
print("saved", OUT / "ch05-fig03-leg-workspace.png")

if want_view():
    # 물리 대신 운동학만: 엉덩이·무릎·벌림을 사인파로 천천히 훑으며 작업 공간을 눈으로 본다
    d = mujoco.MjData(model)
    def _step():
        d.time += model.opt.timestep
        t = d.time
        d.qpos[:] = [0.4 * np.sin(0.5 * t), 0.8 + 0.8 * np.sin(t), -1.5 + 0.9 * np.sin(1.3 * t)]
        mujoco.mj_forward(model, d)
    play(model, d, step_fn=_step, duration=20.0, title="다리 하나: 관절 훑기 (운동학만)")
