# labs/ch05-mjcf-modeling/01_torso_only.py
"""모델 v0: 몸통 하나. MJCF 트리의 최소 단위와 nq/nv, 질량·관성 자동 계산을 확인한다.

실행: uv run python ch05-mjcf-modeling/01_torso_only.py
--view 를 붙이면 같은 장면을 viewer 창에서 실시간으로 본다 (macOS: uv run mjpython ... --view).
"""

import numpy as np

from pathlib import Path

import mujoco
from quadbook.render import play, snapshots, want_view

XML = """
<mujoco model="v0_torso">
  <option timestep="0.002" integrator="implicitfast"/>
  <asset><texture type="skybox" builtin="gradient" rgb1="0.95 0.95 1" rgb2="0.6 0.7 0.9" width="32" height="512"/></asset>
  <worldbody>
    <light pos="1 1 3" dir="-0.3 -0.3 -1" castshadow="true"/>
    <camera name="side" pos="1.5 -1.5 0.8" xyaxes="0.7 0.7 0 -0.3 0.3 0.9"/>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="torso" pos="0 0 0.5">
      <freejoint name="root"/>
      <geom name="torso_geom" type="box" size="0.2 0.08 0.05" mass="4.0" rgba="0.25 0.35 0.6 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

print(f"nbody={model.nbody} (world 포함), njnt={model.njnt}, nq={model.nq}, nv={model.nv}")
print("torso 질량:", model.body("torso").mass[0], "kg")
print("torso 관성 대각(자동 계산, kg·m²):", np.round(model.body("torso").inertia, 4))
# 상자 관성의 손 계산: I_x = m(b²+c²)/3 등 (반치수 b, c)
m, a, b, c = 4.0, 0.2, 0.08, 0.05
print("손 계산  I_x, I_y, I_z:", np.round([m*(b*b+c*c)/3, m*(a*a+c*c)/3, m*(a*a+b*b)/3], 4))

for _ in range(500):
    mujoco.mj_step(model, data)
print("1초 뒤 torso z:", round(data.body("torso").xpos[2], 3), "m | 접촉:", data.ncon)
data = mujoco.MjData(model)
print("saved", snapshots(model, data, (0.0, 0.25, 0.6), Path(__file__).resolve().parent / "out" / "ch05-fig01-torso-drop.png", camera="side"))

if want_view():
    play(model, mujoco.MjData(model), duration=3.0, title="모델 v0: 몸통 낙하")
