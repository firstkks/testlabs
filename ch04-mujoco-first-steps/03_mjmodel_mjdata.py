# labs/ch04-mujoco-first-steps/03_mjmodel_mjdata.py
"""mjModel과 mjData의 분업을 눈으로 확인한다.

두 관절짜리 팔(어깨 hinge + 팔꿈치 hinge)을 만들고,
- mjModel에는 무엇이 들어 있고(상수), mjData에는 무엇이 들어 있는지(변수) 찍어 본다.
- qpos를 손으로 바꾼 뒤 mj_forward를 부르면 xpos(월드 좌표)가 따라 바뀌는 것을 본다.
- mj_forward(계산만)와 mj_step(계산 + 시간 전진)의 차이를 본다.
실행: uv run python ch04-mujoco-first-steps/03_mjmodel_mjdata.py
--view 를 붙이면 같은 장면을 viewer 창에서 실시간으로 본다 (macOS: uv run mjpython ... --view).
"""

import numpy as np

from pathlib import Path

import mujoco
from quadbook.render import play, poses, want_view

XML = """
<mujoco model="two_link_arm">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <asset><texture type="skybox" builtin="gradient" rgb1="0.95 0.95 1" rgb2="0.6 0.7 0.9" width="32" height="512"/></asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <camera name="front" pos="0.3 -1.6 0.6" xyaxes="1 0 0 0 0.2 1"/>
    <geom type="plane" size="1 1 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.5">
      <geom type="cylinder" size="0.05 0.02" rgba="0.3 0.3 0.3 1"/>
      <body name="upper" pos="0 0 0">
        <joint name="shoulder" type="hinge" axis="0 1 0" damping="0.5"/>
        <geom name="upper_geom" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/>
        <body name="lower" pos="0.3 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" damping="0.5"/>
          <geom name="lower_geom" type="capsule" fromto="0 0 0 0.25 0 0" size="0.025" mass="0.6"/>
          <site name="tip" pos="0.25 0 0" size="0.01"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="shoulder_motor" joint="shoulder" gear="1" ctrlrange="-5 5"/>
    <motor name="elbow_motor" joint="elbow" gear="1" ctrlrange="-5 5"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

print("=== mjModel: 컴파일된 상수 ===")
print(f"nq={model.nq}, nv={model.nv}, nu={model.nu} (actuator 수), nbody={model.nbody}, njnt={model.njnt}")
print("body 이름:", [model.body(i).name for i in range(model.nbody)])
print("joint 이름:", [model.joint(i).name for i in range(model.njnt)])
print("timestep:", model.opt.timestep, "| gravity:", model.opt.gravity)
print("upper_geom 질량:", model.body("upper").mass[0], "kg | lower 질량:", model.body("lower").mass[0], "kg")

print("\n=== mjData: 시간에 따라 바뀌는 상태 ===")
print("초기 qpos:", data.qpos, "| qvel:", data.qvel, "| time:", data.time)
print("tip 월드 위치 (아직 계산 전):", data.site("tip").xpos)

# qpos를 손으로 넣고 mj_forward: 시간은 흐르지 않고 파생량(월드 좌표 등)만 계산된다
data.qpos[:] = [np.deg2rad(45), np.deg2rad(-60)]
mujoco.mj_forward(model, data)
print("qpos를 (45°, -60°)로 놓고 mj_forward →  tip:", np.round(data.site("tip").xpos, 3), "| time:", data.time)

# mj_step: 힘을 계산하고 시간을 한 칸 전진시킨다. 중력에 팔이 처지기 시작한다.
for _ in range(500):          # 1초
    mujoco.mj_step(model, data)
print("1초 자유 낙하 후 qpos(deg):", np.round(np.rad2deg(data.qpos), 1), "| time:", round(data.time, 3))

# actuator에 힘을 넣어 본다: ctrl은 mjData에 있다 (모델이 아니라 상태이므로)
data.ctrl[:] = [3.0, 0.0]     # 어깨에 3 N·m
for _ in range(500):
    mujoco.mj_step(model, data)
print("어깨 토크 3 N·m 1초 후 qpos(deg):", np.round(np.rad2deg(data.qpos), 1))

# 같은 mjModel로 두 번째 세계를 만들 수 있다: 상수와 상태의 분리가 주는 자유
data2 = mujoco.MjData(model)
print("\n같은 model, 새 data2의 time:", data2.time, "| 기존 data의 time:", round(data.time, 3))

# 세 자세를 그림으로: 초기(0, 0), 손으로 넣은 (45°, -60°), 토크를 준 뒤의 자세
final_q = data.qpos.copy()
out = poses(model, data2, [[0, 0], [np.deg2rad(45), np.deg2rad(-60)], final_q],
            Path(__file__).resolve().parent / "out" / "ch04-fig03-two-link-poses.png", camera="front",
            labels=["qpos = (0, 0)", "qpos = (45, -60) deg, mj_forward", "after 3 N.m for 1 s"])
print("saved", out)

if want_view():
    d = mujoco.MjData(model)
    d.qpos[:] = [np.deg2rad(45), np.deg2rad(-60)]
    def _step():
        d.ctrl[:] = [3.0, 0.0] if d.time >= 1.0 else [0.0, 0.0]   # 1초 뒤 어깨에 3 N·m
        mujoco.mj_step(model, d)
    play(model, d, step_fn=_step, duration=4.0, title="두 관절 팔: 1초 뒤 어깨 토크")
