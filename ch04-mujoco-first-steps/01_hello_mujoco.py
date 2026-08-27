# labs/ch04-mujoco-first-steps/01_hello_mujoco.py
"""MuJoCo 첫 실행: 상자 하나를 바닥에 떨어뜨린다.

모델을 XML 문자열로 만들고(mjModel), 상태 그릇(mjData)을 준비한 뒤,
mj_step을 반복해 1초를 시뮬레이션한다. 화면은 띄우지 않는다.
실행: uv run python ch04-mujoco-first-steps/01_hello_mujoco.py
--view 를 붙이면 같은 장면을 viewer 창에서 실시간으로 본다 (macOS: uv run mjpython ... --view).
"""

from pathlib import Path

import mujoco
from quadbook.render import play, snapshots, want_view

XML = """
<mujoco model="hello">
  <option timestep="0.002"/>
  <asset><texture type="skybox" builtin="gradient" rgb1="0.95 0.95 1" rgb2="0.6 0.7 0.9" width="32" height="512"/></asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <camera name="side" pos="2.2 -2.2 1.1" xyaxes="0.7 0.7 0 -0.25 0.25 0.94"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="box" pos="0 0 1.0">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" mass="1" rgba="0.2 0.5 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)   # 상수: 모델의 설명
data = mujoco.MjData(model)                    # 변수: 시간에 따라 바뀌는 상태

print(f"MuJoCo {mujoco.__version__}")
print(f"nq={model.nq} (위치 좌표 수), nv={model.nv} (속도 좌표 수), "
      f"nbody={model.nbody}, ngeom={model.ngeom}, timestep={model.opt.timestep}")

# 1초 = 500스텝. 매 0.1초마다 상자 높이와 접촉 개수를 찍는다.
steps_per_report = int(0.1 / model.opt.timestep)
for i in range(int(1.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)
    if (i + 1) % steps_per_report == 0:
        z = data.body("box").xpos[2]
        print(f"t={data.time:4.2f}s  box z={z:6.3f} m  contacts={data.ncon}")

# 같은 장면을 처음부터 다시 돌리며 세 순간을 그림으로 남긴다
data = mujoco.MjData(model)
out = snapshots(model, data, (0.05, 0.3, 0.6), Path(__file__).resolve().parent / "out" / "ch04-fig01-hello-drop.png", camera="side")
print("saved", out)

if want_view():
    play(model, mujoco.MjData(model), duration=3.0, title="상자 낙하")
