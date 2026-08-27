# labs/ch04-mujoco-first-steps/02_viewer.py
"""viewer로 세계 만나기: 상자 여러 개를 떨어뜨리고 창을 띄운다.

macOS에서는 launch_passive를 쓰려면 python 대신 mjpython으로 실행해야 한다.
    uv run mjpython ch04-mujoco-first-steps/02_viewer.py
Linux/Windows:
    uv run python ch04-mujoco-first-steps/02_viewer.py
--speed 0.25 처럼 배속을 줄 수 있다(1/4 슬로모션).
창 안에서 마우스 드래그로 시점을 돌리고, 더블클릭으로 물체를 고르고,
Ctrl(맥은 Cmd)+드래그로 물체를 잡아 끌 수 있다. 스페이스로 일시정지.
"""

import mujoco
from quadbook.render import play

XML = """
<mujoco model="viewer_demo">
  <option timestep="0.002"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.85 0.85 0.85" rgb2="0.65 0.65 0.65" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="2 2 0.1" material="grid"/>
    <body name="box1" pos="0 0 0.6"><freejoint/><geom type="box" size="0.1 0.1 0.1" mass="1" rgba="0.2 0.5 0.9 1"/></body>
    <body name="box2" pos="0.05 0.02 1.1"><freejoint/><geom type="box" size="0.08 0.08 0.08" mass="0.5" rgba="0.9 0.4 0.2 1"/></body>
    <body name="ball" pos="-0.3 0.2 0.8"><freejoint/><geom type="sphere" size="0.07" mass="0.3" rgba="0.3 0.8 0.3 1"/></body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

# 벽시계에 맞춰 실시간으로 재생한다. --speed 0.25 를 붙이면 4배 느리게 본다.
play(model, data, duration=30.0, title="상자 두 개와 공")
