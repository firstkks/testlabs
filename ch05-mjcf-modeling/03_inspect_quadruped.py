# labs/ch05-mjcf-modeling/03_inspect_quadruped.py
"""완성된 로봇(models/quadruped.xml)의 속을 들여다보고, 서 있는 자세를 그림으로 남긴다.

결과 그림: out/ch05-fig04-quadruped-stand.png
실행: uv run python ch05-mjcf-modeling/03_inspect_quadruped.py
--view 를 붙이면 같은 장면을 viewer 창에서 실시간으로 본다 (macOS: uv run mjpython ... --view).
"""

from pathlib import Path

import numpy as np
from PIL import Image

import mujoco
from quadbook import model_path
from quadbook.render import play, want_view
from quadbook.robot import JOINT_NAMES, STAND_POSE, load, pd_torque

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

model, data = load(model_path())

print(f"nbody={model.nbody}, njnt={model.njnt}, nq={model.nq}, nv={model.nv}, nu={model.nu}, nsensor={model.nsensor}")
print("총 질량:", round(float(model.body_subtreemass[model.body('torso').id]), 3), "kg")
print("관절 순서:", JOINT_NAMES)
print("관절 범위(rad):")
for name in JOINT_NAMES[:3]:
    print(f"  {name:14s}", np.round(model.joint(name).range, 2))
print("액추에이터 토크 한계:", model.actuator("LF_hip").ctrlrange, "N·m")
print("센서 데이터 길이:", model.nsensordata, "| 예: torso_quat =", np.round(data.sensor("torso_quat").data, 3))

# 서 있는 자세(keyframe)에서의 몸통 높이와 발 위치
print("몸통 높이:", round(data.body("torso").xpos[2], 3), "m")
for leg in ("LF", "RF", "LH", "RH"):
    p = data.geom(f"{leg}_foot").xpos
    print(f"  {leg} 발: x={p[0]:+.3f} y={p[1]:+.3f} z={p[2]:+.3f}")

# 좌우 대칭 검사: 왼쪽 다리 질량 합 == 오른쪽 다리 질량 합
def subtree_mass(body):
    return float(model.body_subtreemass[model.body(body).id])
print("좌우 질량:", subtree_mass("LF_hip") + subtree_mass("LH_hip"), "vs", subtree_mass("RF_hip") + subtree_mass("RH_hip"))

# 두 카메라로 서 있는 자세를 찍는다
renderer = mujoco.Renderer(model, height=1000, width=1600)
frames = []
for cam in ("side", "front"):
    renderer.update_scene(data, camera=cam)
    frames.append(renderer.render().copy())
Image.fromarray(np.concatenate(frames, axis=1)).save(OUT / "ch05-fig04-quadruped-stand.png")
renderer.close()
print("saved", OUT / "ch05-fig04-quadruped-stand.png")

if want_view():
    # PD로 서기 자세를 붙든 채 창을 띄운다. Cmd(맥)/Ctrl+드래그로 로봇을 잡아 끌어 보자
    m2, d2 = load(model_path())
    def _step():
        d2.ctrl[:] = pd_torque(d2, STAND_POSE); mujoco.mj_step(m2, d2)
    play(m2, d2, step_fn=_step, title="완성 로봇: 서기 자세 (드래그로 밀어 보기)")
