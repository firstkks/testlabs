# labs/ch04-mujoco-first-steps/04_timestep_contact.py
"""물리엔진의 최소 교양: timestep, 적분기, 접촉을 실험으로 확인한다.

실험 1) timestep을 키워 가며 두 관절 진자를 돌린다. 너무 크면 에너지가 폭주한다.
실험 2) timestep 0.04 s에서 적분기(Euler / implicitfast / implicit / RK4)를 바꿔 본다.
실험 3) 상자를 떨어뜨리고 접촉(mjContact)의 내용을 읽는다. solref로 접촉의 '무름'을 바꿔 본다.
결과 그림: out/ch04-fig05-timestep-stability.png
실행: uv run python ch04-mujoco-first-steps/04_timestep_contact.py
--view 를 붙이면 같은 장면을 viewer 창에서 실시간으로 본다 (macOS: uv run mjpython ... --view).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # 화면 없이 파일로만 그린다 (mjpython 작업 스레드에서도 동작)
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook.render import play, snapshots, want_view

matplotlib.rcParams["font.family"] = "DejaVu Sans"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

PENDULUM = """
<mujoco model="double_pendulum">
  <option timestep="{dt}" integrator="{integrator}"/>
  <asset><texture type="skybox" builtin="gradient" rgb1="0.95 0.95 1" rgb2="0.6 0.7 0.9" width="32" height="512"/></asset>
  <worldbody>
    <light pos="0 -1 3" dir="0 0.3 -1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1"/>
    <camera name="front" pos="0 -2.2 0.6" xyaxes="1 0 0 0 0 1"/>
    <!-- 받침대: 바닥에서 회전축까지 올라오는 기둥과 가로대. 보기용이라 충돌하지 않는다 (contype 0) -->
    <geom type="cylinder" fromto="0 0.15 0 0 0.15 1" size="0.015" rgba="0.35 0.35 0.35 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" fromto="0 0.15 1 0 0 1" size="0.012" rgba="0.35 0.35 0.35 1" contype="0" conaffinity="0"/>
    <geom type="sphere" pos="0 0 1" size="0.03" rgba="0.8 0.2 0.2 1" contype="0" conaffinity="0"/>
    <body pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.02" mass="1"/>
      <body pos="0 0 -0.4">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.02" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def run_pendulum(dt, integrator, duration=5.0):
    """감쇠 없는 이중 진자를 duration초 돌리고 시간에 따른 총 에너지를 돌려준다."""
    model = mujoco.MjModel.from_xml_string(PENDULUM.format(dt=dt, integrator=integrator))
    data = mujoco.MjData(model)
    data.qpos[:] = [np.deg2rad(120), 0.0]
    n = int(duration / dt)
    t, e = [], []
    for i in range(n):
        mujoco.mj_step(model, data)
        if i % max(1, n // 500) == 0:
            # 총 에너지 = 위치 에너지 + 운동 에너지 (mjData.energy는 flag를 켜야 계산됨)
            mujoco.mj_energyPos(model, data)
            mujoco.mj_energyVel(model, data)
            t.append(data.time)
            e.append(data.energy[0] + data.energy[1])
        if not np.all(np.isfinite(data.qpos)) or np.abs(data.qvel).max() > 1e4:
            break   # 발산
    return np.array(t), np.array(e)


# ---- 실험 1: timestep 스윕 (implicitfast 고정) ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
for dt in [0.001, 0.005, 0.02, 0.05]:
    t, e = run_pendulum(dt, "implicitfast")
    ax.plot(t, e, label=f"timestep = {dt} s")
    print(f"[timestep {dt:5.3f}, implicitfast] 5초 후 에너지 {e[-1]:8.2f} J (시작 {e[0]:.2f} J), 샘플 {len(t)}")
ax.set_title("(a) integrator = implicitfast, varying timestep")
ax.set_xlabel("time [s]")
ax.set_ylabel("total energy [J]")
ax.set_ylim(-15, 40)
ax.legend(fontsize=8)

# ---- 실험 2: 적분기 비교 (timestep 0.04 고정: Euler 계열이 무너지는 경계) ----
ax = axes[1]
for integ in ["Euler", "implicitfast", "implicit", "RK4"]:
    t, e = run_pendulum(0.04, integ)
    ax.plot(t, e, label=integ)
    print(f"[timestep 0.040, {integ:12s}] 5초 후 에너지 {e[-1]:10.2f} J, 샘플 {len(t)}")
ax.set_title("(b) timestep = 0.04 s, varying integrator")
ax.set_xlabel("time [s]")
ax.set_ylim(-15, 80)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "ch04-fig05-timestep-stability.png", dpi=200)
plt.close(fig)

# 이중 진자가 흔들리는 모습 (timestep 0.002, implicitfast)
_m = mujoco.MjModel.from_xml_string(PENDULUM.format(dt=0.002, integrator="implicitfast"))
_d = mujoco.MjData(_m); _d.qpos[:] = [np.deg2rad(120), 0.0]
print("saved", snapshots(_m, _d, (0.0, 0.4, 0.8, 1.2), OUT / "ch04-fig04-double-pendulum.png", camera="front", width=600, height=600))

# ---- 실험 3: 접촉 읽기와 solref ----
BOX = """
<mujoco model="contact_demo">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="box" pos="0 0 0.5">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" mass="1" solref="{solref}"/>
    </body>
  </worldbody>
</mujoco>
"""
print("\n=== 접촉 실험: solref에 따른 침투 깊이 ===")
for solref in ["0.02 1", "0.05 1", "0.2 1"]:
    model = mujoco.MjModel.from_xml_string(BOX.format(solref=solref))
    data = mujoco.MjData(model)
    for _ in range(1500):        # 3초, 충분히 정지
        mujoco.mj_step(model, data)
    c = data.contact[0]
    g1, g2 = model.geom(c.geom1).name, model.geom(c.geom2).name
    print(f"solref='{solref}': 접촉 {data.ncon}개, 첫 접촉 {g1}-{g2}, dist={c.dist*1000:6.2f} mm "
          f"(음수 = 침투), 법선={np.round(c.frame[:3], 2)}")

if want_view():
    _m = mujoco.MjModel.from_xml_string(PENDULUM.format(dt=0.002, integrator="implicitfast"))
    _d = mujoco.MjData(_m); _d.qpos[:] = [np.deg2rad(120), 0.0]
    play(_m, _d, duration=10.0, title="이중 진자 (timestep 0.002, implicitfast)")
