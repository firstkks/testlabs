# labs/ch05-mjcf-modeling/04_sensors.py
"""센서 읽기: 로봇을 조금 높은 곳에서 떨어뜨리며 IMU와 발 접촉 센서가 무엇을 말하는지 본다.
관절은 PD로 서기 자세에 붙들어 둔다 (제어기 자체는 6장에서 설명).

결과 그림: out/ch05-fig06-sensors-drop.png
실행: uv run python ch05-mjcf-modeling/04_sensors.py
--view 를 붙이면 같은 장면을 viewer 창에서 실시간으로 본다 (macOS: uv run mjpython ... --view).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # 화면 없이 파일로만 그린다 (mjpython 작업 스레드에서도 동작)
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook import model_path
from quadbook.render import play, snapshots, want_view
from quadbook.robot import STAND_POSE, TOUCH_SENSORS, load, pd_torque

matplotlib.rcParams["font.family"] = "DejaVu Sans"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

model, data = load(model_path())
data.qpos[2] = 0.5                      # 서기 높이(0.31)보다 높이 들어 떨어뜨린다
mujoco.mj_forward(model, data)

t, height, acc_z, touch = [], [], [], []
while data.time < 1.5:
    data.ctrl[:] = pd_torque(data, STAND_POSE)
    mujoco.mj_step(model, data)
    t.append(data.time)
    height.append(data.body("torso").xpos[2])
    acc_z.append(data.sensor("torso_acc").data[2])
    touch.append([data.sensor(n).data[0] for n in TOUCH_SENSORS])
t, height, acc_z, touch = map(np.array, (t, height, acc_z, touch))

land = t[np.argmax(touch.sum(axis=1) > 0)]
print(f"착지 시각 {land:.3f} s | 착지 직후 최대 수직 가속도 {acc_z.max():.1f} m/s² | 정지 시 몸통 높이 {height[-1]:.3f} m")
print("정지 시 발 접촉력(N):", np.round(touch[-1], 1), "| 합계", round(float(touch[-1].sum()), 1), "N ≈ 몸무게", round(float(model.body_subtreemass[model.body('torso').id] * 9.81), 1), "N")
q = data.sensor("torso_quat").data
print("정지 시 몸통 자세 쿼터니언 (w x y z):", np.round(q, 3), "→ 거의 (1,0,0,0) = 수평")

# 낙하 장면 네 순간
_m, _d = load(model_path()); _d.qpos[2] = 0.5
def _step():
    _d.ctrl[:] = pd_torque(_d, STAND_POSE); mujoco.mj_step(_m, _d)
print("saved", snapshots(_m, _d, (0.0, 0.2, 0.3, 1.5), OUT / "ch05-fig05-drop-frames.png", camera="side", step_fn=_step, width=700, height=500))

fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
axes[0].plot(t, height); axes[0].set_ylabel("torso z [m]")
axes[1].plot(t, acc_z); axes[1].set_ylabel("IMU acc z [m/s²]")
for k, n in enumerate(TOUCH_SENSORS):
    axes[2].plot(t, touch[:, k], label=n)
axes[2].set_ylabel("foot touch [N]"); axes[2].set_xlabel("time [s]"); axes[2].legend(fontsize=8, ncol=4)
for ax in axes:
    ax.axvline(land, color="#999999", ls=":", lw=1)
axes[0].set_title("drop from 0.5 m with PD holding the standing pose")
fig.tight_layout()
fig.savefig(OUT / "ch05-fig06-sensors-drop.png", dpi=200)
print("saved", OUT / "ch05-fig06-sensors-drop.png")

if want_view():
    m2, d2 = load(model_path()); d2.qpos[2] = 0.5
    def _step():
        d2.ctrl[:] = pd_torque(d2, STAND_POSE); mujoco.mj_step(m2, d2)
    play(m2, d2, step_fn=_step, duration=5.0, title="0.5 m 낙하 (PD로 자세 유지)")
