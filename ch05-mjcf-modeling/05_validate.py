# labs/ch05-mjcf-modeling/05_validate.py
"""모델 검증: 떨어뜨려보고, 흔들어보고, 의심하기.

검사 1) 제어 없이 떨어뜨린다: 다리가 접히며 주저앉아야 정상 (관절이 자유롭다는 뜻).
검사 2) PD로 자세를 붙들고 떨어뜨린다: 서야 한다.
검사 3) 서 있는 로봇의 몸통을 옆에서 민다(xfrc_applied): 흔들리다 돌아오거나, 힘이 크면 넘어져야 한다.
검사 4) 의심 목록: 대칭, 관절 범위 안에 있는가, 침투 깊이, 에너지 폭주 여부, 접촉 개수.
결과 그림: out/ch05-fig08-validation.png
실행: uv run python ch05-mjcf-modeling/05_validate.py
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
from quadbook.robot import STAND_POSE, foot_contacts, joint_qpos, load, pd_torque

matplotlib.rcParams["font.family"] = "DejaVu Sans"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def run(controlled: bool, push_N: float = 0.0, push_at: float = 1.0, push_dur: float = 0.2, duration: float = 3.0):
    model, data = load(model_path())
    data.qpos[2] = 0.4
    mujoco.mj_forward(model, data)
    t, z, roll = [], [], []
    while data.time < duration:
        data.ctrl[:] = pd_torque(data, STAND_POSE) if controlled else 0.0
        data.xfrc_applied[model.body("torso").id, :] = 0.0
        if push_at <= data.time < push_at + push_dur:
            data.xfrc_applied[model.body("torso").id, 1] = push_N     # +y 방향(왼쪽)으로 민다
        mujoco.mj_step(model, data)
        t.append(data.time)
        z.append(data.body("torso").xpos[2])
        # 몸통 roll 각도: 몸통 y축이 세계 z축과 이루는 각
        ymat = data.body("torso").xmat.reshape(3, 3)[:, 1]
        roll.append(np.degrees(np.arcsin(np.clip(ymat[2], -1, 1))))
    return model, data, np.array(t), np.array(z), np.array(roll)


# 검사 1·2
m0, d0, t0, z0, _ = run(controlled=False)
m1, d1, t1, z1, _ = run(controlled=True)
print(f"[검사 1] 제어 없음: 3초 뒤 몸통 높이 {z0[-1]:.3f} m (서기 높이 0.31 m보다 낮으면 주저앉은 것)")
print(f"[검사 2] PD 유지  : 3초 뒤 몸통 높이 {z1[-1]:.3f} m, 발 접촉 {foot_contacts(m1, d1)}")

# 검사 3: 밀기 강도 두 가지
results = {}
for push in (20.0, 60.0):
    m, d, t, z, roll = run(controlled=True, push_N=push)
    results[push] = (t, z, roll)
    fell = z[-1] < 0.2
    print(f"[검사 3] 몸통을 {push:.0f} N으로 0.2초 밀기: 최대 roll {np.abs(roll).max():.1f}°, 3초 뒤 높이 {z[-1]:.3f} m → {'넘어짐' if fell else '회복'}")

# 검사 4: 의심 목록
m, d = load(model_path())
for _ in range(1000):
    d.ctrl[:] = pd_torque(d, STAND_POSE)
    mujoco.mj_step(m, d)
q = joint_qpos(d)
lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]          # 0번은 freejoint
inside = np.all((q >= lo) & (q <= hi))
pen = min((d.contact[i].dist for i in range(d.ncon)), default=0.0)
mujoco.mj_energyPos(m, d); mujoco.mj_energyVel(m, d)
print(f"[검사 4] 관절이 범위 안: {inside} | 최대 침투 {pen*1000:.2f} mm | 접촉 {d.ncon}개 | 운동에너지 {d.energy[1]:.4f} J (정지면 거의 0)")
print(f"        서 있을 때 관절 토크 최대 {np.abs(d.ctrl).max():.2f} N·m (한계 12)")

# 세 결말을 그림으로: 제어 없이 주저앉음 / PD로 섬 / 60 N에 넘어짐 (각각 3초 뒤)
from PIL import Image
import quadbook.render as _r
_frames = []
for ctrl, push in ((False, 0.0), (True, 0.0), (True, 60.0)):
    _m, _d, *_ = run(controlled=ctrl, push_N=push)
    _ren = mujoco.Renderer(_m, height=500, width=700); _ren.update_scene(_d, camera="side")
    _frames.append(_r._label(_ren.render().copy(), ["no control", "PD holds pose", "pushed 60 N"][len(_frames)])); _ren.close()
Image.fromarray(np.concatenate(_frames, axis=1)).save(OUT / "ch05-fig07-validation-frames.png"); print("saved", OUT / "ch05-fig07-validation-frames.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(t0, z0, label="no control (joints free)")
axes[0].plot(t1, z1, label="PD holds standing pose")
axes[0].axhline(0.31, color="#999999", ls=":", lw=1)
axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("torso height [m]"); axes[0].set_title("(a) drop test"); axes[0].legend(fontsize=8)
for push, (t, z, roll) in results.items():
    axes[1].plot(t, roll, label=f"push {push:.0f} N for 0.2 s")
axes[1].axvspan(1.0, 1.2, color="#eeeeee")
axes[1].set_xlabel("time [s]"); axes[1].set_ylabel("torso roll [deg]"); axes[1].set_title("(b) push test"); axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "ch05-fig08-validation.png", dpi=200)
print("saved", OUT / "ch05-fig08-validation.png")

if want_view():
    # 낙하 → 1초에 20 N 밀기(회복) → 3초에 60 N 밀기(전도)를 한 번에 본다
    m2, d2 = load(model_path()); d2.qpos[2] = 0.4
    torso = m2.body("torso").id
    def _step():
        d2.ctrl[:] = pd_torque(d2, STAND_POSE)
        d2.xfrc_applied[torso, :] = 0.0
        if 1.0 <= d2.time < 1.2: d2.xfrc_applied[torso, 1] = 20.0
        if 3.0 <= d2.time < 3.2: d2.xfrc_applied[torso, 1] = 60.0
        mujoco.mj_step(m2, d2)
    play(m2, d2, step_fn=_step, duration=6.0, title="검증: 낙하 → 20 N 밀기(회복) → 60 N 밀기(전도)")
