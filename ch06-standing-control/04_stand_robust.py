# labs/ch06-standing-control/04_stand_robust.py
"""6.4 외란에도 버티고 서 있는 로봇.

단계 1) 하중 균형: 5장에서 앞발 14 N, 뒷발 28 N으로 치우쳤던 서기 자세. 엉덩이 각도를 바꿔 가며 앞뒤 하중을 맞춘다.
단계 2) 밀기 한계: 관절 PD만으로 버티는 최대 옆 밀기 힘.
단계 3) 지지 다각형을 넓히고(발 벌림) 무게중심을 낮추면(웅크림) 한계가 얼마나 오르는지 본다 (3.5절의 정적 안정 여유).
단계 4) IMU로 몸통 기울기를 읽어 관절을 돌리는 '자세 피드백'은 왜 도움이 안 되는가 (발이 땅에 붙어 있는 한).
그림: out/ch06-fig09-stance-balance.png, out/ch06-fig10-push-limit.png, out/ch06-fig11-push-frames.png
실행: uv run python ch06-standing-control/04_stand_robust.py
--view 를 붙이면 발을 벌린 로봇을 1초마다 점점 세게 민다 (--speed 0.5 권장).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import mujoco
import quadbook.render as R
from quadbook import model_path
from quadbook.control import JointPD, critical_kd, joint_inertia_diag, torso_roll_pitch
from quadbook.render import play, want_view
from quadbook.robot import STAND_POSE, TOUCH_SENSORS, load

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
KP = 40.0


def make():
    model, data = load(model_path())
    pd = JointPD(KP, critical_kd(KP, joint_inertia_diag(model, data)))
    return model, data, pd


def pose(abd=0.0, hip=0.8, knee=-1.5):
    """다리 순서 LF, RF, LH, RH. 벌림은 왼쪽 +, 오른쪽 −로 넣어야 네 발이 바깥으로 벌어진다."""
    return np.array([abd, hip, knee, -abd, hip, knee, abd, hip, knee, -abd, hip, knee])


def settle(p, seconds=2.0):
    model, data, pd = make()
    data.qpos[7:19] = p; mujoco.mj_forward(model, data)
    while data.time < seconds:
        data.ctrl[:] = pd.torque(data, p); mujoco.mj_step(model, data)
    return model, data, pd


# ---- 단계 1: 하중 균형 ----
print("단계 1  엉덩이 각도별 발 하중 (LF RF LH RH, N)와 무게중심·발 중심의 x")
hips = (0.8, 0.9, 1.0, 1.1)
ratios = []
for hip in hips:
    model, data, _ = settle(pose(hip=hip))
    f = np.array([data.sensor(n).data[0] for n in TOUCH_SENSORS])
    com_x = data.subtree_com[model.body("torso").id][0]
    feet_x = np.mean([data.geom(f"{l}_foot").xpos[0] for l in ("LF", "RF", "LH", "RH")])
    ratios.append(f[:2].sum() / f[2:].sum())
    print(f"        hip {hip:.1f}: {np.round(f, 1)}  앞/뒤 {ratios[-1]:.2f}  CoM x {com_x:+.3f}  발 중심 x {feet_x:+.3f}  높이 {data.body('torso').xpos[2]:.3f}")
HIP_BALANCED = hips[int(np.argmin(np.abs(np.array(ratios) - 1)))]
BALANCED = pose(hip=HIP_BALANCED)
print(f"        → 균형 자세: 엉덩이 {HIP_BALANCED} rad (quadbook.robot.STAND_POSE_BALANCED)")
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(hips, ratios, "o-"); ax.axhline(1.0, color="#999999", ls=":")
ax.set_xlabel("hip angle [rad]"); ax.set_ylabel("front / rear foot load"); ax.set_title("stance balance vs hip angle")
fig.tight_layout(); fig.savefig(OUT / "ch06-fig09-stance-balance.png", dpi=200); plt.close(fig)


# ---- 단계 2·3: 밀기 한계 ----
def push_limit(p, target_fn=None, forces=range(10, 200, 10)):
    """0.2초 동안 +y로 미는 힘을 10 N씩 올리며, 넘어지기 직전의 힘을 돌려준다."""
    last = 0
    for F in forces:
        model, data, pd = settle(p, 1.0)
        torso = model.body("torso").id
        while data.time < 3.0:
            tgt = target_fn(data, p) if target_fn else p
            data.ctrl[:] = pd.torque(data, tgt)
            data.xfrc_applied[torso, :] = 0.0
            if 1.0 <= data.time < 1.2:
                data.xfrc_applied[torso, 1] = F
            mujoco.mj_step(model, data)
        if data.body("torso").xpos[2] < 0.2:
            return last
        last = F
    return last


cases = {
    "balanced stance": BALANCED,
    "feet apart 0.15 rad": pose(abd=0.15, hip=HIP_BALANCED),
    "feet apart 0.30 rad": pose(abd=0.30, hip=HIP_BALANCED),
    "crouched (hip 1.2, knee -2.1)": pose(hip=1.2, knee=-2.1),
}
print("단계 2·3  자세별 밀기 한계 (kp 40)")
limits = {}
for name, p in cases.items():
    model, data, _ = settle(p)
    width = abs(data.geom("LF_foot").xpos[1] - data.geom("RF_foot").xpos[1])
    limits[name] = push_limit(p)
    print(f"        {name:30s}: 버틴 최대 {limits[name]:3d} N | 발 좌우 간격 {width:.2f} m, 몸통 높이 {data.body('torso').xpos[2]:.3f} m")

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.barh(list(limits.keys()), list(limits.values()), color="#4a90d9")
ax.set_xlabel("max side push survived, 0.2 s [N]"); ax.set_title("push limit vs stance (joint PD, kp 40)")
fig.tight_layout(); fig.savefig(OUT / "ch06-fig10-push-limit.png", dpi=200); plt.close(fig)


# ---- 단계 4: 자세 피드백은 왜 안 되는가 ----
def attitude_feedback(k_roll):
    """몸통 roll에 비례해 네 다리 벌림 목표를 같은 방향으로 옮긴다 (발을 기운 쪽으로 보내려는 시도)."""
    def fn(data, p):
        roll, _ = torso_roll_pitch(data)
        q = p.copy(); q[0::3] += -k_roll * roll
        return q
    return fn

print("단계 4  IMU roll → 벌림 관절 보정 (균형 자세, kp 40)")
for k in (0.0, 0.5, 1.5):
    print(f"        k_roll {k:.1f}: 버틴 최대 {push_limit(BALANCED, attitude_feedback(k) if k else None):3d} N")

# ---- 프레임: 같은 60 N 밀기, 기본 자세 vs 발 벌림 0.3 (밀기 0.5초 뒤) ----
frames = []
for name, p in (("balanced, 60 N", BALANCED), ("feet apart 0.3, 60 N", pose(abd=0.30, hip=HIP_BALANCED))):
    model, data, pd = settle(p, 1.0); torso = model.body("torso").id
    while data.time < 1.7:
        data.ctrl[:] = pd.torque(data, p); data.xfrc_applied[torso, :] = 0.0
        if 1.0 <= data.time < 1.2: data.xfrc_applied[torso, 1] = 60.0
        mujoco.mj_step(model, data)
    ren = mujoco.Renderer(model, height=500, width=700); ren.update_scene(data, camera="front")
    frames.append(R._label(ren.render().copy(), f"{name} (t = 1.7 s)")); ren.close()
Image.fromarray(np.concatenate(frames, axis=1)).save(OUT / "ch06-fig11-push-frames.png")
print("saved", OUT / "ch06-fig09-stance-balance.png", OUT / "ch06-fig10-push-limit.png", OUT / "ch06-fig11-push-frames.png")

if want_view():
    p = pose(abd=0.30, hip=HIP_BALANCED)
    model, data, pd = settle(p, 1.0); torso = model.body("torso").id
    def _step():
        data.ctrl[:] = pd.torque(data, p); data.xfrc_applied[torso, :] = 0.0
        k = int(data.time)
        if k >= 2 and data.time - k < 0.2:
            data.xfrc_applied[torso, 1] = 20.0 * (k - 1) * (1 if k % 2 else -1)   # 20, 40, 60, ... N, 좌우 번갈아
        mujoco.mj_step(model, data)
    play(model, data, step_fn=_step, duration=9.0, title="발을 벌린 로봇을 점점 세게 밀기 (20, 40, 60, ... N)")
