# labs/ch06-standing-control/03_gain_tuning.py
"""6.3 gain 튜닝: 뻣뻣함과 출렁임 사이.

실험 A) 스텝 응답. 관절을 서기 자세에서 0.3 rad 어긋난 곳에서 출발시켜 서기 자세로 돌아오게 한다.
        kp를 바꾸고 kd는 감쇠비 ζ = 1(임계 감쇠)에 맞게 관절 관성에서 계산한다.
        지표: 정착 시간, 넘침, 요구 토크(한계 12 N·m를 넘으면 잘린다), 정지 높이.
실험 B) 같은 kp에서 ζ만 0.3과 1.0으로 바꿔 댐퍼의 역할을 본다.
실험 C) 제어 주기. 물리는 500 Hz(0.002 s)인데 제어기를 500/100/50 Hz로 돌리면, 큰 kp는 지연 때문에 떨린다.
그림: out/ch06-fig06-gain-step.png, out/ch06-fig07-gain-frames.png, out/ch06-fig08-control-rate.png
실행: uv run python ch06-standing-control/03_gain_tuning.py
--view 를 붙이면 kp 40 응답을 viewer에서 본다 (--speed 0.5 권장).
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
from quadbook.control import JointPD, critical_kd, joint_inertia_diag
from quadbook.render import play, want_view
from quadbook.robot import STAND_POSE, joint_qpos, load

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

OFFSET = np.array([0.0, 0.3, -0.3] * 4)     # 엉덩이 +0.3, 무릎 −0.3 rad 어긋난 출발 자세


def step_response(kp, zeta=1.0, control_every=1, duration=2.0):
    """어긋난 자세에서 STAND_POSE로 돌아오는 응답. control_every 스텝마다 한 번만 토크를 새로 계산한다."""
    model, data = load(model_path())
    kd = critical_kd(kp, joint_inertia_diag(model, data), zeta)
    pd = JointPD(kp, kd)
    data.qpos[7:19] = STAND_POSE + OFFSET
    mujoco.mj_forward(model, data)
    t, q_hip, z, tau_req = [], [], [], []
    torque = np.zeros(12)
    i = 0
    while data.time < duration:
        if i % control_every == 0:
            torque = pd.torque(data, STAND_POSE)
        data.ctrl[:] = torque
        mujoco.mj_step(model, data)
        i += 1
        t.append(data.time); q_hip.append(joint_qpos(data)[1]); z.append(data.body("torso").xpos[2]); tau_req.append(np.abs(torque).max())
    t, q_hip, z, tau_req = map(np.array, (t, q_hip, z, tau_req))
    err = q_hip - STAND_POSE[1]
    settled = np.where(np.abs(err) > 0.02)[0]
    t_settle = t[settled[-1]] if len(settled) else 0.0
    overshoot = max(0.0, -err.min()) / 0.3 * 100
    jitter = np.degrees(q_hip[t > 1.0].std())          # 1초 이후 흔들림의 표준편차 (떨림 지표)
    return model, data, kd, t, q_hip, z, tau_req, t_settle, overshoot, jitter


# ---- 실험 A: kp 스윕 (ζ = 1, 500 Hz) ----
print("실험 A  kp     kd(hip)  정착시간  넘침   요구토크(최대)  정지높이")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
frames = []
for kp in (10.0, 40.0, 160.0):
    model, data, kd, t, q, z, tau, ts, ov, _ = step_response(kp)
    print(f"        {kp:5.0f}  {kd[1]:6.2f}   {ts:5.2f} s  {ov:4.0f}%   {tau.max():6.1f} N·m     {z[-1]:.3f} m" + ("   ← 한계 12를 넘어 잘림" if tau.max() > 12 else ""))
    axes[0].plot(t, np.degrees(q), label=f"kp = {kp:.0f}, kd = {kd[1]:.1f}")
    axes[1].plot(t, z, label=f"kp = {kp:.0f}")
    ren = mujoco.Renderer(model, height=500, width=700); ren.update_scene(data, camera="side")
    frames.append(R._label(ren.render().copy(), f"kp = {kp:.0f} (t = 2 s)")); ren.close()
axes[0].axhline(np.degrees(STAND_POSE[1]), color="#999999", ls=":", lw=1)
axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("LF hip angle [deg]"); axes[0].set_title("(a) hip step response, critical damping"); axes[0].legend(fontsize=8)
axes[1].axhline(0.31, color="#999999", ls=":", lw=1)
axes[1].set_xlabel("time [s]"); axes[1].set_ylabel("torso height [m]"); axes[1].set_title("(b) torso height"); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch06-fig06-gain-step.png", dpi=200); plt.close(fig)
Image.fromarray(np.concatenate(frames, axis=1)).save(OUT / "ch06-fig07-gain-frames.png")

# ---- 실험 B: 같은 kp, 다른 ζ ----
print("실험 B  kp 40에서 감쇠비만 바꾸면")
for zeta in (0.3, 1.0):
    *_, ts, ov, _ = step_response(40.0, zeta=zeta)
    print(f"        ζ = {zeta}: 정착 {ts:.2f} s, 넘침 {ov:.0f}%")

# ---- 실험 C: 제어 주기 × kp → 떨림 ----
print("실험 C  제어 주기별 1초 이후 엉덩이 각도 흔들림(표준편차, °)")
rates = {1: "500 Hz", 5: "100 Hz", 10: "50 Hz"}
kps = (40.0, 400.0, 3000.0)
table = np.zeros((len(kps), len(rates)))
fig, ax = plt.subplots(figsize=(7, 3.6))
for i, kp in enumerate(kps):
    for j, (every, name) in enumerate(rates.items()):
        model, data, kd, t, q, z, tau, ts, ov, jitter = step_response(kp, control_every=every)
        table[i, j] = jitter
        if kp == 400.0:
            ax.plot(t, np.degrees(q), lw=0.9, label=f"kp 400, control {name}")
    print(f"        kp {kp:5.0f}: " + "  ".join(f"{name} {table[i, j]:5.2f}°" for j, name in enumerate(rates.values())))
ax.axhline(np.degrees(STAND_POSE[1]), color="#999999", ls=":", lw=1)
ax.set_xlim(0.8, 2.0); ax.set_ylim(np.degrees(STAND_POSE[1]) - 8, np.degrees(STAND_POSE[1]) + 8)
ax.set_xlabel("time [s]"); ax.set_ylabel("LF hip angle [deg]"); ax.set_title("(c) same gains, slower control loop: chatter"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch06-fig08-control-rate.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch06-fig06-gain-step.png", OUT / "ch06-fig07-gain-frames.png", OUT / "ch06-fig08-control-rate.png")

if want_view():
    model, data = load(model_path())
    pd = JointPD(40.0, critical_kd(40.0, joint_inertia_diag(model, data)))
    data.qpos[7:19] = STAND_POSE + OFFSET; mujoco.mj_forward(model, data)
    def _step():
        data.ctrl[:] = pd.torque(data, STAND_POSE); mujoco.mj_step(model, data)
    play(model, data, step_fn=_step, duration=4.0, title="kp 40, 임계 감쇠: 어긋난 자세에서 서기 자세로")
