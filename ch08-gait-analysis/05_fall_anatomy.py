# labs/ch08-gait-analysis/05_fall_anatomy.py
"""8.5 넘어짐의 해부: 실패 데이터에서 배우는 것.

넘어짐 두 건을 3장의 언어로 해부한다.
(1) 7.6절의 60 N 옆 밀기: 안정 여유(무게중심 투영과 지지 다각형의 거리), roll, 접촉을 한 시간축에 놓고 '돌이킬 수 없는 순간'을 찾는다.
(2) 7.6절의 10도 오르막: 넘어지지 않는데 왜 나아가지 못하는가. 지지기 발의 미끄러짐과 토크 포화를 잰다.
(3) 1 Hz trot: 넘어지지는 않지만 걸음이 다른 걸음으로 변한 사례. 주기당 접촉 조각 수와 위상으로 본다.
그림: out/ch08-fig09-fall-frames.png, out/ch08-fig10-fall-timeline.png, out/ch08-fig11-slope-stall.png
실행: uv run python ch08-gait-analysis/05_fall_anatomy.py   (--view --speed 0.25: 60 N 밀기 넘어짐을 슬로모션으로)
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
from quadbook.analysis import gait_metrics, support_margin
from quadbook.cpg import GaitGenerator
from quadbook.control import torso_roll_pitch
from quadbook.gait import LEG_ORDER
from quadbook.render import play, track_camera, want_view
from quadbook.robot import foot_contacts, joint_qpos, load
from quadbook.sim import WALK_KP, fallen, standard_pd

OUT = Path(__file__).resolve().parent / "out"
TROT = dict(gait="trot", freq=2.0, duty=0.5, step_length=0.08, step_height=0.04)
CE = 5
KNEE_IDX = [2, 5, 8, 11]


def run(duration, push=None, slope_deg=0.0, frames_at=(), cam=None):
    model, data = load(model_path())
    if slope_deg:
        a = np.radians(slope_deg); model.opt.gravity[:] = [-9.81 * np.sin(a), 0.0, -9.81 * np.cos(a)]
    pd = standard_pd(model, data, kp=WALK_KP); gen = GaitGenerator(**TROT)
    torso = model.body("torso").id; dt_ctrl = model.opt.timestep * CE
    q, qd = gen.targets_at(gen.osc.theta), np.zeros(12); i = 0
    log = {k: [] for k in ("t", "margin", "roll", "contact", "n_feet", "z", "tau_max", "y", "pitch", "x", "lf_slip")}
    prev_lf = None
    frames = []
    ren = None
    if frames_at:
        R._ensure_framebuffer(model, 600, 450); ren = mujoco.Renderer(model, height=450, width=600)
    while data.time < duration:
        if data.time >= 0.5 and i % CE == 0:
            q, qd = gen.targets_with_velocity(dt_ctrl)
        data.ctrl[:] = pd.torque(data, q, qd)
        data.xfrc_applied[torso, :] = 0.0
        if push and push[0] <= data.time < push[0] + 0.2:
            data.xfrc_applied[torso, 1] = push[1]
        mujoco.mj_step(model, data); i += 1
        margin, com, hull = support_margin(model, data)
        r, _ = torso_roll_pitch(data); c = foot_contacts(model, data)
        log["t"].append(data.time); log["margin"].append(margin); log["roll"].append(r); log["contact"].append(c); log["n_feet"].append(c.sum())
        log["z"].append(data.body("torso").xpos[2]); log["tau_max"].append(np.abs(data.ctrl).max()); log["y"].append(data.body("torso").xpos[1])
        log["pitch"].append(torso_roll_pitch(data)[1]); log["x"].append(data.body("torso").xpos[0])
        lf = data.geom("LF_foot").xpos.copy()
        log["lf_slip"].append((lf[0] - prev_lf[0]) / model.opt.timestep if (prev_lf is not None and c[0]) else np.nan)   # 지지기 LF 발의 x 속도
        prev_lf = lf
        for ft in frames_at:
            if abs(data.time - ft) < model.opt.timestep / 2:
                ren.update_scene(data, camera=cam)
                frames.append(R._label(ren.render().copy(), f"t = {data.time:.2f} s  margin {100*margin:+.1f} cm  roll {np.degrees(r):+.0f} deg  feet {c.sum()}"))
    if ren: ren.close()
    return {k: np.array(v) for k, v in log.items()}, frames


# ---- (1) 60 N 밀기 ----
model0, _ = load(model_path()); cam = track_camera(model0, distance=1.4, azimuth=90, elevation=-10)
log, frames = run(4.5, push=(3.0, 60.0), frames_at=(2.95, 3.15, 3.30, 3.50), cam=cam)
Image.fromarray(np.concatenate(frames, axis=1)).save(OUT / "ch08-fig09-fall-frames.png")
t = log["t"]
print("(1) trot 중 3.0초 60 N 옆 밀기의 해부")
before = (t >= 2.0) & (t < 3.0)
print(f"    밀기 전 1초: 안정 여유 평균 {100*log['margin'][before].mean():+.1f} cm (trot의 지지 다각형은 선분이라 늘 음수), 땅에 닿은 발 평균 {log['n_feet'][before].mean():.2f}")
t_fall = t[np.argmax([fallen_ for fallen_ in (np.abs(log["roll"]) > 0.8)])] if (np.abs(log["roll"]) > 0.8).any() else None
for tt in (3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.7, 4.0):
    k = np.argmin(np.abs(t - tt))
    print(f"    t = {tt:.1f} s: roll {np.degrees(log['roll'][k]):+6.1f}°, 안정 여유 {100*log['margin'][k]:+6.1f} cm, 닿은 발 {log['n_feet'][k]} ({' '.join(l for j, l in enumerate(LEG_ORDER) if log['contact'][k][j]) or '없음'}), y {log['y'][k]:+.2f} m")
print(f"    roll이 45°를 넘는 시각: {t_fall:.2f} s" if t_fall else "    넘어지지 않음")

fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
m = (t >= 2.5) & (t <= 4.5)
axes[0].plot(t[m], 100 * log["margin"][m], lw=1); axes[0].axhline(0, color="#999999", ls=":"); axes[0].axvspan(3.0, 3.2, color="#d62728", alpha=0.15, label="60 N push")
axes[0].set_ylabel("support margin [cm]"); axes[0].set_title("(a) CoM projection vs support polygon (+ inside)", fontsize=10); axes[0].legend(fontsize=8)
axes[1].plot(t[m], np.degrees(log["roll"][m]), lw=1, color="#ff7f0e"); axes[1].axvspan(3.0, 3.2, color="#d62728", alpha=0.15)
axes[1].set_ylabel("torso roll [deg]"); axes[1].set_title("(b) torso roll", fontsize=10)
for i, leg in enumerate(LEG_ORDER):
    axes[2].fill_between(t[m], i, i + 0.8, where=log["contact"][m][:, i].astype(bool), color="#4a90d9")
axes[2].set_yticks([i + 0.4 for i in range(4)]); axes[2].set_yticklabels(LEG_ORDER); axes[2].axvspan(3.0, 3.2, color="#d62728", alpha=0.15)
axes[2].set_xlabel("time [s]"); axes[2].set_title("(c) foot contacts", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "ch08-fig10-fall-timeline.png", dpi=200); plt.close(fig)

# ---- (2) 10도 오르막: 제자리걸음의 해부 ----
print("(2) 10도 오르막: 지지기 발의 미끄러짐과 토크 포화")
log_flat, _ = run(6.0)
log_slope, _ = run(6.0, slope_deg=10.0)
for name, lg in (("flat", log_flat), ("slope 10 deg", log_slope)):
    w = lg["t"] >= 2.0
    slip = lg["lf_slip"][w]; slip = slip[~np.isnan(slip)]
    print(f"    {name:12s}: 속도 {(lg['x'][-1]-lg['x'][w][0])/(lg['t'][-1]-lg['t'][w][0]):.3f} m/s | 지지기 LF 발의 x 속도 평균 {slip.mean():+.3f} m/s, 뒤로(< −0.02) 미끄러진 시간 비율 {(slip < -0.02).mean():.2f} | 토크가 한계 12에 붙은 시간 비율 {(lg['tau_max'][w] >= 11.99).mean():.2f} | pitch 평균 {np.degrees(lg['pitch'][w].mean()):+.1f}°")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for name, lg in (("flat", log_flat), ("slope 10 deg", log_slope)):
    m = (lg["t"] >= 3.0) & (lg["t"] < 4.0)
    axes[0].plot(lg["t"][m], lg["lf_slip"][m], lw=0.9, label=name)
    axes[1].plot(lg["t"][m], np.minimum(lg["tau_max"][m], 30), lw=0.9, label=name)
    axes[2].plot(lg["t"], lg["x"] - lg["x"][0], lw=1, label=name)
axes[0].axhline(0, color="#999999", ls=":"); axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("LF foot x velocity in stance [m/s]"); axes[0].set_title("(a) stance foot slip (negative = sliding back)", fontsize=10); axes[0].legend(fontsize=8)
axes[1].axhline(12, color="#d62728", ls=":", label="motor limit"); axes[1].set_xlabel("time [s]"); axes[1].set_ylabel("max |torque demand| [N m]"); axes[1].set_title("(b) torque demand (clipped at 30 for display)", fontsize=10); axes[1].legend(fontsize=8)
axes[2].set_xlabel("time [s]"); axes[2].set_ylabel("distance [m]"); axes[2].set_title("(c) progress", fontsize=10); axes[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch08-fig11-slope-stall.png", dpi=200); plt.close(fig)

# ---- (3) 1 Hz trot: 걸음이 바뀐 사례 ----
L = np.load(OUT / "logs.npz")
for name in ("trot2", "trot1"):
    mtr = gait_metrics(L[f"{name}_t"], L[f"{name}_contact"], t_from=2.0, x=L[f"{name}_x"])
    print(f"(3) {name}: β {np.round(mtr['duty'], 2)}, φ {np.round(mtr['phase'], 2)}, 조각/주기 {np.round(mtr['fragments'], 1)}, 속도 {mtr['speed']:.3f} m/s")
print("saved", OUT / "ch08-fig09-fall-frames.png", OUT / "ch08-fig10-fall-timeline.png", OUT / "ch08-fig11-payload-torque.png")

if want_view():
    model, data = load(model_path()); pd = standard_pd(model, data, kp=WALK_KP); gen = GaitGenerator(**TROT)
    torso = model.body("torso").id; state = {"i": 0, "q": gen.targets_at(gen.osc.theta), "qd": np.zeros(12), "said": False}
    def _step():
        if data.time >= 0.5 and state["i"] % CE == 0: state["q"], state["qd"] = gen.targets_with_velocity(model.opt.timestep * CE)
        data.ctrl[:] = pd.torque(data, state["q"], state["qd"]); data.xfrc_applied[torso, :] = 0.0
        if 3.0 <= data.time < 3.2:
            data.xfrc_applied[torso, 1] = 60.0
            if not state["said"]: state["said"] = True; print(f"[{data.time:.2f} s] 60 N 밀기 시작")
        mujoco.mj_step(model, data); state["i"] += 1
        if state["i"] % 50 == 0 and 3.0 <= data.time < 4.0:
            mg, _, _ = support_margin(model, data); r, _ = torso_roll_pitch(data)
            print(f"[{data.time:.2f} s] 안정 여유 {100*mg:+.1f} cm, roll {np.degrees(r):+.0f}°, 닿은 발 {foot_contacts(model, data).sum()}")
    play(model, data, step_fn=_step, duration=4.5, title="trot 중 60 N 밀기: 안정 여유와 roll을 중계 (--speed 0.25 권장)")
