# labs/ch07-first-gait/06_open_loop_limits.py
"""7.6 우리의 걸음은 왜 딱딱한가: 개루프 제어의 한계 목격하기.

생성기는 로봇의 상태를 읽지 않는다. 그 결과를 네 가지 상황에서 본다.
(1) 평지에서도: 명령한 발 높이와 실제 발 접촉이 어긋난다 (발이 계획보다 일찍 닿거나 늦게 뜬다).
(2) 옆에서 밀기: 서 있을 때 40 N을 버티던 로봇이 걷는 중에는 얼마에 넘어지는가.
(3) 오르막: 중력 벡터를 기울여 5°, 10° 경사를 만든다.
(4) 짐: 몸통에 2 kg, 6 kg을 더 싣는다. 마찰: 발과 바닥의 마찰 계수를 1.0에서 0.4, 0.15로 낮춘다.
그림: out/ch07-fig11-push-fall-frames.png, out/ch07-fig12-open-loop-traces.png
실행: uv run python ch07-first-gait/06_open_loop_limits.py   (--view --speed 0.5: 걷는 중 3초 20 N, 6초 40 N, 9초 60 N 밀기)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook import model_path
from quadbook.cpg import GaitGenerator
from quadbook.render import play, snapshots, track_camera, want_view
from quadbook.robot import load, reset_stand
from quadbook.sim import WALK_KP, rollout, standard_pd, summary

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
TROT = dict(gait="trot", freq=2.0, duty=0.5, step_length=0.08, step_height=0.04)
CE = 5


def setup(slope_deg=0.0, payload=0.0, friction=None):
    model, data = load(model_path())
    if slope_deg:
        a = np.radians(slope_deg)                       # 오르막: 중력이 뒤(−x)로 기운다
        model.opt.gravity[:] = [-9.81 * np.sin(a), 0.0, -9.81 * np.cos(a)]
    if payload:
        model.body_mass[model.body("torso").id] += payload
        mujoco.mj_setConst(model, data)             # 질량 파생량 재계산. 주의: data가 qpos0(다리를 편 자세)로 돌아간다
        reset_stand(model, data)                    # 그래서 서기 keyframe으로 다시 초기화한다
    if friction is not None:
        model.geom_friction[:, 0] = friction            # 접촉 쌍의 마찰은 두 geom 중 큰 값이므로 발과 바닥을 함께 낮춘다
    mujoco.mj_forward(model, data)
    return model, data, standard_pd(model, data, kp=WALK_KP), GaitGenerator(**TROT)


def pusher(F, t0=3.0, dur=0.2):
    def fn(model, data):
        torso = model.body("torso").id
        data.xfrc_applied[torso, :] = 0.0
        if t0 <= data.time < t0 + dur:
            data.xfrc_applied[torso, 1] = F
    return fn


# ---- (1) 평지: 명령 vs 실제 접촉 ----
print("(1) 평지 trot, LF 다리 2초 동안: 명령한 지지기와 실제 접촉의 어긋남")
slow = None
for f in (2.0, 1.0):
    model, data, pd, gen = setup(); gen = GaitGenerator(**{**TROT, "freq": f})
    log = rollout(model, data, gen, pd, duration=6.0, control_every=CE)
    m = (log["t"] >= 3.0) & (log["t"] < 5.0)
    cmd_stance = log["foot_cmd"][m][:, 0] < 1e-6
    act_stance = log["contact"][m][:, 0].astype(bool)
    early = (act_stance & ~cmd_stance).mean(); late = (~act_stance & cmd_stance).mean()
    print(f"    {f:.0f} Hz: 명령 지지기 {cmd_stance.mean():.2f}, 실제 접촉 {act_stance.mean():.2f} | 계획보다 일찍 닿음 {early:.2f}, 계획엔 지지기인데 떠 있음 {late:.2f} | 속도 {summary(log)['speed']:.3f} m/s ({100*summary(log)['speed']/gen.ideal_speed:.0f}%)")
    if f == 2.0: base = log
    else: slow = log
s0 = summary(base)

# ---- (2) 밀기 ----
print("(2) 걷는 중 옆 밀기 0.2초 (서 있을 때 한계 40 N)")
push_logs = {}
for F in (20, 40, 60, 80, 100):
    model, data, pd, gen = setup()
    log = rollout(model, data, gen, pd, duration=6.0, control_every=CE, on_step=pusher(F))
    push_logs[F] = log
    print(f"    {F:3d} N: 넘어짐 {summary(log)['fell']}, 밀린 뒤 y 이탈 {log['y'][-1]-log['y'][0]:+.2f} m")

# ---- (3)(4) 경사, 짐, 마찰 ----
print("(3)(4) 환경을 바꾸면 (같은 생성기, 같은 gain)")
cases = {"flat": {}, "slope 5 deg": dict(slope_deg=5), "slope 10 deg": dict(slope_deg=10), "payload +2 kg": dict(payload=2.0), "payload +6 kg": dict(payload=6.0), "friction 0.4": dict(friction=0.4), "friction 0.15": dict(friction=0.15)}
logs = {}
for name, kw in cases.items():
    model, data, pd, gen = setup(**kw)
    log = rollout(model, data, gen, pd, duration=6.0, control_every=CE); logs[name] = log; s = summary(log)
    print(f"    {name:14s}: 속도 {s['speed']:+.3f} m/s ({100*s['speed']/gen.ideal_speed:+.0f}% of ideal), 높이 {s['height']:.3f} m, pitch σ {s['pitch_std']:.2f}°, 넘어짐 {s['fell']}")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
t = slow["t"]; mm = (t >= 3.0) & (t < 5.0)
axes[0].plot(t[mm], 1000 * slow["foot_cmd"][mm][:, 0], label="commanded foot height [mm]")
axes[0].fill_between(t[mm], 0, 50, where=slow["contact"][mm][:, 0].astype(bool), color="#2ca02c", alpha=0.25, label="actual contact (LF)")
axes[0].set_xlabel("time [s]"); axes[0].set_title("(a) LF at 1 Hz: planned vs actual contact", fontsize=10); axes[0].legend(fontsize=8)
for F, log in push_logs.items():
    axes[1].plot(log["t"], log["y"] - log["y"][0], lw=0.9, label=f"{F} N push at 3 s")
axes[1].set_xlabel("time [s]"); axes[1].set_ylabel("sideways drift y [m]"); axes[1].set_title("(b) side push while trotting: no return to path", fontsize=10); axes[1].legend(fontsize=8)
for name, log in logs.items():
    axes[2].plot(log["t"], log["x"] - log["x"][0], lw=0.9, label=name)
axes[2].set_xlabel("time [s]"); axes[2].set_ylabel("distance [m]"); axes[2].set_title("(c) progress in changed environments", fontsize=10); axes[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch07-fig12-open-loop-traces.png", dpi=200); plt.close(fig)

# 프레임: 20 N 밀기 뒤
model, data, pd, gen = setup()
cam = track_camera(model, distance=1.4, azimuth=90, elevation=-12)
push = pusher(60); dt_ctrl = model.opt.timestep * CE
state = {"i": 0, "q": gen.targets_at(gen.osc.theta), "qd": np.zeros(12)}
def _step():
    if data.time >= 0.5 and state["i"] % CE == 0: state["q"], state["qd"] = gen.targets_with_velocity(dt_ctrl)
    data.ctrl[:] = pd.torque(data, state["q"], state["qd"]); push(model, data); mujoco.mj_step(model, data); state["i"] += 1
print("saved", snapshots(model, data, (2.9, 3.2, 3.6, 5.0), OUT / "ch07-fig11-push-fall-frames.png", camera=cam, step_fn=_step, width=600, height=450))
print("saved", OUT / "ch07-fig12-open-loop-traces.png")

if want_view():
    # 걷는 로봇을 3초마다 점점 세게 민다: 3초 20 N(밀리지만 걷는다), 6초 40 N(크게 밀리지만 버틴다), 9초 60 N(넘어진다).
    # 각 밀기는 0.2초뿐이라 눈에 잘 안 띄므로 터미널에 중계한다. 몸통이 옆으로 튀는 순간이 밀기다.
    from quadbook.sim import fallen
    model, data, pd, gen = setup(); state = {"i": 0, "q": gen.targets_at(gen.osc.theta), "qd": np.zeros(12)}
    torso = model.body("torso").id
    schedule = {3.0: 20.0, 6.0: 40.0, 9.0: 60.0}
    said = set()
    def _step_view():
        if data.time >= 0.5 and state["i"] % CE == 0: state["q"], state["qd"] = gen.targets_with_velocity(dt_ctrl)
        data.ctrl[:] = pd.torque(data, state["q"], state["qd"])
        data.xfrc_applied[torso, :] = 0.0
        for t0, F in schedule.items():
            if t0 <= data.time < t0 + 0.2:
                data.xfrc_applied[torso, 1] = F
                if t0 not in said:
                    said.add(t0); print(f"[{data.time:4.1f} s] 옆(+y)에서 {F:.0f} N, 0.2초 밀기 시작")
            if (t0 + 1.0) <= data.time < t0 + 1.0 + model.opt.timestep and (t0, "r") not in said:
                said.add((t0, "r")); print(f"[{data.time:4.1f} s] 밀기 1초 뒤: " + ("넘어짐" if fallen(data) else f"버팀, 옆으로 {data.body('torso').xpos[1]:+.2f} m 밀림 (원래 경로로 돌아오지 않는다)"))
        mujoco.mj_step(model, data); state["i"] += 1
    print("[viewer] 3초 20 N → 6초 40 N → 9초 60 N. 몸통이 옆으로 튀는 순간이 밀기다. 12초에 끝난다.")
    play(model, data, step_fn=_step_view, duration=12.0, title="trot 중 옆 밀기: 20 N(3초), 40 N(6초), 60 N(9초)")
