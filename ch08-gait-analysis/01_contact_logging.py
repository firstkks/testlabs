# labs/ch08-gait-analysis/01_contact_logging.py
"""8.1 발 접촉 데이터 로깅.

접촉을 아는 방법 두 가지: (a) data.contact 목록(기하: 발 구와 바닥이 닿았는가), (b) touch 센서(힘: 몇 N으로 눌리는가).
둘을 같은 걸음에서 나란히 기록하고, 힘 문턱값이 이진 신호를 어떻게 바꾸는지, 접지 순간의 튐(chatter)이 어떻게 보이는지 확인한다.
7장의 생성기로 trot 2 Hz, trot 1 Hz, walk 1 Hz를 8초씩 걷고 out/logs.npz 에 저장한다. 02·03 스크립트가 이 파일을 읽는다.
그림: out/ch08-fig01-contact-signals.png, out/ch08-fig02-contact-frames.png
실행: uv run python ch08-gait-analysis/01_contact_logging.py   (--view --speed 0.5: 발이 닿고 뜰 때마다 터미널에 찍힌다)
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
from quadbook.cpg import GaitGenerator
from quadbook.gait import LEG_ORDER
from quadbook.render import play, track_camera, want_view
from quadbook.robot import foot_contacts, load, touch
from quadbook.sim import WALK_KP, rollout, standard_pd

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
GAITS = {
    "trot2": dict(gait="trot", freq=2.0, duty=0.5, step_length=0.08, step_height=0.04),
    "trot1": dict(gait="trot", freq=1.0, duty=0.5, step_length=0.08, step_height=0.04),
    "walk1": dict(gait="walk", freq=1.0, duty=0.75, step_length=0.08, step_height=0.04),
}

logs = {}
for name, kw in GAITS.items():
    model, data = load(model_path()); pd = standard_pd(model, data, kp=WALK_KP); gen = GaitGenerator(**kw)
    log = rollout(model, data, gen, pd, duration=8.0, control_every=5)     # 매 스텝(500 Hz) 기록
    logs[name] = log
    print(f"{name}: {len(log['t'])} 스텝 기록, 접촉 비율(LF RF LH RH) {np.round(log['contact'][log['t'] >= 2].mean(axis=0), 2)}, 명령 duty {kw['duty']}")
np.savez(OUT / "logs.npz", **{f"{n}_{k}": v for n, log in logs.items() for k, v in log.items()},
         **{f"{n}_freq": kw["freq"] for n, kw in GAITS.items()}, **{f"{n}_duty": kw["duty"] for n, kw in GAITS.items()})

# ---- 힘 문턱값과 기하 접촉 비교 (trot 2 Hz, LF) ----
log = logs["trot2"]; t = log["t"]; m = (t >= 3.0) & (t < 4.0)
force = log["touch"][m][:, 0]; geom = log["contact"][m][:, 0].astype(bool)
for thr in (0.0, 1.0, 5.0, 20.0):
    b = force > thr
    print(f"  LF 접촉 비율: 힘 > {thr:4.1f} N 기준 {b.mean():.3f}" + (f" | 기하 접촉 기준 {geom.mean():.3f}, 둘의 불일치 {(b != geom).mean():.3f}" if thr == 0 else ""))
edges = np.abs(np.diff(geom.astype(int))).sum()
print(f"  1초(2주기) 동안 LF의 기하 접촉 상태 변화 횟수 {edges} (깨끗한 2주기라면 4)")

fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
axes[0].plot(t[m], force, lw=1, label="LF touch sensor [N]")
for thr, c in ((1.0, "#2ca02c"), (5.0, "#ff7f0e"), (20.0, "#d62728")):
    axes[0].axhline(thr, color=c, ls=":", lw=1, label=f"threshold {thr:g} N")
axes[0].set_ylabel("normal force [N]"); axes[0].legend(fontsize=8, loc="best"); axes[0].set_title("(a) LF foot force during trot", fontsize=10)
axes[1].fill_between(t[m], 0, 1, where=geom, color="#4a90d9", alpha=0.6, label="contact list (geometry)")
axes[1].plot(t[m], 1000 * log["foot_cmd"][m][:, 0] / 40 + 1.1, color="#555555", lw=1, label="commanded foot height (scaled)")
axes[1].set_yticks([]); axes[1].set_xlabel("time [s]"); axes[1].legend(fontsize=8, loc="best"); axes[1].set_title("(b) binary contact vs commanded swing", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "ch08-fig01-contact-signals.png", dpi=200); plt.close(fig)

# ---- 프레임: 접촉 상태를 라벨로 ----
model, data = load(model_path()); pd = standard_pd(model, data, kp=WALK_KP); gen = GaitGenerator(**GAITS["trot2"])
cam = track_camera(model, distance=1.3, azimuth=140, elevation=-15)
dt_ctrl = model.opt.timestep * 5; state = {"i": 0, "q": gen.targets_at(gen.osc.theta), "qd": np.zeros(12)}
def _step():
    if data.time >= 0.5 and state["i"] % 5 == 0: state["q"], state["qd"] = gen.targets_with_velocity(dt_ctrl)
    data.ctrl[:] = pd.torque(data, state["q"], state["qd"]); mujoco.mj_step(model, data); state["i"] += 1
R._ensure_framebuffer(model, 600, 450); ren = mujoco.Renderer(model, height=450, width=600); frames = []
for tt in (3.0, 3.125, 3.25, 3.375):
    while data.time < tt: _step()
    c = foot_contacts(model, data); f = touch(model, data)
    ren.update_scene(data, camera=cam)
    down = " ".join(f"{leg}:{f[i]:.0f}N" for i, leg in enumerate(LEG_ORDER) if c[i]) or "airborne"
    frames.append(R._label(ren.render().copy(), f"t = {data.time:.3f} s  down: {down}"))
ren.close(); Image.fromarray(np.concatenate(frames, axis=1)).save(OUT / "ch08-fig02-contact-frames.png")
print("saved", OUT / "logs.npz", OUT / "ch08-fig01-contact-signals.png", OUT / "ch08-fig02-contact-frames.png")

if want_view():
    model, data = load(model_path()); pd = standard_pd(model, data, kp=WALK_KP); gen = GaitGenerator(**GAITS["trot2"])
    state = {"i": 0, "q": gen.targets_at(gen.osc.theta), "qd": np.zeros(12), "c": np.zeros(4, dtype=int)}
    def _step_view():
        _step()
        c = foot_contacts(model, data)
        for i, leg in enumerate(LEG_ORDER):
            if c[i] != state["c"][i]:
                print(f"[{data.time:6.3f} s] {leg} {'접지' if c[i] else '이지'}   (닿은 발: {' '.join(l for j, l in enumerate(LEG_ORDER) if c[j]) or '없음'})")
        state["c"] = c
    play(model, data, step_fn=_step_view, duration=4.0, title="trot 2 Hz: 발이 닿고 뜰 때마다 터미널에 기록")
