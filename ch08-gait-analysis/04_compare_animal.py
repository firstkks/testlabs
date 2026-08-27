# labs/ch08-gait-analysis/04_compare_animal.py
"""8.4 동물과 비교하기: 우리 trot은 개의 trot과 얼마나 다른가.

3.4절의 동적 유사성(Alexander & Jayes 1983): 다리 길이가 다른 동물도 같은 Froude 수 Fr = v²/(g·L)에서는 같은 걸음, 같은 상대 보폭 λ/L을 쓴다.
회귀식 λ/L ≈ 2.4·Fr^0.34 (cursorial 포유류), walk→trot 전환 Fr ≈ 0.5, trot→gallop 전환 Fr ≈ 2.5.
우리 로봇(엉덩이 높이 0.30 m)의 trot들을 이 좌표에 찍고, 개(다리 0.5 m, trot 2.5 m/s 가정)와 나란히 놓는다.
그림: out/ch08-fig07-froude.png, out/ch08-fig08-dog-vs-robot.png
실행: uv run python ch08-gait-analysis/04_compare_animal.py   (--view --speed 0.5: 가장 빠른 trot 3 Hz·보폭 0.12)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mujoco
from quadbook import model_path
from quadbook.analysis import alexander_stride, froude, gait_metrics, relative_stride
from quadbook.cpg import GaitGenerator
from quadbook.gait import GAIT_PHASES, gait_intervals, plot_gait_diagram, LEG_ORDER, contacts_to_intervals
from quadbook.render import play, want_view
from quadbook.robot import load
from quadbook.sim import WALK_KP, rollout, standard_pd

OUT = Path(__file__).resolve().parent / "out"
LEG = 0.30          # 서기 자세의 엉덩이 높이 (5장). Froude 수의 L
G = 9.81

# ---- 우리 로봇: 여러 trot ----
cases = [("trot 1 Hz L0.08", 1.0, 0.08), ("trot 2 Hz L0.08", 2.0, 0.08), ("trot 3 Hz L0.08", 3.0, 0.08), ("trot 3 Hz L0.12", 3.0, 0.12), ("trot 4 Hz L0.12", 4.0, 0.12)]
robot = []
print(f"우리 로봇 (L = {LEG} m):  Fr = v²/(g·L),  상대 보폭 λ/L")
for name, f, Lstep in cases:
    model, data = load(model_path()); pd = standard_pd(model, data, kp=WALK_KP)
    gen = GaitGenerator(gait="trot", freq=f, duty=0.5, step_length=Lstep, step_height=0.04)
    log = rollout(model, data, gen, pd, duration=6.0, control_every=5)
    mtr = gait_metrics(log["t"], log["contact"], t_from=2.0, x=log["x"])
    fr = froude(mtr["speed"], LEG); rs = relative_stride(mtr["stride"], LEG)
    robot.append((name, fr, rs, mtr))
    print(f"    {name}: v {mtr['speed']:.3f} m/s, λ {mtr['stride']:.3f} m → Fr {fr:.3f}, λ/L {rs:.2f} (Alexander 예측 {alexander_stride(fr):.2f}), β {mtr['duty'].mean():.2f}, 넘어짐 {bool(log['fell'].any())}")

# ---- 동물 예시 (가정값) ----
animals = {"dog trot (L 0.5 m, 2.5 m/s)": (0.5, 2.5), "dog walk (L 0.5 m, 1.0 m/s)": (0.5, 1.0), "horse trot (L 1.3 m, 4.0 m/s)": (1.3, 4.0)}
print("동물 (다리 길이·속도는 대표값 가정)")
for name, (Lh, v) in animals.items():
    fr = froude(v, Lh); print(f"    {name}: Fr {fr:.2f}, Alexander 상대 보폭 {alexander_stride(fr):.2f}")
v_needed = np.sqrt(0.5 * G * LEG)
print(f"우리 로봇이 동물의 walk→trot 전환점 Fr 0.5 에 이르려면 v = √(0.5·g·L) = {v_needed:.2f} m/s 가 필요")

fig, ax = plt.subplots(figsize=(7, 4))
frs = np.logspace(-2, 0.7, 100)
ax.plot(frs, alexander_stride(frs), color="#999999", label="Alexander & Jayes: 2.4 Fr^0.34")
ax.axvline(0.5, color="#bbbbbb", ls=":"); ax.text(0.5, 3.9, "walk / trot", fontsize=8, color="#666666", ha="center")
ax.axvline(2.5, color="#bbbbbb", ls=":"); ax.text(2.5, 3.9, "trot / gallop", fontsize=8, color="#666666", ha="center")
for name, fr, rs, _ in robot:
    ax.plot(fr, rs, "o", color="#1f77b4"); ax.annotate(name, (fr, rs), textcoords="offset points", xytext=(6, -3 if "L0.12" not in name else 6), fontsize=7)
for k, (name, (Lh, v)) in enumerate(animals.items()):
    fr = froude(v, Lh); ax.plot(fr, alexander_stride(fr), "s", color="#d62728"); ax.annotate(name.split(" (")[0], (fr, alexander_stride(fr)), textcoords="offset points", xytext=(6, -3 - 10 * (k == 2)), fontsize=7)
ax.set_xscale("log"); ax.set_ylim(0, 4.3); ax.set_xlabel("Froude number v^2 / (g L)"); ax.set_ylabel("relative stride length  stride / L")
ax.set_title("dynamic similarity: our trots (circles) vs animals (squares)", fontsize=10); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "ch08-fig07-froude.png", dpi=200); plt.close(fig)

# ---- gait diagram 나란히: 개의 전형적 trot (β 0.42, 위상표) vs 우리 trot 2 Hz 실측 ----
L = np.load(OUT / "logs.npz")
t, c, f = L["trot2_t"], L["trot2_contact"], float(L["trot2_freq"]); T = 1.0 / f
t0 = [s for s, _ in contacts_to_intervals(t, c[:, 0]) if s > 3.0][0]
m = (t >= t0) & (t <= t0 + 2 * T)
ours = {leg: [((s - t0) / T, (e - t0) / T) for s, e in contacts_to_intervals(t[m], c[m][:, i])] for i, leg in enumerate(LEG_ORDER)}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
plot_gait_diagram(axes[0], gait_intervals(GAIT_PHASES["trot"], 0.42), x_max=2, color="#d62728", title="(a) typical dog trot (beta 0.42, with flight)")
plot_gait_diagram(axes[1], ours, x_max=2, color="#333333", title="(b) our trot 2 Hz, measured (beta 0.55)")
fig.tight_layout(); fig.savefig(OUT / "ch08-fig08-dog-vs-robot.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch08-fig07-froude.png", OUT / "ch08-fig08-dog-vs-robot.png")

if want_view():
    model, data = load(model_path()); pd = standard_pd(model, data, kp=WALK_KP)
    gen = GaitGenerator(gait="trot", freq=3.0, duty=0.5, step_length=0.12, step_height=0.04)
    state = {"i": 0, "q": gen.targets_at(gen.osc.theta), "qd": np.zeros(12)}
    def _step():
        if data.time >= 0.5 and state["i"] % 5 == 0: state["q"], state["qd"] = gen.targets_with_velocity(model.opt.timestep * 5)
        data.ctrl[:] = pd.torque(data, state["q"], state["qd"]); mujoco.mj_step(model, data); state["i"] += 1
    play(model, data, step_fn=_step, duration=6.0, title="가장 빠른 trot: 3 Hz, 보폭 0.12 (약 0.74 m/s, Fr 0.19)")
