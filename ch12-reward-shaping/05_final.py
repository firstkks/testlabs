# labs/ch12-reward-shaping/05_final.py
"""12.5 어디까지 좋아졌나: 최종 gait 분석과 동물 보행과의 재비교.

이 장의 사슬(baseline → ppo_first → R1 → R2 → R3 → R4 → C_curriculum)을 한 표에 놓고, 최종 정책을 8.4절의 동적 유사성 좌표와 개의 trot 옆에 놓는다.
그림: out/ch12-fig09-chain.png, out/ch12-fig10-froude-final.png, out/ch12-fig11-final-vs-dog.png, out/ch12-fig12-final-frames.png
실행: uv run python 05_final.py   (--view --speed 0.5: 최종 정책 재생)
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from quadbook.analysis import alexander_stride, froude
from quadbook.gait import GAIT_PHASES, LEG_ORDER, contacts_to_intervals, gait_intervals, plot_gait_diagram
from quadbook.render import play, want_view
from train_lib import OUT, load_policy, run_episode

surv = json.load(open(OUT / "push_survival.json")); forces = surv["forces"]
cands = [n for n in ("C_curriculum_push", "C_curriculum_6M", "C_curriculum_80", "C_curriculum_6M_wide") if abs(json.load(open(OUT / f"metrics_{n}.json"))["yaw_deg"]) < 15]
FINAL = max(cands, key=lambda n: (surv[n][forces.index(60)], surv[n][forces.index(50)], surv[n][forces.index(70)]))   # 걸음을 지킨 것 중 가장 잘 버티는 것
print("최종 정책으로 고른 것:", FINAL)
chain = ("baseline", "ppo_first", "R1_heading", "R2_track", "R3_contact", "R4_full", FINAL)
M = {n: json.load(open(OUT / f"metrics_{n}.json")) for n in chain}
print("사슬 (5 seed 평균)            속도    yaw[°]  넘어짐  β 뒤(LH RH)   조각 뒤(LH RH)  충격[N]  여유[cm]  CoT   |δ|[°]  밀기[N]")
for n in chain:
    m = M[n]; d = m["duty"]; f = m["fragments"]
    print(f"  {n:18s} {m['speed']:.3f}  {m['yaw_deg']:+6.0f}  {m['fall_rate']:5.0%}  {[round(b, 2) for b in d[2:]] if d else 'n/a'}   {[round(x, 1) for x in f[2:]] if f else 'n/a'}     {m['impact']:5.0f}  {100*m['margin']:+6.1f}  {m['cot']:.2f}   {m['delta_deg']:4.1f}   {m['push_limit']}")

fig, axes = plt.subplots(2, 3, figsize=(12, 6))
keys = [("speed", "speed [m/s]"), ("yaw_deg", "yaw after 10 s [deg]"), ("impact", "touchdown peak force [N]"), ("cot", "cost of transport"), ("frag_hind", "hind-leg contact fragments per cycle"), ("margin", "support margin [m]")]
for n in chain: M[n]["frag_hind"] = float(np.mean(M[n]["fragments"][2:]))
short = [n.replace("_push", "").replace("C_curriculum", "curriculum").replace("_", "\n") for n in chain]
for ax, (k, title) in zip(axes.flat, keys):
    vals = [abs(M[n][k]) if k == "yaw_deg" else M[n][k] for n in chain]
    ax.bar(range(len(chain)), vals, color=["#999999", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b", "#17becf"]); ax.set_xticks(range(len(chain))); ax.set_xticklabels(short, fontsize=6); ax.set_title(title, fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "ch12-fig09-chain.png", dpi=200); plt.close(fig)

# 최종 정책 상세: Froude, gait diagram, 프레임
model, venv, env = load_policy(FINAL)
lg = run_episode(venv, env, lambda o: model.predict(o, deterministic=True)[0], seed=7, frames_at=(3.0, 3.125, 3.25, 3.375), label="final policy")
Image.fromarray(np.concatenate(lg["frames"], axis=1)).save(OUT / "ch12-fig12-final-frames.png")
from quadbook.analysis import gait_metrics
mtr = gait_metrics(lg["t"], lg["contact"], t_from=2.0, x=lg["x"])
L = 0.30
pts = {"baseline trot (ch7)": (M["baseline"]["speed"], 0.145), "ch11 first policy": (M["ppo_first"]["speed"], None), "final (ch12)": (mtr["speed"], mtr["stride"])}
print(f"최종 정책 (seed 7): 속도 {mtr['speed']:.3f} m/s, 보폭 {mtr['stride']:.3f} m, Fr {froude(mtr['speed'], L):.3f}, 상대 보폭 {mtr['stride']/L:.2f} (Alexander 예측 {alexander_stride(froude(mtr['speed'], L)):.2f}), β {np.round(mtr['duty'], 2)}, φ {np.round(mtr['phase'], 2)}, 조각 {np.round(mtr['fragments'], 1)}")
fig, ax = plt.subplots(figsize=(7, 4)); frs = np.logspace(-2, 0.7, 100)
ax.plot(frs, alexander_stride(frs), color="#999999", label="Alexander & Jayes: 2.4 Fr^0.34")
ax.axvline(0.5, color="#bbbbbb", ls=":"); ax.text(0.5, 3.9, "walk / trot", fontsize=8, color="#666666", ha="center")
for name, (v, stride), c in (("baseline trot (ch7)", (M["baseline"]["speed"], 0.145), "#999999"), ("final policy (ch12)", (mtr["speed"], mtr["stride"]), "#17becf")):
    mk = "o"
    fr = froude(v, L); ax.plot(fr, stride / L, mk, color=c, ms=8); ax.annotate(name, (fr, stride / L), textcoords="offset points", xytext=(6, -3), fontsize=8)
for name, (Lh, v) in {"dog walk": (0.5, 1.0), "dog trot": (0.5, 2.5)}.items():
    fr = froude(v, Lh); ax.plot(fr, alexander_stride(fr), "s", color="#d62728"); ax.annotate(name, (fr, alexander_stride(fr)), textcoords="offset points", xytext=(6, -3), fontsize=8)
ax.set_xscale("log"); ax.set_ylim(0, 4.3); ax.set_xlabel("Froude number"); ax.set_ylabel("relative stride length"); ax.set_title("dynamic similarity, revisited", fontsize=10); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "ch12-fig10-froude-final.png", dpi=200); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
plot_gait_diagram(axes[0], gait_intervals(GAIT_PHASES["trot"], 0.42), x_max=2, color="#d62728", title="typical dog trot (beta 0.42)")
t, c = lg["t"], lg["contact"]; T = 0.5; t0 = [s for s, _ in contacts_to_intervals(t, c[:, 0]) if s > 3.0][0]; m = (t >= t0) & (t <= t0 + 2 * T)
iv = {leg: [((s - t0) / T, (e - t0) / T) for s, e in contacts_to_intervals(t[m], c[m][:, i])] for i, leg in enumerate(LEG_ORDER)}
plot_gait_diagram(axes[1], iv, x_max=2, color="#17becf", title=f"final policy, measured (beta {mtr['duty'].mean():.2f})")
fig.tight_layout(); fig.savefig(OUT / "ch12-fig11-final-vs-dog.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch12-fig09-chain.png", OUT / "ch12-fig10-froude-final.png", OUT / "ch12-fig11-final-vs-dog.png", OUT / "ch12-fig12-final-frames.png")

if want_view():
    state = {"obs": venv.reset()}
    def _step():
        a = model.predict(state["obs"], deterministic=True)[0]; state["obs"], _, _, _ = venv.step(a)
    play(env.model, env.data, step_fn=_step, duration=10.0, title="최종 정책 (R4 보상 + 밀기 커리큘럼)")
