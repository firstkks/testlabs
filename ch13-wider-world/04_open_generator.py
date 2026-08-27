# labs/ch13-wider-world/04_open_generator.py
"""13.1 (후반) 생성기를 정책에 열기: 목표 0.8 m/s에서 잔차만(12차원) vs 주파수·보폭까지 조절(14차원).

12장 끝의 한계('생성기의 상수를 정책이 바꾸지 못하는 한 거기까지')에 대한 실험. 정책이 고른 주파수·보폭의 추이와 8.4절 동적 유사성 좌표.
그림: out/ch13-fig03-modulation.png, out/ch13-fig04-froude-modulated.png
실행: uv run python 04_open_generator.py   (00_train_all.py 뒤)
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quadbook.analysis import alexander_stride, froude
from train_lib import OUT, load_policy, run_episode

M = {n: json.load(open(OUT / f"metrics_{n}.json")) for n in ("B_residual_08", "B_modulate_08")}
print("5 seed 평균 (목표 0.8 m/s)      속도    yaw[°]  넘어짐  β(LF RF LH RH)             조각             충격[N]  CoT   주파수[Hz]  보폭 L[m]  실측 보폭 λ[m]")
for n, m in M.items():
    print(f"  {n:14s} {m['speed']:.3f}  {m['yaw_deg']:+6.0f}  {m['fall_rate']:5.0%}  {[round(b, 2) for b in m['duty']]}  {[round(f, 1) for f in m['fragments']]}   {m['impact']:5.0f}  {m['cot']:.2f}   {m['freq']:.2f}       {m['step_length']:.3f}      {m['stride']:.3f}")

logs = {}
for n in ("B_residual_08", "B_modulate_08"):
    model, venv, env = load_policy(n); logs[n] = run_episode(venv, env, lambda o: model.predict(o, deterministic=True)[0], seed=7); venv.close()
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for n, c in (("B_residual_08", "#8c564b"), ("B_modulate_08", "#17becf")):
    lg = logs[n]; k = 50; sm = np.convolve(lg["forward"], np.ones(k) / k, mode="valid")
    axes[0].plot(lg["t"][k - 1:], sm, color=c, label=n); axes[1].plot(lg["t"], lg["freq"], color=c, label=n); axes[2].plot(lg["t"], lg["step_length"], color=c, label=n)
axes[0].axhline(0.8, color="#d62728", ls=":", label="v_target 0.8"); axes[0].set_title("(a) forward speed", fontsize=10); axes[0].set_xlabel("time [s]"); axes[0].legend(fontsize=7)
axes[1].set_title("(b) generator frequency chosen by the policy [Hz]", fontsize=10); axes[1].set_xlabel("time [s]"); axes[1].set_ylim(1.4, 3.1)
axes[2].set_title("(c) generator step length chosen [m]", fontsize=10); axes[2].set_xlabel("time [s]"); axes[2].set_ylim(0.04, 0.15)
fig.tight_layout(); fig.savefig(OUT / "ch13-fig03-modulation.png", dpi=200); plt.close(fig)

L = 0.30
fig, ax = plt.subplots(figsize=(7, 4)); frs = np.logspace(-2, 0.7, 100)
ax.plot(frs, alexander_stride(frs), color="#999999", label="Alexander & Jayes: 2.4 Fr^0.34"); ax.axvline(0.5, color="#bbbbbb", ls=":"); ax.text(0.5, 3.9, "walk / trot", fontsize=8, color="#666666", ha="center")
pts = [("baseline trot (ch7)", 0.286, 0.145, "#999999"), ("final policy (ch12)", 0.486, 0.238, "#17becf"), ("residual only, 0.8 m/s", M["B_residual_08"]["speed"], M["B_residual_08"]["stride"], "#8c564b"), ("generator opened, 0.8 m/s", M["B_modulate_08"]["speed"], M["B_modulate_08"]["stride"], "#d62728")]
for name, v, stride, c in pts:
    fr = froude(v, L); ax.plot(fr, stride / L, "o", color=c, ms=8); ax.annotate(name, (fr, stride / L), textcoords="offset points", xytext=(6, -3), fontsize=8)
    print(f"  {name:26s}: Fr {fr:.3f}, 상대 보폭 {stride/L:.2f} (Alexander 예측 {alexander_stride(fr):.2f})")
for name, (Lh, v) in {"dog walk": (0.5, 1.0), "dog trot": (0.5, 2.5)}.items():
    fr = froude(v, Lh); ax.plot(fr, alexander_stride(fr), "s", color="#d62728"); ax.annotate(name, (fr, alexander_stride(fr)), textcoords="offset points", xytext=(6, -3), fontsize=8)
ax.set_xscale("log"); ax.set_ylim(0, 4.3); ax.set_xlabel("Froude number"); ax.set_ylabel("relative stride length"); ax.set_title("dynamic similarity with the generator opened", fontsize=10); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "ch13-fig04-froude-modulated.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch13-fig03-modulation.png", OUT / "ch13-fig04-froude-modulated.png")
