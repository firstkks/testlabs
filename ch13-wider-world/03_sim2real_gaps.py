# labs/ch13-wider-world/03_sim2real_gaps.py
"""13.3 시뮬레이션 밖으로: sim-to-real의 현실.

실제 로봇에서 반드시 만나는 간극 셋을 시뮬레이션 안에서 흉내 내어 최종 정책을 시험한다.
(a) 관측 잡음: 정규화 전 관측에 표준편차 0.01~0.1의 가우시안 잡음.
(b) 행동 지연: 정책의 출력이 1~3 제어 스텝(10~30 ms) 늦게 도착.
(c) 없는 센서: 몸통 선속도를 지운 채 학습한 A_no_linvel과 R4_full(선속도 있음)을 같은 자로 비교. 선속도를 지운 정책에 지형·밀기를 준다.
그림: out/ch13-fig06-noise-latency.png, out/ch13-fig07-no-linvel.png
실행: uv run python 03_sim2real_gaps.py   (00_train_all.py 뒤)
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_lib import OUT, evaluate

FINAL = "C_curriculum_6M"
print("(a) 관측 잡음 (정규화 전 관측에 더한 가우시안, 5 seed)")
noise_res = {}
for sd in (0.0, 0.01, 0.03, 0.1):
    m = evaluate(FINAL, obs_noise=sd); noise_res[sd] = m
    print(f"    σ {sd:4.2f}: 속도 {m['speed']:.3f} m/s, 넘어짐 {m['fall_rate']:.0%}, yaw {m['yaw_deg']:+.0f}°, 충격 {m['impact']:.0f} N")
print("(b) 행동 지연 (제어 스텝 = 10 ms)")
lat_res = {}
for k in (0, 1, 2, 3):
    m = evaluate(FINAL, latency_steps=k); lat_res[k] = m
    print(f"    {k} 스텝 ({10*k} ms): 속도 {m['speed']:.3f} m/s, 넘어짐 {m['fall_rate']:.0%}, yaw {m['yaw_deg']:+.0f}°, 충격 {m['impact']:.0f} N")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].plot(list(noise_res), [noise_res[s]["speed"] for s in noise_res], "o-", label="speed [m/s]"); axes[0].plot(list(noise_res), [noise_res[s]["fall_rate"] for s in noise_res], "s--", label="fall rate")
axes[0].set_xlabel("observation noise std"); axes[0].set_title("(a) observation noise", fontsize=10); axes[0].legend(fontsize=8)
axes[1].plot(list(lat_res), [lat_res[k]["speed"] for k in lat_res], "o-", label="speed [m/s]"); axes[1].plot(list(lat_res), [lat_res[k]["fall_rate"] for k in lat_res], "s--", label="fall rate")
axes[1].set_xlabel("action latency [control steps of 10 ms]"); axes[1].set_title("(b) action latency", fontsize=10); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch13-fig06-noise-latency.png", dpi=200); plt.close(fig)

print("(c) 몸통 선속도 없이 배운 정책 (A_no_linvel) vs 있는 정책 (R4_full), 같은 보상·같은 3M")
rows = {}
for name in ("R4_full", "A_no_linvel"):
    base = evaluate(name); rough = evaluate(name, model_file=str(__import__("quadbook").model_path().parent / "quadruped_rough.xml")); lat = evaluate(name, latency_steps=2)
    rows[name] = (base, rough, lat)
    print(f"    {name:12s} 평지: 속도 {base['speed']:.3f}, yaw {base['yaw_deg']:+.0f}°, 충격 {base['impact']:.0f} N, CoT {base['cot']:.2f} | 거친 지형: 속도 {rough['speed']:.3f}, 넘어짐 {rough['fall_rate']:.0%} | 지연 2스텝: 속도 {lat['speed']:.3f}, 넘어짐 {lat['fall_rate']:.0%}")
fig, ax = plt.subplots(figsize=(7, 3.2)); x = np.arange(3)
for k, (name, c) in enumerate((("R4_full", "#8c564b"), ("A_no_linvel", "#e377c2"))):
    ax.bar(x + (k - 0.5) * 0.4, [r["speed"] for r in rows[name]], 0.4, color=c, label=name + (" (with body velocity)" if name == "R4_full" else " (without)"))
ax.set_xticks(x); ax.set_xticklabels(["flat", "rough terrain", "latency 20 ms"]); ax.set_ylabel("speed [m/s]"); ax.set_title("does the policy need the simulator's body velocity?", fontsize=10); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch13-fig07-no-linvel.png", dpi=200); plt.close(fig)
json.dump({"noise": {str(k): v for k, v in noise_res.items()}, "latency": {str(k): v for k, v in lat_res.items()}, "linvel": {n: [r for r in rows[n]] for n in rows}}, open(OUT / "sim2real.json", "w"), indent=1, default=float)
print("saved", OUT / "ch13-fig06-noise-latency.png", OUT / "ch13-fig07-no-linvel.png")
