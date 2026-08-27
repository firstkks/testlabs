# labs/ch13-wider-world/02_mjx_benchmark.py
"""13.2 더 빠른 학습: MJX와 대규모 병렬화로 가는 길.

같은 quadruped.xml을 MJX(JAX)에 올려 배치 크기별 처리량을 잰다. 이 책의 노트북에는 GPU가 없으므로 CPU 위의 JAX다.
비교: MuJoCo C 엔진 한 인스턴스(4장), SubprocVecEnv 8개(11장).
그림: out/ch13-fig05-mjx-throughput.png
실행: uv run python 02_mjx_benchmark.py   (uv sync --extra mjx 필요)
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from quadbook import model_path
from quadbook.robot import load
from train_lib import OUT

model, data = load(model_path())
print("JAX 장치:", jax.devices())

# MuJoCo C: 한 인스턴스
n = 2000; t0 = time.perf_counter()
for _ in range(n): mujoco.mj_step(model, data)
c_rate = n / (time.perf_counter() - t0)
print(f"MuJoCo (C, 1 instance): {c_rate:,.0f} 물리 스텝/s")

# MJX: 모델을 장치에 올리고 배치 step을 jit
mx = mjx.put_model(model)
dx0 = mjx.put_data(model, data)
step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
rates = {}
for B in (1, 16, 128, 1024):
    batch = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None], B, axis=0), dx0)
    t0 = time.perf_counter(); batch = step(mx, batch); jax.block_until_ready(batch); compile_s = time.perf_counter() - t0
    k = 20; t0 = time.perf_counter()
    for _ in range(k): batch = step(mx, batch)
    jax.block_until_ready(batch); el = time.perf_counter() - t0
    rates[B] = B * k / el
    print(f"MJX batch {B:5d}: 컴파일 {compile_s:5.1f} s | {rates[B]:10,.0f} 물리 스텝/s (배치 하나당 {1000*el/k:.1f} ms)")

fig, ax = plt.subplots(figsize=(7, 3.4))
labels = ["MuJoCo C x1"] + [f"MJX B={B}" for B in rates]; vals = [c_rate] + list(rates.values())
ax.bar(labels, vals, color=["#4a90d9"] + ["#9467bd"] * len(rates)); ax.set_yscale("log"); ax.set_ylabel("physics steps / s (log)"); ax.set_title("throughput on this laptop CPU: MuJoCo vs MJX (JAX on CPU)", fontsize=10); ax.tick_params(axis="x", labelsize=8)
fig.tight_layout(); fig.savefig(OUT / "ch13-fig05-mjx-throughput.png", dpi=200); plt.close(fig)
print("saved", OUT / "ch13-fig05-mjx-throughput.png")
