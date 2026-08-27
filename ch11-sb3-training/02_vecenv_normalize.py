# labs/ch11-sb3-training/02_vecenv_normalize.py
"""11.2 병렬 환경(VecEnv)과 정규화(VecNormalize).

(a) 환경 1개, DummyVecEnv 8개(한 프로세스), SubprocVecEnv 8개(프로세스 8개)의 처리량.
(b) VecNormalize가 관측의 묶음별 크기를 어떻게 맞추는지: 정규화 전후의 표준편차. 보상 정규화의 크기.
그림: out/ch11-fig02-throughput.png, out/ch11-fig03-normalization.png
실행: uv run python ch11-sb3-training/02_vecenv_normalize.py
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import OUT, make_envs, normalized

GROUPS = [("joint offset", 0, 12), ("joint vel", 12, 24), ("gravity", 24, 27), ("gyro", 27, 30), ("lin vel", 30, 33), ("phase", 33, 41), ("prev action", 41, 53)]


def throughput(venv, steps=400):
    venv.reset(); n = venv.num_envs; t0 = time.perf_counter()
    for _ in range(steps):
        venv.step(np.random.uniform(-1, 1, (n, 12)))
    return steps * n / (time.perf_counter() - t0)


if __name__ == "__main__":
    rates = {}
    rates["1 env"] = throughput(make_envs(n_envs=1, subproc=False))
    rates["8 envs, one process"] = throughput(make_envs(n_envs=8, subproc=False))
    sub = make_envs(n_envs=8, subproc=True); rates["8 envs, 8 processes"] = throughput(sub); sub.close()
    print("(a) 처리량 [제어 스텝/s]")
    for k, v in rates.items(): print(f"    {k:22s} {v:8,.0f}  ({v*0.01:.0f}배속)")
    fig, ax = plt.subplots(figsize=(6, 3.2)); ax.bar(list(rates), list(rates.values()), color="#4a90d9"); ax.set_ylabel("control steps / s"); ax.set_title("environment throughput", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "ch11-fig02-throughput.png", dpi=200); plt.close(fig)

    # (b) 정규화
    raw = make_envs(n_envs=8, subproc=False); venv = normalized(raw)
    obs = venv.reset(); raw_obs, norm_obs, raw_r, norm_r = [], [], [], []
    for _ in range(1500):
        a = np.random.uniform(-0.3, 0.3, (8, 12))
        obs, r, done, info = venv.step(a)
        norm_obs.append(obs.copy()); raw_obs.append(venv.get_original_obs().copy()); norm_r.append(r.copy()); raw_r.append(venv.get_original_reward().copy())
    raw_obs, norm_obs = np.concatenate(raw_obs), np.concatenate(norm_obs); raw_r, norm_r = np.concatenate(raw_r), np.concatenate(norm_r)
    print("(b) 12,000 스텝 뒤 VecNormalize의 통계: 묶음별 표준편차 (원 관측 → 정규화 관측)")
    for name, a, b in GROUPS:
        print(f"    {name:12s} {raw_obs[:, a:b].std():.3f} → {norm_obs[:, a:b].std():.3f}")
    print(f"    보상: 원 보상 σ {raw_r.std():.3f}, 정규화 보상 σ {norm_r.std():.3f} (수익의 이동 표준편차로 나눈다)")
    fig, ax = plt.subplots(figsize=(8, 3.4)); x = np.arange(len(GROUPS))
    ax.bar(x - 0.2, [raw_obs[:, a:b].std() for _, a, b in GROUPS], 0.4, label="raw observation"); ax.bar(x + 0.2, [norm_obs[:, a:b].std() for _, a, b in GROUPS], 0.4, label="after VecNormalize")
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in GROUPS], fontsize=8); ax.set_yscale("log"); ax.set_ylabel("std (log)"); ax.set_title("VecNormalize equalizes observation groups", fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "ch11-fig03-normalization.png", dpi=200); plt.close(fig)
    print("saved", OUT / "ch11-fig02-throughput.png", OUT / "ch11-fig03-normalization.png")
