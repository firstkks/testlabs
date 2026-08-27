# labs/ch11-sb3-training/01_ppo_setup.py
"""11.1 SB3 구조와 PPO 설정.

PPO 객체를 만들고 안을 들여다본다: 정책 신경망(배우·비평가), 매개변수 수, 행동 분포의 표준편차(log_std), 한 번의 갱신이 처리하는 데이터 양.
log_std_init을 0(기본)과 −1로 두었을 때 학습 전 정책이 내는 행동의 분포를 비교한다 (11.3절 '베이스라인이 무너지지 않게'의 열쇠).
그림: out/ch11-fig01-initial-actions.png
실행: uv run python ch11-sb3-training/01_ppo_setup.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

from common import OUT, PPO_KWARGS, make_envs

if __name__ == "__main__":
    venv = make_envs(n_envs=1, subproc=False)
    model = PPO("MlpPolicy", venv, verbose=0, **PPO_KWARGS)
    print("정책 신경망:"); print(model.policy)
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"학습 매개변수 {n_params:,}개 | 행동 분포 log_std 초기값 {model.policy.log_std.data[0].item():.2f} → 표준편차 {np.exp(model.policy.log_std.data[0].item()):.2f}")
    print(f"한 번의 갱신: n_steps {PPO_KWARGS['n_steps']} × 환경 8 = {PPO_KWARGS['n_steps']*8:,} 스텝 수집 → batch {PPO_KWARGS['batch_size']} × {PPO_KWARGS['n_steps']*8//PPO_KWARGS['batch_size']} 미니배치 × {PPO_KWARGS['n_epochs']} epochs = {PPO_KWARGS['n_steps']*8//PPO_KWARGS['batch_size']*PPO_KWARGS['n_epochs']} 경사 스텝")
    print(f"제어 주기 0.01 s 이므로 갱신 한 번에 모으는 경험은 시뮬레이션 시간 {PPO_KWARGS['n_steps']*8*0.01:.0f} s = 10초 에피소드 약 {PPO_KWARGS['n_steps']*8*0.01/10:.0f}개")

    # 학습 전 정책의 행동 분포: log_std_init 0 vs -1
    obs = venv.reset()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, ls in zip(axes, (0.0, -1.0)):
        kw = dict(PPO_KWARGS); kw["policy_kwargs"] = dict(log_std_init=ls)
        m = PPO("MlpPolicy", venv, verbose=0, **kw)
        acts = np.array([m.predict(obs, deterministic=False)[0][0] for _ in range(2000)])
        sat = np.mean(np.abs(acts) >= 0.999)
        print(f"log_std_init {ls:+.0f}: 학습 전 행동 표준편차 {acts.std():.2f}, |a| = 1 로 잘린 비율 {100*sat:.0f}%, 관절 보정으로는 ±{0.1*acts.std():.3f} rad")
        ax.hist(acts.ravel(), bins=40, range=(-1, 1), color="#4a90d9"); ax.set_title(f"log_std_init = {ls:+.0f}: std {acts.std():.2f}, clipped {100*sat:.0f}%", fontsize=10); ax.set_xlabel("action (before x 0.1 rad)")
    axes[0].set_ylabel("count"); fig.tight_layout(); fig.savefig(OUT / "ch11-fig01-initial-actions.png", dpi=200); plt.close(fig)
    print("saved", OUT / "ch11-fig01-initial-actions.png")
