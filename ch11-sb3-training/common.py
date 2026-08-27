# labs/ch11-sb3-training/common.py
"""11장 공용: 9.2절 표 9-2의 PPO 설정과 환경 묶음 생성. 03이 학습, 04·05가 읽는다."""

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from quadbook.env import QuadrupedEnv

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
N_ENVS = 8
PPO_KWARGS = dict(                    # 표 9-2
    n_steps=2048, batch_size=256, n_epochs=10, clip_range=0.2, gamma=0.99, gae_lambda=0.95,
    learning_rate=3e-4, ent_coef=0.0, policy_kwargs=dict(log_std_init=-1.0), device="cpu", seed=0,
)


def make_envs(n_envs=N_ENVS, seed=0, subproc=True, **env_kwargs):
    """Monitor로 감싼 QuadrupedEnv n개를 병렬로. 반환값은 아직 정규화 전."""
    cls = SubprocVecEnv if subproc else None
    return make_vec_env(QuadrupedEnv, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs, vec_env_cls=cls)


def normalized(venv, training=True):
    return VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=PPO_KWARGS["gamma"], training=training)


def load_trained(name="ppo_first", n_envs=1, **env_kwargs):
    """학습된 정책과 정규화 통계를 불러 평가용 환경과 함께 돌려준다 (통계는 고정)."""
    venv = make_envs(n_envs=n_envs, seed=123, subproc=False, **env_kwargs)
    venv = VecNormalize.load(str(OUT / f"{name}_vecnormalize.pkl"), venv)
    venv.training = False; venv.norm_reward = False
    model = PPO.load(str(OUT / f"{name}.zip"), env=venv, device="cpu")
    return model, venv
