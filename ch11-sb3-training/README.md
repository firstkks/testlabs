# 11장. SB3로 학습시키기

`common.py`에 표 9-2의 PPO 설정(`PPO_KWARGS`)과 환경 묶음(`make_envs`, `normalized`, `load_trained`)이 있다. 12장이 그대로 쓴다.
03이 학습해 `out/ppo_first.zip`, `out/ppo_first_vecnormalize.pkl`, `out/progress_ppo_first.csv`, `out/ppo_first_eval.npz`, `out/tb_ppo_first/`를 남기고 04·05가 읽는다.

| 스크립트 | 절 | 내용 |
|---|---|---|
| `01_ppo_setup.py` | 11.1 | 정책 신경망 구조·매개변수 수·갱신 한 번의 데이터 양, log_std_init 0 vs −1의 학습 전 행동 분포 |
| `02_vecenv_normalize.py` | 11.2 | 환경 1개 / 8개 한 프로세스 / 8개 8프로세스의 처리량, VecNormalize 전후의 관측·보상 스케일 |
| `03_train_first.py` | 11.3 | log_std_init 비교(각 300k) + 본 학습 3M 스텝(약 8분, CPU), 100k마다 결정론적 평가 |
| `04_curves.py` | 11.4 | progress.csv의 곡선 여섯 개(수익, 길이, KL, clip fraction, explained variance, std)와 평가 곡선 |
| `05_compare.py` | 11.5 | 베이스라인 vs 학습된 정책: 속도·CoT·8장 gait_metrics·안정 여유·접지 충격·δ의 모양·위에서 본 경로·밀기 |

실행은 labs/ 에서 `cd ch11-sb3-training && uv run python 03_train_first.py` 순서로(03 → 04 → 05). 텐서보드: `uv run tensorboard --logdir out/tb_ppo_first`. `05 --view --speed 0.5`로 학습된 정책을 본다(macOS는 `mjpython`).
