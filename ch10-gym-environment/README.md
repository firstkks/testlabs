# 10장. 환경 만들기: Gymnasium

이 장에서 `quadbook/env.py`의 `QuadrupedEnv`(Gymnasium 환경, id `QuadBook/Quadruped-v0`)가 생기고 11~12장이 그대로 학습에 쓴다.
행동 = 7장 생성기 목표 + 0.1 rad 오프셋, 관측 53차원, 보상 = 가중합(항은 `info["terms"]`), 넘어지면 terminated, 10초면 truncated.

| 스크립트 | 절 | 내용 |
|---|---|---|
| `01_env_interface.py` | 10.1 | 공간·env_checker·행동 0 vs 난수 행동·처리량(초당 스텝) |
| `02_observation.py` | 10.2 | 관측 53차원의 묶음별 추이와 스케일 |
| `03_action.py` | 10.3 | 보정 크기 0.02~0.4 rad 스윕, 절대 모드(행동 = 관절 목표) 비교 |
| `04_reward.py` | 10.4 | 네 가지 행동에서 보상 항의 날것 크기와 가중치 적용 후 기여 |
| `05_termination.py` | 10.5 | 에피소드 길이 분포(terminated/truncated), 종료 순간 장면, 리셋 무작위화 |

실행은 labs/ 에서 `uv run python ch10-gym-environment/<script>.py`, 장면을 보려면 `--view --speed 0.5`(macOS는 `mjpython`).
