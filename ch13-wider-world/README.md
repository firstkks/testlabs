# 13장. 더 넓은 세계로

`train_lib.py`에 13장의 학습 설정 셋(선속도 없는 관측, 목표 0.8 m/s 잔차 전용, 생성기 개방)과 평가 도우미가 있다(12장 train_lib 복사, 장 폴더 간 import 금지).
12장 산출물(`../ch12-reward-shaping/out/C_curriculum_6M*`, `R4_full*`)을 읽는다. MJX 벤치마크는 `uv sync --extra mjx` 뒤에 돈다.

| 스크립트 | 절 | 내용 |
|---|---|---|
| `00_train_all.py` | 13.1, 13.3 | A_no_linvel, B_residual_08, B_modulate_08 학습 + 평가 (약 20분) |
| `01_robustness.py` | 13.1 | 최종 정책 vs 개루프: 거친 지형(`models/quadruped_rough.xml`), 경사, 마찰, 짐 |
| `02_mjx_benchmark.py` | 13.2 | 같은 XML을 MJX에 올려 배치별 처리량 (CPU JAX) |
| `03_sim2real_gaps.py` | 13.3 | 관측 잡음, 행동 지연, 몸통 선속도 없이 배운 정책 비교 |
| `04_open_generator.py` | 13.1 | 생성기 주파수·보폭을 정책에 열었을 때: 속도 추이, 정책이 고른 매개변수, 동적 유사성 좌표 |
| `05_slope_anatomy.py` | 13.3 | 중력을 기울인 경사 vs 바닥을 기울인 경사(`models/quadruped_slope*.xml`)에서의 정책 |

환경의 13장 옵션(`quadbook/env.py`): `use_lin_vel`, `modulate_generator`(행동 14차원), `model_file`, `obs_noise`, `latency_steps`.
실행은 labs/ 에서 `cd ch13-wider-world && uv run python 00_train_all.py` 뒤 01~05. `01 --view --speed 0.5`로 거친 지형을 본다(macOS는 `mjpython`).
