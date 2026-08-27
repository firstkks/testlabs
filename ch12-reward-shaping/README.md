# 12장. 보상을 조각하다

`train_lib.py`에 보상 설정(`CONFIGS`: R1~R4 통제 실험, H1~H4 해킹 박물관, C_scratch/C_curriculum 커리큘럼), `train()`, `evaluate()`(8장 지표 + 밀기 한계 → `out/metrics_<name>.json`)가 있다.
`00_train_all.py`가 모든 학습을 순서대로 돌리고(14개 설정, 약 2시간, CPU) 바로 평가한다. 이미 있는 결과는 건너뛴다. 01~05는 그 결과를 읽는다.

| 스크립트 | 절 | 내용 |
|---|---|---|
| `00_train_all.py` | 전체 | 10개 설정 학습 + 평가 (`uv run python 00_train_all.py [이름 ...]`) |
| `01_controlled.py` | 12.1 | 방향 항 하나만 더한 R1 vs 11장 정책: 위에서 본 경로, 지표 막대 |
| `02_terms.py` | 12.2 | R1 → R2(속도 추종) → R3(접촉 일치): 속도 추이, gait diagram |
| `03_hacking.py` | 12.3 | H1(전진만) H2(살아 있기+에너지) H3(충격 과다) H4(접촉 과다)의 장면과 지표 (`--view H1` 등으로 재생) |
| `04_curriculum.py` | 12.4 | 외란 학습 여섯 조건(scratch 60 N, 20 N 고정, 램프 3M, 램프 6M+유지, 80 N 램프, 보정 폭 0.15): 학습 곡선, 밀기 생존율(`out/push_survival.json`) |
| `05_final.py` | 12.5 | 사슬 전체 지표, 최종 정책(걸음을 지킨 것 중 가장 잘 버티는 커리큘럼 = C_curriculum_6M)의 Froude 좌표·개 trot과의 gait diagram·프레임 (`--view` 재생) |

11장 산출물(`../ch11-sb3-training/out/ppo_first*`)을 읽으므로 11장 03을 먼저 돌려 두어야 한다. 새 보상 항과 무작위 밀기는 `quadbook/env.py`에 있다(`DEFAULT_WEIGHTS`, `push_max`).

주의: VecEnv 안의 환경 속성은 `env_method("set_push_max", v)`처럼 메서드로 바꾼다. `set_attr`/`__setattr__`는 Monitor 래퍼에만 쓰여 환경에 반영되지 않는다.
