# 7장. 첫 보행: 하드코딩으로 걷게 하기

이 장에서 `quadbook/cpg.py`(LegKinematics, FootTrajectory, PhaseOscillators, GaitGenerator)와 `quadbook/sim.py`(standard_pd, rollout, summary, WALK_KP)가 생기고, 8장 이후가 그대로 쓴다.

| 스크립트 | 절 | 내용 |
|---|---|---|
| `01_foot_trajectory.py` | 7.1 | 관절 사인파가 그리는 발 궤적 vs 설계한 발 궤적(지지기 직선 + 유각기 호), 역기구학, FK 검증 |
| `02_cpg_oscillator.py` | 7.2 | 결합 위상 진동자: 제멋대로 시작한 위상이 trot 위상표로 수렴, 외란 뒤 복귀. 진동자 → 궤적 → IK 파이프라인 |
| `03_trot.py` | 7.3 | trot 2 Hz. kp 40 → 속도 피드포워드 → kp 80의 세 단계 (24% → 60% → 91%) |
| `04_walk.py` | 7.4 | lateral-sequence walk (duty 0.75, 1 Hz)와 trot 비교 |
| `05_param_sweep.py` | 7.5 | 주파수 × 보폭, 발 높이, 위상표(trot·pace·bound·walk·pronk) |
| `06_open_loop_limits.py` | 7.6 | 명령 접촉 vs 실제 접촉, 걷는 중 옆 밀기, 경사(중력 기울임), 짐 2 kg, 마찰 |

실행은 labs/ 에서 `uv run python ch07-first-gait/<script>.py`, 장면을 보려면 `--view --speed 0.5`(macOS는 `mjpython`).
