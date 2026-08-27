# 6장. 서기부터: 관절 제어의 기초

이 장에서 `quadbook/control.py`(JointPD, gravity_compensation, joint_inertia_diag, critical_kd, torso_roll_pitch)와 `quadbook.robot.STAND_POSE_BALANCED`가 생기고, 7장 이후가 그대로 쓴다.

| 스크립트 | 절 | 내용 |
|---|---|---|
| `01_torque_vs_position.py` | 6.1 | 시험대 다리: 토크만 주면 중력 균형 각도에서 멈춤 vs position 액추에이터. 완성 로봇 착지: kp 40 vs 400의 충격 |
| `02_pd_intuition.py` | 6.2 | 단진자: P만 / PD / PD+중력 보상(qfrc_bias). 관절 관성으로 ω·ζ·overshoot 예측과 측정 대조 |
| `03_gain_tuning.py` | 6.3 | 완성 로봇 스텝 응답(kp 10·40·160, ζ=1), ζ 비교, 제어 주기(500/100/50 Hz)와 떨림 |
| `04_stand_robust.py` | 6.4 | 하중 균형(엉덩이 0.9 rad), 밀기 한계, 발 벌림·웅크림의 효과, IMU 자세 피드백이 안 되는 이유 |

실행은 labs/ 에서 `uv run python ch06-standing-control/<script>.py`, 장면을 보려면 `--view --speed 0.5`(macOS는 `mjpython`).
