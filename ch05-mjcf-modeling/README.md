# 5장. 나의 사족보행 로봇 만들기: MJCF 모델링

산출물은 `labs/models/quadruped.xml`(이 책의 주인공 로봇)과 `quadbook/robot.py`(이름·순서·서기 자세·도우미)이며, 6장 이후 모든 장이 이 둘을 읽는다.

| 스크립트 | 절 | 내용 |
|---|---|---|
| `01_torso_only.py` | 5.1 | 모델 v0: 몸통 하나. 트리·nq/nv·관성 자동 계산 확인 → `out/ch05-fig01-torso-drop.png` |
| `02_one_leg.py` | 5.2 | 모델 v1: 다리 하나를 시험대에 고정, 관절 범위와 발끝 작업 공간 → `out/ch05-fig02-one-leg-poses.png`, `out/ch05-fig03-leg-workspace.png` |
| `03_inspect_quadruped.py` | 5.3 | 완성 모델 검사(질량·대칭·발 위치)와 서기 자세 렌더링 → `out/ch05-fig04-quadruped-stand.png` |
| `04_sensors.py` | 5.4 | 낙하 중 IMU·발 접촉 센서 읽기 → `out/ch05-fig05-drop-frames.png`, `out/ch05-fig06-sensors-drop.png` |
| `05_validate.py` | 5.5 | 낙하(제어 유무)·밀기·의심 목록 → `out/ch05-fig07-validation-frames.png`, `out/ch05-fig08-validation.png` |

실행은 labs/ 에서 `uv run python ch05-mjcf-modeling/<script>.py`.
