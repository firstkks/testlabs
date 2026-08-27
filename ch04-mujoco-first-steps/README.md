# 4장. MuJoCo 첫걸음

| 스크립트 | 절 | 내용 |
|---|---|---|
| `01_hello_mujoco.py` | 4.2 | XML 문자열로 모델 만들기, mj_step으로 1초 시뮬레이션, 접촉 개수 확인 → `out/ch04-fig01-hello-drop.png` |
| `02_viewer.py` | 4.2 | launch_passive로 창 띄우기. **macOS는 `uv run mjpython`으로 실행** |
| `03_mjmodel_mjdata.py` | 4.3 | mjModel(상수)과 mjData(상태)의 분업, mj_forward와 mj_step의 차이, ctrl → `out/ch04-fig03-two-link-poses.png` |
| `04_timestep_contact.py` | 4.4 | timestep·적분기에 따른 에너지 폭주, mjContact 읽기, solref와 침투 깊이 → `out/ch04-fig04-double-pendulum.png`, `out/ch04-fig05-timestep-stability.png` |
| `05_python_control.py` | 4.5 | motor + 파이썬 PD vs position 액추에이터 → `out/ch04-fig06-pendulum-control.png`, `out/ch04-fig07-control-response.png` |
| `06_render_offscreen.py` | 4.2 | mujoco.Renderer로 창 없이 그림 저장 → `out/ch04-fig02-first-scene.png` |

실행은 labs/ 에서 `uv run python ch04-mujoco-first-steps/<script>.py`.
