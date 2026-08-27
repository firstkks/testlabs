# 9장. 강화학습이라는 보정 도구

9장은 코드 없는 개념 장이다. 이 폴더에는 보조 실험(학습 전 무작위 정책이 얼마나 걷는가) 하나만 둔다. 본문 도해 스크립트는 labs_dev/figures 에 있다. 학습 코드는 10~12장에 있다.

| 스크립트 | 절 | 내용 |
|---|---|---|
| `01_random_policies.py` | 9.3 | 무작위 토크 / 무작위 관절 목표 / CPG + 무작위 잔차, 20 에피소드씩 전진 거리와 넘어짐 비율 |

실행은 labs/ 에서 `uv run python ch09-rl-refinement/<script>.py`. `01 --view --speed 0.5`로 CPG + 무작위 잔차 에피소드를 볼 수 있다(macOS는 `mjpython`).
