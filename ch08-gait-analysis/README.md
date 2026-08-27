# 8장. 걸음을 측정하다: Gait 분석 실습

이 장에서 `quadbook/analysis.py`(dominant_period, gait_metrics, support_margin, froude, relative_stride, alexander_stride)가 생기고, 12장의 보상 설계가 그대로 쓴다.
01이 접촉 로그를 `out/logs.npz`에 저장하고 02·03·04·05가 읽는다. 반드시 01부터 실행한다.

| 스크립트 | 절 | 내용 |
|---|---|---|
| `01_contact_logging.py` | 8.1 | 접촉 목록(기하) vs touch 센서(힘), 문턱값, 접지 충격. trot 2 Hz·trot 1 Hz·walk 1 Hz 로그 저장 |
| `02_gait_diagram.py` | 8.2 | 실측 gait diagram과 명령 도해를 나란히 (3장의 plot_gait_diagram) |
| `03_gait_metrics.py` | 8.3 | β·T·φ·λ·v 를 3.2절 정의대로 계산, trot 주파수 스윕에서 β와 φ의 변화 |
| `04_compare_animal.py` | 8.4 | Froude 수와 상대 보폭: Alexander & Jayes의 동적 유사성 선 위에 우리 trot과 개·말을 놓는다 |
| `05_fall_anatomy.py` | 8.5 | 60 N 밀기 넘어짐의 시간축(안정 여유·roll·접촉), 10도 오르막 제자리걸음(발 미끄러짐·토크 포화), 1 Hz trot의 걸음 변형 |

실행은 labs/ 에서 `uv run python ch08-gait-analysis/<script>.py`, 장면을 보려면 `--view --speed 0.5`(macOS는 `mjpython`).
