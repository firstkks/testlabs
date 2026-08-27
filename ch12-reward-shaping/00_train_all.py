# labs/ch12-reward-shaping/00_train_all.py
"""12장의 모든 학습을 순서대로 돌리고 각각을 바로 평가한다 (약 75분, CPU). 이미 있는 결과는 건너뛴다.
실행: cd ch12-reward-shaping && uv run python 00_train_all.py [이름 ...]
"""

import sys
from pathlib import Path

from train_lib import CONFIGS, OUT, evaluate, evaluate_baseline, train

if __name__ == "__main__":
    names = sys.argv[1:] or list(CONFIGS)
    if not (OUT / "metrics_baseline.json").exists():
        m = evaluate_baseline(); print(f"baseline: 속도 {m['speed']:.3f}, 밀기 한계 {m['push_limit']} N", flush=True)
    if not (OUT / "metrics_ppo_first.json").exists():
        m = evaluate("ppo_first"); print(f"ppo_first: 속도 {m['speed']:.3f}, yaw {m['yaw_deg']:+.0f}°, 밀기 한계 {m['push_limit']} N", flush=True)
    for name in names:
        if (OUT / f"metrics_{name}.json").exists():
            print(f"{name}: 이미 있음, 건너뜀", flush=True); continue
        el = train(name)
        m = evaluate(name)
        print(f"{name}: {el/60:.1f}분 | 속도 {m['speed']:.3f} m/s, yaw {m['yaw_deg']:+.0f}°, 조각 {[round(f, 1) for f in m['fragments']] if m['fragments'] else None}, 충격 {m['impact']:.0f} N, CoT {m['cot']:.2f}, 넘어짐 {m['fall_rate']:.0%}, 밀기 한계 {m['push_limit']} N", flush=True)
