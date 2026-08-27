# labs/ch13-wider-world/00_train_all.py
"""13장의 학습 세 개를 순서대로 돌리고 평가한다 (약 20분, CPU). 실행: cd ch13-wider-world && uv run python 00_train_all.py"""
import json
import sys
from train_lib import CONFIGS, OUT, evaluate, train

if __name__ == "__main__":
    for name in (sys.argv[1:] or list(CONFIGS)):
        if (OUT / f"metrics_{name}.json").exists():
            print(f"{name}: 이미 있음", flush=True); continue
        el = train(name); m = evaluate(name); json.dump(m, open(OUT / f"metrics_{name}.json", "w"), indent=1)
        print(f"{name}: {el/60:.1f}분 | 속도 {m['speed']:.3f} m/s, yaw {m['yaw_deg']:+.0f}°, 넘어짐 {m['fall_rate']:.0%}, 조각 {[round(f,1) for f in m['fragments']] if m['fragments'] else None}, 충격 {m['impact']:.0f} N, CoT {m['cot']:.2f}, freq {m['freq']:.2f} Hz, L {m['step_length']:.3f} m", flush=True)
