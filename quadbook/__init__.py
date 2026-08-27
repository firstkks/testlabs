"""책 전체에서 공유하는 실습 유틸리티.

각 장의 스크립트는 여기 있는 함수를 import해서 쓴다.
- assets: 로봇 모델(MJCF) 경로
- gait:   발 접촉 로깅과 gait diagram (3장·8장·12장)
"""

from pathlib import Path

try:  # 그림 기본값: 범례는 반투명하게, 자동 위치로 (데이터를 가리지 않게)
    import matplotlib as _mpl
    _mpl.rcParams["legend.framealpha"] = 0.55
    _mpl.rcParams["legend.loc"] = "best"
except Exception:
    pass

try:  # 그림 기본값: 범례는 반투명하게, 자동 위치로 (데이터를 가리지 않게)
    import matplotlib as _mpl
    _mpl.rcParams["legend.framealpha"] = 0.55
    _mpl.rcParams["legend.loc"] = "best"
except Exception:
    pass

try:  # 그림 기본값: 범례는 반투명하게, 자동 위치로 (데이터를 가리지 않게)
    import matplotlib as _mpl
    _mpl.rcParams["legend.framealpha"] = 0.55
    _mpl.rcParams["legend.loc"] = "best"
except Exception:
    pass

LABS_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = LABS_ROOT / "models"


def model_path(name: str = "quadruped.xml") -> Path:
    """models/ 아래 MJCF 파일의 절대 경로를 돌려준다."""
    path = MODELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {path} (5장에서 만듭니다)")
    return path
