"""gait diagram 유틸리티 (3장 도해, 8장·12장 실측 분석에서 공유).

- gait_intervals(): 위상표(φ)와 duty factor(β)로부터 각 다리의 지지기 구간을 만든다 (이상적 걸음).
- plot_gait_diagram(): 지지기 구간을 막대로 그린다. 행 순서는 책의 규약(LF, RF, LH, RH).
- contacts_to_intervals(): 실측 접촉 이진 신호 c_i(t)를 구간 목록으로 바꾼다 (8장에서 사용).
"""

from __future__ import annotations

import numpy as np

LEG_ORDER = ("LF", "RF", "LH", "RH")

# 3.3절 위상표의 전형값 (LF 기준, φ_LF = 0)
GAIT_PHASES: dict[str, dict[str, float]] = {
    "walk": {"LF": 0.0, "RF": 0.5, "LH": 0.75, "RH": 0.25},
    "trot": {"LF": 0.0, "RF": 0.5, "LH": 0.5, "RH": 0.0},
    "pace": {"LF": 0.0, "RF": 0.5, "LH": 0.0, "RH": 0.5},
    "bound": {"LF": 0.0, "RF": 0.0, "LH": 0.5, "RH": 0.5},
    "gallop": {"LF": 0.0, "RF": 0.1, "LH": 0.5, "RH": 0.6},
}

# 걸음별 전형적 duty factor (표의 범위 안에서 고른 대표값)
GAIT_DUTY: dict[str, float] = {"walk": 0.7, "trot": 0.45, "pace": 0.45, "bound": 0.4, "gallop": 0.35}


def gait_intervals(phases: dict[str, float], duty: float | dict[str, float], n_cycles: int = 2):
    """다리별 지지기 구간 [(시작, 끝), ...]을 주기 단위(0 = 첫 LF 접지)로 돌려준다."""
    out: dict[str, list[tuple[float, float]]] = {}
    for leg in LEG_ORDER:
        beta = duty[leg] if isinstance(duty, dict) else duty
        segs = []
        for k in range(-1, n_cycles + 1):
            start = k + phases[leg]
            end = start + beta
            # 그리는 범위 [0, n_cycles]로 자른다
            s, e = max(start, 0.0), min(end, float(n_cycles))
            if e > s:
                segs.append((s, e))
        out[leg] = segs
    return out


def contacts_to_intervals(t: np.ndarray, contact: np.ndarray):
    """접촉 이진 신호(1 = 지지기)를 (시작, 끝) 시간 구간 목록으로 바꾼다."""
    c = np.asarray(contact).astype(bool)
    if c.size == 0:
        return []
    edges = np.diff(c.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if c[0]:
        starts.insert(0, 0)
    if c[-1]:
        ends.append(len(c) - 1)
    return [(float(t[s]), float(t[e])) for s, e in zip(starts, ends)]


def plot_gait_diagram(ax, intervals: dict[str, list[tuple[float, float]]], *, x_max: float,
                      color="#333333", title: str | None = None, xlabel: str = "gait cycle"):
    """intervals: 다리 이름 -> [(시작, 끝), ...]. 행 순서는 LEG_ORDER (위가 LF)."""
    n = len(LEG_ORDER)
    for row, leg in enumerate(LEG_ORDER):
        y = n - 1 - row
        for s, e in intervals.get(leg, []):
            ax.broken_barh([(s, e - s)], (y + 0.15, 0.7), color=color)
    ax.set_yticks([n - 1 - r + 0.5 for r in range(n)])
    ax.set_yticklabels(LEG_ORDER)
    ax.set_ylim(0, n)
    ax.set_xlim(0, x_max)
    ax.set_xlabel(xlabel)
    for x in np.arange(0, x_max + 1e-9, 0.5):
        ax.axvline(x, color="#bbbbbb", lw=0.6, ls="--", zorder=0)
    if title:
        ax.set_title(title)
    return ax
