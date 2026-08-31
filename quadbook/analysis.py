"""걸음 측정 도구 (8장). 접촉 로그에서 3장의 수치를 계산하고, 지지 다각형과 안정 여유를 잰다. 12장의 보상 설계도 이 함수들을 쓴다.

- touchdowns(): 접촉 이진 신호에서 접지·지면 이탈 시각을 뽑는다.
- gait_metrics(): 다리별 duty factor β, 주기 T, LF 기준 상대 위상 φ, 보폭 λ (3.2절의 정의 그대로).
- support_margin(): 닿아 있는 발들이 만드는 지지 다각형과 무게중심 투영점 사이의 부호 있는 거리 (3.5절).
- froude(), relative_stride(): 동적 유사성 비교 (3.4절, Alexander & Jayes 1983).
"""

from __future__ import annotations

import numpy as np

from .gait import LEG_ORDER, contacts_to_intervals
from .robot import FOOT_GEOMS, foot_contacts


def touchdowns(t, contact):
    """접촉 이진 신호 → (접지 시각 목록, 지면 이탈 시각 목록)."""
    iv = contacts_to_intervals(np.asarray(t), np.asarray(contact))
    return np.array([s for s, _ in iv]), np.array([e for _, e in iv])


def dominant_period(t, contacts, leg=0):
    """한 다리(기본 LF) 접촉 신호의 지배 주기 (FFT). 접지 순간의 튐이나 지지기 중간의 이탈에 흔들리지 않는다.
    네 발의 합을 쓰면 trot에서는 대각 쌍이 번갈아 닿아 반주기가 잡히므로 한 다리의 신호를 쓴다."""
    t = np.asarray(t); sig = np.asarray(contacts, dtype=float)[:, leg]
    sig = sig - sig.mean()
    dt = float(np.mean(np.diff(t)))
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    freqs = np.fft.rfftfreq(len(sig), dt)
    spec[freqs < 0.2] = 0                     # 0.2 Hz 미만(느린 표류)은 무시
    return float(1.0 / freqs[np.argmax(spec)])


def gait_metrics(t, contacts, t_from=2.0, t_to=None, x=None):
    """t: (N,), contacts: (N, 4) 0/1 (LF RF LH RH). t_from 이후의 정수 주기 창에서 계산한다.

    돌려주는 dict:
      period      지배 주기 T (FFT), period_td: LF 접지 간격으로 잰 T (T의 ±30% 안에 드는 간격의 평균)
      duty[4]     β = 접촉 시간 비율 (정수 주기 창)
      phase[4]    LF 기준 상대 위상. 주기마다 '가장 긴 접촉 구간'의 접지 시각을 쓰고 원형 평균한다
      fragments[4] 주기당 접촉 구간 수 (1이면 깨끗한 지지기, 크면 튐·이탈)
      n_cycles, stride(x 주어지면 LF 접지 사이 이동 거리 평균), speed
    """
    t = np.asarray(t); contacts = np.asarray(contacts).astype(bool)
    m = t >= t_from
    if t_to is not None:
        m &= t <= t_to
    t, contacts = t[m], contacts[m]
    x = None if x is None else np.asarray(x)[m]
    T = dominant_period(t, contacts)
    n_cycles = int((t[-1] - t[0]) // T)
    if n_cycles < 2:
        return None
    # LF의 주기별 대표 접지: 각 주기 창에서 가장 긴 LF 접촉 구간의 시작
    def main_touchdowns(col):
        iv = contacts_to_intervals(t, col)
        out = []
        for k in range(n_cycles):
            w0, w1 = t[0] + k * T, t[0] + (k + 1) * T
            cand = [(e - s, s) for s, e in iv if w0 <= s < w1]
            if cand:
                out.append(max(cand)[1])
        return np.array(out)
    lf = main_touchdowns(contacts[:, 0])
    if len(lf) >= 2:
        gaps = np.diff(lf); ok = gaps[(gaps > 0.7 * T) & (gaps < 1.3 * T)]
        T_td = float(ok.mean()) if len(ok) else float("nan")
    else:
        T_td = float("nan")
    win = t <= t[0] + n_cycles * T
    duty = contacts[win].mean(axis=0)
    fragments = np.array([len(contacts_to_intervals(t[win], contacts[win][:, i])) / n_cycles for i in range(4)])
    phase = []
    for i in range(4):
        td = main_touchdowns(contacts[:, i])
        ph = []
        for ti in td:
            prev = lf[lf <= ti]
            if len(prev):
                ph.append(((ti - prev[-1]) / T) % 1.0)
        phase.append(float(np.angle(np.mean(np.exp(2j * np.pi * np.array(ph)))) / (2 * np.pi) % 1.0) if ph else float("nan"))
    out = {"period": T, "period_td": T_td, "duty": np.asarray(duty), "phase": np.array(phase), "fragments": fragments,
           "n_cycles": n_cycles, "lf_touchdowns": lf}
    if x is not None:
        xs = np.interp(lf, t, x) if len(lf) >= 2 else None
        out["stride"] = float(np.diff(xs).mean()) if xs is not None else float("nan")
        out["speed"] = float((x[win][-1] - x[0]) / (t[win][-1] - t[0]))
    return out


def _convex_hull(pts):
    """2D 점들의 볼록 껍질 (반시계). 점이 3개 미만이면 그대로."""
    pts = sorted(set(map(tuple, pts)))
    if len(pts) < 3:
        return np.array(pts)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _point_segment_distance(p, a, b):
    ab = b - a
    tt = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + tt * ab)))


def support_margin(model, data):
    """(안정 여유 [m], 무게중심 투영 xy, 지지 다각형 꼭짓점 xy). 여유는 다각형 안이면 +, 밖이면 −.
    발이 2개면 지지 '선분'이므로 여유는 −(선분까지의 거리), 1개 이하면 −∞ 대신 −1을 돌려준다."""
    c = foot_contacts(model, data)
    feet = np.array([data.geom(g).xpos[:2] for g, on in zip(FOOT_GEOMS, c) if on])
    com = data.subtree_com[model.body("torso").id][:2].copy()
    if len(feet) == 0:
        return -1.0, com, feet
    if len(feet) == 1:
        return -float(np.linalg.norm(com - feet[0])), com, feet
    hull = _convex_hull(feet)
    if len(hull) == 2:
        return -_point_segment_distance(com, hull[0], hull[1]), com, hull
    n = len(hull)
    inside = True
    dists = []
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        crossv = (b[0] - a[0]) * (com[1] - a[1]) - (b[1] - a[1]) * (com[0] - a[0])
        if crossv < 0:
            inside = False
        dists.append(_point_segment_distance(com, a, b))
    d = min(dists)
    return (d if inside else -d), com, hull


def froude(v, leg_length, g=9.81):
    """Fr = v² / (g · L)  (3.4절)."""
    return float(v * v / (g * leg_length))


def relative_stride(stride, leg_length):
    """상대 보폭 λ / L."""
    return float(stride / leg_length)


def alexander_stride(fr):
    """Alexander & Jayes (1983)의 동적 유사성 회귀: 상대 보폭 ≈ 2.4 · Fr^0.34 (cursorial 포유류)."""
    return 2.4 * np.power(fr, 0.34)
