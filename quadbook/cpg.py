"""리듬 생성과 발 궤적 (7장). 사인파에서 CPG까지, 이후 장이 그대로 쓰는 걸음 생성 도구.

- LegKinematics: 시상면 2링크 다리(허벅지 0.2 m, 종아리 0.2 m)의 정기구학(FK)과 역기구학(IK).
- FootTrajectory: 위상 φ ∈ [0, 1) → 발 위치 (x, z). 지지기는 바닥을 따라 뒤로 미는 직선, 유각기는 앞으로 넘어가는 호.
- PhaseOscillators: 결합 위상 진동자. 다리마다 위상을 하나씩 갖고, 결합항이 위상차를 목표 위상표로 끌어당긴다.
- GaitGenerator: 위 셋을 묶어 시간 → 관절 목표 12개 (벌림 관절은 0).
"""

from __future__ import annotations

import numpy as np

from .gait import GAIT_PHASES, LEG_ORDER
from .robot import STAND_POSE_BALANCED

THIGH = 0.2   # models/quadruped.xml 의 허벅지 길이
CALF = 0.2    # 종아리 길이 (발 구는 종아리 끝에 있다)


class LegKinematics:
    """엉덩이 관절을 원점으로 하는 시상면 좌표. x는 앞(+), z는 위(+). 엉덩이 각도 +는 발을 뒤로 보낸다(모델 규약)."""

    def __init__(self, l1: float = THIGH, l2: float = CALF):
        self.l1, self.l2 = l1, l2

    def forward(self, hip: float, knee: float) -> tuple[float, float]:
        """관절 각도 → 발 위치 (x, z)."""
        x = -(self.l1 * np.sin(hip) + self.l2 * np.sin(hip + knee))
        z = -(self.l1 * np.cos(hip) + self.l2 * np.cos(hip + knee))
        return float(x), float(z)

    def inverse(self, x: float, z: float) -> tuple[float, float]:
        """발 위치 (x, z) → 관절 각도 (hip, knee). 무릎은 뒤로 접히는 해(knee < 0)를 고른다."""
        u, v = -x, -z                                     # 아래(−z)를 기준축으로 재는 표준 2링크 좌표
        r2 = u * u + v * v
        c = (r2 - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2)
        knee = -np.arccos(np.clip(c, -1.0, 1.0))          # 닿지 않는 점은 가장 가까운 점으로 잘린다
        hip = np.arctan2(u, v) - np.arctan2(self.l2 * np.sin(knee), self.l1 + self.l2 * np.cos(knee))
        return float(hip), float(knee)


class FootTrajectory:
    """위상 하나로 발 궤적을 만든다.

    φ < duty  : 지지기. 발이 x0 + L/2 에서 x0 − L/2 로 직선으로 뒤로 간다 (몸은 그만큼 앞으로).
    φ ≥ duty  : 유각기. 발이 높이 h 의 호를 그리며 앞으로 돌아온다.
    """

    def __init__(self, x0: float, z0: float, duty: float = 0.5, step_length: float = 0.08, step_height: float = 0.04):
        self.x0, self.z0, self.duty = x0, z0, duty
        self.L, self.h = step_length, step_height

    def __call__(self, phase: float) -> tuple[float, float]:
        phase = phase % 1.0
        if phase < self.duty:
            s = phase / self.duty                         # 0 → 1 동안 앞에서 뒤로
            return self.x0 + self.L * (0.5 - s), self.z0
        s = (phase - self.duty) / (1.0 - self.duty)       # 0 → 1 동안 뒤에서 앞으로
        x = self.x0 + self.L * (-0.5 + 0.5 * (1 - np.cos(np.pi * s)))   # 양 끝에서 속도 0 (부드럽게 뜨고 내린다)
        z = self.z0 + self.h * np.sin(np.pi * s)
        return float(x), float(z)


class PhaseOscillators:
    """다리 4개의 위상 θ_i (주기 단위, [0, 1)).

    dθ_i/dt = f + (k / 2π) · Σ_j sin(2π · (θ_j − θ_i − Δ_ij)),   Δ_ij = φ_j − φ_i (목표 위상표의 차)
    결합 k = 0 이면 각자 f 로만 돌고, k > 0 이면 위상차가 목표로 끌려간다.
    """

    def __init__(self, freq: float, offsets: dict[str, float], coupling: float = 2.0, init=None):
        self.freq, self.k = freq, coupling
        self.target = np.array([offsets[leg] for leg in LEG_ORDER])
        self.theta = self.target.copy() if init is None else np.asarray(init, dtype=float) % 1.0

    def step(self, dt: float) -> np.ndarray:
        d = self.theta[None, :] - self.theta[:, None]                 # θ_j − θ_i
        delta = self.target[None, :] - self.target[:, None]           # Δ_ij
        coupling = (self.k / (2 * np.pi)) * np.sin(2 * np.pi * (d - delta)).sum(axis=1)
        self.theta = (self.theta + (self.freq + coupling) * dt) % 1.0
        return self.theta


class GaitGenerator:
    """시간 → 관절 목표 12개. 벌림 관절은 0으로 고정한다(7장의 걸음은 시상면 안에서만 움직인다)."""

    def __init__(self, gait: str = "trot", freq: float = 2.0, duty: float = 0.5, step_length: float = 0.08,
                 step_height: float = 0.04, stance_pose=STAND_POSE_BALANCED, coupling: float = 2.0,
                 offsets: dict[str, float] | None = None, init_phase=None):
        self.kin = LegKinematics()
        self.offsets = offsets if offsets is not None else GAIT_PHASES[gait]
        self.osc = PhaseOscillators(freq, self.offsets, coupling, init=init_phase)
        x0, z0 = self.kin.forward(stance_pose[1], stance_pose[2])       # 서기 자세의 발 위치가 궤적의 중심
        self.foot = FootTrajectory(x0, z0, duty, step_length, step_height)
        self.duty, self.freq = duty, freq

    def targets_at(self, phases) -> np.ndarray:
        """위상 4개 → 관절 목표 12개 (LF, RF, LH, RH 순으로 [벌림, 엉덩이, 무릎])."""
        q = np.zeros(12)
        for i, ph in enumerate(phases):
            x, z = self.foot(ph)
            q[3 * i + 1], q[3 * i + 2] = self.kin.inverse(x, z)
        return q

    def targets(self, dt: float) -> np.ndarray:
        """dt 만큼 위상을 전진시키고 관절 목표를 돌려준다."""
        q = self.targets_at(self.osc.step(dt))
        self._last_q = q
        return q

    def targets_with_velocity(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """관절 목표와 그 변화율(목표 각속도). 뒤 값은 JointPD의 qd_target 자리에 들어간다 (6.2절에서 비워 둔 자리)."""
        q_prev = getattr(self, "_last_q", self.targets_at(self.osc.theta))
        q = self.targets(dt)
        return q, (q - q_prev) / dt

    def foot_height_cmd(self, phases) -> np.ndarray:
        """명령한 발 높이(궤적 기준 z − z0). 0이면 지지기."""
        return np.array([self.foot(ph)[1] - self.foot.z0 for ph in phases])

    @property
    def ideal_speed(self) -> float:
        """미끄러짐이 없다면 몸 속도 = 보폭 / 지지기 시간 = L · f / β."""
        return self.foot.L * self.freq / self.duty
