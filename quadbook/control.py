"""관절 제어의 기초 (6장). 이후 장이 그대로 쓰는 제어 도구.

- JointPD: 관절마다 스프링(kp)과 댐퍼(kd)를 두는 PD 제어기. 토크 = kp·(목표 − 각도) − kd·각속도
- gravity_compensation: MuJoCo가 계산해 둔 편향력(중력 + 원심력 + 코리올리)을 그대로 피드포워드 토크로 쓴다
- joint_inertia_diag: 현재 자세에서 관절 공간 관성 행렬의 대각 성분. kd를 kp에서 계산할 때 쓴다
- critical_kd: 감쇠비 ζ에 맞는 kd = 2·ζ·sqrt(kp·I)
- torso_roll_pitch: 몸통 자세 쿼터니언에서 roll, pitch (측정·분석용)
"""

from __future__ import annotations

import numpy as np

import mujoco

from .robot import NV_ROOT, joint_qpos, joint_qvel


class JointPD:
    """관절 공간 PD 제어기. kp, kd는 스칼라 또는 관절 수 길이의 배열."""

    def __init__(self, kp, kd, n_joints: int = 12):
        self.kp = np.broadcast_to(np.asarray(kp, dtype=float), (n_joints,)).copy()
        self.kd = np.broadcast_to(np.asarray(kd, dtype=float), (n_joints,)).copy()

    def torque(self, data: mujoco.MjData, q_target: np.ndarray, qd_target=0.0) -> np.ndarray:
        """스프링 항 kp·(q_t − q) 과 댐퍼 항 kd·(qd_t − qd) 의 합."""
        return self.kp * (q_target - joint_qpos(data)) + self.kd * (qd_target - joint_qvel(data))


def gravity_compensation(data: mujoco.MjData) -> np.ndarray:
    """관절 12개에 대한 편향력 c(q, qd). 정지 상태에서는 순수한 중력 토크다.
    mj_step 안에서 매 스텝 계산되므로 직전 스텝의 값을 읽는 셈이고, 서 있는 로봇에서는 그 차이가 무시할 만하다."""
    return data.qfrc_bias[NV_ROOT:NV_ROOT + 12].copy()


def joint_inertia_diag(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """현재 자세에서의 관절 공간 관성 행렬 M(q)의 대각 성분 (관절 12개). armature가 포함된다."""
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)        # MuJoCo 3.12 시그니처: (model, data, dst). 희소 행렬을 꽉 찬 행렬로 편다
    return np.diag(M)[NV_ROOT:NV_ROOT + 12].copy()


def critical_kd(kp, inertia, zeta: float = 1.0) -> np.ndarray:
    """2차계 τ = kp·e − kd·ė 에서 감쇠비 ζ를 만드는 kd. ζ = 1이 임계 감쇠(넘침 없이 가장 빠르게 수렴)."""
    return 2.0 * zeta * np.sqrt(np.asarray(kp, dtype=float) * np.asarray(inertia, dtype=float))


def torso_roll_pitch(data: mujoco.MjData) -> tuple[float, float]:
    """몸통 자세 쿼터니언(w, x, y, z)에서 roll(x축 둘레)과 pitch(y축 둘레)를 rad로 돌려준다."""
    w, x, y, z = data.qpos[3:7]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return float(roll), float(pitch)
