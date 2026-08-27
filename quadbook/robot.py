"""이 책의 사족 로봇(models/quadruped.xml)에 대한 상수와 작은 도우미.

6장 이후 모든 장이 여기의 이름과 순서를 공유한다. 관절 순서는 XML의 actuator 순서와 같다.
"""

from __future__ import annotations

import numpy as np

import mujoco

LEGS = ("LF", "RF", "LH", "RH")                      # 3장 표기와 동일
JOINTS_PER_LEG = ("abduction", "hip", "knee")
JOINT_NAMES = [f"{leg}_{j}" for leg in LEGS for j in JOINTS_PER_LEG]   # 12개, actuator 순서와 동일
FOOT_GEOMS = [f"{leg}_foot" for leg in LEGS]
TOUCH_SENSORS = [f"{leg}_touch" for leg in LEGS]

# 서 있는 자세 (keyframe "stand"와 동일). 벌림 0, 엉덩이 0.8 rad 앞으로, 무릎 -1.5 rad 접힘
STAND_POSE = np.array([0.0, 0.8, -1.5] * 4)
# 6장에서 앞뒤 하중을 맞춘 서기 자세 (엉덩이 0.9 rad). 7장부터 이 자세를 기본으로 쓴다
STAND_POSE_BALANCED = np.array([0.0, 0.9, -1.5] * 4)
NQ_ROOT, NV_ROOT = 7, 6                              # freejoint가 차지하는 qpos, qvel 크기


def load(model_path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """모델을 읽고 'stand' 키프레임으로 초기화한 (model, data)를 돌려준다."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    reset_stand(model, data)
    return model, data


def reset_stand(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)


def joint_qpos(data: mujoco.MjData) -> np.ndarray:
    """관절 각도 12개 (freejoint 7개를 건너뛴다)."""
    return data.qpos[NQ_ROOT:NQ_ROOT + 12]


def joint_qvel(data: mujoco.MjData) -> np.ndarray:
    """관절 각속도 12개 (freejoint 6개를 건너뛴다)."""
    return data.qvel[NV_ROOT:NV_ROOT + 12]


def pd_torque(data: mujoco.MjData, q_target: np.ndarray, kp: float = 40.0, kd: float = 1.0) -> np.ndarray:
    """관절 PD 제어 토크. 6장에서 자세히 다루며, 5장 검증에서는 '자세를 붙드는 도구'로만 빌려 쓴다."""
    return kp * (q_target - joint_qpos(data)) - kd * joint_qvel(data)


def foot_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """발 4개가 바닥에 닿았는지 (LF, RF, LH, RH 순 0/1). 접촉 목록을 직접 읽는다."""
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in FOOT_GEOMS]
    out = np.zeros(4, dtype=int)
    for i in range(data.ncon):
        c = data.contact[i]
        for k, f in enumerate(feet):
            if (c.geom1 == floor and c.geom2 == f) or (c.geom2 == floor and c.geom1 == f):
                out[k] = 1
    return out


def touch(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """발 접촉 센서 4개의 값 (법선력 크기, N)."""
    return np.array([data.sensor(n).data[0] for n in TOUCH_SENSORS])
