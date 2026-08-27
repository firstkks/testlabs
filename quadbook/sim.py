"""걸음 실험 공용 루프 (7장 이후). 관절 PD는 매 물리 스텝, 리듬 생성기는 control_every 스텝마다 갱신한다."""

from __future__ import annotations

import numpy as np

import mujoco

from .control import JointPD, critical_kd, joint_inertia_diag, torso_roll_pitch
from .robot import foot_contacts, joint_qpos, load, touch

KP = 40.0          # 6장의 결론: 서 있을 때의 관절 강성
WALK_KP = 80.0     # 7.3절의 결론: 걸을 때는 추종 오차를 줄이려고 두 배로 올린다 (속도 피드포워드와 함께)


def standard_pd(model, data, kp: float = KP, zeta: float = 1.0) -> JointPD:
    return JointPD(kp, critical_kd(kp, joint_inertia_diag(model, data), zeta))


def fallen(data, z_min: float = 0.15, tilt_max: float = 0.8) -> bool:
    roll, pitch = torso_roll_pitch(data)
    return bool(data.body("torso").xpos[2] < z_min or abs(roll) > tilt_max or abs(pitch) > tilt_max)


def rollout(model, data, gen, pd, duration: float, *, control_every: int = 5, settle: float = 0.5,
            feedforward: bool = True, on_step=None, log_every: int = 1) -> dict:
    """gen.targets(dt)를 control_every 스텝마다 부르고 PD로 따라간다.

    settle 초 동안은 서기 자세(생성기의 위상 0 목표가 아닌 gen.targets_at(초기 위상))로 서 있다가 걸음을 시작한다.
    on_step(model, data) 은 매 스텝 외란을 넣는 자리. 돌려주는 로그는 시간, 몸통 위치, roll/pitch, 발 접촉, 명령 발 높이.
    """
    dt_ctrl = model.opt.timestep * control_every
    log = {k: [] for k in ("t", "x", "y", "z", "roll", "pitch", "contact", "touch", "foot_cmd", "q_cmd", "q_err", "tau", "fell")}
    q_cmd, qd_cmd = gen.targets_at(gen.osc.theta), np.zeros(12)
    i = 0
    while data.time < duration:
        if data.time >= settle and i % control_every == 0:
            q_cmd, qd_cmd = gen.targets_with_velocity(dt_ctrl)      # 목표 각도와 목표 각속도
        data.ctrl[:] = pd.torque(data, q_cmd, qd_cmd if feedforward else 0.0)
        if on_step:
            on_step(model, data)
        mujoco.mj_step(model, data)
        if i % log_every == 0:
            r, p = torso_roll_pitch(data)
            pos = data.body("torso").xpos
            log["t"].append(data.time); log["x"].append(pos[0]); log["y"].append(pos[1]); log["z"].append(pos[2])
            log["roll"].append(r); log["pitch"].append(p)
            log["contact"].append(foot_contacts(model, data)); log["touch"].append(touch(model, data))
            log["foot_cmd"].append(gen.foot_height_cmd(gen.osc.theta)); log["q_cmd"].append(q_cmd.copy())
            log["q_err"].append(np.abs(q_cmd - joint_qpos(data))); log["tau"].append(np.abs(data.ctrl).max())
            log["fell"].append(fallen(data))
        i += 1
    return {k: np.array(v) for k, v in log.items()}


def summary(log: dict, t_from: float = 2.0) -> dict:
    """t_from 이후의 평균 속도(x 방향), roll/pitch 표준편차, 넘어졌는지."""
    m = log["t"] >= t_from
    if m.sum() < 2:
        return {"speed": 0.0, "roll_std": 0.0, "pitch_std": 0.0, "fell": bool(log["fell"].any()), "height": float(log["z"][-1])}
    speed = (log["x"][m][-1] - log["x"][m][0]) / (log["t"][m][-1] - log["t"][m][0])
    return {"speed": float(speed), "roll_std": float(np.degrees(log["roll"][m].std())),
            "pitch_std": float(np.degrees(log["pitch"][m].std())), "fell": bool(log["fell"].any()),
            "height": float(log["z"][m].mean()), "track_err": float(np.degrees(log["q_err"][m][:, [1, 2, 4, 5, 7, 8, 10, 11]].mean())),
            "tau_max": float(log["tau"][m].max())}
