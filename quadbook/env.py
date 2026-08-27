"""Gymnasium 환경 (10장). 7장의 생성기 위에 정책의 보정을 얹는 잔차 학습 환경. 11~12장이 그대로 학습에 쓴다.

- 행동 a ∈ [−1, 1]^12  →  관절 목표 = q_cpg(t) + action_scale · a   (기본 0.1 rad)
- 관측 53차원: 관절 각도 오프셋 12, 관절 각속도 12(×0.05), 몸통 좌표계의 중력 벡터 3, 각속도 3(×0.25),
             몸통 좌표계의 선속도 3, CPG 위상의 sin·cos 8, 직전 행동 12
- 보상: 가중합 Σ w_i · term_i. 항(term)은 info["terms"]에 매 스텝 그대로 실려 12장에서 하나씩 조각한다.
- 종료: 넘어짐(높이 < 0.15 m 또는 기울기 > 0.8 rad)이면 terminated, 에피소드 시간이 다하면 truncated.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import mujoco

from . import model_path
from .control import torso_roll_pitch
from .cpg import GaitGenerator
from .gait import LEG_ORDER
from .robot import STAND_POSE_BALANCED, foot_contacts, joint_qpos, joint_qvel, load, reset_stand
from .sim import WALK_KP, fallen, standard_pd

OBS_DIM = 53
DEFAULT_WEIGHTS = {"forward": 1.0, "attitude": 0.5, "energy": 0.002, "alive": 0.0,
                   # 12장에서 하나씩 켜는 항 (기본 가중치 0)
                   "heading": 0.0, "lateral": 0.0, "track": 0.0, "contact_match": 0.0, "impact": 0.0, "action_rate": 0.0}


class QuadrupedEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(self, gait="trot", freq=2.0, duty=0.5, step_length=0.08, step_height=0.04,
                 action_scale=0.1, action_mode="residual", control_every=5, episode_seconds=10.0,
                 reward_weights=None, random_init_phase=True, init_noise=0.05, render_mode=None,
                 v_target=0.5, push_max=0.0, push_interval=2.0,
                 use_lin_vel=True, modulate_generator=False, model_file=None, obs_noise=0.0, latency_steps=0):
        super().__init__()
        self.use_lin_vel = use_lin_vel                  # 13장: 실기에서 못 재는 몸통 선속도를 관측에서 지운다(0으로)
        self.modulate_generator = modulate_generator    # 13장: 행동 2개를 더해 생성기의 주파수·보폭을 정책이 조절한다 (PMTG)
        self.obs_noise, self.latency_steps = obs_noise, latency_steps   # 13.3절: 관측 잡음(표준편차), 행동 지연(제어 스텝)
        self.model_file = model_file
        self.v_target = v_target                        # 12.2절 속도 추종 항의 목표 [m/s]
        self.push_max, self.push_interval = push_max, push_interval   # 12.4절: 평균 push_interval 초마다 0.2초 동안 최대 push_max N 옆 밀기
        self._push_until, self._next_push = -1.0, float("inf")
        self.model, self.data = load(model_file or model_path())
        self.pd = standard_pd(self.model, self.data, kp=WALK_KP)
        self.gen_kwargs = dict(gait=gait, freq=freq, duty=duty, step_length=step_length, step_height=step_height)
        self.gen = GaitGenerator(**self.gen_kwargs)
        self.action_scale, self.action_mode, self.control_every = action_scale, action_mode, control_every
        self.dt = self.model.opt.timestep * control_every                     # 제어 주기 (0.01 s = 100 Hz)
        self.max_steps = int(round(episode_seconds / self.dt))
        self.weights = dict(DEFAULT_WEIGHTS, **(reward_weights or {}))
        self.random_init_phase, self.init_noise, self.render_mode = random_init_phase, init_noise, render_mode
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.n_act = 14 if modulate_generator else 12
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.n_act,), np.float32)
        self._act_queue = []
        self.torso = self.model.body("torso").id
        self._renderer = None
        self._cam = None
        self.prev_action = np.zeros(12)
        self.step_count = 0
        self.q_cmd, self.qd_cmd = STAND_POSE_BALANCED.copy(), np.zeros(12)

    # ---------- 13장: 생성기 조절 ----------
    FREQ_RANGE, STEP_RANGE = (1.5, 3.0), (0.05, 0.14)

    def _apply_modulation(self, extra):
        """행동 13·14번([−1, 1])을 생성기의 주파수와 보폭으로. 주파수는 진동자에, 보폭은 발 궤적에 바로 쓴다."""
        f = self.FREQ_RANGE[0] + (extra[0] + 1) / 2 * (self.FREQ_RANGE[1] - self.FREQ_RANGE[0])
        L = self.STEP_RANGE[0] + (extra[1] + 1) / 2 * (self.STEP_RANGE[1] - self.STEP_RANGE[0])
        self.gen.osc.freq = f; self.gen.freq = f; self.gen.foot.L = L

    # ---------- 관측 ----------
    def _torso_rotation(self):
        w, x, y, z = self.data.qpos[3:7]
        R = np.empty((3, 3)); mujoco.mju_quat2Mat(R.reshape(-1), np.array([w, x, y, z]))
        return R

    def _get_obs(self):
        R = self._torso_rotation()
        gravity_body = R.T @ np.array([0.0, 0.0, -1.0])                        # 몸통이 보는 '아래' 방향 (IMU에서 얻는다)
        gyro = self.data.sensor("torso_gyro").data.copy()
        lin_vel_body = (R.T @ self.data.qvel[0:3]) if self.use_lin_vel else np.zeros(3)   # 몸통 좌표계 선속도 (실기에서는 추정값; 13장에서 지워 본다)
        ph = self.gen.osc.theta
        phase = np.concatenate([np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)])
        obs = np.concatenate([
            joint_qpos(self.data) - STAND_POSE_BALANCED,     # 12: 서기 자세 기준 관절 오프셋
            joint_qvel(self.data) * 0.05,                     # 12: 각속도 (수십 rad/s → 1 안팎)
            gravity_body,                                     # 3
            gyro * 0.25,                                      # 3
            lin_vel_body,                                     # 3
            phase,                                            # 8
            self.prev_action,                                 # 12
        ])
        if self.obs_noise > 0:
            obs = obs + self.np_random.normal(0.0, self.obs_noise, obs.shape)
        return obs.astype(np.float32)

    # ---------- 보상 ----------
    def _yaw(self):
        w, x, y, z = self.data.qpos[3:7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def _reward_terms(self, tau_power, action, impact_force):
        R = self._torso_rotation()
        v_body = R.T @ self.data.qvel[0:3]
        roll, pitch = torso_roll_pitch(self.data)
        planned_stance = self.gen.foot_height_cmd(self.gen.osc.theta) < 1e-6      # 생성기가 지지기라고 한 발
        actual = foot_contacts(self.model, self.data).astype(bool)
        return {
            "forward": float(v_body[0]),                       # 몸통 전진 속도 [m/s]
            "attitude": -float(roll ** 2 + pitch ** 2),        # 기울기 벌점 [rad²]
            "energy": -float(tau_power),                       # 관절 기계적 일률의 합 [W]
            "alive": 1.0,
            "heading": -float(self._yaw() ** 2),               # 세계 x축에서 벗어난 방향 [rad²]
            "lateral": -float(v_body[1] ** 2),                 # 옆 방향 속도 [m²/s²]
            "track": float(np.exp(-((v_body[0] - self.v_target) / 0.25) ** 2)),   # 목표 속도 추종, 0~1
            "contact_match": float(np.mean(planned_stance == actual)),           # 계획과 실제 접촉의 일치 비율, 0~1
            "impact": -float(impact_force),                    # 접지 충격: 발 힘이 60 N을 넘은 만큼의 합 [N]
            "action_rate": -float(np.sum((action - self.prev_action) ** 2)),      # 행동 변화율
        }

    # ---------- gym.Env ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        reset_stand(self.model, self.data)
        self.data.qpos[7:19] = STAND_POSE_BALANCED + self.np_random.uniform(-self.init_noise, self.init_noise, 12)
        mujoco.mj_forward(self.model, self.data)
        init = self.np_random.random(4) if self.random_init_phase else None
        self.gen = GaitGenerator(**self.gen_kwargs, init_phase=init)
        self.prev_action = np.zeros(12)
        self.step_count = 0
        self._act_queue = []
        self.q_cmd, self.qd_cmd = self.gen.targets_at(self.gen.osc.theta), np.zeros(12)
        self.data.xfrc_applied[self.torso, :] = 0.0
        self._push_until = -1.0
        self._next_push = self.np_random.exponential(self.push_interval) + 1.0 if self.push_max > 0 else float("inf")
        return self._get_obs(), {}

    def set_push_max(self, value):
        """VecEnv.env_method로 부른다. 래퍼(Monitor)의 __setattr__는 래퍼에만 값을 쓰므로 메서드로 진짜 환경에 쓴다."""
        self.push_max = float(value)

    def get_push_max(self):
        return float(self.push_max)

    def _maybe_push(self):
        """12.4절: 무작위 시각에 옆에서 0.2초 동안 민다. 크기는 push_max의 50~100%, 방향은 무작위."""
        t = self.data.time
        if t >= self._next_push and self.push_max > 0:
            F = self.np_random.uniform(0.5, 1.0) * self.push_max * self.np_random.choice([-1.0, 1.0])
            self.data.xfrc_applied[self.torso, 1] = F
            self._push_until = t + 0.2
            self._next_push = t + 0.2 + self.np_random.exponential(self.push_interval)
        elif self._push_until > 0 and t >= self._push_until:       # 내가 넣은 밀기만 거둔다 (밖에서 넣은 외력은 건드리지 않는다)
            self.data.xfrc_applied[self.torso, :] = 0.0
            self._push_until = -1.0

    def step(self, action):
        a_full = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if self.latency_steps > 0:                                   # 13.3절: 행동이 latency_steps 제어 스텝 늦게 도착한다
            self._act_queue.append(a_full); a_full = self._act_queue.pop(0) if len(self._act_queue) > self.latency_steps else np.zeros_like(a_full)
        a = a_full[:12]
        if self.modulate_generator:
            self._apply_modulation(a_full[12:14])
        q_cpg, qd_cpg = self.gen.targets_with_velocity(self.dt)
        if self.action_mode == "residual":
            q_target, qd_target = q_cpg + self.action_scale * a, qd_cpg     # 생성기 목표 + 보정
        else:                                                                # 비교용: 행동이 관절 목표 그 자체
            lo, hi = self.model.jnt_range[1:13, 0], self.model.jnt_range[1:13, 1]
            q_target, qd_target = lo + (a + 1) / 2 * (hi - lo), np.zeros(12)
        power, impact = 0.0, 0.0
        self._maybe_push()
        for _ in range(self.control_every):
            tau = self.pd.torque(self.data, q_target, qd_target)
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)
            power += np.abs(np.clip(tau, -12, 12) * joint_qvel(self.data)).sum() / self.control_every
            f = np.array([self.data.sensor(n).data[0] for n in ("LF_touch", "RF_touch", "LH_touch", "RH_touch")])
            impact = max(impact, float(np.maximum(f - 60.0, 0.0).sum()))
        self.q_cmd, self.qd_cmd = q_target, qd_target
        self.step_count += 1
        terms = self._reward_terms(power, a, impact)
        self.prev_action = a.copy()
        reward = float(sum(self.weights[k] * v for k, v in terms.items()))
        terminated = fallen(self.data)
        truncated = self.step_count >= self.max_steps
        info = {"terms": terms, "x": float(self.data.body("torso").xpos[0]), "y": float(self.data.body("torso").xpos[1]),
                "z": float(self.data.body("torso").xpos[2]), "yaw": self._yaw(),
                "contact": foot_contacts(self.model, self.data), "fell": bool(terminated), "t": float(self.data.time),
                "freq": float(self.gen.freq), "step_length": float(self.gen.foot.L)}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            from .render import _ensure_framebuffer, track_camera
            _ensure_framebuffer(self.model, 600, 450)
            self._renderer = mujoco.Renderer(self.model, height=450, width=600)
            self._cam = track_camera(self.model, distance=1.3, azimuth=140, elevation=-15)
        self._renderer.update_scene(self.data, camera=self._cam)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close(); self._renderer = None


gym.register(id="QuadBook/Quadruped-v0", entry_point="quadbook.env:QuadrupedEnv")
