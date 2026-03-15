import time

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

# ----------------- Environment geometry -----------------
SIZE = 10.0
UAV_R = 0.2
GOAL_R = 0.25

DT = 1.0 / 240.0
EPISODE_SECONDS = 18.0

# ----------------- Static obstacle field -----------------
OBSTACLE_MARGIN = 0.25
GOAL_POS = np.array([0.0, 0.0], dtype=np.float32)

SHAPE_CODES = {
    "sphere": 0.0,
    "box": 1.0,
    "cylinder": 2.0,
}

STATIC_OBSTACLE_LAYOUT = (
    {
        "shape": "sphere",
        "position": (-4.8, 4.4),
        "radius": 0.50,
        "color": (0.08, 0.43, 0.95, 1.0),
    },
    {
        "shape": "box",
        "position": (-1.8, 3.2),
        "half_extents": (0.45, 0.75, 0.30),
        "color": (0.10, 0.50, 0.92, 1.0),
    },
    {
        "shape": "cylinder",
        "position": (3.1, 4.1),
        "radius": 0.45,
        "height": 0.60,
        "color": (0.15, 0.58, 0.95, 1.0),
    },
    {
        "shape": "box",
        "position": (5.5, 1.8),
        "half_extents": (0.75, 0.28, 0.30),
        "color": (0.12, 0.47, 0.88, 1.0),
    },
    {
        "shape": "sphere",
        "position": (2.4, -2.2),
        "radius": 0.55,
        "color": (0.05, 0.40, 0.92, 1.0),
    },
    {
        "shape": "cylinder",
        "position": (-1.4, -2.9),
        "radius": 0.42,
        "height": 0.55,
        "color": (0.18, 0.60, 0.92, 1.0),
    },
    {
        "shape": "box",
        "position": (-4.6, -4.1),
        "half_extents": (0.50, 0.68, 0.30),
        "color": (0.10, 0.54, 0.90, 1.0),
    },
    {
        "shape": "sphere",
        "position": (4.2, -4.6),
        "radius": 0.38,
        "color": (0.10, 0.46, 0.97, 1.0),
    },
)

# ----------------- Reward shaping (Stage 1+) -----------------
GOAL_REACHED_DIST = 0.65
GOAL_BONUS = 35.0
COLLISION_PENALTY = 20.0
SAFETY_VIOLATION_PENALTY = 2.5
TIME_PENALTY = 0.30
PROGRESS_GAIN = 12.0
APPROACH_BONUS_GAIN = 5.0
APPROACH_BONUS_SCALE = 0.8
ATTRACTIVE_REWARD_CLIP = 8.0
ALIGNMENT_GAIN = 0.75
REPULSION_GAIN = 0.85
REPULSION_RANGE = 1.8
REPULSION_REWARD_CLIP = 4.0
STALL_PROGRESS_THRESH = 0.002
STALL_SPEED_THRESH = 0.35
STALL_GOAL_DIST = 1.0
STALL_PENALTY = 0.35

# Hover-based success: staying near goal for sustained period
HOVER_SUCCESS_DIST = 1.0
HOVER_SUCCESS_STEPS = 120   # 0.5 s at 240 Hz

# ----------------- Trajectory shaping (Stage 2+) -----------------
CLEARANCE_BONUS_GAIN = 0.4
CLEARANCE_BONUS_RANGE = 2.5
SMOOTHNESS_PENALTY = 0.15
NEAR_MISS_RANGE = 0.5
NEAR_MISS_PENALTY = 3.0

# ----------------- Sensor noise & Kalman filter (Stage 3+) --------
SENSOR_NOISE_POS = 0.05
SENSOR_NOISE_VEL = 0.10
SENSOR_NOISE_GOAL = 0.05
SENSOR_NOISE_OBST = 0.08
KF_PROCESS_NOISE = 0.5
KF_MEAS_NOISE_POS = 0.05
KF_MEAS_NOISE_VEL = 0.10

# ----------------- Air drag & wind (Stage 4+) --------------------
DRAG_COEFF = 0.12
WIND_NOISE_STD = 0.35

# ----------------- Anchored obstacle drift -----------------
OBSTACLE_DRIFT_STD = 0.018
OBSTACLE_DRIFT_DAMPING = 0.965
OBSTACLE_DRIFT_MAX = 0.30

# ----------------- Heuristic controller -----------------
CTRL_GOAL_GAIN = 1.45
CTRL_REPULSION_GAIN = 0.9
CTRL_REPULSION_RANGE = 2.0


def _clearance_radius(spec):
    if spec["shape"] in {"sphere", "cylinder"}:
        return float(spec["radius"])
    half_extents = np.array(spec["half_extents"][:2], dtype=np.float32)
    return float(np.linalg.norm(half_extents))


class UAV2DAvoidSimple1NoKF(gym.Env):
    """
    Goal-directed navigation in a 2D PyBullet plane with mixed-shape obstacles
    that drift slightly around fixed nominal positions.

    Supports a 4-stage curriculum controlled by ``curriculum_stage``:

        Stage 1 — Basic navigation (clean dynamics, base rewards)
        Stage 2 — + trajectory shaping (clearance bonus, smoothness, near-miss)
        Stage 3 — + sensor noise with Kalman filter
        Stage 4 — + air drag and wind gusts

    Action:
        2D velocity command in [-1, 1]^2 scaled by ``uav_speed_max``.
    Observation:
        [uav_x, uav_y,
         uav_vx, uav_vy,
         goal_dx, goal_dy, goal_dist,
         obst_x[0..N-1], obst_y[0..N-1],
         obst_radius[0..N-1], obst_shape_code[0..N-1]]
    """

    metadata = {"render_modes": ["human"], "render_fps": 240}

    def __init__(self, render_mode=None, curriculum_stage=1):
        super().__init__()
        self.gui = render_mode == "human"
        self.time_limit_steps = int(round(EPISODE_SECONDS / DT))
        self._client = None
        self._rng = np.random.default_rng(0)

        self.uav_speed_max = 15.0
        self.num_obstacles = len(STATIC_OBSTACLE_LAYOUT)
        self.curriculum_stage = int(curriculum_stage)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        low = []
        high = []
        low += [-SIZE, -SIZE]
        high += [SIZE, SIZE]
        low += [-self.uav_speed_max, -self.uav_speed_max]
        high += [self.uav_speed_max, self.uav_speed_max]
        low += [-2.0 * SIZE, -2.0 * SIZE, 0.0]
        high += [2.0 * SIZE, 2.0 * SIZE, 2.0 * SIZE]
        low += [-SIZE] * self.num_obstacles
        high += [SIZE] * self.num_obstacles
        low += [-SIZE] * self.num_obstacles
        high += [SIZE] * self.num_obstacles
        low += [0.0] * self.num_obstacles
        high += [SIZE] * self.num_obstacles
        low += [0.0] * self.num_obstacles
        high += [max(SHAPE_CODES.values())] * self.num_obstacles

        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

        self._uav = None
        self._goal = None
        self._uav_v = np.zeros(2, dtype=np.float32)
        self._prev_uav_v = np.zeros(2, dtype=np.float32)
        self._obstacle_body_ids = [None] * self.num_obstacles
        self._obstacle_positions = np.zeros((self.num_obstacles, 2), dtype=np.float32)
        self._obstacle_nominal_positions = np.zeros((self.num_obstacles, 2), dtype=np.float32)
        self._obstacle_offsets = np.zeros((self.num_obstacles, 2), dtype=np.float32)
        self._obstacle_collision_radii = np.zeros(self.num_obstacles, dtype=np.float32)
        self._obstacle_safety_radii = np.zeros(self.num_obstacles, dtype=np.float32)
        self._obstacle_shape_codes = np.zeros(self.num_obstacles, dtype=np.float32)
        self._obstacle_z = np.zeros(self.num_obstacles, dtype=np.float32)
        self._steps = 0
        self._prev_goal_dist = 0.0
        self._hover_counter = 0

        obs_dim = 7 + 4 * self.num_obstacles
        self._obs_buffer = np.zeros(obs_dim, dtype=np.float32)
        self._obs_uav_pos = slice(0, 2)
        self._obs_uav_vel = slice(2, 4)
        self._obs_goal_vec = slice(4, 6)
        self._obs_goal_dist = 6
        self._obs_obst_x = slice(7, 7 + self.num_obstacles)
        self._obs_obst_y = slice(7 + self.num_obstacles, 7 + 2 * self.num_obstacles)
        self._obs_obst_r = slice(7 + 2 * self.num_obstacles, 7 + 3 * self.num_obstacles)
        self._obs_obst_shape = slice(7 + 3 * self.num_obstacles, 7 + 4 * self.num_obstacles)

        # Kalman filter state  (always allocated; active at stage >= 3)
        self._kf_x = np.zeros(4, dtype=np.float64)   # [x, y, vx, vy]
        self._kf_P = np.eye(4, dtype=np.float64)

    # --------- Curriculum control ----------
    def set_curriculum_stage(self, stage: int) -> None:
        """Change the curriculum difficulty at runtime (called by callback)."""
        self.curriculum_stage = int(stage)

    # --------- Kalman filter ----------
    def _kf_init(self, pos, vel):
        """Reset filter to known ground-truth state."""
        self._kf_x[:2] = pos.astype(np.float64)
        self._kf_x[2:] = vel.astype(np.float64)
        self._kf_P = np.eye(4, dtype=np.float64) * 0.01

    def _kf_predict(self):
        """Propagate with constant-velocity model."""
        F = np.eye(4, dtype=np.float64)
        F[0, 2] = DT
        F[1, 3] = DT
        q = KF_PROCESS_NOISE
        Q = np.diag([q * DT ** 2, q * DT ** 2, q * DT, q * DT])
        self._kf_x = F @ self._kf_x
        self._kf_P = F @ self._kf_P @ F.T + Q

    def _kf_update(self, z_pos, z_vel):
        """Fuse noisy position & velocity measurements; return filtered estimates."""
        z = np.array([z_pos[0], z_pos[1], z_vel[0], z_vel[1]], dtype=np.float64)
        R = np.diag([
            KF_MEAS_NOISE_POS ** 2, KF_MEAS_NOISE_POS ** 2,
            KF_MEAS_NOISE_VEL ** 2, KF_MEAS_NOISE_VEL ** 2,
        ])
        S = self._kf_P + R
        K = self._kf_P @ np.linalg.inv(S)
        self._kf_x = self._kf_x + K @ (z - self._kf_x)
        self._kf_P = (np.eye(4, dtype=np.float64) - K) @ self._kf_P
        return self._kf_x[:2].astype(np.float32), self._kf_x[2:].astype(np.float32)

    # ---------------- Bullet setup ----------------
    def _connect(self):
        if self._client is not None and p.isConnected(self._client):
            return
        self._client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.setTimeStep(DT)

    def _clear(self):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.setTimeStep(DT)

    def _make_walls(self):
        thick, height = 0.3, 1.0
        col_x = p.createCollisionShape(p.GEOM_BOX, halfExtents=[thick, SIZE, height])
        vis_x = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[thick, SIZE, height],
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
        )
        col_y = p.createCollisionShape(p.GEOM_BOX, halfExtents=[SIZE, thick, height])
        vis_y = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[SIZE, thick, height],
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
        )
        p.createMultiBody(0, col_x, vis_x, basePosition=[+SIZE, 0.0, height])
        p.createMultiBody(0, col_x, vis_x, basePosition=[-SIZE, 0.0, height])
        p.createMultiBody(0, col_y, vis_y, basePosition=[0.0, +SIZE, height])
        p.createMultiBody(0, col_y, vis_y, basePosition=[0.0, -SIZE, height])

    def _make_sphere(self, radius, rgba, mass, xy):
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
        body_id = p.createMultiBody(mass, col, vis, basePosition=[xy[0], xy[1], radius])
        p.changeDynamics(
            body_id,
            -1,
            restitution=1.0,
            linearDamping=0.0,
            angularDamping=0.0,
            lateralFriction=0.0,
            spinningFriction=0.0,
            rollingFriction=0.0,
        )
        return body_id

    def _make_static_obstacle(self, spec):
        position = spec["position"]
        color = spec["color"]

        if spec["shape"] == "sphere":
            radius = float(spec["radius"])
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
            z = radius
        elif spec["shape"] == "cylinder":
            radius = float(spec["radius"])
            height = float(spec["height"])
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
            z = height / 2.0
        elif spec["shape"] == "box":
            half_extents = spec["half_extents"]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
            z = half_extents[2]
        else:
            raise ValueError(f"Unsupported obstacle shape: {spec['shape']}")

        body_id = p.createMultiBody(0.0, col, vis, basePosition=[position[0], position[1], z])
        return body_id, z

    def _set_xy(self, body_id, xy, z):
        _, orn = p.getBasePositionAndOrientation(body_id)
        p.resetBasePositionAndOrientation(body_id, [xy[0], xy[1], z], orn)

    def _spawn_static_obstacles(self):
        for idx, spec in enumerate(STATIC_OBSTACLE_LAYOUT):
            body_id, z = self._make_static_obstacle(spec)
            collision_radius = _clearance_radius(spec)
            nominal_position = np.array(spec["position"], dtype=np.float32)
            self._obstacle_body_ids[idx] = body_id
            self._obstacle_positions[idx] = nominal_position
            self._obstacle_nominal_positions[idx] = nominal_position
            self._obstacle_offsets[idx].fill(0.0)
            self._obstacle_collision_radii[idx] = collision_radius
            self._obstacle_safety_radii[idx] = collision_radius + OBSTACLE_MARGIN
            self._obstacle_shape_codes[idx] = SHAPE_CODES[spec["shape"]]
            self._obstacle_z[idx] = z

    def _goal_position(self):
        goal_pos, _ = p.getBasePositionAndOrientation(self._goal)
        return np.array([goal_pos[0], goal_pos[1]], dtype=np.float32)

    def _uav_position(self):
        uav_pos, _ = p.getBasePositionAndOrientation(self._uav)
        return np.array([uav_pos[0], uav_pos[1]], dtype=np.float32)

    def _obstacle_metrics(self, u_xy):
        deltas = u_xy[None, :] - self._obstacle_positions
        center_dists = np.linalg.norm(deltas, axis=1) + 1e-9
        collision_clearances = center_dists - (UAV_R + self._obstacle_collision_radii)
        safety_clearances = center_dists - (UAV_R + self._obstacle_safety_radii)
        strengths = np.clip(
            (REPULSION_RANGE - safety_clearances) / REPULSION_RANGE,
            0.0,
            2.0,
        )
        repulsion = -REPULSION_GAIN * np.square(strengths).sum(dtype=np.float32)

        return (
            float(repulsion),
            float(collision_clearances.min()),
            float(safety_clearances.min()),
        )

    def _update_obstacle_positions(self):
        offsets = (
            OBSTACLE_DRIFT_DAMPING * self._obstacle_offsets
            + self._rng.normal(0.0, OBSTACLE_DRIFT_STD, size=(self.num_obstacles, 2)).astype(np.float32)
        )

        norms = np.linalg.norm(offsets, axis=1, keepdims=True)
        scale = np.minimum(1.0, OBSTACLE_DRIFT_MAX / np.maximum(norms, 1e-9))
        offsets *= scale.astype(np.float32)

        candidates = self._obstacle_nominal_positions + offsets
        wall_limits = (SIZE - self._obstacle_collision_radii)[:, None]
        candidates = np.clip(candidates, -wall_limits, wall_limits).astype(np.float32)

        self._obstacle_offsets[:] = candidates - self._obstacle_nominal_positions
        self._obstacle_positions[:] = candidates

        for body_id, position, z in zip(
            self._obstacle_body_ids,
            self._obstacle_positions,
            self._obstacle_z,
        ):
            self._set_xy(body_id, position, z)

    def potential_field_action(self):
        if self._uav is None or self._goal is None:
            raise RuntimeError("Call reset() before requesting a controller action.")

        u_xy = self._uav_position()
        goal_pos = self._goal_position()
        goal_vec = goal_pos - u_xy
        goal_dist = float(np.linalg.norm(goal_vec) + 1e-9)
        action_vec = CTRL_GOAL_GAIN * (goal_vec / goal_dist)

        deltas = u_xy[None, :] - self._obstacle_positions
        center_dists = np.linalg.norm(deltas, axis=1) + 1e-9
        safety_clearances = center_dists - (UAV_R + self._obstacle_safety_radii)
        strengths = CTRL_REPULSION_GAIN * np.clip(
            (CTRL_REPULSION_RANGE - safety_clearances) / CTRL_REPULSION_RANGE,
            0.0,
            2.0,
        )
        action_vec += ((strengths[:, None] * deltas) / center_dists[:, None]).sum(axis=0)

        norm = float(np.linalg.norm(action_vec))
        if norm > 1.0:
            action_vec /= norm
        return np.clip(action_vec, -1.0, 1.0).astype(np.float32)

    # ---------------- Gym API ----------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._connect()
        self._clear()
        p.loadURDF("plane.urdf")
        self._make_walls()

        corner = int(self._rng.integers(0, 4))
        if corner == 0:
            ux, uy = -SIZE + UAV_R, -SIZE + UAV_R
        elif corner == 1:
            ux, uy = SIZE - UAV_R, -SIZE + UAV_R
        elif corner == 2:
            ux, uy = -SIZE + UAV_R, SIZE - UAV_R
        else:
            ux, uy = SIZE - UAV_R, SIZE - UAV_R

        self._uav = self._make_sphere(UAV_R, [0.0, 1.0, 0.0, 1.0], 0.3, (ux, uy))
        self._uav_v[:] = 0.0
        self._prev_uav_v[:] = 0.0
        self._hover_counter = 0

        self._goal = self._make_sphere(GOAL_R, [1.0, 0.95, 0.1, 1.0], 0.0, GOAL_POS)
        self._spawn_static_obstacles()

        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=20.0,
                cameraYaw=90.0,
                cameraPitch=-89.99,
                cameraTargetPosition=[0.0, 0.0, 0.0],
            )

        u_xy = self._uav_position()
        goal_pos = self._goal_position()
        self._prev_goal_dist = float(np.linalg.norm(goal_pos - u_xy) + 1e-9)
        self._steps = 0

        # Initialise Kalman filter from ground truth (removed in NoKF version)
        # self._kf_init(u_xy, self._uav_v)

        return self._get_obs(), {}

    def _get_obs(self):
        u_xy = self._uav_position()
        goal_pos = self._goal_position()

        if self.curriculum_stage >= 3:
            # ---------- noisy measurements ----------
            noisy_pos = u_xy + self._rng.normal(
                0.0, SENSOR_NOISE_POS, size=2
            ).astype(np.float32)
            noisy_vel = self._uav_v + self._rng.normal(
                0.0, SENSOR_NOISE_VEL, size=2
            ).astype(np.float32)
            noisy_obst_x = self._obstacle_positions[:, 0] + self._rng.normal(
                0.0, SENSOR_NOISE_OBST, size=self.num_obstacles
            ).astype(np.float32)
            noisy_obst_y = self._obstacle_positions[:, 1] + self._rng.normal(
                0.0, SENSOR_NOISE_OBST, size=self.num_obstacles
            ).astype(np.float32)

            # ---------- No Kalman filter (Ablation) ----------
            # We directly pass the noisy measurements downstream
            filt_pos = noisy_pos
            filt_vel = noisy_vel

            goal_vec = goal_pos - filt_pos + self._rng.normal(
                0.0, SENSOR_NOISE_GOAL, size=2
            ).astype(np.float32)
            goal_dist = float(np.linalg.norm(goal_vec) + 1e-9)

            self._obs_buffer[self._obs_uav_pos] = filt_pos
            self._obs_buffer[self._obs_uav_vel] = filt_vel
            self._obs_buffer[self._obs_goal_vec] = goal_vec
            self._obs_buffer[self._obs_goal_dist] = goal_dist
            self._obs_buffer[self._obs_obst_x] = noisy_obst_x
            self._obs_buffer[self._obs_obst_y] = noisy_obst_y
        else:
            # ---------- clean observations ----------
            goal_vec = goal_pos - u_xy
            goal_dist = float(np.linalg.norm(goal_vec) + 1e-9)
            self._obs_buffer[self._obs_uav_pos] = u_xy
            self._obs_buffer[self._obs_uav_vel] = self._uav_v
            self._obs_buffer[self._obs_goal_vec] = goal_vec
            self._obs_buffer[self._obs_goal_dist] = goal_dist
            self._obs_buffer[self._obs_obst_x] = self._obstacle_positions[:, 0]
            self._obs_buffer[self._obs_obst_y] = self._obstacle_positions[:, 1]

        # ---- always clean (shape doesn't change) ----
        self._obs_buffer[self._obs_obst_r] = self._obstacle_collision_radii
        self._obs_buffer[self._obs_obst_shape] = self._obstacle_shape_codes
        return self._obs_buffer.copy()

    def step(self, action):
        self._steps += 1

        action = np.clip(np.array(action, dtype=np.float32), -1.0, 1.0)
        target_v = action * self.uav_speed_max
        self._prev_uav_v[:] = self._uav_v
        self._uav_v = 0.5 * self._uav_v + 0.5 * target_v

        # ---- Stage 4: air drag & wind gusts ----
        if self.curriculum_stage >= 4:
            speed = float(np.linalg.norm(self._uav_v))
            drag_factor = max(0.0, 1.0 - DRAG_COEFF * speed * DT)
            self._uav_v *= drag_factor
            self._uav_v += self._rng.normal(
                0.0, WIND_NOISE_STD * DT, size=2
            ).astype(np.float32)

        u_xy = self._uav_position()
        new_u = u_xy + self._uav_v * DT

        if new_u[0] <= -(SIZE - UAV_R) or new_u[0] >= (SIZE - UAV_R):
            self._uav_v[0] *= -1.0
            new_u[0] = np.clip(new_u[0], -(SIZE - UAV_R), SIZE - UAV_R)
        if new_u[1] <= -(SIZE - UAV_R) or new_u[1] >= (SIZE - UAV_R):
            self._uav_v[1] *= -1.0
            new_u[1] = np.clip(new_u[1], -(SIZE - UAV_R), SIZE - UAV_R)

        self._set_xy(self._uav, new_u, UAV_R)
        self._update_obstacle_positions()
        p.stepSimulation()
        if self.gui:
            time.sleep(DT)

        u_xy = self._uav_position()
        goal_pos = self._goal_position()
        goal_vec = goal_pos - u_xy
        goal_dist = float(np.linalg.norm(goal_vec) + 1e-9)

        # ======== Reward computation ========

        # ---- Base rewards (Stage 1+) ----
        progress = self._prev_goal_dist - goal_dist
        if progress > 0.0:
            approach_bonus = APPROACH_BONUS_GAIN * np.exp(-goal_dist / APPROACH_BONUS_SCALE)
        else:
            approach_bonus = 0.0
        attraction = PROGRESS_GAIN * progress + approach_bonus
        attraction = float(np.clip(attraction, -ATTRACTIVE_REWARD_CLIP, ATTRACTIVE_REWARD_CLIP))

        speed = float(np.linalg.norm(self._uav_v))
        goal_dir = goal_vec / goal_dist
        if speed > 1e-6:
            vel_dir = self._uav_v / speed
            alignment = ALIGNMENT_GAIN * float(np.dot(goal_dir, vel_dir))
        else:
            alignment = 0.0

        repulsion, min_collision_clearance, min_safety_clearance = self._obstacle_metrics(u_xy)
        repulsion = float(np.clip(repulsion, -REPULSION_REWARD_CLIP, 0.0))
        collision = min_collision_clearance < 0.0
        safety_violation = min_safety_clearance < 0.0
        goal_reached = goal_dist < GOAL_REACHED_DIST

        # Hover-based success: sustained proximity counts as reaching goal
        if goal_dist < HOVER_SUCCESS_DIST:
            self._hover_counter += 1
        else:
            self._hover_counter = 0
        if self._hover_counter >= HOVER_SUCCESS_STEPS:
            goal_reached = True
        stall_penalty = 0.0
        if (
            goal_dist > STALL_GOAL_DIST
            and progress < STALL_PROGRESS_THRESH
            and speed < STALL_SPEED_THRESH
        ):
            stall_penalty = STALL_PENALTY

        reward = attraction + alignment + repulsion - TIME_PENALTY - stall_penalty

        # ---- Stage 2+: trajectory shaping ----
        clearance_bonus_val = 0.0
        smoothness_penalty_val = 0.0
        near_miss_penalty_val = 0.0

        if self.curriculum_stage >= 2:
            # Clearance bonus — reward for keeping safe distance
            deltas = u_xy[None, :] - self._obstacle_positions
            center_dists = np.linalg.norm(deltas, axis=1) + 1e-9
            clearances = center_dists - (UAV_R + self._obstacle_safety_radii)
            normed = np.clip(clearances / CLEARANCE_BONUS_RANGE, 0.0, 1.0)
            clearance_bonus_val = CLEARANCE_BONUS_GAIN * float(normed.mean())
            reward += clearance_bonus_val

            # Smoothness penalty — penalise sudden velocity changes (jerk)
            dv = self._uav_v - self._prev_uav_v
            jerk = float(np.linalg.norm(dv))
            smoothness_penalty_val = SMOOTHNESS_PENALTY * jerk
            reward -= smoothness_penalty_val

            # Near-miss penalty — strong penalty when barely avoiding obstacles
            min_clearance_val = float(clearances.min())
            if 0.0 < min_clearance_val < NEAR_MISS_RANGE:
                near_miss_frac = 1.0 - min_clearance_val / NEAR_MISS_RANGE
                near_miss_penalty_val = NEAR_MISS_PENALTY * near_miss_frac ** 2
                reward -= near_miss_penalty_val

        # ---- Terminal rewards / penalties ----
        if safety_violation:
            reward -= SAFETY_VIOLATION_PENALTY
        if collision:
            reward -= COLLISION_PENALTY
        if goal_reached:
            reward += GOAL_BONUS

        terminated = bool(goal_reached or collision)
        truncated = bool(self._steps >= self.time_limit_steps)

        self._prev_goal_dist = goal_dist

        info = {
            "is_success": bool(goal_reached and not collision),
            "collision": bool(collision),
            "safety_violation": bool(safety_violation),
            "goal_dist": float(goal_dist),
            "min_collision_clearance": float(min_collision_clearance),
            "min_safety_clearance": float(min_safety_clearance),
            "reward_attraction": float(attraction),
            "reward_alignment": float(alignment),
            "reward_repulsion": float(repulsion),
            "reward_time": float(-TIME_PENALTY),
            "reward_stall": float(-stall_penalty),
            "reward_clearance_bonus": float(clearance_bonus_val),
            "reward_smoothness_penalty": float(-smoothness_penalty_val),
            "reward_near_miss_penalty": float(-near_miss_penalty_val),
            "curriculum_stage": self.curriculum_stage,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def close(self):
        if self._client is not None and p.isConnected(self._client):
            p.disconnect(self._client)
        self._client = None

    def render(self):
        pass


gym.register(
    id="UAV2DAvoidNoKF-v1",
    entry_point=UAV2DAvoidSimple1NoKF,
    max_episode_steps=int(EPISODE_SECONDS / DT),
)
