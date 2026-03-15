import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch.nn as nn
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
    sync_envs_normalization,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gym_pybullet_drones.envs.constrained_environment import UAV2DAvoidSimple1


ALGO_CHOICES = ("ppo", "a2c", "sac", "td3", "ddpg")
ALGO_ORDER = ["ppo", "a2c", "sac", "td3", "ddpg"]
MODEL_CLASSES = {
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
    "td3": TD3,
    "ddpg": DDPG,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the modified static navigation environment with multiple "
            "model-free RL algorithms."
        )
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=["ppo"],
        choices=[*ALGO_CHOICES, "all"],
        help="Algorithms to train. Use 'all' to run the full suite sequentially.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=300_000,
        help="Total environment timesteps per algorithm.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments.",
    )
    parser.add_argument(
        "--vec-env",
        choices=["auto", "dummy", "subproc"],
        default="auto",
        help="Vectorized environment backend. 'auto' picks subproc when n-envs > 1.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for PPO and off-policy algorithms.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device passed to Stable-Baselines3.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("gym_pybullet_drones/examples/runs/static_nav"),
        help="Root directory for training outputs.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional suffix added to each run directory name.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=20_000,
        help="Evaluation frequency in environment timesteps. Set 0 to disable.",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
        help="Episodes per evaluation cycle.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="Checkpoint frequency in environment timesteps. Set 0 to disable.",
    )
    parser.add_argument(
        "--curriculum-stage",
        type=int,
        default=1,
        help="Initial curriculum stage. To test single-shot stability, use 4.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=300_000,
        help="Replay buffer size for off-policy algorithms.",
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=5_000,
        help="Replay warm-up steps for off-policy algorithms.",
    )
    parser.add_argument(
        "--train-freq",
        type=int,
        default=1,
        help="Train frequency in steps for off-policy algorithms.",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="Gradient updates per rollout step for off-policy algorithms.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Target-network smoothing coefficient for off-policy algorithms.",
    )
    parser.add_argument(
        "--action-noise-std",
        type=float,
        default=0.10,
        help="Exploration noise std for TD3/DDPG.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Rollout steps per environment for PPO/A2C.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="PPO optimization epochs per update.",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda for PPO/A2C.",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO clip range.",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.0,
        help="Entropy coefficient for PPO/A2C.",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="Value loss coefficient for PPO/A2C.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Gradient clipping norm for PPO/A2C.",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="Optional PPO target KL threshold for early stopping within a policy update.",
    )
    parser.add_argument(
        "--use-vecnormalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wrap vectorized environments with VecNormalize.",
    )
    parser.add_argument(
        "--vecnormalize-obs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize observations when VecNormalize is enabled.",
    )
    parser.add_argument(
        "--vecnormalize-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize training rewards when VecNormalize is enabled.",
    )
    parser.add_argument(
        "--vecnormalize-clip-obs",
        type=float,
        default=10.0,
        help="Absolute observation clipping bound used by VecNormalize.",
    )
    parser.add_argument(
        "--vecnormalize-clip-reward",
        type=float,
        default=10.0,
        help="Absolute discounted-reward clipping bound used by VecNormalize.",
    )
    parser.add_argument(
        "--net-arch",
        type=str,
        default="256,256,256",
        help="Comma-separated hidden layer sizes shared across policies.",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Enable Stable-Baselines3 progress bars.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="SB3 log interval passed to learn().",
    )
    parser.add_argument(
        "--max-curriculum-stage",
        type=int,
        default=4,
        help="Maximum curriculum stage allowed during training. Use the same value as --curriculum-stage to keep a fixed stage.",
    )
    parser.add_argument(
        "--curriculum-gate",
        type=float,
        default=0.90,
        help="Success-rate threshold required before advancing the curriculum.",
    )
    parser.add_argument(
        "--curriculum-consecutive",
        type=int,
        default=3,
        help="Number of consecutive evaluations above the gate needed to advance the curriculum.",
    )
    parser.add_argument(
        "--early-stop-after-perfect-success",
        action="store_true",
        help=(
            "Stop training when the final-stage evaluation has already reached the target "
            "success rate and then regresses for several consecutive evaluations."
        ),
    )
    parser.add_argument(
        "--perfect-success-threshold",
        type=float,
        default=1.0,
        help="Success rate considered good enough to activate regression-based early stopping.",
    )
    parser.add_argument(
        "--success-drop-patience",
        type=int,
        default=3,
        help="Consecutive regressed evaluations tolerated after reaching perfect success.",
    )
    return parser.parse_args()


def parse_net_arch(spec: str) -> list[int]:
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if not parts:
        raise ValueError("net-arch must contain at least one layer size")
    return [int(part) for part in parts]


def resolve_algorithms(requested: list[str]) -> list[str]:
    if "all" in requested:
        return ALGO_ORDER.copy()
    ordered = [algo for algo in ALGO_ORDER if algo in requested]
    if not ordered:
        raise ValueError("No valid algorithms requested")
    return ordered


def choose_vec_env_backend(requested: str, n_envs: int) -> str:
    if requested == "auto":
        return "subproc" if n_envs > 1 else "dummy"
    if requested == "dummy" and n_envs > 1:
        raise ValueError(
            "DummyVecEnv with n-envs > 1 is unsafe for this PyBullet environment "
            "because the env does not isolate physicsClientId across instances. "
            "Use --vec-env subproc or leave --vec-env auto."
        )
    return requested


def make_env(seed: int, rank: int, stage: int = 1):
    def _init():
        env = UAV2DAvoidSimple1(render_mode=None)
        env = Monitor(env)
        env.unwrapped.set_curriculum_stage(stage)
        env.reset(seed=seed + rank)
        return env

    return _init


def vecnormalize_enabled(args: argparse.Namespace | None) -> bool:
    return bool(args is not None and getattr(args, "use_vecnormalize", False))


def wrap_vecnormalize(
    vec_env,
    args: argparse.Namespace | None,
    training: bool,
    vecnormalize_path: Path | None = None,
):
    if not vecnormalize_enabled(args):
        return vec_env

    norm_reward = bool(getattr(args, "vecnormalize_reward", True) and training)
    if vecnormalize_path is not None and vecnormalize_path.exists():
        wrapped = VecNormalize.load(str(vecnormalize_path), vec_env)
        wrapped.training = training
        wrapped.norm_obs = bool(getattr(args, "vecnormalize_obs", True))
        wrapped.norm_reward = norm_reward
        wrapped.clip_obs = float(getattr(args, "vecnormalize_clip_obs", 10.0))
        wrapped.clip_reward = float(getattr(args, "vecnormalize_clip_reward", 10.0))
        return wrapped

    return VecNormalize(
        vec_env,
        training=training,
        norm_obs=bool(getattr(args, "vecnormalize_obs", True)),
        norm_reward=norm_reward,
        clip_obs=float(getattr(args, "vecnormalize_clip_obs", 10.0)),
        clip_reward=float(getattr(args, "vecnormalize_clip_reward", 10.0)),
        gamma=float(getattr(args, "gamma", 0.99)),
    )


def build_vec_env(
    seed: int,
    n_envs: int,
    backend: str,
    monitor_file: Path,
    stage: int = 1,
    args: argparse.Namespace | None = None,
    training: bool = True,
    vecnormalize_path: Path | None = None,
):
    env_fns = [make_env(seed, rank, stage) for rank in range(n_envs)]
    if backend == "dummy":
        vec_env = DummyVecEnv(env_fns)
    else:
        start_method = "spawn" if os.name == "nt" else None
        vec_env = SubprocVecEnv(env_fns, start_method=start_method)
    vec_env = wrap_vecnormalize(
        vec_env,
        args=args,
        training=training,
        vecnormalize_path=vecnormalize_path,
    )
    return VecMonitor(vec_env, filename=str(monitor_file))


def resolved_device(algo: str, args: argparse.Namespace) -> str:
    return "cpu" if args.device == "auto" and algo in {"ppo", "a2c"} else args.device


def policy_kwargs_for(algo: str, net_arch: list[int]) -> dict:
    if algo in {"ppo", "a2c"}:
        return {
            "activation_fn": nn.ReLU,
            "net_arch": {"pi": net_arch, "vf": net_arch},
        }
    return {
        "activation_fn": nn.ReLU,
        "net_arch": {"pi": net_arch, "qf": net_arch},
    }


def build_model(algo: str, env, args: argparse.Namespace, run_dir: Path):
    policy_kwargs = policy_kwargs_for(algo, parse_net_arch(args.net_arch))
    tensorboard_dir = run_dir / "tensorboard"
    device = resolved_device(algo, args)

    if algo == "ppo":
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(tensorboard_dir),
            device=device,
            seed=args.seed,
            verbose=1,
        )

    if algo == "a2c":
        return A2C(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(tensorboard_dir),
            device=device,
            seed=args.seed,
            verbose=1,
        )

    common_kwargs = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=(args.train_freq, "step"),
        gradient_steps=args.gradient_steps,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tensorboard_dir),
        device=device,
        seed=args.seed,
        verbose=1,
    )

    if algo == "sac":
        return SAC(
            **common_kwargs,
            ent_coef="auto",
        )

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=args.action_noise_std * np.ones(n_actions),
    )

    if algo == "td3":
        return TD3(
            **common_kwargs,
            action_noise=action_noise,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
        )

    if algo == "ddpg":
        return DDPG(
            **common_kwargs,
            action_noise=action_noise,
        )

    raise ValueError(f"Unsupported algorithm: {algo}")


def validate_algorithm_args(algo: str, args: argparse.Namespace) -> None:
    if algo == "ppo":
        rollout_batch = args.n_steps * args.n_envs
        if args.batch_size > rollout_batch:
            raise ValueError(
                f"PPO batch-size ({args.batch_size}) cannot exceed "
                f"n-steps * n-envs ({rollout_batch})."
            )
        if args.target_kl is not None and args.target_kl <= 0:
            raise ValueError("PPO target-kl must be positive when provided.")


def final_vecnormalize_path(run_dir: Path) -> Path:
    return run_dir / "final_vecnormalize.pkl"


def best_vecnormalize_path(run_dir: Path) -> Path:
    return run_dir / "best_model" / "vecnormalize.pkl"


def evaluate_policy_metrics(model, eval_env, n_eval_episodes: int) -> dict[str, float]:
    rewards = []
    episode_lengths = []
    path_lengths = []
    final_goal_dists = []
    min_collision_clearances = []
    min_safety_clearances = []

    success_count = 0
    collision_count = 0
    safety_violation_count = 0
    timeout_count = 0

    success_rewards = []
    success_lengths = []

    for _ in range(n_eval_episodes):
        obs = eval_env.reset()
        prev_pos = np.array(obs[0, :2], dtype=np.float32)

        episode_reward = 0.0
        episode_length = 0
        path_length = 0.0
        success = False
        collision = False
        safety_violation = False
        final_goal_dist = np.nan
        min_collision_clearance = np.inf
        min_safety_clearance = np.inf

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards_step, dones, infos = eval_env.step(action)

            reward = float(rewards_step[0])
            info = infos[0]
            done = bool(dones[0])

            episode_reward += reward
            episode_length += 1

            pos = np.array(obs[0, :2], dtype=np.float32)
            path_length += float(np.linalg.norm(pos - prev_pos))
            prev_pos = pos

            success = success or bool(info.get("is_success", False))
            collision = collision or bool(info.get("collision", False) or info.get("conflict", False))
            safety_violation = safety_violation or bool(info.get("safety_violation", False))

            if "goal_dist" in info:
                final_goal_dist = float(info["goal_dist"])
            if "min_collision_clearance" in info:
                min_collision_clearance = min(
                    min_collision_clearance,
                    float(info["min_collision_clearance"]),
                )
            if "min_safety_clearance" in info:
                min_safety_clearance = min(
                    min_safety_clearance,
                    float(info["min_safety_clearance"]),
                )

        timeout = not success and not collision

        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        path_lengths.append(path_length)
        final_goal_dists.append(final_goal_dist)
        min_collision_clearances.append(min_collision_clearance)
        min_safety_clearances.append(min_safety_clearance)

        success_count += int(success)
        collision_count += int(collision)
        safety_violation_count += int(safety_violation)
        timeout_count += int(timeout)

        if success:
            success_rewards.append(episode_reward)
            success_lengths.append(episode_length)

    def safe_mean(values):
        return float(np.mean(values)) if values else float("nan")

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "success_rate": float(success_count / n_eval_episodes),
        "collision_rate": float(collision_count / n_eval_episodes),
        "safety_violation_rate": float(safety_violation_count / n_eval_episodes),
        "timeout_rate": float(timeout_count / n_eval_episodes),
        "mean_path_length": float(np.mean(path_lengths)),
        "mean_final_goal_dist": safe_mean(final_goal_dists),
        "mean_min_collision_clearance": safe_mean(min_collision_clearances),
        "mean_min_safety_clearance": safe_mean(min_safety_clearances),
        "success_mean_reward": safe_mean(success_rewards),
        "success_mean_episode_length": safe_mean(success_lengths),
    }


class PaperEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        run_dir: Path,
        eval_freq: int,
        n_eval_episodes: int,
        start_stage: int = 1,
        max_stage: int = 4,
        curriculum_gate: float = 0.90,
        curriculum_consecutive: int = 3,
        early_stop_after_perfect_success: bool = False,
        perfect_success_threshold: float = 1.0,
        success_drop_patience: int = 3,
    ):
        super().__init__(verbose=1)
        self.eval_env = eval_env
        self.run_dir = run_dir
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.best_model_dir = run_dir / "best_model"
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = run_dir / "paper_metrics.csv"
        self.npz_path = run_dir / "paper_metrics.npz"
        self.latest_json_path = run_dir / "latest_eval.json"
        self.best_json_path = self.best_model_dir / "best_metrics.json"
        self.history = []
        self.best_success_rate = -np.inf
        self.best_mean_reward = -np.inf

        # ---- Curriculum gating ----
        self.curriculum_stage = start_stage
        self.max_curriculum_stages = max_stage
        self.curriculum_gate = curriculum_gate
        self.curriculum_consecutive = curriculum_consecutive
        self._gate_counter = 0
        self.early_stop_after_perfect_success = early_stop_after_perfect_success
        self.perfect_success_threshold = perfect_success_threshold
        self.success_drop_patience = success_drop_patience
        self._perfect_success_seen = False
        self._success_regression_counter = 0

    def _on_training_start(self) -> None:
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self._fieldnames())
                writer.writeheader()
        else:
            try:
                with self.csv_path.open("r", newline="", encoding="utf-8") as csv_file:
                    reader = csv.DictReader(csv_file)
                    self.history = []
                    for row in reader:
                        parsed = {}
                        for key in self._fieldnames():
                            if key == "timesteps" or key == "curriculum_stage":
                                parsed[key] = int(float(row[key]))
                            else:
                                parsed[key] = float(row[key])
                        self.history.append(parsed)
            except (OSError, ValueError, KeyError):
                self.history = []
        if self.best_json_path.exists():
            try:
                best_metrics = json.loads(self.best_json_path.read_text(encoding="utf-8"))
                self.best_success_rate = float(best_metrics.get("success_rate", self.best_success_rate))
                self.best_mean_reward = float(best_metrics.get("mean_reward", self.best_mean_reward))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
        if self.best_success_rate >= self.perfect_success_threshold:
            self._perfect_success_seen = True

    def _fieldnames(self) -> list[str]:
        return [
            "timesteps",
            "curriculum_stage",
            "mean_reward",
            "std_reward",
            "mean_episode_length",
            "std_episode_length",
            "success_rate",
            "collision_rate",
            "safety_violation_rate",
            "timeout_rate",
            "mean_path_length",
            "mean_final_goal_dist",
            "mean_min_collision_clearance",
            "mean_min_safety_clearance",
            "success_mean_reward",
            "success_mean_episode_length",
        ]

    def _save_history(self) -> None:
        if not self.history:
            return
        arrays = {}
        for key in self._fieldnames():
            arrays[key] = np.array([entry[key] for entry in self.history], dtype=np.float64)
        np.savez_compressed(self.npz_path, **arrays)

    def _append_csv(self, metrics: dict[str, float]) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self._fieldnames())
            writer.writerow(metrics)

    def _is_better(self, metrics: dict[str, float]) -> bool:
        success_rate = metrics["success_rate"]
        mean_reward = metrics["mean_reward"]
        if success_rate > self.best_success_rate:
            return True
        if success_rate == self.best_success_rate and mean_reward > self.best_mean_reward:
            return True
        return False

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or (self.num_timesteps % self.eval_freq) != 0:
            return True

        if self.model.get_vec_normalize_env() is not None:
            try:
                sync_envs_normalization(self.training_env, self.eval_env)
            except AttributeError as exc:
                raise AssertionError(
                    "Training and eval env are not wrapped the same way. "
                    "If VecNormalize is enabled, both envs must share the same wrapper stack."
                ) from exc

        metrics = evaluate_policy_metrics(
            model=self.model,
            eval_env=self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
        )
        metrics["timesteps"] = int(self.num_timesteps)
        metrics["curriculum_stage"] = self.curriculum_stage
        self.history.append(metrics)

        for key, value in metrics.items():
            if key == "timesteps":
                continue
            self.logger.record(f"paper_eval/{key}", value)
        self.logger.dump(self.num_timesteps)

        self._append_csv(metrics)
        self._save_history()
        self.latest_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        if self._is_better(metrics):
            self.best_success_rate = metrics["success_rate"]
            self.best_mean_reward = metrics["mean_reward"]
            self.model.save(str(self.best_model_dir / "best_model.zip"))
            vec_normalize_env = self.model.get_vec_normalize_env()
            if vec_normalize_env is not None:
                vec_normalize_env.save(str(best_vecnormalize_path(self.run_dir)))
            self.best_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            if self.best_success_rate >= self.perfect_success_threshold:
                self._perfect_success_seen = True

        print(
            f"[paper-eval {self.num_timesteps}  stage={self.curriculum_stage}/4] "
            f"R={metrics['mean_reward']:.3f} "
            f"S={100.0 * metrics['success_rate']:.1f}% "
            f"C={100.0 * metrics['collision_rate']:.1f}% "
            f"SV={100.0 * metrics['safety_violation_rate']:.1f}% "
            f"T={100.0 * metrics['timeout_rate']:.1f}% "
            f"L={metrics['mean_episode_length']:.1f}"
        )

        if (
            self.early_stop_after_perfect_success
            and self.curriculum_stage >= self.max_curriculum_stages
            and self._perfect_success_seen
        ):
            regressed = metrics["success_rate"] < self.perfect_success_threshold
            if regressed:
                self._success_regression_counter += 1
            else:
                self._success_regression_counter = 0

            if self._success_regression_counter >= self.success_drop_patience:
                print(
                    f"[early-stop] perfect success had already been reached, but evaluation "
                    f"success regressed for {self._success_regression_counter} consecutive "
                    f"checks at stage {self.curriculum_stage}. Stopping training to keep the "
                    f"best success-first checkpoint."
                )
                return False

        # ---- Curriculum gating ----
        if self.curriculum_stage < self.max_curriculum_stages:
            if metrics["success_rate"] >= self.curriculum_gate:
                self._gate_counter += 1
            else:
                self._gate_counter = 0

            if self._gate_counter >= self.curriculum_consecutive:
                self.curriculum_stage += 1
                self._gate_counter = 0
                # Reset best-model tracking (rewards change across stages)
                self.best_success_rate = -np.inf
                self.best_mean_reward = -np.inf
                self._perfect_success_seen = False
                self._success_regression_counter = 0
                # Update all environments
                train_env = self.model.get_env()
                train_env.env_method("set_curriculum_stage", self.curriculum_stage)
                self.eval_env.env_method("set_curriculum_stage", self.curriculum_stage)
                print(
                    f"\n{'='*60}\n"
                    f"  [CURRICULUM] ★ Advanced to Stage {self.curriculum_stage}/4 ★\n"
                    f"{'='*60}\n"
                )

        return True


def build_callbacks(
    algo: str,
    eval_env,
    args: argparse.Namespace,
    run_dir: Path,
) -> CallbackList | None:
    callbacks = []

    if args.checkpoint_freq > 0:
        save_freq = max(args.checkpoint_freq // args.n_envs, 1)
        callbacks.append(
            CheckpointCallback(
                save_freq=save_freq,
                save_path=str(run_dir / "checkpoints"),
                name_prefix=f"{algo}_static_nav",
                save_replay_buffer=algo in {"sac", "td3", "ddpg"},
                save_vecnormalize=vecnormalize_enabled(args),
            )
        )

    if args.eval_freq > 0:
        eval_freq = max(args.eval_freq // args.n_envs, 1)
        callbacks.append(
            PaperEvalCallback(
                eval_env=eval_env,
                run_dir=run_dir,
                eval_freq=eval_freq,
                n_eval_episodes=args.n_eval_episodes,
                start_stage=args.curriculum_stage,
                max_stage=args.max_curriculum_stage,
                curriculum_gate=args.curriculum_gate,
                curriculum_consecutive=args.curriculum_consecutive,
                early_stop_after_perfect_success=args.early_stop_after_perfect_success,
                perfect_success_threshold=args.perfect_success_threshold,
                success_drop_patience=args.success_drop_patience,
            )
        )

    if not callbacks:
        return None
    return CallbackList(callbacks)


def make_run_dir(args: argparse.Namespace, algo: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"_{args.run_tag}" if args.run_tag else ""
    run_dir = args.log_dir / f"{algo}_modified_static_nav{tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir: Path, args: argparse.Namespace, algo: str, backend: str) -> None:
    config = vars(args).copy()
    config["algo"] = algo
    config["vec_env_resolved"] = backend
    config["log_dir"] = str(args.log_dir)
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True)


def checkpoint_name_prefix(algo: str) -> str:
    return f"{algo}_static_nav"


def checkpoint_step(path: Path) -> int:
    match = re.search(r"_(\d+)_steps$", path.stem)
    if match is None:
        return -1
    return int(match.group(1))


def latest_checkpoint_path(run_dir: Path, algo: str) -> Path | None:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(
        checkpoint_dir.glob(f"{checkpoint_name_prefix(algo)}_*_steps.zip"),
        key=checkpoint_step,
    )
    return checkpoints[-1] if checkpoints else None


def replay_buffer_checkpoint_path(model_checkpoint: Path) -> Path:
    match = re.search(r"^(.*)_(\d+)_steps$", model_checkpoint.stem)
    if match is None:
        return model_checkpoint.with_suffix(".pkl")
    prefix, step = match.groups()
    return model_checkpoint.with_name(f"{prefix}_replay_buffer_{step}_steps.pkl")


def checkpoint_vecnormalize_path(model_checkpoint: Path) -> Path:
    match = re.search(r"^(.*)_(\d+)_steps$", model_checkpoint.stem)
    if match is None:
        return model_checkpoint.with_name("vecnormalize.pkl")
    prefix, step = match.groups()
    return model_checkpoint.with_name(f"{prefix}_vecnormalize_{step}_steps.pkl")


def load_model_for_resume(
    algo: str,
    args: argparse.Namespace,
    train_env,
    run_dir: Path,
    checkpoint_path: Path | None = None,
):
    checkpoint_path = checkpoint_path or latest_checkpoint_path(run_dir, algo)
    if checkpoint_path is None:
        return None, 0

    model = MODEL_CLASSES[algo].load(
        str(checkpoint_path),
        env=train_env,
        device=resolved_device(algo, args),
    )
    replay_buffer_path = replay_buffer_checkpoint_path(checkpoint_path)
    if replay_buffer_path.exists() and hasattr(model, "load_replay_buffer"):
        model.load_replay_buffer(str(replay_buffer_path))
    return model, int(getattr(model, "num_timesteps", checkpoint_step(checkpoint_path)))


def train_one_algorithm(
    algo: str,
    args: argparse.Namespace,
    backend: str,
    run_dir: Path | None = None,
    resume: bool = False,
) -> Path:
    run_dir = Path(run_dir) if run_dir is not None else make_run_dir(args, algo)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(run_dir, args, algo, backend)

    print(f"[{algo}] run directory: {run_dir}")
    validate_algorithm_args(algo, args)
    if algo == "a2c":
        print("[a2c] note: --batch-size is ignored by A2C.")

    checkpoint_path = latest_checkpoint_path(run_dir, algo) if resume else None
    vecnormalize_path = None
    if checkpoint_path is not None:
        candidate = checkpoint_vecnormalize_path(checkpoint_path)
        if candidate.exists():
            vecnormalize_path = candidate

    train_env = build_vec_env(
        seed=args.seed,
        n_envs=args.n_envs,
        backend=backend,
        monitor_file=run_dir / "train.monitor.csv",
        stage=args.curriculum_stage,
        args=args,
        training=True,
        vecnormalize_path=vecnormalize_path,
    )
    eval_env = build_vec_env(
        seed=args.seed + 50_000,
        n_envs=1,
        backend="dummy",
        monitor_file=run_dir / "eval.monitor.csv",
        stage=args.curriculum_stage,
        args=args,
        training=False,
        vecnormalize_path=vecnormalize_path,
    )

    try:
        model = None
        current_timesteps = 0
        if resume:
            model, current_timesteps = load_model_for_resume(
                algo,
                args,
                train_env,
                run_dir,
                checkpoint_path=checkpoint_path,
            )
            if model is not None:
                print(f"[{algo}] resuming from checkpoint at {current_timesteps} timesteps")

        if model is None:
            model = build_model(algo, train_env, args, run_dir)

        callbacks = build_callbacks(algo, eval_env, args, run_dir)
        remaining_timesteps = max(args.total_timesteps - current_timesteps, 0)

        if remaining_timesteps <= 0:
            print(f"[{algo}] target timesteps already reached, skipping training")
        else:
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=callbacks,
                log_interval=args.log_interval,
                progress_bar=args.progress_bar,
                reset_num_timesteps=current_timesteps == 0,
            )

        model.save(str(run_dir / f"final_{algo}_model.zip"))
        vec_normalize_env = model.get_vec_normalize_env()
        if vec_normalize_env is not None:
            vec_normalize_env.save(str(final_vecnormalize_path(run_dir)))
        if algo in {"sac", "td3", "ddpg"} and hasattr(model, "save_replay_buffer"):
            model.save_replay_buffer(str(run_dir / f"final_{algo}_replay_buffer.pkl"))
    finally:
        train_env.close()
        eval_env.close()

    print(f"[{algo}] training complete")
    return run_dir


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    backend = choose_vec_env_backend(args.vec_env, args.n_envs)
    algorithms = resolve_algorithms(args.algos)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    completed_runs = []
    for algo in algorithms:
        run_dir = train_one_algorithm(algo, args, backend)
        completed_runs.append((algo, run_dir))

    print("\nCompleted runs:")
    for algo, run_dir in completed_runs:
        print(f"  {algo}: {run_dir}")


if __name__ == "__main__":
    main()
