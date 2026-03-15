import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gym_pybullet_drones.envs.constrained_environment import (
    GOAL_POS,
    SIZE,
    STATIC_OBSTACLE_LAYOUT,
    UAV2DAvoidSimple1,
    _clearance_radius,
)
from sb3_model_utils import load_sb3_model_for_inference
from shap_distillation import compute_action_from_obs, equation_lines, load_result


PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Run spec must be ALGO=RUN_DIR.")
    algo, run_dir_str = spec.split("=", 1)
    run_dir = Path(run_dir_str).expanduser().resolve()
    if not run_dir.exists():
        raise argparse.ArgumentTypeError(f"Run directory not found: {run_dir}")
    return algo.lower().strip(), run_dir


def parse_override_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Override spec must be ALGO=PATH.")
    algo, path_str = spec.split("=", 1)
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Override path not found: {path}")
    return algo.lower().strip(), path


def resolve_run_artifacts(algo: str, run_dir: Path) -> dict:
    best_model = run_dir / "best_model" / "best_model.zip"
    final_model = run_dir / f"final_{algo}_model.zip"
    model_path = best_model if best_model.exists() else final_model
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found for {algo}: {run_dir}")

    equation_json = run_dir / "policy_analysis" / "shap_equation" / "shap_distillation_equation.json"
    if not equation_json.exists():
        raise FileNotFoundError(f"SHAP equation not found for {algo}: {equation_json}")

    vecnormalize_path = None
    best_vecnormalize = run_dir / "best_model" / "vecnormalize.pkl"
    final_vecnormalize = run_dir / "final_vecnormalize.pkl"
    if best_vecnormalize.exists():
        vecnormalize_path = best_vecnormalize
    elif final_vecnormalize.exists():
        vecnormalize_path = final_vecnormalize

    return {
        "algo": algo,
        "label": algo.upper(),
        "run_dir": run_dir,
        "model_path": model_path,
        "equation_json": equation_json,
        "vecnormalize_path": vecnormalize_path,
    }


def run_episode(env, controller_id: str, seed: int, model=None, equation=None):
    obs, info = env.reset(seed=seed)
    positions = [obs[:2].copy()]
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        if controller_id == "policy":
            action, _ = model.predict(obs, deterministic=True)
        elif controller_id == "equation":
            action = compute_action_from_obs(obs, equation)
        else:
            raise ValueError(f"Unsupported controller id: {controller_id}")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        positions.append(obs[:2].copy())
        done = terminated or truncated

    trajectory = np.asarray(positions, dtype=np.float32)
    deltas = np.diff(trajectory, axis=0)
    path_length = float(np.linalg.norm(deltas, axis=1).sum()) if len(deltas) else 0.0
    return trajectory, {
        "success": bool(info.get("is_success", False)),
        "collision": bool(info.get("collision", False)),
        "timeout": bool(not info.get("is_success", False) and not info.get("collision", False)),
        "goal_dist": float(info.get("goal_dist", 0.0)),
        "steps": int(steps),
        "path_length": path_length,
        "total_reward": float(total_reward),
    }


def resample_trajectory(trajectory: np.ndarray, num_points: int = 200) -> np.ndarray:
    if len(trajectory) == 1:
        return np.repeat(trajectory, num_points, axis=0)

    sample_t = np.linspace(0.0, len(trajectory) - 1, num_points)
    base_t = np.arange(len(trajectory), dtype=float)
    x = np.interp(sample_t, base_t, trajectory[:, 0])
    y = np.interp(sample_t, base_t, trajectory[:, 1])
    return np.column_stack([x, y]).astype(np.float32)


def trajectory_match_metrics(policy_traj: np.ndarray, equation_traj: np.ndarray) -> dict[str, float]:
    policy_resampled = resample_trajectory(policy_traj)
    equation_resampled = resample_trajectory(equation_traj)
    pointwise_l2 = np.linalg.norm(policy_resampled - equation_resampled, axis=1)
    endpoint_distance = float(np.linalg.norm(policy_traj[-1] - equation_traj[-1]))
    return {
        "mean_resampled_l2": float(np.mean(pointwise_l2)),
        "max_resampled_l2": float(np.max(pointwise_l2)),
        "endpoint_distance": endpoint_distance,
    }


def controller_summary(episodes: list[dict]) -> dict:
    return {
        "successes": int(sum(item["success"] for item in episodes)),
        "collisions": int(sum(item["collision"] for item in episodes)),
        "timeouts": int(sum(item["timeout"] for item in episodes)),
        "mean_goal_dist": float(np.mean([item["goal_dist"] for item in episodes])),
        "mean_path_length": float(np.mean([item["path_length"] for item in episodes])),
        "mean_reward": float(np.mean([item["total_reward"] for item in episodes])),
        "mean_steps": float(np.mean([item["steps"] for item in episodes])),
    }


def draw_world(ax):
    for spec in STATIC_OBSTACLE_LAYOUT:
        radius = _clearance_radius(spec)
        circle = plt.Circle(spec["position"], radius, color="steelblue", alpha=0.35)
        ax.add_patch(circle)

    ax.plot(*GOAL_POS, "y*", markersize=18, markeredgecolor="k")
    ax.set_xlim(-SIZE, SIZE)
    ax.set_ylim(-SIZE, SIZE)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.22)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run_spec,
        help="Run specification in ALGO=RUN_DIR format. Repeat for multiple algorithms.",
    )
    parser.add_argument(
        "--model-override",
        action="append",
        default=[],
        type=parse_override_spec,
        help="Optional override in ALGO=MODEL_PATH format.",
    )
    parser.add_argument(
        "--equation-override",
        action="append",
        default=[],
        type=parse_override_spec,
        help="Optional override in ALGO=SHAP_EQUATION_JSON format.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("policy_analysis"))
    parser.add_argument("--base-name", type=str, default="9_td3_ddpg_shap_distilled_vs_policy_trajectories")
    parser.add_argument("--curriculum-stage", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    args = parser.parse_args()

    run_specs = [resolve_run_artifacts(algo, run_dir) for algo, run_dir in args.run]
    model_overrides = dict(args.model_override)
    equation_overrides = dict(args.equation_override)
    for spec in run_specs:
        if spec["algo"] in model_overrides:
            spec["model_path"] = model_overrides[spec["algo"]]
        if spec["algo"] in equation_overrides:
            spec["equation_json"] = equation_overrides[spec["algo"]]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(run_specs), figsize=(7.2 * len(run_specs), 6.8))
    if len(run_specs) == 1:
        axes = [axes]

    legend_handles = [
        Line2D([0], [0], color="#111111", linewidth=2.8, linestyle="-", label="Original policy"),
        Line2D([0], [0], color="#111111", linewidth=2.4, linestyle="--", label="SHAP equation"),
    ]

    metrics = {
        "curriculum_stage": int(args.curriculum_stage),
        "seeds": [int(seed) for seed in args.seeds],
        "algorithms": [],
    }

    for ax, spec in zip(axes, run_specs):
        equation = load_result(spec["equation_json"])
        line_x, line_y = equation_lines(equation)
        model = load_sb3_model_for_inference(
            spec["algo"],
            spec["model_path"],
            vecnormalize_path=spec["vecnormalize_path"],
        )

        policy_env = UAV2DAvoidSimple1(render_mode=None, curriculum_stage=args.curriculum_stage)
        equation_env = UAV2DAvoidSimple1(render_mode=None, curriculum_stage=args.curriculum_stage)
        policy_env.set_curriculum_stage(args.curriculum_stage)
        equation_env.set_curriculum_stage(args.curriculum_stage)

        policy_episodes = []
        equation_episodes = []
        match_episodes = []

        try:
            for idx, seed in enumerate(args.seeds):
                color = PALETTE[idx % len(PALETTE)]
                policy_traj, policy_metrics = run_episode(
                    policy_env,
                    "policy",
                    seed=seed,
                    model=model,
                )
                equation_traj, equation_metrics = run_episode(
                    equation_env,
                    "equation",
                    seed=seed,
                    equation=equation,
                )
                match = trajectory_match_metrics(policy_traj, equation_traj)

                policy_metrics["seed"] = int(seed)
                equation_metrics["seed"] = int(seed)
                match["seed"] = int(seed)
                policy_episodes.append(policy_metrics)
                equation_episodes.append(equation_metrics)
                match_episodes.append(match)

                ax.plot(policy_traj[:, 0], policy_traj[:, 1], "-", color=color, linewidth=2.8, alpha=0.9)
                ax.plot(equation_traj[:, 0], equation_traj[:, 1], "--", color=color, linewidth=2.2, alpha=0.85)
                ax.plot(policy_traj[0, 0], policy_traj[0, 1], "s", color=color, markersize=6)
                ax.plot(policy_traj[-1, 0], policy_traj[-1, 1], "o", color=color, markersize=6, markeredgecolor="k")
                ax.plot(equation_traj[-1, 0], equation_traj[-1, 1], "x", color=color, markersize=7, markeredgewidth=1.6)
        finally:
            policy_env.close()
            equation_env.close()

        policy_summary = controller_summary(policy_episodes)
        equation_summary = controller_summary(equation_episodes)
        match_summary = {
            "mean_resampled_l2": float(np.mean([item["mean_resampled_l2"] for item in match_episodes])),
            "max_resampled_l2": float(np.max([item["max_resampled_l2"] for item in match_episodes])),
            "mean_endpoint_distance": float(np.mean([item["endpoint_distance"] for item in match_episodes])),
        }

        ax.set_title(
            f"{spec['label']}: policy {policy_summary['successes']}/{len(args.seeds)}, "
            f"equation {equation_summary['successes']}/{len(args.seeds)}\n"
            f"mean trace deviation = {match_summary['mean_resampled_l2']:.3f} m",
            fontsize=13,
        )
        draw_world(ax)

        metrics["algorithms"].append(
            {
                "algo": spec["algo"],
                "label": spec["label"],
                "model_path": str(spec["model_path"]),
                "equation_json": str(spec["equation_json"]),
                "equations": {
                    "action_x": line_x,
                    "action_y": line_y,
                    "r2_full": float(equation["metrics"]["r2_full"]),
                    "r2_x": float(equation["metrics"]["r2_x"]),
                    "r2_y": float(equation["metrics"]["r2_y"]),
                    "mae": float(equation["metrics"]["mae"]),
                    "candidate": equation["candidate"]["name"],
                },
                "policy_summary": policy_summary,
                "equation_summary": equation_summary,
                "trajectory_match_summary": match_summary,
                "policy_episodes": policy_episodes,
                "equation_episodes": equation_episodes,
                "trajectory_match_episodes": match_episodes,
            }
        )

    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    plot_path = args.out_dir / f"{args.base_name}.png"
    metrics_path = args.out_dir / f"{args.base_name}.json"
    fig.savefig(plot_path, dpi=180)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved trajectory plot to {plot_path.resolve()}")
    print(f"Saved trajectory metrics to {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
