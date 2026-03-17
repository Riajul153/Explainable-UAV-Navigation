from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from gym_pybullet_drones.envs.constrained_environment import (
    GOAL_POS,
    SHAPE_CODES,
    SIZE,
    STATIC_OBSTACLE_LAYOUT,
    UAV2DAvoidSimple1,
    UAV_R,
    _clearance_radius,
)
from shap_distillation import compute_action_from_obs, load_result

FIG_DIR = ROOT / "paper_springer" / "figures"
SAC_STAGE4_MODEL = ROOT / "gym_pybullet_drones/examples/runs/static_nav/sac_modified_static_nav_ablation_single_shot_20260313-042738/best_model/best_model.zip"
DISTILL_SAC_RUN = ROOT / "gym_pybullet_drones/examples/runs/static_nav/sac_modified_static_nav_sac_30m_curriculum_v5_seed0_20260313-003023"
DISTILL_SAC_MODEL = DISTILL_SAC_RUN / "best_model" / "best_model.zip"
DISTILL_SAC_EQUATION_JSON = DISTILL_SAC_RUN / "policy_analysis" / "shap_equation" / "shap_distillation_equation.json"

ALGO_SERIES = {
    "SAC": ROOT / "gym_pybullet_drones/examples/runs/static_nav/sac_modified_static_nav_ablation_single_shot_20260313-042738/paper_metrics.csv",
    "TD3": ROOT / "gym_pybullet_drones/examples/runs/kf_nav_suite/kf_baselines_20260314-164705/runs/td3_modified_static_nav_kf_baselines_20260314-173657/paper_metrics.csv",
    "DDPG": ROOT / "gym_pybullet_drones/examples/runs/kf_nav_suite/kf_baselines_20260314-164705/runs/ddpg_modified_static_nav_kf_stage4_optuna_20260314-204207/paper_metrics.csv",
    "PPO": ROOT / "gym_pybullet_drones/examples/runs/kf_nav_suite/kf_baselines_20260314-164705/runs/ppo_modified_static_nav_kf_stage4_optuna_20260314-211046/paper_metrics.csv",
}

KALMAN_SERIES = {
    "SAC + Kalman": ROOT / "gym_pybullet_drones/examples/runs/static_nav/sac_modified_static_nav_ablation_single_shot_20260313-042738/paper_metrics.csv",
    "SAC without Kalman": ROOT / "gym_pybullet_drones/examples/runs/static_nav/sac_modified_static_nav_single_shot_no_kf_20260313-115635/paper_metrics.csv",
}

COPIED_FIGURES = {
    "paper_first_1050k_clean_success_rate.pdf": ROOT / "gym_pybullet_drones/examples/runs/kf_nav_suite/kf_baselines_20260314-164705/comparison_plots/paper_first_1050k_clean_success_rate.pdf",
    "paper_first_1050k_clean_mean_reward.pdf": ROOT / "gym_pybullet_drones/examples/runs/kf_nav_suite/kf_baselines_20260314-164705/comparison_plots/paper_first_1050k_clean_mean_reward.pdf",
    "q_value_landscape.png": ROOT / "policy_analysis/2_q_value_landscape.png",
    "trajectory_traces.png": ROOT / "policy_analysis/5_trajectory_traces.png",
    "stage4_shap_trajectories.png": ROOT / "policy_analysis/16_sac_td3_ddpg_final_selected_stage4_shap_distilled_vs_policy_trajectories.png",
    "sac_shap_action_x.png": ROOT / "gym_pybullet_drones/examples/runs/static_nav/sac_modified_static_nav_sac_30m_curriculum_v5_seed0_20260313-003023/policy_analysis/explain/shap_summary_action_x.png",
    "sac_shap_action_y.png": ROOT / "gym_pybullet_drones/examples/runs/static_nav/sac_modified_static_nav_sac_30m_curriculum_v5_seed0_20260313-003023/policy_analysis/explain/shap_summary_action_y.png",
    "sac_lime_action_x.png": ROOT / "gym_pybullet_drones/examples/runs/static_nav/sac_modified_static_nav_sac_30m_curriculum_v5_seed0_20260313-003023/policy_analysis/explain/lime_explain_action_x.png",
    "sac_lime_action_y.png": ROOT / "gym_pybullet_drones/examples/runs/static_nav/sac_modified_static_nav_sac_30m_curriculum_v5_seed0_20260313-003023/policy_analysis/explain/lime_explain_action_y.png",
}

PALETTE = {
    "SAC": "#1f77b4",
    "TD3": "#ff7f0e",
    "DDPG": "#2ca02c",
    "PPO": "#d62728",
    "SAC + Kalman": "#1f77b4",
    "SAC without Kalman": "#7f7f7f",
}

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.9,
        "lines.linewidth": 2.1,
        "savefig.dpi": 450,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

OBSTACLE_POS = np.array([spec["position"] for spec in STATIC_OBSTACLE_LAYOUT], dtype=float)
OBSTACLE_RADII = np.array([_clearance_radius(spec) for spec in STATIC_OBSTACLE_LAYOUT], dtype=float)
OBSTACLE_SHAPES = np.array([SHAPE_CODES[spec["shape"]] for spec in STATIC_OBSTACLE_LAYOUT], dtype=float)
OBSTACLE_SAFETY_RADII = OBSTACLE_RADII + 0.25
DEEP_TREE_TRAJECTORY_JSON = FIG_DIR / "deep_tree_trajectory_traces.json"
SEED_COLORS = {
    42: "#2ca58d",
    43: "#f18f01",
    44: "#6c5ce7",
    45: "#e84393",
}


def smooth_by_timestep(timesteps: np.ndarray, values: np.ndarray, window_timesteps: int) -> np.ndarray:
    smoothed = np.empty_like(values, dtype=float)
    half_window = window_timesteps / 2.0
    for idx, timestep in enumerate(timesteps):
        mask = np.abs(timesteps - timestep) <= half_window
        smoothed[idx] = float(np.mean(values[mask]))
    return smoothed


def rolling_spread_by_timestep(timesteps: np.ndarray, values: np.ndarray, window_timesteps: int) -> np.ndarray:
    spread = np.empty_like(values, dtype=float)
    half_window = window_timesteps / 2.0
    for idx, timestep in enumerate(timesteps):
        mask = np.abs(timesteps - timestep) <= half_window
        spread[idx] = float(np.std(values[mask]))
    return spread


def load_series(series_map: dict[str, Path], max_timesteps: int) -> list[tuple[str, pd.DataFrame]]:
    loaded: list[tuple[str, pd.DataFrame]] = []
    for label, csv_path in series_map.items():
        df = pd.read_csv(csv_path).sort_values("timesteps")
        df = df[df["timesteps"] <= max_timesteps].copy()
        loaded.append((label, df))
    return loaded


def style_axis(ax, ylabel: str, x_max_millions: float, y_range=None) -> None:
    ax.set_xlabel("Timesteps (Millions)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, x_max_millions)
    if y_range is not None:
        ax.set_ylim(*y_range)
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.28, color="#9f9f9f")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", width=0.8, length=4)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))


def save(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_obs(uav_xy: np.ndarray, uav_v: np.ndarray) -> np.ndarray:
    goal_vec = GOAL_POS - uav_xy
    goal_dist = float(np.linalg.norm(goal_vec) + 1e-9)
    return np.concatenate(
        [
            uav_xy,
            uav_v,
            goal_vec,
            [goal_dist],
            OBSTACLE_POS[:, 0],
            OBSTACLE_POS[:, 1],
            OBSTACLE_RADII,
            OBSTACLE_SHAPES,
        ]
    ).astype(np.float32)


def point_clearance(xy: np.ndarray) -> float:
    center_dists = np.linalg.norm(xy[None, :] - OBSTACLE_POS, axis=1)
    clearances = center_dists - (UAV_R + OBSTACLE_RADII)
    return float(np.min(clearances))


def obstacle_mask(xy: np.ndarray, pad: float = 0.0) -> bool:
    center_dists = np.linalg.norm(xy[None, :] - OBSTACLE_POS, axis=1)
    return bool(np.any(center_dists <= (UAV_R + OBSTACLE_RADII + pad)))


def draw_obstacles(ax, opaque: bool = True) -> None:
    for spec in STATIC_OBSTACLE_LAYOUT:
        color = spec["color"][:3]
        edge = "#0d3b66"
        facealpha = 1.0 if opaque else 0.75
        if spec["shape"] == "sphere":
            patch = Circle(spec["position"], radius=spec["radius"], facecolor=color, edgecolor=edge, linewidth=1.2, alpha=facealpha, zorder=5)
        elif spec["shape"] == "cylinder":
            patch = Circle(spec["position"], radius=spec["radius"], facecolor=color, edgecolor=edge, linewidth=1.2, alpha=facealpha, zorder=5)
        else:
            hx, hy, _ = spec["half_extents"]
            patch = Rectangle((spec["position"][0] - hx, spec["position"][1] - hy), 2 * hx, 2 * hy, facecolor=color, edgecolor=edge, linewidth=1.2, alpha=facealpha, zorder=5)
        ax.add_patch(patch)


def style_workspace(ax, title: str | None = None) -> None:
    ax.set_xlim(-SIZE, SIZE)
    ax.set_ylim(-SIZE, SIZE)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.22)
    ax.scatter([GOAL_POS[0]], [GOAL_POS[1]], marker="*", s=180, color="#f4d35e", edgecolor="#7a5c00", linewidth=1.0, zorder=8)


def generate_action_vector_field(out_base: Path) -> None:
    model = SAC.load(str(SAC_STAGE4_MODEL))
    xs = np.linspace(-9.2, 9.2, 25)
    ys = np.linspace(-9.2, 9.2, 25)
    x_grid, y_grid = np.meshgrid(xs, ys)
    u = np.full_like(x_grid, np.nan, dtype=float)
    v = np.full_like(y_grid, np.nan, dtype=float)

    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            xy = np.array([x_grid[i, j], y_grid[i, j]], dtype=np.float32)
            if obstacle_mask(xy, pad=0.18):
                continue
            obs = build_obs(xy, np.zeros(2, dtype=np.float32))
            action, _ = model.predict(obs, deterministic=True)
            u[i, j] = float(action[0])
            v[i, j] = float(action[1])

    fig, ax = plt.subplots(figsize=(3.35, 3.25), constrained_layout=True)
    q = ax.quiver(
        x_grid,
        y_grid,
        u,
        v,
        np.hypot(np.nan_to_num(u), np.nan_to_num(v)),
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=2.8,
        width=0.006,
        alpha=0.95,
        zorder=2,
    )
    draw_obstacles(ax, opaque=True)
    style_workspace(ax)
    ax.scatter([-9.8], [-9.8], s=28, color="#111111", zorder=7)
    cbar = fig.colorbar(q, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Action magnitude")
    save(fig, out_base)


def apf_potential(xy: np.ndarray, k_goal: float, k_rep: float, d0: float) -> float:
    goal_term = k_goal * float(np.linalg.norm(xy - GOAL_POS))
    deltas = xy[None, :] - OBSTACLE_POS
    dists = np.linalg.norm(deltas, axis=1)
    clearances = dists - (UAV_R + OBSTACLE_RADII)
    repulsion = 0.0
    for clearance in clearances:
        if clearance < d0:
            clearance = max(float(clearance), 1e-3)
            repulsion += 0.5 * k_rep * (1.0 / clearance - 1.0 / d0) ** 2
    return goal_term + repulsion


def apf_grad(xy: np.ndarray, k_goal: float, k_rep: float, d0: float, eps: float = 1e-3) -> np.ndarray:
    grad = np.zeros(2, dtype=float)
    for i in range(2):
        step = np.zeros(2, dtype=float)
        step[i] = eps
        grad[i] = (
            apf_potential(xy + step, k_goal, k_rep, d0) - apf_potential(xy - step, k_goal, k_rep, d0)
        ) / (2.0 * eps)
    return grad


def rollout_apf_trap(start: np.ndarray, k_goal: float, k_rep: float, d0: float) -> tuple[np.ndarray, np.ndarray]:
    xy = start.astype(float).copy()
    traj = [xy.copy()]
    for _ in range(1600):
        grad = apf_grad(xy, k_goal, k_rep, d0)
        norm = float(np.linalg.norm(grad))
        if norm < 1e-6:
            break
        step = -0.10 * grad / max(norm, 1e-9)
        candidate = xy + step
        if obstacle_mask(candidate, pad=0.02):
            step *= 0.35
            candidate = xy + step
        xy = candidate
        traj.append(xy.copy())
        if np.linalg.norm(xy - GOAL_POS) < 0.7:
            break
        if len(traj) > 80:
            recent = np.asarray(traj[-80:])
            span = float(np.max(np.linalg.norm(recent - recent.mean(axis=0), axis=1)))
            if span < 0.06:
                break
    return np.asarray(traj), xy


def rollout_stage4_sac(start: np.ndarray, seed: int = 0) -> np.ndarray:
    model = SAC.load(str(SAC_STAGE4_MODEL))
    env = UAV2DAvoidSimple1(render_mode=None, curriculum_stage=4)
    env.set_curriculum_stage(4)
    obs, _ = env.reset(seed=seed)
    env._set_xy(env._uav, start, UAV_R)
    env._uav_v[:] = 0.0
    env._prev_uav_v[:] = 0.0
    env._prev_goal_dist = float(np.linalg.norm(env._goal_position() - start) + 1e-9)
    env._kf_init(start, env._uav_v)
    obs = env._get_obs()
    traj = [env._uav_position().copy()]
    for _ in range(env.time_limit_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        traj.append(env._uav_position().copy())
        if terminated or truncated:
            break
    env.close()
    return np.asarray(traj)


def generate_policy_vs_potential_field(out_base: Path) -> None:
    start = np.array([-9.8, -9.8], dtype=np.float32)
    k_goal, k_rep, d0 = 0.2, 8.0, 4.0
    apf_traj, stuck_xy = rollout_apf_trap(start, k_goal, k_rep, d0)
    sac_traj = rollout_stage4_sac(start, seed=0)

    xs = np.linspace(-SIZE, SIZE, 220)
    ys = np.linspace(-SIZE, SIZE, 220)
    xx, yy = np.meshgrid(xs, ys)
    potential = np.full_like(xx, np.nan, dtype=float)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            xy = np.array([xx[i, j], yy[i, j]], dtype=float)
            if obstacle_mask(xy, pad=0.0):
                continue
            potential[i, j] = apf_potential(xy, k_goal, k_rep, d0)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), constrained_layout=True)

    im = axes[0].contourf(xx, yy, potential, levels=32, cmap="magma_r", alpha=0.92, zorder=1)
    draw_obstacles(axes[0], opaque=True)
    axes[0].plot(apf_traj[:, 0], apf_traj[:, 1], color="#0b6e4f", linewidth=2.0, zorder=7)
    axes[0].scatter([start[0]], [start[1]], marker="o", s=28, color="#111111", zorder=8)
    axes[0].scatter([stuck_xy[0]], [stuck_xy[1]], marker="X", s=80, color="#b80c09", edgecolor="white", linewidth=0.6, zorder=9)
    axes[0].annotate("Local minimum", xy=stuck_xy, xytext=(stuck_xy[0] + 1.3, stuck_xy[1] - 1.2), arrowprops={"arrowstyle": "->", "lw": 1.0}, fontsize=8)
    style_workspace(axes[0], "Untuned APF Trap")

    axes[1].plot(sac_traj[:, 0], sac_traj[:, 1], color="#1f77b4", linewidth=2.2, zorder=7)
    draw_obstacles(axes[1], opaque=True)
    axes[1].scatter([start[0]], [start[1]], marker="o", s=28, color="#111111", zorder=8)
    style_workspace(axes[1], "Stage-4 SAC Rollout")

    cbar = fig.colorbar(im, ax=axes, fraction=0.026, pad=0.02)
    cbar.set_label("Untuned APF potential")
    save(fig, out_base)


def load_deep_tree_trajectories() -> list[dict[str, object]]:
    with DEEP_TREE_TRAJECTORY_JSON.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return list(payload["trajectories"])


def seed_label_offset(seed: int, xy: np.ndarray) -> tuple[float, float]:
    fixed_offsets = {
        42: (0.35, 0.45),
        43: (0.50, -0.35),
        44: (0.20, -0.95),
        45: (-0.85, -0.55),
    }
    if seed in fixed_offsets:
        return fixed_offsets[seed]
    dx = 0.45 if xy[0] < 0.0 else -0.95
    dy = 0.45 if xy[1] < 0.0 else -0.65
    return dx, dy


def generate_policy_trajectory_traces(out_base: Path) -> None:
    trajectories = load_deep_tree_trajectories()
    fig, ax = plt.subplots(figsize=(3.2, 3.05), constrained_layout=True)

    for entry in trajectories:
        seed = int(entry["seed"])
        color = SEED_COLORS.get(seed, "#1f77b4")
        policy = np.asarray(entry["policy"], dtype=float)
        ax.plot(policy[:, 0], policy[:, 1], color=color, linewidth=1.9, alpha=0.95, zorder=7)
        ax.scatter([policy[0, 0]], [policy[0, 1]], s=18, color=color, edgecolor="white", linewidth=0.4, zorder=8)
        dx, dy = seed_label_offset(seed, policy[0])
        ax.text(
            policy[0, 0] + dx,
            policy[0, 1] + dy,
            str(seed),
            color=color,
            fontsize=7,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 0.2},
        )

    draw_obstacles(ax, opaque=False)
    style_workspace(ax)
    save(fig, out_base)


def generate_deep_tree_trajectory_traces(out_base: Path) -> None:
    trajectories = load_deep_tree_trajectories()
    fig, ax = plt.subplots(figsize=(3.45, 3.75), constrained_layout=True)

    for entry in trajectories:
        seed = int(entry["seed"])
        color = SEED_COLORS.get(seed, "#1f77b4")
        policy = np.asarray(entry["policy"], dtype=float)
        tree = np.asarray(entry["tree"], dtype=float)
        ax.plot(policy[:, 0], policy[:, 1], color=color, linewidth=2.0, alpha=0.95, zorder=7)
        ax.plot(tree[:, 0], tree[:, 1], color=color, linewidth=1.8, linestyle="--", alpha=0.95, zorder=7)
        ax.scatter([policy[0, 0]], [policy[0, 1]], s=20, color=color, edgecolor="white", linewidth=0.4, zorder=8)
        dx, dy = seed_label_offset(seed, policy[0])
        ax.text(
            policy[0, 0] + dx,
            policy[0, 1] + dy,
            str(seed),
            color=color,
            fontsize=7,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 0.2},
        )

    style_handles = [
        Line2D([0], [0], color="#222222", linewidth=2.0, linestyle="-", label="SAC policy"),
        Line2D([0], [0], color="#222222", linewidth=1.8, linestyle="--", label="Depth-15 tree"),
    ]
    draw_obstacles(ax, opaque=False)
    style_workspace(ax)
    ax.legend(
        handles=style_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.12),
        handlelength=2.2,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    save(fig, out_base)


def rollout_shap_equation(seed: int, equation: dict) -> np.ndarray:
    env = UAV2DAvoidSimple1(render_mode=None, curriculum_stage=4)
    env.set_curriculum_stage(4)
    obs, _ = env.reset(seed=seed)
    positions = [obs[:2].copy()]

    try:
        done = False
        while not done:
            action = compute_action_from_obs(obs, equation)
            obs, reward, terminated, truncated, info = env.step(action)
            positions.append(obs[:2].copy())
            done = terminated or truncated
    finally:
        env.close()

    return np.asarray(positions, dtype=np.float32)


def plot_seed_label(ax, seed: int, xy: np.ndarray) -> None:
    color = SEED_COLORS.get(seed, "#1f77b4")
    dx, dy = seed_label_offset(seed, xy)
    ax.text(
        xy[0] + dx,
        xy[1] + dy,
        str(seed),
        color=color,
        fontsize=7,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 0.2},
    )


def generate_surrogate_trace_panel(out_base: Path) -> None:
    trajectories = load_deep_tree_trajectories()
    equation = load_result(DISTILL_SAC_EQUATION_JSON)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), constrained_layout=True)

    for ax, title in zip(
        axes,
        ("(a) Neural policy", "(b) Depth-15 tree", "(c) Compact equation"),
    ):
        draw_obstacles(ax, opaque=False)
        style_workspace(ax, title)

    for entry in trajectories:
        seed = int(entry["seed"])
        color = SEED_COLORS.get(seed, "#1f77b4")
        policy = np.asarray(entry["policy"], dtype=float)
        tree = np.asarray(entry["tree"], dtype=float)
        equation_traj = rollout_shap_equation(seed, equation)

        axes[0].plot(policy[:, 0], policy[:, 1], color=color, linewidth=2.0, alpha=0.95, zorder=7)
        axes[1].plot(tree[:, 0], tree[:, 1], color=color, linewidth=2.0, alpha=0.95, zorder=7)
        axes[2].plot(equation_traj[:, 0], equation_traj[:, 1], color=color, linewidth=2.0, alpha=0.95, zorder=7)

        for ax, traj in zip(axes, (policy, tree, equation_traj)):
            ax.scatter([traj[0, 0]], [traj[0, 1]], s=18, color=color, edgecolor="white", linewidth=0.4, zorder=8)

        plot_seed_label(axes[0], seed, policy[0])

    save(fig, out_base)


def draw_block(ax, xy: tuple[float, float], w: float, h: float, text: str, face: str) -> None:
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="#304050",
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2.0, xy[1] + h / 2.0, text, ha="center", va="center", fontsize=7.3)


def draw_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color="#405060",
        )
    )


def draw_architecture_panel(
    ax,
    title: str,
    actor_text: str,
    head_text: str,
    critic_text: str,
    note_text: str,
    twin_critics: bool = False,
) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title(title, fontsize=9.2, pad=3)

    draw_block(ax, (0.03, 0.68), 0.18, 0.17, "39-d\nobservation", "#e9f2ff")
    draw_block(ax, (0.30, 0.68), 0.22, 0.17, actor_text, "#fff3db")
    draw_block(ax, (0.62, 0.68), 0.20, 0.17, head_text, "#e8f7e8")
    draw_block(ax, (0.62, 0.14), 0.20, 0.16, critic_text, "#f6e8ff")
    if twin_critics:
        draw_block(ax, (0.62, 0.36), 0.20, 0.16, "Second critic", "#f6e8ff")

    draw_arrow(ax, (0.21, 0.765), (0.30, 0.765))
    draw_arrow(ax, (0.52, 0.765), (0.62, 0.765))
    draw_arrow(ax, (0.43, 0.68), (0.68, 0.30 if not twin_critics else 0.52))
    draw_arrow(ax, (0.72, 0.68), (0.72, 0.30 if not twin_critics else 0.52))
    if twin_critics:
        draw_arrow(ax, (0.72, 0.46), (0.72, 0.30))

    ax.text(
        0.03,
        0.04,
        note_text,
        ha="left",
        va="bottom",
        fontsize=6.8,
        wrap=True,
    )


def generate_algorithm_architecture_panel(out_base: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.05), constrained_layout=True)
    axes = axes.ravel()

    draw_architecture_panel(
        axes[0],
        "PPO",
        "Actor MLP\n512-256-256",
        "Gaussian policy\n(on-policy)",
        "Value MLP\n512-256-256",
        "Separate policy and value streams; no replay buffer.",
    )
    draw_architecture_panel(
        axes[1],
        "DDPG",
        "Actor MLP\n256-256",
        "Deterministic\naction",
        "Single Q critic",
        "Shallowest actor among the tested off-policy controllers.",
    )
    draw_architecture_panel(
        axes[2],
        "TD3",
        "Actor MLP\n256-256-256",
        "Deterministic\naction",
        "Twin Q critics",
        "Same actor depth as SAC, but deterministic head and delayed policy updates.",
        twin_critics=True,
    )
    draw_architecture_panel(
        axes[3],
        "SAC",
        "Actor MLP\n256-256-256",
        "Gaussian mean\n+ log std",
        "Twin Q critics",
        "Same backbone depth as TD3, but stochastic actor and entropy-regularised objective.",
        twin_critics=True,
    )

    save(fig, out_base)


def plot_two_panel(series_map: dict[str, Path], out_base: Path, max_timesteps: int = 1_050_000, window_timesteps: int = 300_000) -> None:
    loaded = load_series(series_map, max_timesteps=max_timesteps)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)
    legend_handles: list[Line2D] = []

    for idx, (label, df) in enumerate(loaded):
        color = PALETTE.get(label, plt.cm.tab10(idx))
        timesteps = df["timesteps"].to_numpy(dtype=float)
        x = timesteps / 1e6

        for ax, metric, ylabel, y_range in (
            (axes[0], "success_rate", "Success Rate", (-0.02, 1.02)),
            (axes[1], "mean_reward", "Mean Reward", None),
        ):
            values = df[metric].to_numpy(dtype=float)
            smoothed = smooth_by_timestep(timesteps, values, window_timesteps)
            spread = rolling_spread_by_timestep(timesteps, values, window_timesteps)
            lower = smoothed - spread
            upper = smoothed + spread
            if metric == "success_rate":
                lower = np.clip(lower, 0.0, 1.0)
                upper = np.clip(upper, 0.0, 1.0)
            ax.fill_between(x, lower, upper, color=color, alpha=0.14, linewidth=0)
            ax.plot(x, smoothed, color=color)

        legend_handles.append(Line2D([0], [0], color=color, lw=2.1, label=label))

    style_axis(axes[0], "Success Rate", max_timesteps / 1e6, y_range=(-0.02, 1.02))
    style_axis(axes[1], "Mean Reward", max_timesteps / 1e6, y_range=None)
    axes[0].set_title("Evaluation Success")
    axes[1].set_title("Evaluation Reward")

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=min(4, len(legend_handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
        handlelength=2.2,
        columnspacing=1.2,
    )
    save(fig, out_base)


def copy_static_figures() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for out_name, src in COPIED_FIGURES.items():
        shutil.copy2(src, FIG_DIR / out_name)


def build_panel(out_base: Path, items: list[tuple[str, Path]], ncols: int = 2, figsize=(7.1, 5.4)) -> None:
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax in axes.ravel():
        ax.axis("off")

    for idx, (label, img_path) in enumerate(items):
        ax = axes[idx // ncols, idx % ncols]
        image = plt.imread(img_path)
        ax.imshow(image)
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5},
        )

    save(fig, out_base)


def main() -> None:
    plot_two_panel(ALGO_SERIES, FIG_DIR / "algorithm_comparison")
    plot_two_panel(KALMAN_SERIES, FIG_DIR / "kalman_ablation")
    copy_static_figures()
    generate_algorithm_architecture_panel(FIG_DIR / "algorithm_architectures")
    generate_action_vector_field(FIG_DIR / "action_vector_field")
    generate_policy_vs_potential_field(FIG_DIR / "policy_vs_potential_field")
    generate_policy_trajectory_traces(FIG_DIR / "trajectory_traces")
    generate_deep_tree_trajectory_traces(FIG_DIR / "deep_tree_trajectory_traces")
    generate_surrogate_trace_panel(FIG_DIR / "surrogate_trace_panel")
    build_panel(
        FIG_DIR / "interpretability_panel",
        [
            ("(a) Action vector field", FIG_DIR / "action_vector_field.png"),
            ("(b) Q-value landscape", FIG_DIR / "q_value_landscape.png"),
            ("(c) Policy vs. potential field", FIG_DIR / "policy_vs_potential_field.png"),
            ("(d) Policy trajectory traces", FIG_DIR / "trajectory_traces.png"),
        ],
        ncols=2,
        figsize=(7.1, 5.5),
    )
    build_panel(
        FIG_DIR / "explainability_panel",
        [
            ("(a) SHAP for action x", FIG_DIR / "sac_shap_action_x.png"),
            ("(b) SHAP for action y", FIG_DIR / "sac_shap_action_y.png"),
            ("(c) LIME for action x", FIG_DIR / "sac_lime_action_x.png"),
            ("(d) LIME for action y", FIG_DIR / "sac_lime_action_y.png"),
        ],
        ncols=2,
        figsize=(7.1, 5.2),
    )


if __name__ == "__main__":
    main()
