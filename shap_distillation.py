import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gym_pybullet_drones.envs.constrained_environment import (
    GOAL_POS,
    SHAPE_CODES,
    SIZE,
    STATIC_OBSTACLE_LAYOUT,
    UAV2DAvoidSimple1,
    UAV_R,
    _clearance_radius,
)

DEFAULT_MODEL_PATH = Path(
    "gym_pybullet_drones/examples/runs/static_nav/"
    "sac_modified_static_nav_sac_30m_curriculum_v5_seed0_20260313-003023/"
    "best_model/best_model.zip"
)
DEFAULT_ANALYSIS_DIR = Path("policy_analysis")
DEFAULT_EQUATION_JSON = DEFAULT_ANALYSIS_DIR / "shap_distillation_equation.json"
DEFAULT_EQUATION_MD = DEFAULT_ANALYSIS_DIR / "shap_distillation_equation.md"
EPS = 1e-9
RIDGE_ALPHA = 0.15
OPTIONAL_FEATURES_PER_AXIS = 4


def nominal_obstacle_data():
    positions = np.array([s["position"] for s in STATIC_OBSTACLE_LAYOUT], dtype=np.float32)
    collision_radii = np.array(
        [_clearance_radius(s) for s in STATIC_OBSTACLE_LAYOUT], dtype=np.float32
    )
    shape_codes = np.array(
        [SHAPE_CODES[s["shape"]] for s in STATIC_OBSTACLE_LAYOUT], dtype=np.float32
    )
    return positions, collision_radii, shape_codes


def build_obs(uav_xy, uav_v, obst_pos, obst_coll_r, obst_shape):
    goal_vec = GOAL_POS - uav_xy
    goal_dist = float(np.linalg.norm(goal_vec) + EPS)
    obs = np.concatenate(
        [
            uav_xy,
            uav_v,
            goal_vec,
            [goal_dist],
            obst_pos[:, 0],
            obst_pos[:, 1],
            obst_coll_r,
            obst_shape,
        ]
    ).astype(np.float32)
    return obs


def obstacle_slices(num_obstacles):
    start = 7
    obs_x = slice(start, start + num_obstacles)
    obs_y = slice(start + num_obstacles, start + 2 * num_obstacles)
    obs_r = slice(start + 2 * num_obstacles, start + 3 * num_obstacles)
    return obs_x, obs_y, obs_r


def extract_obstacle_arrays(obs, num_obstacles):
    obs_x, obs_y, obs_r = obstacle_slices(num_obstacles)
    positions = np.column_stack((obs[obs_x], obs[obs_y])).astype(np.float32)
    radii = np.asarray(obs[obs_r], dtype=np.float32)
    return positions, radii


def _allocate_dataset(n_samples, sampling_mode, seed, curriculum_stage):
    data = {
        "uav_x": np.zeros(n_samples, dtype=np.float32),
        "uav_y": np.zeros(n_samples, dtype=np.float32),
        "vx": np.zeros(n_samples, dtype=np.float32),
        "vy": np.zeros(n_samples, dtype=np.float32),
        "goal_dx": np.zeros(n_samples, dtype=np.float32),
        "goal_dy": np.zeros(n_samples, dtype=np.float32),
        "goal_dist": np.zeros(n_samples, dtype=np.float32),
        "away_x": np.zeros(n_samples, dtype=np.float32),
        "away_y": np.zeros(n_samples, dtype=np.float32),
        "min_clear": np.zeros(n_samples, dtype=np.float32),
        "actions": np.zeros((n_samples, 2), dtype=np.float32),
        "sampling_mode": sampling_mode,
        "seed": int(seed),
        "curriculum_stage": curriculum_stage,
    }
    return data


def _record_dataset_row(dataset, idx, obs, action):
    obs = np.asarray(obs, dtype=np.float32)
    num_obstacles = int((len(obs) - 7) // 4)
    obstacle_positions, obstacle_radii = extract_obstacle_arrays(obs, num_obstacles)

    deltas = obs[:2][None, :] - obstacle_positions
    dists = np.linalg.norm(deltas, axis=1)
    clearances = dists - (UAV_R + obstacle_radii)
    nearest_idx = int(np.argmin(clearances))
    safe_dist = float(dists[nearest_idx] + EPS)
    away_vec = deltas[nearest_idx] / safe_dist

    dataset["uav_x"][idx] = float(obs[0])
    dataset["uav_y"][idx] = float(obs[1])
    dataset["vx"][idx] = float(obs[2])
    dataset["vy"][idx] = float(obs[3])
    dataset["goal_dx"][idx] = float(obs[4])
    dataset["goal_dy"][idx] = float(obs[5])
    dataset["goal_dist"][idx] = float(obs[6])
    dataset["away_x"][idx] = float(away_vec[0])
    dataset["away_y"][idx] = float(away_vec[1])
    dataset["min_clear"][idx] = float(clearances[nearest_idx])
    dataset["actions"][idx] = np.asarray(action, dtype=np.float32)


def sample_policy_dataset(model, n_samples=30000, seed=7, env=None):
    if env is None:
        rng = np.random.default_rng(seed)
        obst_pos, obst_coll_r, obst_shape = nominal_obstacle_data()
        dataset = _allocate_dataset(
            n_samples=n_samples,
            sampling_mode="random_nominal",
            seed=seed,
            curriculum_stage=None,
        )
        for idx in range(n_samples):
            uav_xy = rng.uniform(-SIZE + 1.0, SIZE - 1.0, size=2).astype(np.float32)
            uav_v = rng.uniform(-1.5, 1.5, size=2).astype(np.float32)
            obs = build_obs(uav_xy, uav_v, obst_pos, obst_coll_r, obst_shape)
            action, _ = model.predict(obs, deterministic=True)
            _record_dataset_row(dataset, idx, obs, action)
        dataset["episodes_collected"] = 0
        return dataset

    dataset = _allocate_dataset(
        n_samples=n_samples,
        sampling_mode="on_policy_rollout",
        seed=seed,
        curriculum_stage=int(getattr(env, "curriculum_stage", 0)),
    )
    idx = 0
    episode_count = 0

    while idx < n_samples:
        obs, _ = env.reset(seed=seed + episode_count)
        done = False
        while not done and idx < n_samples:
            action, _ = model.predict(obs, deterministic=True)
            _record_dataset_row(dataset, idx, obs, action)
            idx += 1
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        episode_count += 1

    dataset["episodes_collected"] = int(episode_count)
    return dataset


def default_candidate_specs():
    thresholds = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5)
    inverse_caps = (50.0, 100.0)
    candidates = []

    for threshold in thresholds:
        for power in (1, 2, 3, 4):
            for cap in inverse_caps:
                candidates.append(
                    {
                        "family": "inverse_power",
                        "name": f"inverse_power_p{power}_thr{threshold:.1f}_cap{cap:.0f}",
                        "threshold": float(threshold),
                        "power": int(power),
                        "cap": float(cap),
                    }
                )
        candidates.append(
            {
                "family": "linear_spring",
                "name": f"linear_spring_thr{threshold:.1f}",
                "threshold": float(threshold),
            }
        )
        candidates.append(
            {
                "family": "quadratic_spring",
                "name": f"quadratic_spring_thr{threshold:.1f}",
                "threshold": float(threshold),
            }
        )
        for decay in (0.5, 1.0, 2.0):
            candidates.append(
                {
                    "family": "exponential",
                    "name": f"exp_decay_{decay:.1f}_thr{threshold:.1f}",
                    "threshold": float(threshold),
                    "decay": float(decay),
                }
            )

    return candidates


def repulsion_magnitude(min_clear, candidate):
    safe_clear = np.maximum(np.asarray(min_clear, dtype=np.float64), 0.01)
    threshold = float(candidate["threshold"])
    family = candidate["family"]
    magnitude = np.zeros_like(safe_clear, dtype=np.float64)
    active = safe_clear < threshold

    if family == "inverse_power":
        power = int(candidate["power"])
        cap = float(candidate["cap"])
        magnitude[active] = np.minimum(cap, 1.0 / np.power(safe_clear[active], power))
    elif family == "linear_spring":
        magnitude = np.maximum(0.0, threshold - safe_clear)
    elif family == "quadratic_spring":
        magnitude = np.square(np.maximum(0.0, threshold - safe_clear))
    elif family == "exponential":
        decay = float(candidate["decay"])
        magnitude[active] = np.exp(-decay * safe_clear[active])
    else:
        raise ValueError(f"Unsupported repulsion family: {family}")

    return magnitude.astype(np.float32)


def _feature_score(shap_scores, lime_items, feature_name):
    mapping = {
        "uav_x": ["uav_x"],
        "uav_y": ["uav_y"],
        "vx": ["uav_vx"],
        "vy": ["uav_vy"],
        "goal_dx": ["goal_dx"],
        "goal_dy": ["goal_dy"],
        "goal_dist": ["goal_dist"],
        "danger_gate": ["obstacle_group"],
        "repulsion_x": ["obstacle_group"],
        "repulsion_y": ["obstacle_group"],
        "repulsion_x_gate": ["obstacle_group"],
        "repulsion_y_gate": ["obstacle_group"],
    }
    obs_tokens = mapping.get(feature_name, [])
    obstacle_shap = float(
        sum(value for key, value in shap_scores.items() if key.startswith("obst_"))
    )
    obstacle_lime = float(
        sum(abs(weight) for label, weight in lime_items if "obst_" in label)
    )

    score = 0.0
    for token in obs_tokens:
        if token == "obstacle_group":
            score += obstacle_shap + obstacle_lime
        else:
            score += float(shap_scores.get(token, 0.0))
            score += float(
                sum(abs(weight) for label, weight in lime_items if token in label)
            )
    return score


def select_feature_orders(explainability_metadata=None):
    feature_orders = {
        "x": ["goal_dx", "vx", "repulsion_x", "repulsion_x_gate"],
        "y": ["goal_dy", "vy", "repulsion_y", "repulsion_y_gate"],
    }

    optional_pool = {
        "x": ["goal_dy", "vy", "uav_x", "uav_y", "goal_dist", "danger_gate"],
        "y": ["goal_dx", "vx", "uav_x", "uav_y", "goal_dist", "danger_gate"],
    }

    if explainability_metadata is None:
        feature_orders["x"].extend(["uav_x", "uav_y", "goal_dist", "goal_dy"])
        feature_orders["y"].extend(["uav_x", "uav_y", "goal_dist", "goal_dx"])
        return feature_orders

    for axis in ("x", "y"):
        shap_scores = explainability_metadata.get(f"shap_action_{axis}_mean_abs", {})
        lime_items = explainability_metadata.get(f"lime_action_{axis}", [])
        ranked = sorted(
            optional_pool[axis],
            key=lambda name: _feature_score(shap_scores, lime_items, name),
            reverse=True,
        )
        for feature_name in ranked[:OPTIONAL_FEATURES_PER_AXIS]:
            if feature_name not in feature_orders[axis]:
                feature_orders[axis].append(feature_name)

    return feature_orders


def compact_feature_orders():
    return {
        "x": ["goal_dx", "vx", "repulsion_x"],
        "y": ["goal_dy", "vy", "repulsion_y"],
    }


def build_contextual_feature_arrays(dataset, candidate):
    rep_mag = repulsion_magnitude(dataset["min_clear"], candidate)
    rep_x = dataset["away_x"] * rep_mag
    rep_y = dataset["away_y"] * rep_mag
    threshold = max(float(candidate["threshold"]), 0.1)
    danger_gate = np.clip((threshold - dataset["min_clear"]) / threshold, 0.0, 1.0).astype(np.float32)

    return {
        "uav_x": dataset["uav_x"],
        "uav_y": dataset["uav_y"],
        "vx": dataset["vx"],
        "vy": dataset["vy"],
        "goal_dx": dataset["goal_dx"],
        "goal_dy": dataset["goal_dy"],
        "goal_dist": dataset["goal_dist"],
        "danger_gate": danger_gate,
        "repulsion_x": rep_x,
        "repulsion_y": rep_y,
        "repulsion_x_gate": rep_x * danger_gate,
        "repulsion_y_gate": rep_y * danger_gate,
    }


def fit_axis_model(feature_arrays, feature_names, targets, sample_weight):
    x_raw = np.column_stack([feature_arrays[name] for name in feature_names]).astype(np.float32)
    scaler = StandardScaler().fit(x_raw)
    x_scaled = scaler.transform(x_raw)

    reg = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    reg.fit(x_scaled, targets, sample_weight=sample_weight)

    raw_coefficients = reg.coef_ / scaler.scale_
    raw_intercept = reg.intercept_ - np.sum(reg.coef_ * scaler.mean_ / scaler.scale_)
    predictions = raw_intercept + x_raw @ raw_coefficients
    return float(raw_intercept), raw_coefficients.astype(np.float32), predictions.astype(np.float32)


def fit_candidate(dataset, candidate, feature_orders, model_form, weighting_mode="near_obstacle"):
    feature_arrays = build_contextual_feature_arrays(dataset, candidate)
    targets = dataset["actions"]
    threshold = max(float(candidate["threshold"]), 0.1)
    if weighting_mode == "near_obstacle":
        danger_gate = np.clip((threshold - dataset["min_clear"]) / threshold, 0.0, 1.0)
        sample_weight = 1.0 + 4.0 * danger_gate + 2.0 * (dataset["min_clear"] < threshold).astype(np.float32)
        weighting_summary = {
            "scheme": "on_policy_near_obstacle_emphasis",
            "min_weight": float(np.min(sample_weight)),
            "max_weight": float(np.max(sample_weight)),
            "mean_weight": float(np.mean(sample_weight)),
        }
    else:
        sample_weight = np.ones(len(targets), dtype=np.float32)
        weighting_summary = {
            "scheme": "uniform",
            "min_weight": 1.0,
            "max_weight": 1.0,
            "mean_weight": 1.0,
        }

    intercept_x, coefficients_x, pred_x = fit_axis_model(
        feature_arrays,
        feature_orders["x"],
        targets[:, 0],
        sample_weight=sample_weight,
    )
    intercept_y, coefficients_y, pred_y = fit_axis_model(
        feature_arrays,
        feature_orders["y"],
        targets[:, 1],
        sample_weight=sample_weight,
    )

    predictions = np.column_stack((pred_x, pred_y))
    result = {
        "model_form": model_form,
        "candidate": candidate,
        "coefficients": {
            "x": {
                "intercept": intercept_x,
                "features": {
                    label: float(coef) for label, coef in zip(feature_orders["x"], coefficients_x)
                },
            },
            "y": {
                "intercept": intercept_y,
                "features": {
                    label: float(coef) for label, coef in zip(feature_orders["y"], coefficients_y)
                },
            },
        },
        "feature_order": {
            "x": list(feature_orders["x"]),
            "y": list(feature_orders["y"]),
        },
        "metrics": {
            "r2_full": float(r2_score(targets, predictions)),
            "r2_x": float(r2_score(targets[:, 0], pred_x)),
            "r2_y": float(r2_score(targets[:, 1], pred_y)),
            "mae": float(np.mean(np.abs(targets - predictions))),
        },
        "weighting": weighting_summary,
    }
    return result


def _run_controller_episode(env, controller_id, seed, model=None, equation=None):
    obs, _ = env.reset(seed=seed)
    positions = [obs[:2].copy()]
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        if controller_id == "policy":
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = compute_action_from_obs(obs, equation)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        positions.append(obs[:2].copy())
        done = terminated or truncated

    trajectory = np.asarray(positions, dtype=np.float32)
    return trajectory, {
        "success": bool(info.get("is_success", False)),
        "collision": bool(info.get("collision", False)),
        "goal_dist": float(info.get("goal_dist", 0.0)),
        "steps": int(steps),
        "reward": float(total_reward),
    }


def _resample_trajectory(trajectory, num_points=200):
    if len(trajectory) == 1:
        return np.repeat(trajectory, num_points, axis=0)
    sample_t = np.linspace(0.0, len(trajectory) - 1, num_points)
    base_t = np.arange(len(trajectory), dtype=float)
    x = np.interp(sample_t, base_t, trajectory[:, 0])
    y = np.interp(sample_t, base_t, trajectory[:, 1])
    return np.column_stack((x, y)).astype(np.float32)


def _trajectory_match_metrics(policy_traj, equation_traj):
    policy_resampled = _resample_trajectory(policy_traj)
    equation_resampled = _resample_trajectory(equation_traj)
    pointwise_l2 = np.linalg.norm(policy_resampled - equation_resampled, axis=1)
    endpoint_distance = float(np.linalg.norm(policy_traj[-1] - equation_traj[-1]))
    return {
        "mean_resampled_l2": float(np.mean(pointwise_l2)),
        "endpoint_distance": endpoint_distance,
    }


def rollout_validate_result(model, result, curriculum_stage, seeds):
    policy_env = build_stage_env(curriculum_stage)
    equation_env = build_stage_env(curriculum_stage)
    policy_episodes = []
    equation_episodes = []
    match_episodes = []

    try:
        for seed in seeds:
            policy_traj, policy_metrics = _run_controller_episode(
                policy_env,
                "policy",
                seed=seed,
                model=model,
            )
            equation_traj, equation_metrics = _run_controller_episode(
                equation_env,
                "equation",
                seed=seed,
                equation=result,
            )
            match = _trajectory_match_metrics(policy_traj, equation_traj)
            policy_episodes.append(policy_metrics)
            equation_episodes.append(equation_metrics)
            match_episodes.append(match)
    finally:
        policy_env.close()
        equation_env.close()

    return {
        "seeds": [int(seed) for seed in seeds],
        "policy_successes": int(sum(item["success"] for item in policy_episodes)),
        "equation_successes": int(sum(item["success"] for item in equation_episodes)),
        "equation_collisions": int(sum(item["collision"] for item in equation_episodes)),
        "mean_goal_dist": float(np.mean([item["goal_dist"] for item in equation_episodes])),
        "mean_reward": float(np.mean([item["reward"] for item in equation_episodes])),
        "mean_steps": float(np.mean([item["steps"] for item in equation_episodes])),
        "mean_trace_deviation": float(
            np.mean([item["mean_resampled_l2"] for item in match_episodes])
        ),
        "mean_endpoint_distance": float(
            np.mean([item["endpoint_distance"] for item in match_episodes])
        ),
    }


def optimize_shap_distillation(
    model,
    n_samples=30000,
    seed=7,
    candidate_specs=None,
    env=None,
    explainability_metadata=None,
):
    dataset = sample_policy_dataset(model, n_samples=n_samples, seed=seed, env=env)
    legacy_dataset = None
    if env is not None:
        legacy_dataset = sample_policy_dataset(model, n_samples=n_samples, seed=seed, env=None)
    candidates = candidate_specs or default_candidate_specs()
    contextual_orders = select_feature_orders(explainability_metadata)
    compact_orders = compact_feature_orders()

    compact_results = [
        fit_candidate(
            dataset,
            candidate,
            compact_orders,
            model_form="compact",
            weighting_mode="near_obstacle",
        )
        for candidate in candidates
    ]
    contextual_results = [
        fit_candidate(
            dataset,
            candidate,
            contextual_orders,
            model_form="contextual",
            weighting_mode="near_obstacle",
        )
        for candidate in candidates
    ]
    legacy_results = []
    if legacy_dataset is not None:
        legacy_results = [
            fit_candidate(
                legacy_dataset,
                candidate,
                compact_orders,
                model_form="legacy_nominal_compact",
                weighting_mode="uniform",
            )
            for candidate in candidates
        ]
    compact_results.sort(key=lambda item: item["metrics"]["r2_full"], reverse=True)
    contextual_results.sort(key=lambda item: item["metrics"]["r2_full"], reverse=True)
    legacy_results.sort(key=lambda item: item["metrics"]["r2_full"], reverse=True)
    scored_results = compact_results + contextual_results + legacy_results
    scored_results.sort(key=lambda item: item["metrics"]["r2_full"], reverse=True)

    compact_best = compact_results[0]
    contextual_best = contextual_results[0]
    candidate_pool = [compact_best, contextual_best]
    if legacy_results:
        candidate_pool.append(legacy_results[0])
    best_result = max(candidate_pool, key=lambda item: item["metrics"]["r2_full"])

    if env is not None and dataset["curriculum_stage"] is not None:
        validation_seeds = [seed + 100 + idx for idx in range(4)]
        compact_best["rollout_validation"] = rollout_validate_result(
            model,
            compact_best,
            curriculum_stage=int(dataset["curriculum_stage"]),
            seeds=validation_seeds,
        )
        contextual_best["rollout_validation"] = rollout_validate_result(
            model,
            contextual_best,
            curriculum_stage=int(dataset["curriculum_stage"]),
            seeds=validation_seeds,
        )
        if legacy_results:
            legacy_results[0]["rollout_validation"] = rollout_validate_result(
                model,
                legacy_results[0],
                curriculum_stage=int(dataset["curriculum_stage"]),
                seeds=validation_seeds,
            )
        best_result = max(
            candidate_pool,
            key=lambda item: (
                item["rollout_validation"]["equation_successes"],
                -item["rollout_validation"]["equation_collisions"],
                -item["rollout_validation"]["mean_trace_deviation"],
                -item["rollout_validation"]["mean_endpoint_distance"],
                item["metrics"]["r2_full"],
            ),
        )

    best_result["sampling"] = {
        "n_samples": int(n_samples),
        "seed": int(seed),
        "mode": str(dataset["sampling_mode"]),
        "curriculum_stage": dataset["curriculum_stage"],
        "episodes_collected": int(dataset.get("episodes_collected", 0)),
    }
    best_result["feature_space"] = {
        "selected_features_x": best_result["feature_order"]["x"],
        "selected_features_y": best_result["feature_order"]["y"],
        "goal_terms": ["goal_dx", "goal_dy", "goal_dist"],
        "position_terms": ["uav_x", "uav_y"],
        "damping_terms": ["vx", "vy"],
        "repulsion_terms": [
            "repulsion_x",
            "repulsion_y",
            "repulsion_x_gate",
            "repulsion_y_gate",
            "danger_gate",
        ],
    }
    best_result["leaderboard"] = [
        {
            "name": item["candidate"]["name"],
            "model_form": item["model_form"],
            "family": item["candidate"]["family"],
            "r2_full": item["metrics"]["r2_full"],
            "mae": item["metrics"]["mae"],
        }
        for item in scored_results[:10]
    ]
    if env is not None and dataset["curriculum_stage"] is not None:
        best_result["selection"] = {
            "rule": "rollout_success_then_trace_deviation_then_r2",
            "legacy_nominal_compact_best": (
                legacy_results[0].get("rollout_validation") if legacy_results else None
            ),
            "compact_best": compact_best.get("rollout_validation"),
            "contextual_best": contextual_best.get("rollout_validation"),
        }
    else:
        best_result["selection"] = {"rule": "highest_r2_full"}
    if explainability_metadata is not None:
        best_result["explainability_guidance"] = {
            "used_shap_lime_metadata": True,
            "selected_features_x": contextual_orders["x"],
            "selected_features_y": contextual_orders["y"],
        }
    return best_result


def _legacy_axis_terms(axis_key, axis_spec):
    if axis_key == "x":
        return {
            "goal_dx": float(axis_spec["goal"]),
            "vx": float(axis_spec["velocity"]),
            "repulsion_x": float(axis_spec["repulsion"]),
        }
    return {
        "goal_dy": float(axis_spec["goal"]),
        "vy": float(axis_spec["velocity"]),
        "repulsion_y": float(axis_spec["repulsion"]),
    }


def _axis_terms(axis_key, axis_spec, result):
    if "features" in axis_spec:
        ordered = result.get("feature_order", {}).get(axis_key, list(axis_spec["features"].keys()))
        return [(name, float(axis_spec["features"].get(name, 0.0))) for name in ordered]
    legacy = _legacy_axis_terms(axis_key, axis_spec)
    return list(legacy.items())


def format_signed_term(value, label):
    sign = "+" if value >= 0.0 else "-"
    return f" {sign} {abs(value):.4f} * {label}"


def equation_lines(result, threshold=0.008):
    coeff_x = result["coefficients"]["x"]
    coeff_y = result["coefficients"]["y"]

    line_x = f"Action X = {float(coeff_x['intercept']):.4f}"
    for label, value in _axis_terms("x", coeff_x, result):
        if abs(float(value)) < threshold:
            continue
        line_x += format_signed_term(float(value), label)

    line_y = f"Action Y = {float(coeff_y['intercept']):.4f}"
    for label, value in _axis_terms("y", coeff_y, result):
        if abs(float(value)) < threshold:
            continue
        line_y += format_signed_term(float(value), label)

    return line_x, line_y


def print_result(result):
    line_x, line_y = equation_lines(result)
    metrics = result["metrics"]
    candidate = result["candidate"]
    sampling = result["sampling"]

    print("\n=======================================================")
    print(" SHAP-INFORMED RAW-FEATURE DISTILLATION")
    print("=======================================================")
    print(f"Best formulation: {candidate['name']}")
    print(
        f"Sampling: mode={sampling['mode']}, stage={sampling['curriculum_stage']}, "
        f"samples={sampling['n_samples']}"
    )
    print(f"Overall R^2: {metrics['r2_full']:.4f}")
    print(f"Axis R^2: vx={metrics['r2_x']:.4f}, vy={metrics['r2_y']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(line_x)
    print(line_y)


def render_markdown(result):
    line_x, line_y = equation_lines(result)
    candidate = result["candidate"]
    metrics = result["metrics"]
    sampling = result["sampling"]

    lines = [
        "# SHAP-Informed Distillation Equation",
        "",
        f"- Samples: {sampling['n_samples']}",
        f"- Sampling seed: {sampling['seed']}",
        f"- Sampling mode: `{sampling['mode']}`",
        f"- Curriculum stage: `{sampling['curriculum_stage']}`",
        f"- Episodes collected: `{sampling['episodes_collected']}`",
        f"- Best repulsion formulation: `{candidate['name']}`",
        f"- Overall R^2: `{metrics['r2_full']:.4f}`",
        f"- Action MAE: `{metrics['mae']:.4f}`",
        "",
        "```text",
        line_x,
        line_y,
        "```",
        "",
        "Selected contextual features:",
        f"- Action X: `{', '.join(result['feature_space']['selected_features_x'])}`",
        f"- Action Y: `{', '.join(result['feature_space']['selected_features_y'])}`",
        "",
        "Top candidates:",
    ]

    for item in result["leaderboard"][:5]:
        lines.append(
            f"- `{item['name']}` -> R^2 `{item['r2_full']:.4f}`, MAE `{item['mae']:.4f}`"
        )

    return "\n".join(lines) + "\n"


def save_result(result, json_path):
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def load_result(json_path):
    json_path = Path(json_path)
    return json.loads(json_path.read_text(encoding="utf-8"))


def save_markdown(result, markdown_path):
    markdown_path = Path(markdown_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(result), encoding="utf-8")


def compute_repulsion_vector(uav_xy, obstacle_positions, obstacle_radii, candidate):
    deltas = np.asarray(uav_xy, dtype=np.float32)[None, :] - np.asarray(
        obstacle_positions, dtype=np.float32
    )
    dists = np.linalg.norm(deltas, axis=1)
    clearances = dists - (UAV_R + np.asarray(obstacle_radii, dtype=np.float32))
    nearest_idx = int(np.argmin(clearances))
    safe_dist = float(dists[nearest_idx] + EPS)
    away_vec = deltas[nearest_idx] / safe_dist
    rep_mag = float(repulsion_magnitude(np.array([clearances[nearest_idx]]), candidate)[0])
    repulsion = away_vec * rep_mag
    return repulsion.astype(np.float32), float(clearances[nearest_idx])


def _contextual_feature_dict(obs, candidate):
    obs = np.asarray(obs, dtype=np.float32)
    num_obstacles = int((len(obs) - 7) // 4)
    obstacle_positions, obstacle_radii = extract_obstacle_arrays(obs, num_obstacles)
    repulsion, min_clear = compute_repulsion_vector(
        obs[:2],
        obstacle_positions,
        obstacle_radii,
        candidate,
    )
    threshold = max(float(candidate["threshold"]), 0.1)
    danger_gate = float(np.clip((threshold - min_clear) / threshold, 0.0, 1.0))
    return {
        "uav_x": float(obs[0]),
        "uav_y": float(obs[1]),
        "vx": float(obs[2]),
        "vy": float(obs[3]),
        "goal_dx": float(obs[4]),
        "goal_dy": float(obs[5]),
        "goal_dist": float(obs[6]),
        "danger_gate": danger_gate,
        "repulsion_x": float(repulsion[0]),
        "repulsion_y": float(repulsion[1]),
        "repulsion_x_gate": float(repulsion[0]) * danger_gate,
        "repulsion_y_gate": float(repulsion[1]) * danger_gate,
    }


def compute_action_from_result(feature_values, result):
    action = np.zeros(2, dtype=np.float32)
    for idx, axis_key in enumerate(("x", "y")):
        axis_spec = result["coefficients"][axis_key]
        value = float(axis_spec["intercept"])
        if "features" in axis_spec:
            for label, coef in axis_spec["features"].items():
                value += float(coef) * float(feature_values.get(label, 0.0))
        else:
            for label, coef in _legacy_axis_terms(axis_key, axis_spec).items():
                value += float(coef) * float(feature_values.get(label, 0.0))
        action[idx] = value

    norm = float(np.linalg.norm(action))
    if norm > 1.0:
        action /= norm
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def compute_action_from_obs(obs, result):
    feature_values = _contextual_feature_dict(obs, result["candidate"])
    return compute_action_from_result(feature_values, result)


def build_stage_env(curriculum_stage):
    env = UAV2DAvoidSimple1(render_mode=None, curriculum_stage=curriculum_stage)
    env.set_curriculum_stage(curriculum_stage)
    return env
