import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gym_pybullet_drones.envs.constrained_environment import GOAL_POS, SIZE, UAV_R
from shap_distillation import (
    build_obs,
    nominal_obstacle_data,
    optimize_shap_distillation,
    save_markdown as save_shap_markdown,
    save_result as save_shap_result,
)


def get_feature_names(num_obstacles=8):
    names = [
        "uav_x",
        "uav_y",
        "uav_vx",
        "uav_vy",
        "goal_dx",
        "goal_dy",
        "goal_dist",
    ]
    for i in range(num_obstacles):
        names.append(f"obst_{i}_x")
    for i in range(num_obstacles):
        names.append(f"obst_{i}_y")
    for i in range(num_obstacles):
        names.append(f"obst_{i}_r")
    for i in range(num_obstacles):
        names.append(f"obst_{i}_shape")
    return names


def collect_background_data(env, model, num_samples=500):
    data = []
    actions = []
    obs, _ = env.reset()
    while len(data) < num_samples:
        action, _ = model.predict(obs, deterministic=True)
        data.append(np.asarray(obs, dtype=np.float32))
        actions.append(np.asarray(action, dtype=np.float32))
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    return np.asarray(data, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def run_shap_lime_analysis(
    env,
    model,
    out_dir,
    background_samples=500,
    shap_clusters=25,
    shap_explain_samples=20,
    lime_num_features=10,
    lime_instance_idx=10,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_names = get_feature_names(env.unwrapped.num_obstacles)

    x_bg, y_bg = collect_background_data(env, model, num_samples=background_samples)

    def predict_fn(x):
        actions, _ = model.predict(x, deterministic=True)
        return actions

    explainer_shap = shap.KernelExplainer(predict_fn, shap.kmeans(x_bg, shap_clusters))
    x_explain = x_bg[:shap_explain_samples]
    shap_values = explainer_shap.shap_values(x_explain)

    if isinstance(shap_values, list):
        sv_x, sv_y = shap_values[0], shap_values[1]
    else:
        shape = np.shape(shap_values)
        if len(shape) == 3 and shape[2] == 2:
            sv_x = shap_values[:, :, 0]
            sv_y = shap_values[:, :, 1]
        elif len(shape) == 3 and shape[0] == 2:
            sv_x = shap_values[0]
            sv_y = shap_values[1]
        else:
            sv_x = shap_values
            sv_y = shap_values

    mean_abs_shap_x = np.mean(np.abs(sv_x), axis=0)
    mean_abs_shap_y = np.mean(np.abs(sv_y), axis=0)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv_x, x_explain, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance (Action: X)")
    plt.savefig(out_dir / "shap_summary_action_x.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv_y, x_explain, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance (Action: Y)")
    plt.savefig(out_dir / "shap_summary_action_y.png", bbox_inches="tight")
    plt.close()

    explainer_lime = LimeTabularExplainer(
        x_bg,
        feature_names=feature_names,
        mode="regression",
        verbose=False,
        random_state=42,
    )

    instance_idx = min(lime_instance_idx, len(x_bg) - 1)
    instance = x_bg[instance_idx]

    def predict_x(x):
        acts, _ = model.predict(x, deterministic=True)
        return acts[:, 0]

    def predict_y(x):
        acts, _ = model.predict(x, deterministic=True)
        return acts[:, 1]

    exp_x = explainer_lime.explain_instance(instance, predict_x, num_features=lime_num_features)
    fig = exp_x.as_pyplot_figure()
    plt.title("LIME Explanation for Action X")
    plt.tight_layout()
    plt.savefig(out_dir / "lime_explain_action_x.png")
    plt.close()

    exp_y = explainer_lime.explain_instance(instance, predict_y, num_features=lime_num_features)
    fig = exp_y.as_pyplot_figure()
    plt.title("LIME Explanation for Action Y")
    plt.tight_layout()
    plt.savefig(out_dir / "lime_explain_action_y.png")
    plt.close()

    results = {
        "background_samples": int(background_samples),
        "shap_explain_samples": int(shap_explain_samples),
        "feature_names": feature_names,
        "shap_action_x_mean_abs": {
            name: float(value) for name, value in zip(feature_names, mean_abs_shap_x)
        },
        "shap_action_y_mean_abs": {
            name: float(value) for name, value in zip(feature_names, mean_abs_shap_y)
        },
        "lime_instance_index": int(instance_idx),
        "lime_action_x": exp_x.as_list(),
        "lime_action_y": exp_y.as_list(),
    }
    (out_dir / "explainability_metadata.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


def generate_distillation_dataset(model, n_samples=25000, seed=7):
    rng = np.random.default_rng(seed)
    obst_pos, obst_coll_r, obst_shape = nominal_obstacle_data()

    features = []
    actions = []
    labels = [
        "goal_dx",
        "goal_dy",
        "vx",
        "vy",
        "repulsion_x",
        "repulsion_y",
    ]

    for _ in range(n_samples):
        uav_xy = rng.uniform(-SIZE + 1.0, SIZE - 1.0, size=2).astype(np.float32)
        uav_v = rng.uniform(-1.5, 1.5, size=2).astype(np.float32)

        obs = build_obs(uav_xy, uav_v, obst_pos, obst_coll_r, obst_shape)
        action, _ = model.predict(obs, deterministic=True)

        goal_vec = GOAL_POS - uav_xy
        deltas = uav_xy[None, :] - obst_pos
        dists = np.linalg.norm(deltas, axis=1)
        clearances = dists - (UAV_R + obst_coll_r)

        nearest_idx = int(np.argmin(clearances))
        min_clear = max(0.01, float(clearances[nearest_idx]))
        away_vec = deltas[nearest_idx] / (float(dists[nearest_idx]) + 1e-9)
        repulsion = away_vec * (1.0 / (min_clear**2))

        features.append(
            [
                float(goal_vec[0]),
                float(goal_vec[1]),
                float(uav_v[0]),
                float(uav_v[1]),
                float(repulsion[0]),
                float(repulsion[1]),
            ]
        )
        actions.append(np.asarray(action, dtype=np.float32))

    return np.asarray(features, dtype=np.float32), np.asarray(actions, dtype=np.float32), labels


def sparse_terms(intercept, coefs, labels, threshold=0.01):
    equation = [f"{intercept:.4f}"]
    terms = []
    for coef, label in zip(coefs, labels):
        if abs(float(coef)) < threshold:
            continue
        sign = "+" if coef >= 0 else "-"
        equation.append(f"{sign} {abs(float(coef)):.4f} * {label}")
        terms.append({"label": label, "coef": float(coef)})
    return equation, terms


def run_distillation_analysis(model, out_dir, dataset_samples=25000, dataset_seed=7):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y, labels = generate_distillation_dataset(
        model,
        n_samples=dataset_samples,
        seed=dataset_seed,
    )
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)

    summary = {
        "dataset_samples": int(dataset_samples),
        "dataset_seed": int(dataset_seed),
        "features": labels,
        "sparse_linear": {},
        "decision_tree": {},
    }

    markdown_lines = [
        "# Policy Distillation Summary",
        "",
        f"- Samples: {dataset_samples}",
        f"- Seed: {dataset_seed}",
        "",
        "## Sparse Linear Equations",
        "",
    ]

    tree_lines = []
    for dim, action_name in enumerate(["action_x", "action_y"]):
        lasso = LassoCV(cv=5, fit_intercept=True, random_state=dataset_seed)
        lasso.fit(x_scaled, y[:, dim])

        raw_coefficients = lasso.coef_ / scaler.scale_
        raw_intercept = lasso.intercept_ - np.sum(lasso.coef_ * scaler.mean_ / scaler.scale_)
        eq_lines, terms = sparse_terms(raw_intercept, raw_coefficients, labels)
        equation_text = f"{action_name} = " + " ".join(eq_lines)
        r2 = float(lasso.score(x_scaled, y[:, dim]))
        summary["sparse_linear"][action_name] = {
            "r2": r2,
            "intercept": float(raw_intercept),
            "coefficients": {label: float(coef) for label, coef in zip(labels, raw_coefficients)},
            "scaled_intercept": float(lasso.intercept_),
            "scaled_coefficients": {label: float(coef) for label, coef in zip(labels, lasso.coef_)},
            "active_terms": terms,
            "equation": equation_text,
        }
        markdown_lines.extend(
            [
                f"### {action_name}",
                "",
                f"- R^2: `{r2:.4f}`",
                "",
                "```text",
                equation_text,
                "```",
                "",
            ]
        )

        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=500, random_state=dataset_seed)
        tree.fit(x, y[:, dim])
        tree_r2 = float(tree.score(x, y[:, dim]))
        rules = export_text(tree, feature_names=labels, spacing=2, decimals=3)
        summary["decision_tree"][action_name] = {"r2": tree_r2, "rules": rules}
        tree_lines.extend(
            [
                f"## Decision Tree: {action_name}",
                "",
                f"- R^2: `{tree_r2:.4f}`",
                "",
                "```text",
                rules,
                "```",
                "",
            ]
        )

    (out_dir / "distillation_summary.md").write_text(
        "\n".join(markdown_lines + tree_lines) + "\n",
        encoding="utf-8",
    )
    (out_dir / "distillation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def run_shap_equation_analysis(
    model,
    env,
    explainability,
    out_dir,
    samples=30000,
    seed=7,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = optimize_shap_distillation(
        model,
        n_samples=samples,
        seed=seed,
        env=env,
        explainability_metadata=explainability,
    )
    json_path = out_dir / "shap_distillation_equation.json"
    markdown_path = out_dir / "shap_distillation_equation.md"
    save_shap_result(result, json_path)
    save_shap_markdown(result, markdown_path)

    return {
        "samples": int(samples),
        "seed": int(seed),
        "json_path": str(json_path.resolve()),
        "markdown_path": str(markdown_path.resolve()),
        "metrics": result["metrics"],
        "candidate": result["candidate"],
        "equations": {
            "action_x": result["coefficients"]["x"],
            "action_y": result["coefficients"]["y"],
        },
    }


def run_full_analysis(
    env,
    model,
    out_dir,
    background_samples=500,
    shap_clusters=25,
    shap_explain_samples=20,
    lime_num_features=10,
    dataset_samples=25000,
    dataset_seed=7,
    shap_equation_samples=30000,
    shap_equation_seed=7,
):
    out_dir = Path(out_dir)
    explain_dir = out_dir / "explain"
    distill_dir = out_dir / "distill"
    shap_equation_dir = out_dir / "shap_equation"

    explainability = run_shap_lime_analysis(
        env=env,
        model=model,
        out_dir=explain_dir,
        background_samples=background_samples,
        shap_clusters=shap_clusters,
        shap_explain_samples=shap_explain_samples,
        lime_num_features=lime_num_features,
    )
    distillation = run_distillation_analysis(
        model=model,
        out_dir=distill_dir,
        dataset_samples=dataset_samples,
        dataset_seed=dataset_seed,
    )
    shap_equation = run_shap_equation_analysis(
        model=model,
        env=env,
        explainability=explainability,
        out_dir=shap_equation_dir,
        samples=shap_equation_samples,
        seed=shap_equation_seed,
    )

    summary = {
        "explain_dir": str(explain_dir.resolve()),
        "distill_dir": str(distill_dir.resolve()),
        "shap_equation_dir": str(shap_equation_dir.resolve()),
        "explainability": explainability,
        "distillation": {
            "dataset_samples": distillation["dataset_samples"],
            "dataset_seed": distillation["dataset_seed"],
        },
        "shap_equation": shap_equation,
    }
    (out_dir / "analysis_manifest.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
