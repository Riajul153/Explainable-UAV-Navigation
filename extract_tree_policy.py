"""
extract_tree_policy.py
Clones the SAC neural network into a robust, deterministic Decision Tree of IF/THEN rules.
It then saves this Decision Tree to be used as a standalone controller.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor
from stable_baselines3 import SAC

import sys
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gym_pybullet_drones.envs.constrained_environment import (
    STATIC_OBSTACLE_LAYOUT, SIZE, UAV_R, GOAL_POS, SHAPE_CODES, _clearance_radius
)

def nominal_obstacle_data():
    positions = np.array([s["position"] for s in STATIC_OBSTACLE_LAYOUT], dtype=np.float32)
    collision_radii = np.array([_clearance_radius(s) for s in STATIC_OBSTACLE_LAYOUT], dtype=np.float32)
    shape_codes = np.array([SHAPE_CODES[s["shape"]] for s in STATIC_OBSTACLE_LAYOUT], dtype=np.float32)
    return positions, collision_radii, shape_codes

def build_obs(uav_xy, uav_v, obst_pos, obst_coll_r, obst_shape):
    """Builds the exact 39D observation the neural network expects."""
    goal_vec = GOAL_POS - uav_xy
    goal_dist = float(np.linalg.norm(goal_vec) + 1e-9)
    obs = np.concatenate([
        uav_xy, uav_v, goal_vec, [goal_dist],
        obst_pos[:, 0], obst_pos[:, 1], obst_coll_r, obst_shape,
    ]).astype(np.float32)
    return obs

def generate_dataset(model, n_samples=100000):
    print(f"Sampling {n_samples} states to extract high-accuracy behavioral rules...")
    obst_pos, obst_coll_r, obst_shape = nominal_obstacle_data()
    
    X_obs = []
    y_actions = []

    for _ in range(n_samples):
        uav_xy = np.random.uniform(-SIZE+1, SIZE-1, size=2)
        uav_v = np.random.uniform(-1.5, 1.5, size=2)
        
        obs = build_obs(uav_xy, uav_v, obst_pos, obst_coll_r, obst_shape)
        action, _ = model.predict(obs, deterministic=True)
        
        X_obs.append(obs)
        y_actions.append(action)
        
    return np.array(X_obs), np.array(y_actions)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "gym_pybullet_drones/examples/runs/static_nav/"
        "sac_modified_static_nav_sac_30m_curriculum_v5_seed0_20260313-003023/"
        "best_model/best_model.zip"
    ))
    parser.add_argument("--out", type=Path, default=Path("distilled_tree.pkl"))
    args = parser.parse_args()

    model_path = args.model.resolve()
    print(f"Loading SAC model from: {model_path}")
    model = SAC.load(str(model_path))
    
    X, y = generate_dataset(model, n_samples=100000)
    
    print("Fitting deep Decision Tree Regressor to extract precise rules...")
    # Depth 15 allows enough branches to capture fine obstacle avoidance
    tree = DecisionTreeRegressor(max_depth=15, min_samples_leaf=5)
    tree.fit(X, y)
    
    r2 = tree.score(X, y)
    print(f"Distillation Complete! Tree R^2 Score vs Neural Network: {r2:.4f}")
    if r2 > 0.90:
        print(" -> Excellent fit. The tree accurately mimics the policy rules.")
        
    out_path = args.out.resolve()
    joblib.dump(tree, str(out_path))
    print(f"Saved distilled rule-based controller to: {out_path}")

if __name__ == "__main__":
    main()
