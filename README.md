# Robust UAV Navigation Code

Minimal code-only repository for the noisy UAV navigation setup built on top of `gym-pybullet-drones`.

This repo contains only the scripts needed to:

- train the Kalman-filtered navigation policy
- train the no-Kalman ablation
- run TD3/DDPG/PPO baseline training with Optuna
- evaluate saved models
- run SHAP/LIME/decision-tree based policy analysis and equation extraction

It does **not** include:

- paper source or PDF
- generated plots and reports
- training runs, checkpoints, or TensorBoard logs

## What The Environment Is

The simulator is PyBullet-based and uses a 3D world while constraining the UAV motion to the horizontal 2D plane.

The "full noisy setting" used in the scripts includes:

- noisy position and velocity observations
- Kalman-filtered state estimates
- aerodynamic drag
- wind gust disturbances
- slowly drifting obstacles

Some scripts still select this configuration using the internal flag `--curriculum-stage 4`. In this repository, that flag simply means the full noisy Kalman-filtered setting above. It is not presented as a curriculum-learning contribution.

## Included Files

Core environment files:

- `gym_pybullet_drones/envs/constrained_environment.py`
- `gym_pybullet_drones/envs/constrained_environment_no_kf.py`

Training and evaluation:

- `gym_pybullet_drones/examples/train_modified_static_nav.py`
- `gym_pybullet_drones/examples/train_no_kf_static_nav.py`
- `gym_pybullet_drones/examples/train_kf_baselines_optuna.py`
- `gym_pybullet_drones/examples/evaluate_modified_static_nav.py`

Policy analysis and distillation:

- `policy_analysis_suite.py`
- `run_policy_analysis_suite.py`
- `shap_distillation.py`
- `extract_shap_equation.py`
- `compare_shap_distilled_trajectories.py`
- `optimize_shap_equation.py`
- `sb3_model_utils.py`

## Setup

Use Python 3.10.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you need CUDA, install the appropriate PyTorch build first, then install the remaining requirements.

## Example Commands

Train SAC in the full noisy Kalman-filtered setting:

```bash
python gym_pybullet_drones/examples/train_modified_static_nav.py ^
  --algos sac ^
  --curriculum-stage 4 ^
  --max-curriculum-stage 4
```

Train the no-Kalman ablation:

```bash
python gym_pybullet_drones/examples/train_no_kf_static_nav.py ^
  --algos sac
```

Run sequential TD3, DDPG, and PPO baseline tuning/training:

```bash
python gym_pybullet_drones/examples/train_kf_baselines_optuna.py ^
  --algos td3 ddpg ppo ^
  --curriculum-stage 4 ^
  --max-curriculum-stage 4
```

Evaluate a saved model:

```bash
python gym_pybullet_drones/examples/evaluate_modified_static_nav.py ^
  --algo sac ^
  --run-dir <run_dir> ^
  --which best ^
  --curriculum-stage 4
```

Run explainability and equation extraction:

```bash
python run_policy_analysis_suite.py ^
  --algo sac ^
  --model-path <model_zip> ^
  --out-dir outputs\sac_analysis ^
  --curriculum-stage 4
```

## Suggested GitHub Scope

Keep this repository code-only.

Store these separately if needed:

- checkpoints
- TensorBoard logs
- generated figures
- paper files
- markdown reports
