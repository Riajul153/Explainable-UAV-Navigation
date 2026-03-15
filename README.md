# Stage-4 UAV Navigation Artifact

This repository is a curated artifact for the stage-4 UAV navigation study built on top of `gym-pybullet-drones`.
It contains the custom noisy navigation environment, the training and evaluation scripts used for the single-shot stage-4 runs, the explainability and policy-distillation pipeline, the best checkpoints for the main algorithms, and the final paper/report assets.

The focus of this artifact is the stage-4 setting:

- noisy observations
- Kalman-filtered state estimation
- aerodynamic drag
- wind gusts
- anchored obstacle drift
- single-shot training, without curriculum scheduling

## Included Contents

- Custom stage-4 environments:
  - `gym_pybullet_drones/envs/constrained_environment.py`
  - `gym_pybullet_drones/envs/constrained_environment_no_kf.py`
- Training and evaluation scripts:
  - `gym_pybullet_drones/examples/train_modified_static_nav.py`
  - `gym_pybullet_drones/examples/train_no_kf_static_nav.py`
  - `gym_pybullet_drones/examples/train_kf_baselines_optuna.py`
  - `gym_pybullet_drones/examples/evaluate_modified_static_nav.py`
- Explainability and equation-distillation pipeline:
  - `policy_analysis_suite.py`
  - `run_policy_analysis_suite.py`
  - `shap_distillation.py`
  - `extract_shap_equation.py`
  - `compare_shap_distilled_trajectories.py`
  - `optimize_shap_equation.py`
- Final analysis outputs:
  - `policy_analysis/Policy_Distillation_Report.md`
  - `paper_springer/build/main.pdf`
- Best checkpoints and per-run metrics for:
  - SAC stage-4 with Kalman filter
  - SAC stage-4 without Kalman filter
  - TD3 stage-4
  - DDPG stage-4
  - PPO stage-4

## Repository Layout

```text
.
|-- gym_pybullet_drones/
|   |-- envs/
|   |-- examples/
|   |   `-- runs/
|-- policy_analysis/
|-- paper_springer/
|-- README.md
`-- requirements.txt
```

## Included Checkpoints

The artifact ships the best saved checkpoints plus their `best_metrics.json` files:

- `gym_pybullet_drones/examples/runs/static_nav/sac_modified_static_nav_ablation_single_shot_20260313-042738/best_model/best_model.zip`
- `gym_pybullet_drones/examples/runs/static_nav/sac_modified_static_nav_single_shot_no_kf_20260313-115635/best_model/best_model.zip`
- `gym_pybullet_drones/examples/runs/kf_nav_suite/kf_baselines_20260314-164705/runs/td3_modified_static_nav_kf_baselines_20260314-173657/best_model/best_model.zip`
- `gym_pybullet_drones/examples/runs/kf_nav_suite/kf_baselines_20260314-164705/runs/ddpg_modified_static_nav_kf_stage4_optuna_20260314-204207/best_model/best_model.zip`
- `gym_pybullet_drones/examples/runs/kf_nav_suite/kf_baselines_20260314-164705/runs/ppo_modified_static_nav_kf_stage4_optuna_20260314-211046/best_model/best_model.zip`

Each included run also contains the saved `config.json` and `paper_metrics.csv` needed for reproducing the comparison plots.

## Setup

Use Python 3.10.

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you need a CUDA-enabled PyTorch build, install the correct `torch` wheel for your machine first and then install the remaining requirements.

## Quick Start

Evaluate the best stage-4 SAC checkpoint:

```bash
python gym_pybullet_drones/examples/evaluate_modified_static_nav.py ^
  --algo sac ^
  --run-dir gym_pybullet_drones/examples/runs/static_nav/sac_modified_static_nav_ablation_single_shot_20260313-042738 ^
  --which best ^
  --curriculum-stage 4 ^
  --episodes 5
```

Run the explainability and SHAP-equation pipeline on the same checkpoint:

```bash
python run_policy_analysis_suite.py ^
  --algo sac ^
  --model-path gym_pybullet_drones/examples/runs/static_nav/sac_modified_static_nav_ablation_single_shot_20260313-042738/best_model/best_model.zip ^
  --out-dir outputs/sac_stage4_analysis ^
  --curriculum-stage 4
```

Regenerate the paper PDF:

```bash
cd paper_springer
pdflatex -interaction=nonstopmode -output-directory=build main.tex
pdflatex -interaction=nonstopmode -output-directory=build main.tex
```

Launch the sequential Optuna baseline suite:

```bash
python gym_pybullet_drones/examples/train_kf_baselines_optuna.py ^
  --algos td3 ddpg ppo ^
  --curriculum-stage 4 ^
  --max-curriculum-stage 4 ^
  --total-timesteps 5000000 ^
  --tuning-timesteps 300000 ^
  --optuna-trials 12 ^
  --n-envs 8
```

## Precomputed Outputs

- Paper PDF: `paper_springer/build/main.pdf`
- Paper source: `paper_springer/main.tex`
- Policy analysis report: `policy_analysis/Policy_Distillation_Report.md`
- Clean paper comparison plots:
  - `gym_pybullet_drones/examples/runs/kf_nav_suite/kf_baselines_20260314-164705/comparison_plots/paper_first_1050k_clean_success_rate.pdf`
  - `gym_pybullet_drones/examples/runs/kf_nav_suite/kf_baselines_20260314-164705/comparison_plots/paper_first_1050k_clean_mean_reward.pdf`

## Scope

This is a research artifact, not a cleaned general-purpose simulator release.
Only the code, checkpoints, metrics, reports, and figures required for the stage-4 navigation study were intentionally kept.
The original upstream simulator contains many additional environments and examples that are not part of this artifact.

## Uploading To GitHub

This folder is prepared to be used as its own repository.
From inside the artifact root:

```bash
git init -b main
git add .
git commit -m "Initial stage-4 navigation artifact"
git remote add origin <your-github-repo-url>
git push -u origin main
```
