import argparse
from pathlib import Path

from gym_pybullet_drones.envs.constrained_environment import UAV2DAvoidSimple1

from policy_analysis_suite import run_full_analysis
from sb3_model_utils import load_sb3_model_for_inference


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SHAP/LIME and interpretable policy distillation for a saved SB3 model."
    )
    parser.add_argument("--algo", required=True, choices=["ppo", "td3", "ddpg", "sac", "a2c"])
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vecnormalize-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--curriculum-stage", type=int, default=3)
    parser.add_argument("--background-samples", type=int, default=500)
    parser.add_argument("--shap-clusters", type=int, default=25)
    parser.add_argument("--shap-explain-samples", type=int, default=20)
    parser.add_argument("--lime-num-features", type=int, default=10)
    parser.add_argument("--dataset-samples", type=int, default=25000)
    parser.add_argument("--dataset-seed", type=int, default=7)
    parser.add_argument("--shap-equation-samples", type=int, default=30000)
    parser.add_argument("--shap-equation-seed", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = UAV2DAvoidSimple1(render_mode=None, curriculum_stage=args.curriculum_stage)
    env.set_curriculum_stage(args.curriculum_stage)
    try:
        model = load_sb3_model_for_inference(
            args.algo,
            args.model_path,
            vecnormalize_path=args.vecnormalize_path,
        )
        summary = run_full_analysis(
            env=env,
            model=model,
            out_dir=args.out_dir,
            background_samples=args.background_samples,
            shap_clusters=args.shap_clusters,
            shap_explain_samples=args.shap_explain_samples,
            lime_num_features=args.lime_num_features,
            dataset_samples=args.dataset_samples,
            dataset_seed=args.dataset_seed,
            shap_equation_samples=args.shap_equation_samples,
            shap_equation_seed=args.shap_equation_seed,
        )
    finally:
        env.close()

    print(f"Saved policy analysis to {args.out_dir.resolve()}")
    print(f"Explainability outputs: {summary['explain_dir']}")
    print(f"Distillation outputs: {summary['distill_dir']}")
    print(f"SHAP equation outputs: {summary['shap_equation_dir']}")


if __name__ == "__main__":
    main()
