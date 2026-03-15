import argparse
from pathlib import Path

from shap_distillation import (
    DEFAULT_EQUATION_JSON,
    DEFAULT_EQUATION_MD,
    DEFAULT_MODEL_PATH,
    build_stage_env,
    optimize_shap_distillation,
    print_result,
    save_markdown,
    save_result,
)
from sb3_model_utils import load_sb3_model_for_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="sac", choices=["ppo", "td3", "ddpg", "sac", "a2c"])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vecnormalize-path", type=Path, default=None)
    parser.add_argument("--curriculum-stage", type=int, default=4)
    parser.add_argument("--samples", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_EQUATION_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_EQUATION_MD)
    args = parser.parse_args()

    model = load_sb3_model_for_inference(
        args.algo,
        args.model,
        vecnormalize_path=args.vecnormalize_path,
    )
    env = build_stage_env(args.curriculum_stage)
    try:
        result = optimize_shap_distillation(
            model,
            n_samples=args.samples,
            seed=args.seed,
            env=env,
        )
    finally:
        env.close()
    result["model"] = {
        "algo": args.algo,
        "model_path": str(args.model.resolve()),
        "vecnormalize_path": (
            str(args.vecnormalize_path.resolve()) if args.vecnormalize_path is not None else None
        ),
        "curriculum_stage": int(args.curriculum_stage),
    }
    save_result(result, args.out_json)
    save_markdown(result, args.out_md)

    print_result(result)
    print("\nTop 5 candidate leaderboard:")
    for item in result["leaderboard"][:5]:
        print(
            f"  {item['name']}: R^2={item['r2_full']:.4f}, MAE={item['mae']:.4f}"
        )
    print(f"\nSaved optimization summary to {args.out_json.resolve()}")
    print(f"Saved optimization markdown to {args.out_md.resolve()}")


if __name__ == "__main__":
    main()
