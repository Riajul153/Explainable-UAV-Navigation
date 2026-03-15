import argparse
from pathlib import Path

from shap_distillation import (
    DEFAULT_EQUATION_JSON,
    DEFAULT_EQUATION_MD,
    DEFAULT_MODEL_PATH,
    build_stage_env,
    load_result,
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
    parser.add_argument("--equation-json", type=Path, default=DEFAULT_EQUATION_JSON)
    parser.add_argument("--equation-md", type=Path, default=DEFAULT_EQUATION_MD)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Load the saved equation JSON instead of refitting from the neural network.",
    )
    args = parser.parse_args()

    if args.reuse_existing and args.equation_json.exists():
        result = load_result(args.equation_json)
    else:
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
        save_result(result, args.equation_json)

    save_markdown(result, args.equation_md)

    print_result(result)
    print(f"\nSaved equation JSON to {args.equation_json.resolve()}")
    print(f"Saved equation markdown to {args.equation_md.resolve()}")


if __name__ == "__main__":
    main()
