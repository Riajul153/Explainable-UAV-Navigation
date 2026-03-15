import argparse
import json
import sys
import time
from pathlib import Path

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gym_pybullet_drones.envs.constrained_environment import UAV2DAvoidSimple1


MODEL_CLASSES = {
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
    "td3": TD3,
    "ddpg": DDPG,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate or render a trained policy on the modified static navigation environment."
    )
    parser.add_argument(
        "--algo",
        required=True,
        choices=sorted(MODEL_CLASSES.keys()),
        help="Algorithm class used to train the checkpoint.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory produced by train_modified_static_nav.py.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Explicit path to a model zip. If set, overrides --run-dir/--which.",
    )
    parser.add_argument(
        "--which",
        choices=["best", "latest", "final"],
        default="best",
        help="Which artifact to load from --run-dir when --model is not provided.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional per-episode step cap. Defaults to the environment time limit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed used for environment resets.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Extra delay in seconds between environment steps.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic actions.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run evaluation without opening the PyBullet GUI.",
    )
    parser.add_argument(
        "--curriculum-stage",
        type=int,
        default=4,
        help="Curriculum stage to evaluate/render. Default: 4 for the stage-4 single-shot setting.",
    )
    parser.add_argument(
        "--vecnormalize-path",
        type=Path,
        default=None,
        help="Optional VecNormalize stats path. When omitted, the script tries to infer it from --run-dir/--which.",
    )
    return parser.parse_args()


def resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model is not None:
        model_path = args.model.expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return model_path

    if args.run_dir is None:
        raise ValueError("Provide either --model or --run-dir.")

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if args.which == "best":
        model_path = run_dir / "best_model" / "best_model.zip"
    elif args.which == "final":
        model_path = run_dir / f"final_{args.algo}_model.zip"
    else:
        checkpoint_dir = run_dir / "checkpoints"
        checkpoints = sorted(checkpoint_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found under: {checkpoint_dir}")
        model_path = checkpoints[-1]

    if not model_path.exists():
        raise FileNotFoundError(f"Resolved model file not found: {model_path}")
    return model_path


def checkpoint_vecnormalize_path(model_path: Path) -> Path:
    if model_path.name.startswith("final_") and model_path.name.endswith("_model.zip"):
        return model_path.with_name("final_vecnormalize.pkl")
    if model_path.name == "best_model.zip":
        return model_path.with_name("vecnormalize.pkl")
    stem = model_path.stem
    if stem.endswith("_steps"):
        prefix, step, _ = stem.rsplit("_", 2)
        return model_path.with_name(f"{prefix}_vecnormalize_{step}_steps.pkl")
    return model_path.with_name("vecnormalize.pkl")


def resolve_vecnormalize_path(args: argparse.Namespace, model_path: Path) -> Path | None:
    if args.vecnormalize_path is not None:
        path = args.vecnormalize_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"VecNormalize stats not found: {path}")
        return path

    run_dir = args.run_dir.expanduser().resolve() if args.run_dir is not None else None
    candidates = []
    if run_dir is not None:
        if args.which == "best":
            candidates.append(run_dir / "best_model" / "vecnormalize.pkl")
        elif args.which == "final":
            candidates.append(run_dir / "final_vecnormalize.pkl")
        else:
            candidates.append(checkpoint_vecnormalize_path(model_path))
    else:
        candidates.append(checkpoint_vecnormalize_path(model_path))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def make_env(curriculum_stage: int, render_mode):
    def _init():
        env = UAV2DAvoidSimple1(render_mode=render_mode, curriculum_stage=curriculum_stage)
        env = Monitor(env)
        env.unwrapped.set_curriculum_stage(curriculum_stage)
        return env

    return _init


def maybe_print_eval_metadata(run_dir: Path, which: str) -> None:
    if which == "best":
        metrics_path = run_dir / "best_model" / "best_metrics.json"
    else:
        metrics_path = run_dir / "latest_eval.json"

    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        print(f"Loaded metrics from {metrics_path}:")
        print(json.dumps(metrics, indent=2))


def maybe_print_run_config(run_dir: Path) -> None:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return

    if "curriculum_stage" in config:
        print(
            f"Run config: curriculum_stage={config['curriculum_stage']}, "
            f"max_curriculum_stage={config.get('max_curriculum_stage', config['curriculum_stage'])}"
        )


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args)
    vecnormalize_path = resolve_vecnormalize_path(args, model_path)
    model_class = MODEL_CLASSES[args.algo]

    if args.run_dir is not None and args.model is None:
        run_dir = args.run_dir.expanduser().resolve()
        maybe_print_eval_metadata(run_dir, args.which)
        maybe_print_run_config(run_dir)

    render_mode = None if args.no_render else "human"
    env = DummyVecEnv([make_env(args.curriculum_stage, render_mode)])
    if vecnormalize_path is not None:
        env = VecNormalize.load(str(vecnormalize_path), env)
        env.training = False
        env.norm_reward = False
        print(f"Using VecNormalize stats from {vecnormalize_path}")
    model = model_class.load(str(model_path), env=env)

    if tuple(model.observation_space.shape) != tuple(env.observation_space.shape):
        env.close()
        raise ValueError(
            "Checkpoint observation space does not match the current environment. "
            "This usually means the environment changed after training."
        )

    try:
        print(
            f"Rendering/evaluating {args.algo} from {model_path} "
            f"at curriculum stage {args.curriculum_stage}"
        )
        for episode in range(args.episodes):
            env.seed(args.seed + episode)
            obs = env.reset()
            episode_reward = 0.0
            step_limit = args.max_steps or env.get_attr("time_limit_steps")[0]

            for step in range(step_limit):
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, done, info = env.step(action)
                episode_reward += float(reward[0])

                if args.sleep > 0.0:
                    time.sleep(args.sleep)

                if bool(done[0]):
                    print(
                        f"Episode {episode + 1}: reward={episode_reward:.3f}, "
                        f"steps={step + 1}, info={info[0]}"
                    )
                    break
            else:
                print(
                    f"Episode {episode + 1}: reward={episode_reward:.3f}, "
                    f"steps={step_limit}, info={{'time_limit_hit': True}}"
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
