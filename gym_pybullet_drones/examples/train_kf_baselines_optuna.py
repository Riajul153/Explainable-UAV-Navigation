import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path

import optuna
import torch
from optuna.trial import TrialState
from stable_baselines3.common.utils import set_random_seed

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gym_pybullet_drones.envs.constrained_environment import UAV2DAvoidSimple1
from gym_pybullet_drones.examples.train_modified_static_nav import (
    best_vecnormalize_path,
    build_model,
    build_vec_env,
    choose_vec_env_backend,
    evaluate_policy_metrics,
    final_vecnormalize_path,
    train_one_algorithm,
    validate_algorithm_args,
)
from policy_analysis_suite import run_full_analysis
from sb3_model_utils import load_sb3_model_for_inference


ALGO_CHOICES = ("ppo", "td3", "ddpg")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially tune, train, resume, and analyze PPO/TD3/DDPG on the "
            "Kalman-filter navigation environment."
        )
    )
    parser.add_argument("--algos", nargs="+", default=list(ALGO_CHOICES), choices=ALGO_CHOICES)
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--tuning-timesteps", type=int, default=300_000)
    parser.add_argument("--optuna-trials", type=int, default=12)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--checkpoint-freq", type=int, default=250_000)
    parser.add_argument("--n-eval-episodes", type=int, default=8)
    parser.add_argument("--curriculum-stage", type=int, default=4)
    parser.add_argument(
        "--max-curriculum-stage",
        type=int,
        default=4,
        help="Keep equal to curriculum-stage for a fixed stage-4 single-shot setting.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--vec-env", choices=["auto", "dummy", "subproc"], default="auto")
    parser.add_argument(
        "--n-envs",
        type=int,
        default=0,
        help="0 means auto-select based on CPU count, capped for this PyBullet setup.",
    )
    parser.add_argument(
        "--suite-root",
        type=Path,
        default=Path("gym_pybullet_drones/examples/runs/kf_nav_suite"),
    )
    parser.add_argument(
        "--resume-suite-root",
        type=Path,
        default=None,
        help="Resume an existing suite root instead of creating a new timestamped one.",
    )
    parser.add_argument("--run-tag", type=str, default="kf_stage4_optuna")
    parser.add_argument("--background-samples", type=int, default=500)
    parser.add_argument("--shap-clusters", type=int, default=25)
    parser.add_argument("--shap-explain-samples", type=int, default=20)
    parser.add_argument("--lime-num-features", type=int, default=10)
    parser.add_argument("--dataset-samples", type=int, default=25000)
    parser.add_argument("--dataset-seed", type=int, default=7)
    parser.add_argument("--shap-equation-samples", type=int, default=30000)
    parser.add_argument("--shap-equation-seed", type=int, default=7)
    parser.add_argument("--skip-optuna", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--force-rerun-completed", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--curriculum-gate", type=float, default=0.90)
    parser.add_argument("--curriculum-consecutive", type=int, default=3)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument(
        "--use-vecnormalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable VecNormalize for training/eval envs.",
    )
    parser.add_argument("--vecnormalize-obs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vecnormalize-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vecnormalize-clip-obs", type=float, default=10.0)
    parser.add_argument("--vecnormalize-clip-reward", type=float, default=10.0)
    parser.add_argument(
        "--early-stop-after-perfect-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable early stopping when success regresses after already reaching perfect success.",
    )
    parser.add_argument(
        "--perfect-success-threshold",
        type=float,
        default=1.0,
        help="Success-rate threshold that activates regression-based early stopping.",
    )
    parser.add_argument(
        "--success-drop-patience",
        type=int,
        default=3,
        help="Consecutive regressed evals tolerated after perfect success has been reached.",
    )
    return parser.parse_args()


def auto_n_envs():
    cpu_count = os.cpu_count() or 4
    return max(1, min(8, cpu_count - 2))


def base_training_args(cli_args, suite_root):
    args = argparse.Namespace()
    args.algos = [*cli_args.algos]
    args.total_timesteps = cli_args.total_timesteps
    args.n_envs = auto_n_envs() if cli_args.n_envs <= 0 else cli_args.n_envs
    args.vec_env = cli_args.vec_env
    args.batch_size = 512
    args.learning_rate = 3e-4
    args.gamma = 0.99
    args.seed = cli_args.seed
    args.device = cli_args.device
    args.log_dir = suite_root / "runs"
    args.run_tag = cli_args.run_tag
    args.eval_freq = cli_args.eval_freq
    args.n_eval_episodes = cli_args.n_eval_episodes
    args.checkpoint_freq = cli_args.checkpoint_freq
    args.curriculum_stage = cli_args.curriculum_stage
    args.max_curriculum_stage = cli_args.max_curriculum_stage
    args.buffer_size = 500_000
    args.learning_starts = 10_000
    args.train_freq = 1
    args.gradient_steps = 1
    args.tau = 0.005
    args.action_noise_std = 0.10
    args.n_steps = 1024
    args.n_epochs = 10
    args.gae_lambda = 0.95
    args.clip_range = 0.2
    args.ent_coef = 0.0
    args.vf_coef = 0.5
    args.max_grad_norm = 0.5
    args.target_kl = cli_args.target_kl
    args.use_vecnormalize = cli_args.use_vecnormalize
    args.vecnormalize_obs = cli_args.vecnormalize_obs
    args.vecnormalize_reward = cli_args.vecnormalize_reward
    args.vecnormalize_clip_obs = cli_args.vecnormalize_clip_obs
    args.vecnormalize_clip_reward = cli_args.vecnormalize_clip_reward
    args.net_arch = "256,256,256"
    args.progress_bar = cli_args.progress_bar
    args.log_interval = cli_args.log_interval
    args.curriculum_gate = cli_args.curriculum_gate
    args.curriculum_consecutive = cli_args.curriculum_consecutive
    args.early_stop_after_perfect_success = cli_args.early_stop_after_perfect_success
    args.perfect_success_threshold = cli_args.perfect_success_threshold
    args.success_drop_patience = cli_args.success_drop_patience
    return args


def suggest_hyperparameters(algo, trial, args):
    trial_args = copy.deepcopy(args)

    trial_args.net_arch = trial.suggest_categorical(
        "net_arch",
        ["256,256", "256,256,256", "512,256,256"],
    )
    trial_args.learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    trial_args.gamma = trial.suggest_categorical("gamma", [0.97, 0.985, 0.99, 0.995])

    if algo == "ppo":
        trial_args.n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
        rollout_batch = trial_args.n_steps * trial_args.n_envs
        batch_candidates = [
            size for size in (256, 512, 1024, 2048) if size <= rollout_batch and rollout_batch % size == 0
        ]
        trial_args.batch_size = trial.suggest_categorical("batch_size", batch_candidates)
        trial_args.n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 15])
        trial_args.gae_lambda = trial.suggest_categorical("gae_lambda", [0.92, 0.95, 0.97, 0.99])
        trial_args.clip_range = trial.suggest_float("clip_range", 0.10, 0.30)
        trial_args.ent_coef = trial.suggest_float("ent_coef", 1e-5, 2e-2, log=True)
        trial_args.vf_coef = trial.suggest_float("vf_coef", 0.30, 0.90)
        trial_args.max_grad_norm = trial.suggest_float("max_grad_norm", 0.30, 1.00)
        trial_args.target_kl = trial.suggest_categorical("target_kl", [0.01, 0.015, 0.02, 0.03])
    else:
        trial_args.batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
        trial_args.buffer_size = trial.suggest_categorical("buffer_size", [300_000, 500_000, 800_000])
        trial_args.learning_starts = trial.suggest_categorical(
            "learning_starts", [5_000, 10_000, 20_000, 50_000]
        )
        trial_args.train_freq = trial.suggest_categorical("train_freq", [1, 2, 4])
        trial_args.gradient_steps = trial.suggest_categorical("gradient_steps", [1, 2, 4])
        trial_args.tau = trial.suggest_categorical("tau", [0.002, 0.005, 0.01])
        trial_args.action_noise_std = trial.suggest_categorical(
            "action_noise_std", [0.05, 0.10, 0.15, 0.20]
        )

    return trial_args


def tuning_score(metrics):
    return (
        1_000_000.0 * metrics["success_rate"]
        - 10_000.0 * metrics["collision_rate"]
        - 1_000.0 * metrics["timeout_rate"]
        + metrics["mean_reward"]
    )


def load_manifest(suite_root: Path) -> dict | None:
    manifest_path = suite_root / "suite_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_manifest(suite_root: Path, manifest: dict) -> None:
    (suite_root / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def initialize_manifest(suite_root: Path, cli_args, args, backend: str, existing_manifest: dict | None):
    if existing_manifest is not None:
        existing_manifest.setdefault("algorithms", {})
        existing_manifest["suite_root"] = str(suite_root.resolve())
        existing_manifest["backend"] = existing_manifest.get("backend", backend)
        existing_manifest["n_envs"] = int(existing_manifest.get("n_envs", args.n_envs))
        existing_manifest["curriculum_stage"] = int(
            existing_manifest.get("curriculum_stage", args.curriculum_stage)
        )
        existing_manifest["max_curriculum_stage"] = int(
            existing_manifest.get("max_curriculum_stage", args.max_curriculum_stage)
        )
        existing_manifest["model_selection_rule"] = "best_success_rate_then_mean_reward"
        return existing_manifest

    return {
        "created_at": time.strftime("%Y%m%d-%H%M%S"),
        "suite_root": str(suite_root.resolve()),
        "backend": backend,
        "n_envs": int(args.n_envs),
        "curriculum_stage": int(args.curriculum_stage),
        "max_curriculum_stage": int(args.max_curriculum_stage),
        "model_selection_rule": "best_success_rate_then_mean_reward",
        "algorithms": {},
    }


def infer_existing_run_dir(suite_root: Path, algo: str) -> Path | None:
    runs_root = suite_root / "runs"
    if not runs_root.exists():
        return None
    candidates = sorted(
        [path for path in runs_root.glob(f"{algo}_modified_static_nav*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def algorithm_completed(record: dict | None) -> bool:
    if not record:
        return False
    run_dir = Path(record["run_dir"]) if record.get("run_dir") else None
    model_path = Path(record["model_path"]) if record.get("model_path") else None
    return bool(run_dir and run_dir.exists() and model_path and model_path.exists())


def analysis_completed(record: dict | None) -> bool:
    if not record or not record.get("analysis_manifest"):
        return False
    return Path(record["analysis_manifest"]).exists()


def count_complete_trials(study: optuna.Study) -> int:
    return sum(1 for trial in study.trials if trial.state == TrialState.COMPLETE)


def tune_algorithm(algo, base_args, backend, algo_root, tuning_timesteps, n_trials):
    algo_root.mkdir(parents=True, exist_ok=True)
    trials_root = algo_root / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)

    storage_path = algo_root / f"{algo}_optuna.db"
    study = optuna.create_study(
        study_name=f"{algo}_kf_nav",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=base_args.seed),
        storage=f"sqlite:///{storage_path.resolve()}",
        load_if_exists=True,
    )

    completed_trials_before = count_complete_trials(study)
    remaining_trials = max(0, n_trials - completed_trials_before)

    def objective(trial):
        trial_args = suggest_hyperparameters(algo, trial, base_args)
        trial_dir = trials_root / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        train_env = build_vec_env(
            seed=trial_args.seed,
            n_envs=trial_args.n_envs,
            backend=backend,
            monitor_file=trial_dir / "train.monitor.csv",
            stage=trial_args.curriculum_stage,
            args=trial_args,
            training=True,
        )
        eval_env = build_vec_env(
            seed=trial_args.seed + 50_000,
            n_envs=1,
            backend="dummy",
            monitor_file=trial_dir / "eval.monitor.csv",
            stage=trial_args.curriculum_stage,
            args=trial_args,
            training=False,
        )

        try:
            validate_algorithm_args(algo, trial_args)
            model = build_model(algo, train_env, trial_args, trial_dir)
            model.learn(
                total_timesteps=tuning_timesteps,
                callback=None,
                log_interval=trial_args.log_interval,
                progress_bar=False,
            )
            metrics = evaluate_policy_metrics(
                model=model,
                eval_env=eval_env,
                n_eval_episodes=max(4, trial_args.n_eval_episodes // 2),
            )
            score = tuning_score(metrics)
            payload = {
                "trial": int(trial.number),
                "score": float(score),
                "metrics": metrics,
                "params": {
                    key: value
                    for key, value in vars(trial_args).items()
                    if key
                    in {
                        "learning_rate",
                        "gamma",
                        "n_steps",
                        "batch_size",
                        "n_epochs",
                        "gae_lambda",
                        "clip_range",
                        "ent_coef",
                        "vf_coef",
                        "max_grad_norm",
                        "target_kl",
                        "buffer_size",
                        "learning_starts",
                        "train_freq",
                        "gradient_steps",
                        "tau",
                        "action_noise_std",
                        "net_arch",
                    }
                },
            }
            (trial_dir / "trial_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return score
        finally:
            train_env.close()
            eval_env.close()
            if "model" in locals():
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, gc_after_trial=True)

    if count_complete_trials(study) == 0:
        raise RuntimeError(f"No completed Optuna trials are available for {algo}.")

    best_summary = {
        "algo": algo,
        "best_value": float(study.best_value),
        "best_params": study.best_trial.params,
        "requested_trials": int(n_trials),
        "completed_trials": int(count_complete_trials(study)),
        "total_trials_in_db": int(len(study.trials)),
        "remaining_trials_run": int(remaining_trials),
    }
    (algo_root / "best_params.json").write_text(json.dumps(best_summary, indent=2), encoding="utf-8")
    try:
        study.trials_dataframe().to_csv(algo_root / "trials.csv", index=False)
    except Exception:
        pass
    return best_summary


def apply_best_params(args, best_params):
    tuned = copy.deepcopy(args)
    for key, value in best_params.items():
        setattr(tuned, key, value)
    return tuned


def resolve_analysis_artifacts(run_dir, algo):
    best_model = run_dir / "best_model" / "best_model.zip"
    if best_model.exists():
        vecnormalize = best_vecnormalize_path(run_dir)
        return best_model, (vecnormalize if vecnormalize.exists() else None)
    final_model = run_dir / f"final_{algo}_model.zip"
    vecnormalize = final_vecnormalize_path(run_dir)
    return final_model, (vecnormalize if vecnormalize.exists() else None)


def ensure_parent_manifest_record(manifest: dict, algo: str) -> dict:
    record = manifest["algorithms"].setdefault(algo, {})
    record.setdefault("selection_rule", "best_success_rate_then_mean_reward")
    return record


def run_analysis_for_record(algo, record: dict, cli_args, args):
    run_dir = Path(record["run_dir"])
    model_path = Path(record["model_path"])
    vecnormalize_path = Path(record["vecnormalize_path"]) if record.get("vecnormalize_path") else None
    analysis_dir = run_dir / "policy_analysis"
    env = UAV2DAvoidSimple1(render_mode=None, curriculum_stage=args.curriculum_stage)
    env.set_curriculum_stage(args.curriculum_stage)

    try:
        model = load_sb3_model_for_inference(algo, model_path, vecnormalize_path=vecnormalize_path)
        analysis = run_full_analysis(
            env=env,
            model=model,
            out_dir=analysis_dir,
            background_samples=cli_args.background_samples,
            shap_clusters=cli_args.shap_clusters,
            shap_explain_samples=cli_args.shap_explain_samples,
            lime_num_features=cli_args.lime_num_features,
            dataset_samples=cli_args.dataset_samples,
            dataset_seed=cli_args.dataset_seed,
            shap_equation_samples=cli_args.shap_equation_samples,
            shap_equation_seed=cli_args.shap_equation_seed,
        )
    finally:
        env.close()
        if "model" in locals():
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    record["analysis_dir"] = str(analysis_dir.resolve())
    record["analysis_manifest"] = str((analysis_dir / "analysis_manifest.json").resolve())
    record["analysis"] = analysis
    record["status"] = "analyzed"


def main():
    cli_args = parse_args()

    if cli_args.resume_suite_root is not None:
        suite_root = cli_args.resume_suite_root.expanduser().resolve()
        suite_root.mkdir(parents=True, exist_ok=True)
        existing_manifest = load_manifest(suite_root)
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        suite_root = (cli_args.suite_root / f"{cli_args.run_tag}_{timestamp}").resolve()
        suite_root.mkdir(parents=True, exist_ok=True)
        existing_manifest = None

    args = base_training_args(cli_args, suite_root)
    if existing_manifest is not None:
        args.n_envs = int(existing_manifest.get("n_envs", args.n_envs))
        args.curriculum_stage = int(existing_manifest.get("curriculum_stage", args.curriculum_stage))
        args.max_curriculum_stage = int(
            existing_manifest.get("max_curriculum_stage", args.max_curriculum_stage)
        )
        backend = existing_manifest.get("backend", choose_vec_env_backend(args.vec_env, args.n_envs))
    else:
        backend = choose_vec_env_backend(args.vec_env, args.n_envs)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    manifest = initialize_manifest(suite_root, cli_args, args, backend, existing_manifest)
    save_manifest(suite_root, manifest)

    try:
        for algo in cli_args.algos:
            record = ensure_parent_manifest_record(manifest, algo)
            if record.get("run_dir") and not record.get("vecnormalize_path"):
                _, inferred_vecnormalize = resolve_analysis_artifacts(Path(record["run_dir"]), algo)
                record["vecnormalize_path"] = (
                    str(inferred_vecnormalize.resolve()) if inferred_vecnormalize is not None else None
                )

            if not cli_args.force_rerun_completed and algorithm_completed(record):
                print(f"[{algo}] already completed; keeping existing run at {record['run_dir']}")
                save_manifest(suite_root, manifest)
                continue

            algo_root = suite_root / algo
            algo_root.mkdir(parents=True, exist_ok=True)
            train_args = copy.deepcopy(args)

            if not cli_args.skip_optuna:
                best = tune_algorithm(
                    algo=algo,
                    base_args=train_args,
                    backend=backend,
                    algo_root=algo_root / "optuna",
                    tuning_timesteps=cli_args.tuning_timesteps,
                    n_trials=cli_args.optuna_trials,
                )
                train_args = apply_best_params(train_args, best["best_params"])
                record["optuna"] = best
            else:
                record.setdefault("optuna", None)

            existing_run_dir = None
            if record.get("run_dir"):
                existing_run_dir = Path(record["run_dir"])
            else:
                existing_run_dir = infer_existing_run_dir(suite_root, algo)

            run_dir = train_one_algorithm(
                algo,
                train_args,
                backend,
                run_dir=existing_run_dir,
                resume=existing_run_dir is not None and existing_run_dir.exists(),
            )
            model_path, vecnormalize_path = resolve_analysis_artifacts(run_dir, algo)

            record["run_dir"] = str(run_dir.resolve())
            record["model_path"] = str(model_path.resolve())
            record["vecnormalize_path"] = (
                str(vecnormalize_path.resolve()) if vecnormalize_path is not None else None
            )
            record["best_metrics_path"] = str((run_dir / "best_model" / "best_metrics.json").resolve())
            record["status"] = "trained"
            save_manifest(suite_root, manifest)

        if not cli_args.skip_analysis:
            for algo in cli_args.algos:
                record = ensure_parent_manifest_record(manifest, algo)
                if not algorithm_completed(record):
                    continue
                if analysis_completed(record) and not cli_args.force_rerun_completed:
                    print(f"[{algo}] analysis already completed; keeping existing outputs")
                    continue
                run_analysis_for_record(algo, record, cli_args, args)
                save_manifest(suite_root, manifest)

    except KeyboardInterrupt:
        save_manifest(suite_root, manifest)
        print("\nInterrupted. Resume this suite with:")
        print(
            f"& \"E:\\Anaconda\\envs\\gym_pybullet_drones\\python.exe\" "
            f"\"{Path(__file__).resolve()}\" --resume-suite-root \"{suite_root}\" "
            f"--algos {' '.join(cli_args.algos)}"
        )
        return

    save_manifest(suite_root, manifest)
    print(f"Suite completed. Manifest: {suite_root / 'suite_manifest.json'}")
    for algo, record in manifest["algorithms"].items():
        print(f"{algo}: {record.get('run_dir')}")


if __name__ == "__main__":
    main()
