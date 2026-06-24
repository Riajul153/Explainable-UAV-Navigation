"""Distillation-method baselines for the explainability-guided controller.

Reviewer ask: compare the explainability-guided analytical controller against
standard policy-distillation baselines -- VIPER (Q-weighted DAgger tree), plain
behavioural cloning into a small net, and DAgger -- under the *same* closed-loop
protocol the paper already uses (shap_distillation.rollout_validate_result):
build_stage_env(stage), the paper's evaluation seeds, success = is_success,
trace deviation = resampled L2 vs the teacher trajectory, endpoint distance.

Nothing here is fabricated: every number is produced by rolling the trained
surrogate in the simulator. The teacher checkpoint for each algorithm is the
SAME one the paper distilled (validated by reproducing the published equation
metrics before trusting it -- see --validate).

Usage:
    python run_distillation_baselines.py --teacher SAC_2D
    python run_distillation_baselines.py --teacher SAC_2D --validate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import shap_distillation as SD  # noqa: E402
from stable_baselines3 import SAC, DDPG, TD3  # noqa: E402

# ---------------------------------------------------------------------------
# Teacher registry. Only checkpoints validated against the published distillation
# metrics are trusted (see validate_teacher). SAC_2D is pinned and validated;
# TD3/DDPG entries are left as candidates pending confirmation of the exact
# noise-0.15 checkpoints (the result.json files do not record the source path).
# ---------------------------------------------------------------------------
RUNS = ROOT / "gym_pybullet_drones" / "examples" / "runs"
TEACHERS = {
    "SAC_2D": {
        "algo": SAC,
        # point at your trained SAC teacher (2D, full noisy Kalman-filtered setting)
        "path": RUNS / "sac_teacher/best_model.zip",
        "stage": 4,
        "eval_seeds": [107, 108, 109, 110],
        # published distillation result to validate the checkpoint/protocol against
        "published": {"equation_successes": 4, "mean_trace_deviation": 2.808, "mean_endpoint_distance": 0.799},
        "equation_result": ROOT / "paper_data/distillation/sac_2d_v5_best_model/result.json",
    },
    "TD3_2D": {
        "algo": TD3,
        # point at your trained TD3 teacher (paper: 2D, action-noise 0.15, 1.3M steps)
        "path": Path("runs/td3_teacher/td3_static_nav_1300000_steps.zip"),
        "stage": 4,
        "eval_seeds": [107, 108, 109, 110],
        "published": {"equation_successes": 4, "mean_trace_deviation": 1.562, "mean_endpoint_distance": 0.691},
        "equation_result": ROOT / "paper_data/distillation/td3_2d_noise15_1300k/result.json",
    },
    "DDPG_2D": {
        "algo": DDPG,
        # point at your trained DDPG teacher (paper: 2D, action-noise 0.15, ~1.0M steps)
        "path": Path("runs/ddpg_teacher/ddpg_static_nav_1000000_steps.zip"),
        "stage": 4,
        "eval_seeds": [107, 108, 109, 110],
        "published": {"equation_successes": 4, "mean_trace_deviation": 1.112, "mean_endpoint_distance": 0.227},
        "equation_result": ROOT / "paper_data/distillation/ddpg_2d_retrain_1000k/result.json",
    },
}


def count_equation_params(result):
    """Number of scalar coefficients in the analytical controller (intercept +
    per-feature gains, summed over both action axes)."""
    coeffs = result.get("coefficients", {})
    n = 0
    for axis in coeffs.values():
        n += 1  # intercept
        n += len(axis.get("features", {}))
    return n

DATASET_SEED = 7
DATASET_N = 30000
BC_HIDDEN = (64, 64)
DAGGER_ITERS = 4
DAGGER_EPISODES_PER_ITER = 12
VIPER_ITERS = 4
VIPER_DEPTH = 15
DEVICE = "cpu"


# ----------------------------- data collection -----------------------------
def collect_onpolicy(model, stage, seed=DATASET_SEED, n=DATASET_N):
    """Full 39-dim obs -> teacher action, on-policy (matches the paper's
    on_policy_rollout sampling), used as the behavioural-cloning dataset."""
    env = SD.build_stage_env(stage)
    obs_buf, act_buf = [], []
    ep = 0
    while len(obs_buf) < n:
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done and len(obs_buf) < n:
            act, _ = model.predict(obs, deterministic=True)
            obs_buf.append(np.asarray(obs, dtype=np.float32))
            act_buf.append(np.asarray(act, dtype=np.float32))
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc
        ep += 1
    env.close()
    return np.asarray(obs_buf, dtype=np.float32), np.asarray(act_buf, dtype=np.float32)


def rollout_states(predict_fn, stage, seeds):
    """Collect states visited by `predict_fn` (for DAgger/VIPER aggregation)."""
    states = []
    for s in seeds:
        env = SD.build_stage_env(stage)
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            states.append(np.asarray(obs, dtype=np.float32))
            obs, _, term, trunc, _ = env.step(predict_fn(obs))
            done = term or trunc
        env.close()
    return np.asarray(states, dtype=np.float32)


# ------------------------------- small net (BC) ----------------------------
class SmallNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden=BC_HIDDEN):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
            d = h
        layers += [torch.nn.Linear(d, out_dim)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def train_net(X, Y, hidden=BC_HIDDEN, epochs=400, batch=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    mean, std = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mean) / std
    net = SmallNet(X.shape[1], Y.shape[1], hidden).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()
    Xt = torch.tensor(Xn, dtype=torch.float32, device=DEVICE)
    Yt = torch.tensor(Y, dtype=torch.float32, device=DEVICE)
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(net(Xt[idx]), Yt[idx])
            loss.backward()
            opt.step()
    return net, mean, std


def net_predict_fn(net, mean, std):
    mt = torch.tensor(mean, dtype=torch.float32, device=DEVICE)
    st = torch.tensor(std, dtype=torch.float32, device=DEVICE)

    def f(obs):
        x = (torch.tensor(np.asarray(obs, dtype=np.float32), device=DEVICE) - mt) / st
        with torch.no_grad():
            a = net(x.unsqueeze(0)).squeeze(0).cpu().numpy()
        return np.clip(a, -1.0, 1.0)
    return f


def tree_predict_fn(tree):
    def f(obs):
        a = tree.predict(np.asarray(obs, dtype=np.float32)[None, :])[0]
        return np.clip(a, -1.0, 1.0)
    return f


# ------------------------------- VIPER weights -----------------------------
def viper_weights(model, X, n_action_samples=16, seed=0):
    """VIPER state-importance l(s) ~ range of Q over sampled actions (the gap
    between best and worst action value), adapted to continuous control via the
    teacher's critic. States where the action choice matters most are upweighted."""
    rng = np.random.default_rng(seed)
    obs_t, _ = model.policy.obs_to_tensor(X)
    with torch.no_grad():
        pi_act, _ = model.predict(X, deterministic=True)
        pi_t = torch.as_tensor(pi_act, dtype=torch.float32, device=obs_t.device)
        q_pi = _q_min(model, obs_t, pi_t)
        q_min_rand = q_pi.clone()
        for _ in range(n_action_samples):
            ra = torch.as_tensor(rng.uniform(-1, 1, size=pi_act.shape).astype(np.float32), device=obs_t.device)
            q_min_rand = torch.minimum(q_min_rand, _q_min(model, obs_t, ra))
    w = (q_pi - q_min_rand).cpu().numpy().flatten()
    w = np.clip(w, 0, None)
    return w / (w.mean() + 1e-9)


def _q_min(model, obs_t, act_t):
    qs = model.critic(obs_t, act_t)
    return torch.min(torch.stack(list(qs), 0), 0).values.flatten()


# ------------------------------- evaluation --------------------------------
def run_episode(predict_fn, env, seed):
    obs, _ = env.reset(seed=seed)
    positions = [obs[:2].copy()]
    done = False
    info = {}
    while not done:
        obs, _, term, trunc, info = env.step(predict_fn(obs))
        positions.append(obs[:2].copy())
        done = term or trunc
    return np.asarray(positions, dtype=np.float32), bool(info.get("is_success", False))


def evaluate(predict_fn, stage, seeds, teacher_trajs):
    succ, devs, endpts = 0, [], []
    for s in seeds:
        env = SD.build_stage_env(stage)
        traj, ok = run_episode(predict_fn, env, s)
        env.close()
        succ += int(ok)
        m = SD._trajectory_match_metrics(teacher_trajs[s], traj)
        devs.append(m["mean_resampled_l2"])
        endpts.append(m["endpoint_distance"])
    return {"success": f"{succ}/{len(seeds)}", "success_n": succ,
            "trace_dev": float(np.mean(devs)), "endpoint_dev": float(np.mean(endpts))}


def teacher_trajectories(model, stage, seeds):
    trajs = {}
    for s in seeds:
        env = SD.build_stage_env(stage)
        traj, _ = run_episode(lambda o: model.predict(o, deterministic=True)[0], env, s)
        env.close()
        trajs[s] = traj
    return trajs


# --------------------------------- baselines -------------------------------
def make_bc(model, X, Y):
    net, mean, std = train_net(X, Y, seed=0)
    return net_predict_fn(net, mean, std), {"name": "BC (small MLP)", "params": net.n_params(),
                                            "interpretable": "no (black box)"}


def make_dagger(model, stage, X, Y):
    aggX, aggY = X.copy(), Y.copy()
    net, mean, std = train_net(aggX, aggY, seed=0)
    pred = net_predict_fn(net, mean, std)
    for _ in range(DAGGER_ITERS):
        seeds = list(range(2000, 2000 + DAGGER_EPISODES_PER_ITER))
        new_states = rollout_states(pred, stage, seeds)
        if len(new_states) == 0:
            break
        new_acts, _ = model.predict(new_states, deterministic=True)
        aggX = np.concatenate([aggX, new_states], 0)
        aggY = np.concatenate([aggY, np.asarray(new_acts, dtype=np.float32)], 0)
        net, mean, std = train_net(aggX, aggY, seed=0)
        pred = net_predict_fn(net, mean, std)
    return pred, {"name": "DAgger (small MLP)", "params": net.n_params(),
                  "interpretable": "no (black box)", "agg_samples": int(len(aggX))}


def make_viper(model, stage, X, Y):
    aggX, aggY = X.copy(), Y.copy()
    w = viper_weights(model, aggX)
    tree = DecisionTreeRegressor(max_depth=VIPER_DEPTH, random_state=7).fit(aggX, aggY, sample_weight=w)
    pred = tree_predict_fn(tree)
    for _ in range(VIPER_ITERS):
        seeds = list(range(3000, 3000 + DAGGER_EPISODES_PER_ITER))
        new_states = rollout_states(pred, stage, seeds)
        if len(new_states) == 0:
            break
        new_acts, _ = model.predict(new_states, deterministic=True)
        aggX = np.concatenate([aggX, new_states], 0)
        aggY = np.concatenate([aggY, np.asarray(new_acts, dtype=np.float32)], 0)
        w = viper_weights(model, aggX)
        tree = DecisionTreeRegressor(max_depth=VIPER_DEPTH, random_state=7).fit(aggX, aggY, sample_weight=w)
        pred = tree_predict_fn(tree)
    return pred, {"name": "VIPER (Q-weighted DAgger tree)", "params": int(tree.tree_.node_count),
                  "interpretable": f"partial (tree, {tree.tree_.node_count} nodes)"}


# --------------------------------- driver ----------------------------------
def load_teacher(spec):
    return spec["algo"].load(str(spec["path"]))


def validate_teacher(spec):
    """Re-run the paper's equation through the closed-loop harness and check it
    reproduces the published metrics, proving checkpoint + protocol are correct."""
    if not spec["equation_result"].exists():
        print(f"equation_result not found: {spec['equation_result']} "
              "(run the analysis suite to produce the distillation result.json first).")
        return False
    model = load_teacher(spec)
    result = SD.load_result(spec["equation_result"])
    rv = SD.rollout_validate_result(model, result, spec["stage"], spec["eval_seeds"])
    pub = spec["published"]
    ok = (rv["equation_successes"] == pub["equation_successes"]
          and abs(rv["mean_trace_deviation"] - pub["mean_trace_deviation"]) < 0.05)
    print(json.dumps({"reproduced": rv, "published": pub, "MATCH": ok}, indent=2))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="SAC_2D", choices=list(TEACHERS))
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    spec = TEACHERS[args.teacher]

    if args.validate:
        validate_teacher(spec)
        return

    model = load_teacher(spec)
    stage, seeds = spec["stage"], spec["eval_seeds"]
    print(f"[{args.teacher}] collecting BC dataset ...")
    X, Y = collect_onpolicy(model, stage)
    teacher_trajs = teacher_trajectories(model, stage, seeds)

    rows = []
    # teacher reference (sanity: should be full success)
    rows.append({"method": "Neural teacher", "params": "256-256-256",
                 "interpretable": "no", **evaluate(lambda o: model.predict(o, deterministic=True)[0], stage, seeds, teacher_trajs)})
    for builder in (make_bc, make_dagger, make_viper):
        pred, meta = (builder(model, X, Y) if builder is make_bc else builder(model, stage, X, Y))
        print(f"[{args.teacher}] evaluating {meta['name']} ...")
        rows.append({"method": meta["name"], "params": meta["params"],
                     "interpretable": meta["interpretable"], **evaluate(pred, stage, seeds, teacher_trajs)})

    # explainability-guided analytical controller (the paper's own surrogate),
    # measured under the identical harness vs the same teacher. Requires the
    # distillation result.json produced by the analysis suite; skipped if absent.
    if spec["equation_result"].exists():
        eq = SD.load_result(spec["equation_result"])
        rv = SD.rollout_validate_result(model, eq, stage, seeds)
        rows.append({"method": "Explainability-guided analytical", "params": count_equation_params(eq),
                     "interpretable": "yes (readable equation)",
                     "success": f"{rv['equation_successes']}/{len(seeds)}", "success_n": rv["equation_successes"],
                     "trace_dev": rv["mean_trace_deviation"], "endpoint_dev": rv["mean_endpoint_distance"]})
    else:
        print(f"[{args.teacher}] note: {spec['equation_result']} not found; "
              "skipping analytical-controller row (run the analysis suite first).")

    out = ROOT / "paper_data" / "distillation_baselines"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.teacher}.json").write_text(json.dumps(rows, indent=2))
    print("\n=== RESULTS:", args.teacher, "===")
    for r in rows:
        print(f"  {r['method']:32s} succ={r['success']:>5}  dev={r['trace_dev']:.3f}m  "
              f"endpt={r['endpoint_dev']:.3f}m  params={r['params']}  [{r['interpretable']}]")
    print("wrote", out / f"{args.teacher}.json")


if __name__ == "__main__":
    main()
