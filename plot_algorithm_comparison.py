"""
Generate paper-style comparison plots from paper_metrics.csv logs.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


PALETTE = {
    "SAC": "#1f77b4",
    "TD3": "#ff7f0e",
    "DDPG": "#2ca02c",
    "PPO": "#d62728",
}

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "font.size": 12,
        "axes.labelsize": 17,
        "axes.titlesize": 20,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.linewidth": 1.7,
        "lines.linewidth": 3.2,
        "savefig.dpi": 450,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def parse_series(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Series must be LABEL=CSV_PATH.")
    label, path_str = spec.split("=", 1)
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"CSV not found: {path}")
    return label, path


def parse_annotation(spec: str) -> tuple[str, int, str]:
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Annotation must be LABEL:TIMESTEPS:TEXT.")
    label, timestep_str, text = parts
    return label, int(timestep_str), text


def smooth_by_timestep(timesteps: np.ndarray, values: np.ndarray, window_timesteps: int) -> np.ndarray:
    if window_timesteps <= 0:
        return values.astype(float).copy()
    smoothed = np.empty_like(values, dtype=float)
    half_window = window_timesteps / 2.0
    for idx, timestep in enumerate(timesteps):
        mask = np.abs(timesteps - timestep) <= half_window
        smoothed[idx] = float(np.mean(values[mask]))
    return smoothed


def rolling_spread_by_timestep(timesteps: np.ndarray, values: np.ndarray, window_timesteps: int) -> np.ndarray:
    if window_timesteps <= 0:
        return np.zeros_like(values, dtype=float)
    spread = np.empty_like(values, dtype=float)
    half_window = window_timesteps / 2.0
    for idx, timestep in enumerate(timesteps):
        mask = np.abs(timesteps - timestep) <= half_window
        window_values = values[mask]
        spread[idx] = float(np.std(window_values))
    return spread


def load_series(series_specs: list[tuple[str, Path]]) -> list[dict]:
    loaded = []
    for label, csv_path in series_specs:
        df = pd.read_csv(csv_path)
        if "timesteps" not in df.columns:
            raise ValueError(f"'timesteps' column missing in {csv_path}")
        loaded.append(
            {
                "label": label,
                "csv_path": csv_path,
                "df": df.sort_values("timesteps").reset_index(drop=True),
            }
        )
    return loaded


def style_axis(ax, ylabel: str, x_max_millions: float | None, title: str | None, y_range=None) -> None:
    ax.set_xlabel("Timesteps (Millions)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=10)
    if x_max_millions is not None:
        ax.set_xlim(0.0, x_max_millions)
    if y_range is not None:
        ax.set_ylim(*y_range)
    ax.grid(True, which="major", linestyle="--", linewidth=1.1, alpha=0.32, color="#b5b5b5")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", width=1.05, length=6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))


def add_annotation_box(
    ax,
    loaded_series: list[dict],
    metric: str,
    annotations: list[tuple[str, int, str]],
) -> None:
    if not annotations:
        return

    text_lines = []
    for label, timestep, text in annotations:
        color = PALETTE.get(label.upper(), "#444444")
        for series in loaded_series:
            if series["label"] != label:
                continue
            df = series["df"]
            rows = df[df["timesteps"] == timestep]
            if rows.empty or metric not in rows.columns:
                continue
            value = float(rows.iloc[0][metric])
            ax.axvline(timestep / 1e6, color=color, linestyle=":", linewidth=2.0, alpha=0.9)
            ax.scatter([timestep / 1e6], [value], color=color, s=28, zorder=6)
            text_lines.append(text)

    if text_lines:
        ax.text(
            0.018,
            0.018,
            "\n".join(text_lines),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#c9c9c9",
                "alpha": 0.93,
            },
        )


def save_figure(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_metric(
    loaded_series: list[dict],
    metric: str,
    ylabel: str,
    out_path: Path,
    window_timesteps: int,
    max_timesteps: int | None,
    title: str | None,
    y_range=None,
    annotations: list[tuple[str, int, str]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    legend_handles = []

    for idx, series in enumerate(loaded_series):
        df = series["df"]
        if max_timesteps is not None:
            df = df[df["timesteps"] <= max_timesteps].copy()
        if df.empty or metric not in df.columns:
            continue

        timesteps = df["timesteps"].to_numpy(dtype=float)
        values = df[metric].to_numpy(dtype=float)
        label = series["label"]
        color = PALETTE.get(label.upper(), plt.cm.tab10(idx))

        smoothed = smooth_by_timestep(timesteps, values, window_timesteps)
        spread = rolling_spread_by_timestep(timesteps, values, window_timesteps)
        lower = smoothed - spread
        upper = smoothed + spread

        if metric == "success_rate":
            lower = np.clip(lower, 0.0, 1.0)
            upper = np.clip(upper, 0.0, 1.0)

        ax.fill_between(
            timesteps / 1e6,
            lower,
            upper,
            color=color,
            alpha=0.15,
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(
            timesteps / 1e6,
            smoothed,
            color=color,
            linewidth=3.2,
            alpha=0.98,
            zorder=3,
        )
        legend_handles.append(Line2D([0], [0], color=color, lw=3.2, label=label))

    style_axis(
        ax,
        ylabel=ylabel,
        x_max_millions=(max_timesteps / 1e6) if max_timesteps is not None else None,
        title=title,
        y_range=y_range,
    )
    add_annotation_box(ax, loaded_series, metric, annotations or [])

    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=False,
        handlelength=2.0,
        borderaxespad=0.3,
    )
    fig.tight_layout()
    save_figure(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        type=parse_series,
        help="Series specification in LABEL=CSV_PATH format.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--window-timesteps", type=int, default=350_000)
    parser.add_argument("--max-timesteps", type=int, default=None)
    parser.add_argument("--base-name", type=str, default="paper")
    parser.add_argument(
        "--annotate-success-point",
        action="append",
        default=[],
        type=parse_annotation,
        help="Optional LABEL:TIMESTEPS:TEXT annotation for success-rate plots.",
    )
    parser.add_argument(
        "--no-titles",
        action="store_true",
        help="Suppress plot titles.",
    )
    args = parser.parse_args()

    loaded_series = load_series(args.series)
    out_dir = args.out_dir.resolve()

    plot_metric(
        loaded_series=loaded_series,
        metric="success_rate",
        ylabel="Evaluation Success Rate",
        out_path=out_dir / f"{args.base_name}_success_rate.png",
        window_timesteps=args.window_timesteps,
        max_timesteps=args.max_timesteps,
        title=None if args.no_titles else "Success Rate vs Timesteps (Paper Comparison)",
        y_range=(0.0, 1.02),
        annotations=args.annotate_success_point,
    )
    plot_metric(
        loaded_series=loaded_series,
        metric="mean_reward",
        ylabel="Evaluation Mean Reward",
        out_path=out_dir / f"{args.base_name}_mean_reward.png",
        window_timesteps=args.window_timesteps,
        max_timesteps=args.max_timesteps,
        title=None if args.no_titles else "Reward vs Timesteps (Paper Comparison)",
        y_range=None,
        annotations=None,
    )


if __name__ == "__main__":
    main()
