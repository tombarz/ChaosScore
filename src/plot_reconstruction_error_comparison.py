from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def finite_values(df: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return values.to_numpy(dtype=float)


def save_violin_plot(
    df: pd.DataFrame,
    *,
    output_path: Path,
    metric: str,
) -> Path:
    task_column = f"task_{metric}"
    decoder_column = f"scfoundation_decoder_{metric}"
    data = [finite_values(df, task_column), finite_values(df, decoder_column)]
    if any(values.size == 0 for values in data):
        raise ValueError(f"No finite values available for metric '{metric}'")

    fig, axis = plt.subplots(figsize=(7, 5))
    parts = axis.violinplot(data, showmeans=False, showmedians=True, widths=0.75)
    colors = ["#4C78A8", "#F58518"]
    for body, color in zip(parts["bodies"], colors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.72)
    for key in ["cmedians", "cbars", "cmins", "cmaxes"]:
        parts[key].set_color("#222222")
        parts[key].set_linewidth(1.2)

    axis.set_xticks([1, 2])
    axis.set_xticklabels(["Our model", "scFoundation decoder"])
    axis.set_ylabel(f"Per-cell masked {metric.upper()}")
    axis.set_title(f"Distribution of Reconstruction {metric.upper()}")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path = ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_summary_bar_plot(
    df: pd.DataFrame,
    *,
    output_path: Path,
    metric: str,
) -> tuple[Path, pd.DataFrame]:
    task_values = finite_values(df, f"task_{metric}")
    decoder_values = finite_values(df, f"scfoundation_decoder_{metric}")
    if task_values.size == 0 or decoder_values.size == 0:
        raise ValueError(f"No finite values available for metric '{metric}'")

    summary = pd.DataFrame(
        [
            {
                "model": "Our model",
                "mean": float(np.mean(task_values)),
                "median": float(np.median(task_values)),
                "n": int(task_values.size),
            },
            {
                "model": "scFoundation decoder",
                "mean": float(np.mean(decoder_values)),
                "median": float(np.median(decoder_values)),
                "n": int(decoder_values.size),
            },
        ]
    )

    x = np.arange(summary.shape[0])
    width = 0.36
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.bar(x - width / 2, summary["mean"], width, label="Mean", color="#4C78A8")
    axis.bar(x + width / 2, summary["median"], width, label="Median", color="#F58518")
    axis.set_xticks(x)
    axis.set_xticklabels(summary["model"])
    axis.set_ylabel(f"Per-cell masked {metric.upper()}")
    axis.set_title(f"Average and Median Reconstruction {metric.upper()}")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path = ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path, summary


def write_json(path: str | Path, payload: dict) -> None:
    path = ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def plot_reconstruction_error_comparison(
    *,
    scores_path: str,
    output_dir: str,
    metrics: list[str],
) -> None:
    df = load_table(scores_path)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, dict[str, str]] = {}
    summaries: dict[str, list[dict]] = {}
    for metric in metrics:
        required = [f"task_{metric}", f"scfoundation_decoder_{metric}"]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for metric '{metric}': {missing}")

        violin_path = save_violin_plot(
            df,
            output_path=outdir / f"{metric}_violin.png",
            metric=metric,
        )
        bar_path, metric_summary = save_summary_bar_plot(
            df,
            output_path=outdir / f"{metric}_mean_median_bar.png",
            metric=metric,
        )
        summary_path = outdir / f"{metric}_mean_median_summary.csv"
        metric_summary.to_csv(summary_path, index=False)

        outputs[metric] = {
            "violin_plot": str(violin_path.resolve()),
            "mean_median_bar_plot": str(bar_path.resolve()),
            "summary_csv": str(summary_path.resolve()),
        }
        summaries[metric] = metric_summary.to_dict(orient="records")

    write_json(
        outdir / "plot_summary.json",
        {
            "scores_path": str(Path(scores_path).resolve()),
            "metrics": metrics,
            "summaries": summaries,
            "outputs": outputs,
        },
    )
    print("Reconstruction comparison plots written")
    for metric, paths in outputs.items():
        print(f"{metric.upper()} violin: {paths['violin_plot']}")
        print(f"{metric.upper()} bar: {paths['mean_median_bar_plot']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot existing reconstruction error comparison results without rerunning inference."
    )
    parser.add_argument("--scores_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metrics", nargs="*", choices=["mse", "mae"], default=["mse", "mae"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_reconstruction_error_comparison(
        scores_path=args.scores_path,
        output_dir=args.output_dir,
        metrics=args.metrics,
    )


if __name__ == "__main__":
    main()
