#!/usr/bin/env python3
"""Plot test AUC versus CMI, with one figure per KL beta."""

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

BRANCHES = ("x", "y", "s", "r")

SERIES = {
    f"downstream_z{branch}_{target}": (task, branch)
    for branch in BRANCHES
    for target, task in (
        ("y", "Y"),
        ("s", "island"),
    )
}
SERIES.update({
    "downstream_zyr_y": ("Y", "yr"),
    "downstream_zsr_s": ("island", "sr"),
})
PANELS = tuple(
    (
        f"downstream_z{branch}_{target}",
        rf"Downstream: $z_{branch} \rightarrow {symbol}$ ({name})",
    )
    for branch in BRANCHES
    for target, symbol, name in (
        ("y", "Y", "venation"),
        ("s", "S", "island"),
    )
) 

PANELS += (
    (
        "downstream_zyr_y",
        r"Downstream: $[z_y,z_r] \rightarrow Y$ (venation)",
    ),
    (
        "downstream_zsr_s",
        r"Downstream: $[z_s,z_r] \rightarrow S$ (island)",
    ),
)

PANELS +=  (
    (
        "vae_cls_y",
        r"VAE head: $[z_y,z_r] \rightarrow Y$ (venation)",
    ),
    (
        "vae_cls_s",
        r"VAE head: $[z_s,z_r] \rightarrow S$ (island)",
    ),
)

PANELS += (
    (
        "vae_opp_y",
        r"Opponent: $z_s \rightarrow Y$ (venation)",
    ),
    (
        "vae_opp_s",
        r"Opponent: $z_y \rightarrow S$ (island)",
    ),
)

BLOCK_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
)

BLOCK_LABELS = {
    176: "Scaffold 10: Block 176",
    136: "Scaffold 1: Block 136",
}

BLOCK_SCAFFOLDS = {
    176: 10,
    136: 1,
}


BLOCK_MARKERS = ("o", "s", "^", "D")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Sweep directory containing block_0, block_1, etc.",
    )
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument(
        "--blocks",
        type=int,
        nargs="+",
        default=[176, 136],
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[1],
    )
    parser.add_argument("--cls", type=float, default=500)
    parser.add_argument("--tc", type=float, default=0)
    parser.add_argument(
        "--lri_values",
        type=float,
        nargs="+",
        default=[0, 20, 40, 60, 80, 100, 500, 1000],
    )
    parser.add_argument("--cmi", type=float, default=500)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--expected_folds", type=int, default=5)

    return parser.parse_args()


def run_directory(args, block, beta, lri):
    scaffold = BLOCK_SCAFFOLDS[block]

    return (
        args.root
        / f"scaffold_{scaffold}"
        / f"block_{block}"
        / "extended"
        / f"epochs_{args.epochs}"
        / f"warmup_{args.warmup}"
        / f"beta_{beta:g}"
        / f"cls_{args.cls:g}"
        / (f"tc_{args.tc:g}" if args.tc is not None else "")
        / f"cmi_{args.cmi:g}"
        / f"lri_{lri:g}"
        / f"seed_{args.seed}"
    )


def add_configuration(frame, args, block, beta, lri):
    return frame.assign(
        block_id=block,
        beta=beta,
        cls=args.cls,
        tc=args.tc,
        cmi=args.cmi,
        lri=lri,
        seed=args.seed,
        epochs=args.epochs,
        warmup=args.warmup,
    )


def collect_auc(args):
    frames = []

    for block in args.blocks:
        for beta in args.betas:
            for lri in args.lri_values:
                directory = run_directory(
                    args, block, beta, lri
                )

                downstream_path = (
                    directory
                    / "downstream_metrics_all_folds.csv"
                )
                vae_path = (
                    directory
                    / "vae_test_metrics_all_folds.csv"
                )

                if (
                    not downstream_path.is_file()
                    or not vae_path.is_file()
                ):
                    warnings.warn(
                        "Missing result files; leaving a gap: "
                        f"{directory}"
                    )
                    continue

                downstream = pd.read_csv(downstream_path)
                vae = pd.read_csv(vae_path)

                if not {"auc_y", "auc_s"}.issubset(vae.columns):
                    raise ValueError(
                        f"{vae_path} lacks auc_y/auc_s. "
                        "Add AUC to trainer.py before "
                        "evaluating the VAE heads."
                    )

                # Collect the downstream prediction tasks.
                for series, (task, branch) in SERIES.items():
                    selected = downstream.loc[
                        (downstream["task"] == task)
                        & (downstream["branch"] == branch),
                        ["fold_id", "auc"],
                    ].copy()

                    selected["series"] = series

                    frames.append(
                        add_configuration(
                            selected, args, block, beta, lri
                        )
                    )

                # Collect the four built-in VAE classifier AUCs.
                for series, column in (
                    ("vae_cls_y", "auc_y"),
                    ("vae_cls_s", "auc_s"),
                    ("vae_opp_y", "auc_opponent_y"),
                    ("vae_opp_s", "auc_opponent_s"),
                ):
                    if column not in vae.columns:
                        warnings.warn(
                            f"{vae_path} lacks {column}; leaving a gap."
                        )
                        continue
                    selected = vae[
                        ["fold_id", column]
                    ].rename(columns={column: "auc"})

                    selected = selected.copy()
                    selected["series"] = series

                    frames.append(
                        add_configuration(
                            selected, args, block, beta, lri
                        )
                    )

    if not frames:
        raise FileNotFoundError(
            f"No matching result files under {args.root}"
        )

    values = pd.concat(frames, ignore_index=True)

    keys = [
        "block_id",
        "beta",
        "cls",
        "tc",
        "cmi",
        "lri",
        "series",
        "fold_id",
    ]

    if values.duplicated(keys).any():
        raise ValueError(
            "Duplicate fold results found for a curve "
            "and configuration."
        )

    return values.sort_values(keys).reset_index(drop=True)


def summarize_auc(values, args):
    keys = [
        "block_id",
        "beta",
        "cmi",
        "lri",
        "seed",
        "epochs",
        "warmup",
    ]

    # Use the mean and sample standard deviation across folds.
    summary = (
        values.groupby(keys + ["series"], sort=True)["auc"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )

    # Include absent curves as NaN to preserve gaps.
    configurations = values[keys].drop_duplicates()

    all_series = pd.DataFrame({
        "series": [series for series, _ in PANELS]
    })

    summary = configurations.merge(
        all_series,
        how="cross",
    ).merge(
        summary,
        on=keys + ["series"],
        how="left",
    )

    summary["n"] = summary["n"].fillna(0).astype(int)

    incomplete = summary.loc[
        summary["n"] != args.expected_folds
    ]

    if not incomplete.empty:
        warnings.warn(
            f"{len(incomplete)} curve points do not contain "
            f"{args.expected_folds} valid fold AUCs; "
            "using available valid folds."
        )

    return summary.sort_values(
        keys + ["series"]
    ).reset_index(drop=True)


def plot_beta(summary, args, beta, output_dir):
    selected = summary.loc[summary["beta"] == beta]

    if selected.empty:
        return

    lri_values = sorted(set(args.lri_values))

    # Treat LRI values as equally spaced parameter settings.
    positions = np.arange(len(lri_values))

    fig, axes = plt.subplots(
        7,
        2,
        figsize=(13, 24),
        sharex=True,
        sharey=True,
    )

    # Keep each block's style consistent across all panels.
    block_styles = {
        block: (
            BLOCK_COLORS[i % len(BLOCK_COLORS)],
            BLOCK_MARKERS[i % len(BLOCK_MARKERS)],
        )
        for i, block in enumerate(dict.fromkeys(args.blocks))
    }

    for ax, (series, title) in zip(axes.flat, PANELS):
        for block, (color, marker) in block_styles.items():
            points = selected.loc[
                (selected["series"] == series)
                & (selected["block_id"] == block)
            ].set_index("lri")

            points = points.reindex(lri_values)

            means = points["mean"].to_numpy(dtype=float)
            stds = points["std"].to_numpy(dtype=float)

            ax.plot(
                positions,
                means,
                color=color,
                marker=marker,
                linewidth=1.8,
                markersize=5,
            )

            valid_errors = (
                np.isfinite(means)
                & np.isfinite(stds)
            )

            ax.errorbar(
                positions[valid_errors],
                means[valid_errors],
                yerr=stds[valid_errors],
                fmt="none",
                ecolor=color,
                capsize=3,
                elinewidth=1,
                alpha=0.8,
            )

        ax.axhline(
            0.5,
            color="0.5",
            linestyle=":",
            linewidth=1,
        )

        ax.set_title(title)

        ax.set_xticks(
            positions,
            [f"{value:g}" for value in lri_values],
        )

        ax.set_ylim(0.4, 1.02)
        ax.grid(alpha=0.1)

    for ax in axes[:, 0]:
        ax.set_ylabel("AUC")

    for ax in axes[-1, :]:
        ax.set_xlabel(
            "LRI weight"
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linewidth=1.8,
            label=BLOCK_LABELS[block],
        )
        for block, (color, marker) in block_styles.items()
    ]

    fig.suptitle(
        f"CAD-VAE | KL beta = {beta:g} | "
        f"CLS = {args.cls:g} | TC = {args.tc:g} | "
        f"CMI = {args.cmi:g}",
        fontsize=15,
        y=0.985,
    )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(block_styles),
        bbox_to_anchor=(0.5, 0.958),
        frameon=False,
    )


    fig.tight_layout(
        rect=(0, 0.05, 1, 0.915)
    )

    output_path = (
        output_dir
        / (
            f"beta_{beta:g}_cls_{args.cls:g}"
            f"_tc_{args.tc:g}_cmi_{args.cmi:g}_auc_vs_lri.png"
        )
    )

    fig.savefig(output_path,dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")

def plot_training_losses(root):
    """Save one loss figure beside each fold's training log."""
    root = Path(root)

    metrics = (
        ("loss", "Total objective loss"),
        ("recon", "Reconstruction loss"),
        ("kl", "KL loss"),
        ("cls_y", "Y classification loss"),
        ("cls_s", "Island classification loss"),
        ("cmi_y", "CMI: Y from z_s"),
        ("cmi_s", "CMI: Island from z_y"),
        ("relevance_y", "LRI: Y relevance"),
        ("relevance_s", "LRI: Island relevance"),
        ("tc", "TC loss"),
        ("opponent_y", "Y opponent classification from z_s"),
        ("opponent_s", "Island opponent classification from z_y"),
    )

    count = 0

    for log_path in sorted(root.rglob("training_log.csv")):
        history = pd.read_csv(log_path)

        fig, axes = plt.subplots(
            4,
            3,
            figsize=(16, 12),
            sharex=True,
        )

        for ax, (metric, title) in zip(axes.flat, metrics):
            for prefix, label, color in (
                ("train", "Train", "#0072B2"),
                ("val", "Validation", "#D55E00"),
            ):
                column = f"{prefix}_{metric}"

                # Some objectives are recorded only during training.
                if column in history.columns:
                    ax.plot(
                        history["epoch"],
                        history[column],
                        label=label,
                        color=color,
                        linewidth=1.2,
                    )

            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)

        # Remove the two unused panels.
        for ax in list(axes.flat)[len(metrics):]:
            fig.delaxes(ax)

        relative_folder = log_path.parent.relative_to(root)

        fig.suptitle(
            f"CAD-VAE training losses\n{relative_folder}",
            fontsize=11,
        )

        fig.supxlabel("Epoch")

        fig.tight_layout(
            rect=(0, 0.03, 1, 0.94)
        )

        output_path = log_path.with_name(
            "training_losses.png"
        )

        fig.savefig(
            output_path,
            dpi=150,
        )

        plt.close(fig)
        count += 1

    print(f"Saved {count} loss figures.")
    
    
def main():
    args = parse_args()
    # plot_training_losses(args.root)

    output_dir = args.output_dir or (
        args.root
        / "auc_plots"
        / (
            f"epochs_{args.epochs}"
            f"_warmup_{args.warmup}"
            f"_cmi_{args.cmi:g}"
            f"_seed_{args.seed}"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    values = collect_auc(args)
    summary = summarize_auc(values, args)

    for beta in args.betas:
        plot_beta(
            summary,
            args,
            beta,
            output_dir,
        )

    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()