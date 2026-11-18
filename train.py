"""Run full-pipeline CAD-VAE cross-validation for one LD window."""

import argparse
import json
import os
import random
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from data import DataConfig, LDWindowDataModule
from model import BRANCHES, CADVAE, ModelConfig
from trainer import CADVAETrainer, TrainConfig

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(values, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2)


def clean_labels(values):
    return np.asarray(
        [str(value).strip() for value in values],
        dtype=object,
    )


def encode_labels(arrays, task_name):
    encoder = LabelEncoder()

    all_labels = np.concatenate([
        arrays[f"{task_name}_{split_name}"]
        for split_name in ("train", "val", "test")
    ])

    encoder.fit(all_labels)

    encoded = {
        split_name: encoder.transform(
            arrays[f"{task_name}_{split_name}"]
        )
        for split_name in ("train", "val", "test")
    }

    return encoder, encoded


def make_classifier(C, class_weight, seed):
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "logreg",
            LogisticRegression(
                max_iter=5000,
                solver="lbfgs",
                penalty="l2",
                C=C,
                class_weight=class_weight,
                random_state=seed,
            ),
        ),
    ])


def macro_ovr_auc(y_true, probabilities, n_classes):
    class_aucs = []

    for class_id in range(n_classes):
        binary_true = (y_true == class_id).astype(int)

        if len(np.unique(binary_true)) == 2:
            class_aucs.append(
                roc_auc_score(
                    binary_true,
                    probabilities[:, class_id],
                )
            )

    return float(np.mean(class_aucs)) if class_aucs else np.nan


def evaluate_probe(
    x_train,
    y_train,
    x_test,
    y_test,
    n_classes,
    C=1.0,
    class_weight="balanced",
    seed=1024,
):
    classifier = make_classifier(
        C,
        class_weight,
        seed,
    )

    classifier.fit(x_train, y_train)

    prediction = classifier.predict(x_test)
    raw_probabilities = classifier.predict_proba(x_test)

    fitted_classes = (
        classifier.named_steps["logreg"].classes_.astype(int)
    )

    probabilities = np.zeros(
        (len(y_test), n_classes),
        dtype=float,
    )

    probabilities[:, fitted_classes] = raw_probabilities

    metrics = {
        "acc": accuracy_score(
            y_test,
            prediction,
        ),
        "balanced_acc": balanced_accuracy_score(
            y_test,
            prediction,
        ),
        "macro_f1": f1_score(
            y_test,
            prediction,
            average="macro",
            zero_division=0,
        ),
        "auc": macro_ovr_auc(
            y_test,
            probabilities,
            n_classes,
        ),
    }

    return (
        metrics,
        prediction,
        probabilities,
        np.arange(n_classes),
    )


def evaluate_downstream(data, latent_mu, fold_id, fold_dir):
    arrays = {
        f"{task}_{split_name}": clean_labels(
            getattr(data, f"{task}_{split_name}")
        )
        for task in ("population", "region", "island")
        for split_name in ("train", "val", "test")
    }

    tasks = [
        (
            "Y",
            data.y_train,
            data.y_test,
            data.cfg.num_y_classes,
        )
    ]

    mapping_rows = []

    for task in ("population", "region", "island"):
        encoder, encoded = encode_labels(arrays, task)

        tasks.append((
            task,
            encoded["train"],
            encoded["test"],
            len(encoder.classes_),
        ))

        for class_id, class_name in enumerate(encoder.classes_):
            mapping_rows.append({
                "task": task,
                "encoded_id": class_id,
                "class_name": class_name,
            })

    pd.DataFrame(mapping_rows).to_csv(
        os.path.join(fold_dir, "class_mapping.csv"),
        index=False,
    )

    metric_rows = []

    prediction_table = {
        "sample": data.sample_ids_test,
    }

    for task, y_train, y_test, n_classes in tasks:
        prediction_table[f"{task}_true"] = y_test

        # Keep all existing single-branch evaluations.
        feature_groups = {
            branch: (branch,)
            for branch in BRANCHES
        }

        # Add joint features for the corresponding prediction task.
        if task == "Y":
            feature_groups["yr"] = ("y", "r")
        elif task == "island":
            feature_groups["sr"] = ("s", "r")

        for branch, names in feature_groups.items():
            x_train = np.concatenate(
                [
                    latent_mu["train"][name]
                    for name in names
                ],
                axis=1,
            )

            x_test = np.concatenate(
                [
                    latent_mu["test"][name]
                    for name in names
                ],
                axis=1,
            )

            metrics, prediction, probability, classes = evaluate_probe(
                x_train,
                y_train,
                x_test,
                y_test,
                n_classes=n_classes,
                seed=data.cfg.seed,
            )

            metric_rows.append({
                "fold_id": fold_id,
                "task": task,
                "branch": branch,
                "n_classes": n_classes,
                "chance_balanced_acc": 1.0 / n_classes,
                **metrics,
            })

            prefix = f"{task}_{branch}"

            prediction_table[
                f"{prefix}_prediction"
            ] = prediction

            for column, class_label in enumerate(classes):
                prediction_table[
                    f"{prefix}_probability_{class_label}"
                ] = probability[:, column]

    metrics = pd.DataFrame(metric_rows)

    metrics.to_csv(
        os.path.join(fold_dir, "downstream_metrics.csv"),
        index=False,
    )

    pd.DataFrame(prediction_table).to_csv(
        os.path.join(fold_dir, "downstream_predictions.csv"),
        index=False,
    )

    return metrics


def summarize_downstream(results):
    rows = []

    for (task, branch), frame in results.groupby(
        ["task", "branch"],
        sort=False,
    ):
        row = {
            "task": task,
            "branch": branch,
        }

        for metric in ("acc", "balanced_acc", "macro_f1", "auc"):
            values = frame[metric].to_numpy(dtype=float)

            row[f"cv_{metric}_mean"] = np.nanmean(values)
            row[f"cv_{metric}_std"] = np.nanstd(values,ddof=1)
            row[f"cv_{metric}_count"] = int(
                np.isfinite(values).sum()
            )

        rows.append(row)

    return pd.DataFrame(rows)


def export_latents(data, trainer, fold_dir, batch_size):
    """Export deterministic posterior means in stored sample order."""
    latent_mu = {}

    for split_name in ("train", "val", "test"):
        mu = trainer.encode_dataset(
            getattr(data, f"x_{split_name}"),
            batch_size=batch_size,
        )

        latent_mu[split_name] = mu

        np.savez_compressed(
            os.path.join(
                fold_dir,
                f"{split_name}_latents.npz",
            ),
            **{
                f"mu_{branch}": mu[branch]
                for branch in BRANCHES
            },
            y=getattr(data, f"y_{split_name}"),
            s=getattr(data, f"island_index_{split_name}"),
            condition=getattr(
                data,
                f"condition_{split_name}",
            ),
            sample_ids=np.asarray(
                getattr(data, f"sample_ids_{split_name}"),
                dtype=str,
            ),
            indices=getattr(
                data,
                f"{split_name}_indices",
            ),
        )

        getattr(
            data,
            f"{split_name}_metadata",
        ).to_csv(
            os.path.join(
                fold_dir,
                f"{split_name}_metadata.csv",
            ),
            index=False,
        )

    return latent_mu


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run five-fold CAD-VAE training "
            "for one LD window."
        )
    )

    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ld_block_pkl", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--block_id", type=int, default=0)

    window_group = parser.add_mutually_exclusive_group()

    window_group.add_argument(
        "--use_ext_window",
        dest="use_ext_window",
        action="store_true",
    )

    window_group.add_argument(
        "--use_core_window",
        dest="use_ext_window",
        action="store_false",
    )

    parser.set_defaults(use_ext_window=True)

    parser.add_argument(
        "--target_column",
        default="venation",
    )

    parser.add_argument(
        "--num_y_classes",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--inner_val_fraction",
        type=float,
        default=0.125,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--device",
        default=None,
    )
    
     # Loss weights.
    parser.add_argument(
        "--beta",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--beta_warmup_epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--lambda_cls",
        type=float,
        default=1000.0,
    )
    parser.add_argument(
        "--lambda_tc",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--lambda_cmi",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--lambda_lri",
        type=float,
        default=100.0,
    )

    return parser.parse_args()

def plot_training_losses(root,title=None):
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

        for ax, (metric, metric_title) in zip(axes.flat, metrics):
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

            ax.set_title(metric_title, fontsize=10)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)

        # Remove the two unused panels.
        for ax in list(axes.flat)[len(metrics):]:
            fig.delaxes(ax)

        relative_folder = (
            title
            if title is not None
            else log_path.parent.relative_to(root)
        )

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
    
    
def main(args):
    set_seed(args.seed)

    os.makedirs(
        args.save_dir,
        exist_ok=True,
    )

    data_cfg = DataConfig(
        out_dir=args.out_dir,
        ld_block_pkl=args.ld_block_pkl,
        block_id=args.block_id,
        use_ext_window=args.use_ext_window,
        target_column=args.target_column,
        num_y_classes=args.num_y_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        n_splits=args.n_splits,
        inner_val_fraction=args.inner_val_fraction,
    )

    model_cfg = ModelConfig(
        dim_x=16,
        dim_y=16,
        dim_s=16,
        dim_r=16,
        target_classes=args.num_y_classes,
    )

    # Keep the remaining loss weights and training settings in TrainConfig.
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,

        # Gaussian KL weight and warmup.
        beta=args.beta,
        beta_warmup_epochs=args.beta_warmup_epochs,

        # Main, opponent, and discriminator classification weight.
        lambda_cls=args.lambda_cls,

        # Encoder TC weight.
        lambda_tc=args.lambda_tc,

        # Use the same CMI weight in both directions.
        lambda_fair_yzs=args.lambda_cmi,
        lambda_fair_szy=args.lambda_cmi,

        # Use the same LRI weight for Y and S.
        lambda_fair_yzy=args.lambda_lri,
        lambda_fair_szs=args.lambda_lri,

        # Internal coefficients of the LRI entropy terms.
        lri_alpha=1.0,
        lri_beta=1.0,

        num_y_classes=args.num_y_classes,
        num_s_classes=3,
    )

    device = args.device or (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    save_json(
        {
            "data": asdict(data_cfg),
            "model": asdict(model_cfg),
            "training": asdict(train_cfg),
            "device": device,
            "downstream": {
                "classifier": (
                    "StandardScaler + LogisticRegression"
                ),
                "C": 1.0,
                "solver": "lbfgs",
                "penalty": "l2",
                "max_iter": 5000,
                "class_weight": "balanced",
                "random_state": args.seed,
                "tasks": [
                    "Y",
                    "population",
                    "region",
                    "island",
                ],
                "features": (
                    "posterior mean of each latent branch"
                ),
            },
        },
        os.path.join(
            args.save_dir,
            "config.json",
        ),
    )

    data = LDWindowDataModule(data_cfg)
    data.setup()

    folds = data.make_cv_splits()

    # Keep the original indices before sampling
    data.metadata_all.to_csv(
        os.path.join(
            args.save_dir,
            "sample_pool_metadata.csv",
        ),
        index=False,
    )

    vae_rows = []
    downstream_results = []

    for fold_id, (
        train_indices,
        val_indices,
        test_indices,
    ) in enumerate(folds, start=1):
        fold_dir = os.path.join(
            args.save_dir,
            f"fold_{fold_id}",
        )

        os.makedirs(
            fold_dir,
            exist_ok=True,
        )

        data.set_split(
            train_indices,
            val_indices,
            test_indices,
            fold_id=fold_id,
        )

        for split_name in ("train", "val", "test"):
            np.save(
                os.path.join(
                    fold_dir,
                    f"{split_name}_indices.npy",
                ),
                getattr(
                    data,
                    f"{split_name}_indices",
                ),
            )

        save_json(
            data.block_info,
            os.path.join(
                fold_dir,
                "block_info.json",
            ),
        )

        # Start each fold with the same seed and a fresh model and optimizers.
        set_seed(args.seed)

        model = CADVAE(
            data.input_dim,
            model_cfg,
        )

        trainer = CADVAETrainer(
            model=model,
            cfg=train_cfg,
            device=device,
        )

        train_loader = data.train_dataloader()
        val_loader = data.val_dataloader()
        test_loader = data.test_dataloader()

        trainer.fit(
            train_loader,
            val_loader,
        )

        trainer.restore_best_model()

        pd.DataFrame(
            trainer.history
        ).to_csv(
            os.path.join(
                fold_dir,
                "training_log.csv",
            ),
            index=False,
        )
        
        plot_training_losses(
            fold_dir,
            title=(
                f"{Path(args.out_dir).parent.name} | "
                f"block_{args.block_id} | fold_{fold_id}\n"
                f"beta={args.beta:g} | "
                f"CLS={args.lambda_cls:g} | "
                f"TC={args.lambda_tc:g} | "
                f"CMI={args.lambda_cmi:g} | "
                f"LRI={args.lambda_lri:g}"
            ),
        )

        # Use the same final loss coefficients in every fold.
        test_metrics = trainer.evaluate(test_loader)

        vae_row = {
            "fold_id": fold_id,
            "best_epoch": trainer.best_epoch,
            "best_val_loss": trainer.best_val_loss,
            **test_metrics,
        }

        vae_rows.append(vae_row)

        save_json(
            vae_row,
            os.path.join(
                fold_dir,
                "final_test_metrics.json",
            ),
        )

        # torch.save(
        #     {
        #         "model_state_dict": {
        #             name: tensor.detach().cpu()
        #             for name, tensor
        #             in model.state_dict().items()
        #         },
        #         "input_dim": data.input_dim,
        #         "model_config": asdict(model_cfg),
        #         "train_config": asdict(train_cfg),
        #         "data_config": asdict(data_cfg),
        #         "fold_id": fold_id,
        #         "best_epoch": trainer.best_epoch,
        #         "best_val_loss": trainer.best_val_loss,
        #     },
        #     os.path.join(
        #         fold_dir,
        #         "best_model.pt",
        #     ),
        # )

        latent_mu = export_latents(
            data,
            trainer,
            fold_dir,
            batch_size=args.batch_size,
        )

        downstream_results.append(
            evaluate_downstream(
                data,
                latent_mu,
                fold_id,
                fold_dir,
            )
        )

        del trainer, model

    vae_results = pd.DataFrame(vae_rows)

    vae_results.to_csv(
        os.path.join(
            args.save_dir,
            "vae_test_metrics_all_folds.csv",
        ),
        index=False,
    )

    vae_summary = (
        vae_results.drop(
            columns=[
                "fold_id",
                "best_epoch",
                "best_val_loss",
            ]
        )
        .agg(["mean", "std", "count"])
        .T
    )

    vae_summary.index.name = "metric"

    vae_summary.to_csv(
        os.path.join(
            args.save_dir,
            "vae_test_summary.csv",
        )
    )

    downstream_results = pd.concat(
        downstream_results,
        ignore_index=True,
    )

    downstream_results.to_csv(
        os.path.join(
            args.save_dir,
            "downstream_metrics_all_folds.csv",
        ),
        index=False,
    )

    downstream_summary = summarize_downstream(
        downstream_results
    )

    downstream_summary.to_csv(
        os.path.join(
            args.save_dir,
            "downstream_summary.csv",
        ),
        index=False,
    )

    # print(
    #     downstream_summary.to_string(index=False)
    # )

    print(
        f"Saved results to: {args.save_dir}"
    )


if __name__ == "__main__":
    main(parse_args())