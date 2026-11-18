"""Load LD-window genotypes for full-pipeline CAD-VAE cross-validation."""

import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import zarr
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset


ISLAND_NAMES = ("Hawaii/Hilo", "Kauai", "Oahu")

ISLAND_TO_INDEX = {
    island: index
    for index, island in enumerate(ISLAND_NAMES)
}


@dataclass
class DataConfig:
    out_dir: str
    ld_block_pkl: str

    block_id: int = 0
    use_ext_window: bool = True

    target_column: str = "venation"
    num_y_classes: int = 2

    batch_size: int = 32
    num_workers: int = 0
    seed: int = 1024

    n_splits: int = 5
    inner_val_fraction: float = 0.125


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def population_to_island(population):
    """Map population names to the three islands."""
    population = str(population).strip()

    if population.startswith(("Hilo", "Hawaii")):
        return "Hawaii/Hilo"

    if population.startswith("Kauai"):
        return "Kauai"

    if population.startswith("Oahu"):
        return "Oahu"

    raise ValueError(
        f"Cannot derive island from population: {population}"
    )


def encode_islands(island_labels):
    """Return integer island labels and one-hot vectors."""
    indices = np.asarray(
        [ISLAND_TO_INDEX[label] for label in island_labels],
        dtype=np.int64,
    )

    one_hot = np.eye(
        len(ISLAND_NAMES),
        dtype=np.float32,
    )[indices]

    return indices, one_hot


def validate_partition(
    train_indices,
    val_indices,
    test_indices,
    n_samples,
):
    combined = np.concatenate(
        [train_indices, val_indices, test_indices]
    )

    if not np.array_equal(
        np.sort(combined),
        np.arange(n_samples),
    ):
        raise ValueError(
            "Train/val/test must be a full partition "
            "of all samples."
        )


class GenotypeDataset(Dataset):
    """Return (genotype, target label, island label)."""

    def __init__(self, x_array, y_array, s_array):
        self.x = torch.from_numpy(
            x_array.astype(np.float32, copy=False)
        )

        self.y = torch.from_numpy(
            y_array.astype(np.int64, copy=False)
        )

        self.s = torch.from_numpy(
            s_array.astype(np.int64, copy=False)
        )

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.s[index]


class LDWindowDataModule:
    """Load one LD window and select samples for each CV fold."""

    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self):
        x_train_zarr = zarr.open(
            os.path.join(self.cfg.out_dir, "x_train.zarr"),
            mode="r",
        )

        x_holdout_zarr = zarr.open(
            os.path.join(self.cfg.out_dir, "x_test.zarr"),
            mode="r",
        )

        original_y_train = np.asarray(
            load_pickle(
                os.path.join(self.cfg.out_dir, "y_train.pkl")
            ),
            dtype=int,
        ).reshape(-1)

        original_y_holdout = np.asarray(
            load_pickle(
                os.path.join(self.cfg.out_dir, "y_test.pkl")
            ),
            dtype=int,
        ).reshape(-1)

        train_metadata = pd.read_pickle(
            os.path.join(self.cfg.out_dir, "train_metadata.pkl")
        ).reset_index(drop=True)

        holdout_metadata = pd.read_pickle(
            os.path.join(self.cfg.out_dir, "test_metadata.pkl")
        ).reset_index(drop=True)

        if not (
            x_train_zarr.shape[0]
            == len(original_y_train)
            == len(train_metadata)
        ):
            raise ValueError(
                "Training genotype/label/metadata lengths differ."
            )

        if not (
            x_holdout_zarr.shape[0]
            == len(original_y_holdout)
            == len(holdout_metadata)
        ):
            raise ValueError(
                "Holdout genotype/label/metadata lengths differ."
            )

        train_metadata["island"] = (
            train_metadata["population"].map(population_to_island)
        )

        holdout_metadata["island"] = (
            holdout_metadata["population"].map(population_to_island)
        )

        ld_object = load_pickle(self.cfg.ld_block_pkl)

        block = ld_object["merged_blocks"].iloc[
            self.cfg.block_id
        ]

        if self.cfg.use_ext_window:
            start = int(block["ext_start_idx_full"])
            end = int(block["ext_end_idx_full"])
            n_snps = int(block["ext_n_snps"])
            window_type = "extended"
        else:
            start = int(block["core_start_idx_full"])
            end = int(block["core_end_idx_full"])
            n_snps = int(block["core_n_snps"])
            window_type = "core"

        x_train = np.asarray(
            x_train_zarr[:, start:end + 1, :],
            dtype=np.float32,
        ).reshape(x_train_zarr.shape[0], -1)

        x_holdout = np.asarray(
            x_holdout_zarr[:, start:end + 1, :],
            dtype=np.float32,
        ).reshape(x_holdout_zarr.shape[0], -1)

        # Merge the old storage partitions before making new CV folds.
        self.x_all = np.concatenate(
            [x_train, x_holdout],
            axis=0,
        )

        self.original_y_all = np.concatenate(
            [original_y_train, original_y_holdout],
            axis=0,
        )

        self.metadata_all = pd.concat(
            [train_metadata, holdout_metadata],
            ignore_index=True,
        )

        self.sample_ids_all = (
            self.metadata_all["sample"].astype(str).to_numpy()
        )

        if pd.Series(self.sample_ids_all).duplicated().any():
            raise ValueError(
                "Duplicate sample IDs in the merged dataset."
            )

        if self.cfg.target_column:
            raw_y = self.metadata_all[
                self.cfg.target_column
            ].to_numpy()
        else:
            raw_y = self.original_y_all

        self.y_all = np.asarray(
            raw_y,
            dtype=np.int64,
        ).reshape(-1)

        self.input_dim = self.x_all.shape[1]

        self._window_info = {
            "block_id": self.cfg.block_id,
            "window_type": window_type,
            "start": start,
            "end": end,
            "n_snps": n_snps,
            "input_dim": self.input_dim,
            "target_source": (
                self.cfg.target_column
                or "original y pickle files"
            ),
            "stratification_source": "original y pickle files",
            "n_splits": self.cfg.n_splits,
            "inner_val_fraction": self.cfg.inner_val_fraction,
        }

    def make_cv_splits(self):
        """Return global (train, val, test) indices for each fold."""
        ordered_indices = np.argsort(
            self.sample_ids_all,
            kind="stable",
        )

        strata = self.original_y_all[ordered_indices]

        splitter = StratifiedKFold(
            n_splits=self.cfg.n_splits,
            shuffle=True,
            random_state=self.cfg.seed,
        )

        folds = []

        for fold_id, (train_val_positions, test_positions) in enumerate(
            splitter.split(ordered_indices, strata),
            start=1,
        ):
            train_val_indices = ordered_indices[train_val_positions]
            test_indices = ordered_indices[test_positions]

            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=self.cfg.inner_val_fraction,
                random_state=self.cfg.seed,
                stratify=self.original_y_all[train_val_indices],
            )

            folds.append(
                (train_indices, val_indices, test_indices)
            )

        return folds

    def set_split(
        self,
        train_indices,
        val_indices,
        test_indices,
        fold_id,
    ):
        """Select the train/val/test samples for one fold."""
        train_indices = np.asarray(train_indices)
        val_indices = np.asarray(val_indices)
        test_indices = np.asarray(test_indices)

        # Validate the partition once when activating a fold.
        validate_partition(
            train_indices,
            val_indices,
            test_indices,
            len(self.x_all),
        )

        self.fold_id = fold_id

        self.train_indices = train_indices.copy()
        self.val_indices = val_indices.copy()
        self.test_indices = test_indices.copy()

        for split_name, indices in (
            ("train", train_indices),
            ("val", val_indices),
            ("test", test_indices),
        ):
            setattr(
                self,
                f"x_{split_name}",
                self.x_all[indices],
            )

            setattr(
                self,
                f"y_{split_name}",
                self.y_all[indices],
            )

            setattr(
                self,
                f"original_y_{split_name}",
                self.original_y_all[indices],
            )

            metadata = self.metadata_all.iloc[
                indices
            ].reset_index(drop=True)

            setattr(
                self,
                f"{split_name}_metadata",
                metadata,
            )

            for column, attribute in (
                ("population", "population"),
                ("region", "region"),
                ("island", "island"),
                ("sample", "sample_ids"),
            ):
                setattr(
                    self,
                    f"{attribute}_{split_name}",
                    metadata[column].astype(str).to_numpy(),
                )

            island_indices, one_hot = encode_islands(
                getattr(self, f"island_{split_name}")
            )

            setattr(
                self,
                f"island_index_{split_name}",
                island_indices,
            )

            # Keep the old export attribute
            setattr(
                self,
                f"condition_{split_name}",
                one_hot,
            )

        self.block_info = {
            **self._window_info,
            "fold_id": fold_id,
            "train_samples": len(self.x_train),
            "val_samples": len(self.x_val),
            "test_samples": len(self.x_test),
        }

        print(self.block_info)

    def _loader(self, split_name, shuffle):
        dataset = GenotypeDataset(
            getattr(self, f"x_{split_name}"),
            getattr(self, f"y_{split_name}"),
            getattr(self, f"island_index_{split_name}"),
        )

        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            drop_last=False,
        )

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("val", shuffle=False)

    def test_dataloader(self):
        return self._loader("test", shuffle=False)