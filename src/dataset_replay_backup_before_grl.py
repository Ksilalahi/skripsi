from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "manifest_three_channel.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_CHANNELS = 3
EXPECTED_MELS = 80


# ============================================================
# REPLAY DATASET
# ============================================================

class ReplayDataset(Dataset):
    """
    Dataset untuk final three-channel features.

    Feature:
        [3,80,T]

    Channel:
        0 = normalized Log-Mel
        1 = normalized Phase-derived
        2 = normalized MGD

    Class label:
        0 = bona fide
        1 = spoof

    Transform label:
        0 = clean
        1 = replay

    Replay mencakup:
        simulated
        real
    """

    def __init__(
        self,
        manifest_path=MANIFEST_PATH,
        split="train",
    ):
        super().__init__()

        self.manifest_path = Path(
            manifest_path
        )

        self.split = str(
            split
        ).strip().lower()

        # ====================================================
        # MANIFEST CHECK
        # ====================================================

        if not self.manifest_path.exists():

            raise FileNotFoundError(
                f"Manifest tidak ditemukan:\n"
                f"{self.manifest_path}"
            )

        # ====================================================
        # LOAD MANIFEST
        # ====================================================

        df = pd.read_csv(
            self.manifest_path,
            low_memory=False,
        )

        if df.empty:

            raise RuntimeError(
                "Three-channel manifest kosong."
            )

        # ====================================================
        # REQUIRED COLUMNS
        # ====================================================

        required_columns = {
            "global_feature_id",
            "feature_path",
            "condition",
            "label",
            "split",
            "n_channels",
            "n_mels",
            "n_frames",
        }

        missing_columns = (
            required_columns
            - set(
                df.columns
            )
        )

        if missing_columns:

            raise RuntimeError(
                "Manifest kehilangan kolom:\n"
                f"{sorted(missing_columns)}"
            )

        # ====================================================
        # SPLIT FILTER
        # ====================================================

        normalized_split = (
            df[
                "split"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df = (
            df[
                normalized_split.eq(
                    self.split
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        if df.empty:

            raise RuntimeError(
                f"Tidak ada sample untuk "
                f"split={self.split}"
            )

        # ====================================================
        # BASIC MANIFEST QC
        # ====================================================

        if (
            df[
                "global_feature_id"
            ]
            .duplicated()
            .any()
        ):

            raise RuntimeError(
                "Duplicate global_feature_id "
                f"pada split={self.split}"
            )

        if not (
            df[
                "n_channels"
            ]
            .astype(int)
            .eq(
                EXPECTED_CHANNELS
            )
            .all()
        ):

            raise RuntimeError(
                "Ada sample dengan channel != 3."
            )

        if not (
            df[
                "n_mels"
            ]
            .astype(int)
            .eq(
                EXPECTED_MELS
            )
            .all()
        ):

            raise RuntimeError(
                "Ada sample dengan Mel != 80."
            )

        # ====================================================
        # CLASS LABEL QC
        # ====================================================

        labels = pd.to_numeric(
            df[
                "label"
            ],
            errors="coerce",
        )

        if labels.isna().any():

            bad = df[
                labels.isna()
            ][
                [
                    "global_feature_id",
                    "label",
                ]
            ]

            raise RuntimeError(
                "Ada class label invalid:\n"
                f"{bad.head(10).to_string(index=False)}"
            )

        labels = labels.astype(
            int
        )

        if not labels.isin(
            [
                0,
                1,
            ]
        ).all():

            raise RuntimeError(
                "Class label harus 0 atau 1."
            )

        df[
            "label"
        ] = labels

        # ====================================================
        # CONDITION QC
        # ====================================================

        allowed_conditions = {
            "clean",
            "simulated",
            "real",
        }

        conditions = set(
            df[
                "condition"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
            .tolist()
        )

        unknown = (
            conditions
            - allowed_conditions
        )

        if unknown:

            raise RuntimeError(
                "Unknown condition(s): "
                f"{sorted(unknown)}"
            )

        self.df = df

        # ====================================================
        # INFO
        # ====================================================

        print()
        print("=" * 75)

        print(
            f"REPLAY DATASET: "
            f"{self.split.upper()}"
        )

        print("=" * 75)

        print()

        print(
            f"Samples : "
            f"{len(self.df)}"
        )

        print()

        print(
            "Class distribution:"
        )

        print(
            self.df[
                "label"
            ]
            .value_counts()
            .sort_index()
        )

        print()

        print(
            "Condition distribution:"
        )

        print(
            self.df[
                "condition"
            ]
            .value_counts()
            .sort_index()
        )


    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(
        self,
    ):

        return len(
            self.df
        )


    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index,
    ):

        row = (
            self.df.iloc[
                index
            ]
        )

        # ====================================================
        # FEATURE PATH
        # ====================================================

        feature_path = (
            ROOT
            / str(
                row[
                    "feature_path"
                ]
            )
        )

        if not feature_path.exists():

            raise FileNotFoundError(
                feature_path
            )

        # ====================================================
        # LOAD FEATURE
        # ====================================================

        with np.load(
            feature_path
        ) as data:

            if (
                "feature"
                not in data.files
            ):

                raise RuntimeError(
                    f"{feature_path} "
                    "tidak memiliki key 'feature'."
                )

            feature = (
                data[
                    "feature"
                ]
                .astype(
                    np.float32,
                    copy=False,
                )
            )

        # ====================================================
        # FEATURE QC
        # ====================================================

        if feature.ndim != 3:

            raise RuntimeError(
                f"Invalid feature shape: "
                f"{feature.shape}"
            )

        if (
            feature.shape[
                0
            ]
            != EXPECTED_CHANNELS
        ):

            raise RuntimeError(
                f"Expected 3 channels, "
                f"got {feature.shape}"
            )

        if (
            feature.shape[
                1
            ]
            != EXPECTED_MELS
        ):

            raise RuntimeError(
                f"Expected 80 Mel bins, "
                f"got {feature.shape}"
            )

        if (
            feature.shape[
                2
            ]
            <= 0
        ):

            raise RuntimeError(
                "Feature has zero frames."
            )

        if not np.isfinite(
            feature
        ).all():

            raise RuntimeError(
                f"{feature_path} contains "
                "NaN/Inf."
            )

        # ====================================================
        # CLASS LABEL
        # ====================================================

        class_label = int(
            row[
                "label"
            ]
        )

        # ====================================================
        # TRANSFORM LABEL
        # ====================================================
        #
        # Guide:
        #
        # 0 = clean
        # 1 = replay
        #
        # simulated + real -> replay
        #
        # ====================================================

        condition = (
            str(
                row[
                    "condition"
                ]
            )
            .strip()
            .lower()
        )

        if condition == "clean":

            transform_label = 0

        elif condition in {
            "simulated",
            "real",
        }:

            transform_label = 1

        else:

            raise RuntimeError(
                f"Unknown condition: "
                f"{condition}"
            )

        # ====================================================
        # NUMPY -> TORCH
        # ====================================================

        feature = torch.from_numpy(
            feature
        ).float()

        class_label = torch.tensor(
            class_label,
            dtype=torch.long,
        )

        transform_label = torch.tensor(
            transform_label,
            dtype=torch.long,
        )

        return {
            "feature":
                feature,

            "class_label":
                class_label,

            "transform_label":
                transform_label,

            "global_feature_id":
                str(
                    row[
                        "global_feature_id"
                    ]
                ),

            "condition":
                condition,

            "n_frames":
                int(
                    feature.shape[
                        2
                    ]
                ),
        }


# ============================================================
# VARIABLE-LENGTH COLLATE FUNCTION
# ============================================================

def replay_collate_fn(
    batch,
):
    """
    Padding T ke panjang maksimum dalam batch.

    Karena feature sudah distandardisasi,
    padding = 0 berarti approximately
    training mean pada masing-masing channel.

    Input per sample:
        [3,80,T_i]

    Output:
        features:
            [B,3,80,max_T]
    """

    if len(
        batch
    ) == 0:

        raise RuntimeError(
            "Empty batch."
        )

    # ========================================================
    # MAX TIME
    # ========================================================

    max_frames = max(
        item[
            "feature"
        ].shape[
            2
        ]
        for item in batch
    )

    batch_size = len(
        batch
    )

    # ========================================================
    # PADDED FEATURE BUFFER
    # ========================================================

    features = torch.zeros(
        batch_size,
        EXPECTED_CHANNELS,
        EXPECTED_MELS,
        max_frames,
        dtype=torch.float32,
    )

    class_labels = torch.empty(
        batch_size,
        dtype=torch.long,
    )

    transform_labels = torch.empty(
        batch_size,
        dtype=torch.long,
    )

    lengths = torch.empty(
        batch_size,
        dtype=torch.long,
    )

    global_ids = []

    conditions = []

    # ========================================================
    # COPY FEATURES
    # ========================================================

    for i, item in enumerate(
        batch
    ):

        feature = (
            item[
                "feature"
            ]
        )

        frames = int(
            feature.shape[
                2
            ]
        )

        features[
            i,
            :,
            :,
            :frames
        ] = feature

        class_labels[
            i
        ] = item[
            "class_label"
        ]

        transform_labels[
            i
        ] = item[
            "transform_label"
        ]

        lengths[
            i
        ] = frames

        global_ids.append(
            item[
                "global_feature_id"
            ]
        )

        conditions.append(
            item[
                "condition"
            ]
        )

    # ========================================================
    # FINAL QC
    # ========================================================

    if not torch.isfinite(
        features
    ).all():

        raise RuntimeError(
            "Batch contains NaN/Inf."
        )

    return {
        "features":
            features,

        "class_labels":
            class_labels,

        "transform_labels":
            transform_labels,

        "lengths":
            lengths,

        "global_feature_ids":
            global_ids,

        "conditions":
            conditions,
    }


# ============================================================
# SELF TEST
# ============================================================

def _self_test():

    print("=" * 75)

    print(
        "REPLAY DATASET / DATALOADER SELF TEST"
    )

    print("=" * 75)

    # ========================================================
    # TRAIN DATASET
    # ========================================================

    dataset = ReplayDataset(
        split="train"
    )

    # ========================================================
    # INDIVIDUAL SAMPLE
    # ========================================================

    sample = dataset[
        0
    ]

    print()

    print(
        "Single sample:"
    )

    print(
        "Feature shape:",
        tuple(
            sample[
                "feature"
            ].shape
        ),
    )

    print(
        "Class label:",
        int(
            sample[
                "class_label"
            ]
        ),
    )

    print(
        "Transform label:",
        int(
            sample[
                "transform_label"
            ]
        ),
    )

    print(
        "Condition:",
        sample[
            "condition"
        ],
    )

    # ========================================================
    # DATALOADER
    # ========================================================

    loader = DataLoader(
        dataset,

        batch_size=4,

        shuffle=False,

        num_workers=0,

        collate_fn=replay_collate_fn,
    )

    batch = next(
        iter(
            loader
        )
    )

    print()

    print(
        "Batch features:",
        tuple(
            batch[
                "features"
            ].shape
        ),
    )

    print(
        "Class labels:",
        tuple(
            batch[
                "class_labels"
            ].shape
        ),
    )

    print(
        "Transform labels:",
        tuple(
            batch[
                "transform_labels"
            ].shape
        ),
    )

    print(
        "Lengths:",
        batch[
            "lengths"
        ].tolist(),
    )

    print(
        "Conditions:",
        batch[
            "conditions"
        ],
    )

    # ========================================================
    # ASSERTIONS
    # ========================================================

    assert (
        batch[
            "features"
        ].ndim
        == 4
    )

    assert (
        batch[
            "features"
        ].shape[
            0
        ]
        == 4
    )

    assert (
        batch[
            "features"
        ].shape[
            1
        ]
        == 3
    )

    assert (
        batch[
            "features"
        ].shape[
            2
        ]
        == 80
    )

    assert (
        batch[
            "class_labels"
        ].shape
        == (
            4,
        )
    )

    assert (
        batch[
            "transform_labels"
        ].shape
        == (
            4,
        )
    )

    assert torch.isfinite(
        batch[
            "features"
        ]
    ).all()

    print()

    print("=" * 75)

    print(
        "DATASET / DATALOADER SELF TEST: PASSED"
    )

    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    _self_test()