from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "manifest_three_channel.csv"
)


# ============================================================
# DOMAIN CONFIGURATION
# ============================================================
#
# Domain supervision untuk Tahap 7 hanya memakai simulated
# replay dengan RIR seen.
#
# Mapping sengaja FIXED, bukan dibuat otomatis berdasarkan
# urutan value_counts(), agar reproducible.
#
# booth dan office TIDAK dimasukkan karena keduanya merupakan
# unseen-RIR test domains.
#
# clean / real / metadata tidak tersedia / unseen RIR:
# domain_label = -1
#
# -1 akan dimask saat menghitung domain loss.
#
# ============================================================

DOMAIN_MAP = {
    "lecture": 0,
    "meeting": 1,
    "phone": 2,
    "stairway": 3,
}

DOMAIN_NAMES = {
    0: "lecture",
    1: "meeting",
    2: "phone",
    3: "stairway",
}

UNKNOWN_DOMAIN = -1

N_DOMAIN_CLASSES = len(
    DOMAIN_MAP
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text_value(
    value,
):
    """
    Normalize metadata text.

    Missing / NaN becomes empty string.
    """

    if pd.isna(
        value
    ):
        return ""

    return str(
        value
    ).strip().lower()


# ============================================================
# TRANSFORM LABEL
# ============================================================

def condition_to_transform_label(
    condition,
):
    """
    Transformation label:

        clean     -> 0
        simulated -> 1
        real       -> 1
    """

    condition = normalize_text_value(
        condition
    )

    if condition == "clean":
        return 0

    if condition in {
        "simulated",
        "real",
    }:
        return 1

    raise ValueError(
        "Unknown condition for transform label: "
        f"{condition!r}"
    )


# ============================================================
# DOMAIN LABEL
# ============================================================

def row_to_domain_label(
    row,
):
    """
    Build domain label for GRL.

    Only simulated replay from SEEN RIR domains receives
    a valid room-domain label.

    Valid domains:
        lecture  -> 0
        meeting  -> 1
        phone    -> 2
        stairway -> 3

    Everything else:
        -1 (unknown / masked)

    This includes:
        clean
        real replay
        simulated unseen booth
        simulated unseen office
        missing metadata
    """

    condition = normalize_text_value(
        row.get(
            "condition",
            "",
        )
    )

    # Domain supervision is restricted to simulated replay.
    if condition != "simulated":
        return UNKNOWN_DOMAIN

    rir_split = normalize_text_value(
        row.get(
            "rir_split",
            "",
        )
    )

    # Only seen RIR is eligible for domain supervision.
    if rir_split != "seen":
        return UNKNOWN_DOMAIN

    rir_room = normalize_text_value(
        row.get(
            "rir_room",
            "",
        )
    )

    return DOMAIN_MAP.get(
        rir_room,
        UNKNOWN_DOMAIN,
    )


# ============================================================
# FEATURE LOADER
# ============================================================

def load_feature_npz(
    feature_path,
):
    """
    Load three-channel feature from .npz.

    Expected output:
        [3, 80, T]

    The loader supports common NPZ key names while requiring
    exactly one valid 3D feature tensor.
    """

    feature_path = Path(
        feature_path
    )

    if not feature_path.is_absolute():
        feature_path = (
            ROOT
            / feature_path
        )

    if not feature_path.exists():

        raise FileNotFoundError(
            "Feature file not found:\n"
            f"{feature_path}"
        )

    with np.load(
        feature_path,
        allow_pickle=False,
    ) as data:

        # ====================================================
        # PREFERRED KEY NAMES
        # ====================================================

        preferred_keys = [
            "features",
            "feature",
            "x",
            "X",
            "three_channel",
        ]

        array = None

        for key in preferred_keys:

            if key in data.files:

                candidate = np.asarray(
                    data[
                        key
                    ]
                )

                if candidate.ndim == 3:

                    array = candidate
                    break

        # ====================================================
        # FALLBACK
        # ====================================================

        if array is None:

            candidates = []

            for key in data.files:

                candidate = np.asarray(
                    data[
                        key
                    ]
                )

                if candidate.ndim == 3:

                    candidates.append(
                        (
                            key,
                            candidate,
                        )
                    )

            if len(
                candidates
            ) == 1:

                array = candidates[
                    0
                ][
                    1
                ]

            elif len(
                candidates
            ) == 0:

                raise RuntimeError(
                    "No 3D feature array found in:\n"
                    f"{feature_path}\n"
                    f"NPZ keys: {data.files}"
                )

            else:

                raise RuntimeError(
                    "Multiple possible 3D feature arrays found "
                    "and key is ambiguous:\n"
                    f"{feature_path}\n"
                    f"Candidates: "
                    f"{[k for k, _ in candidates]}"
                )

    array = np.asarray(
        array,
        dtype=np.float32,
    )

    # ========================================================
    # SHAPE QC
    # ========================================================

    if array.ndim != 3:

        raise RuntimeError(
            "Feature must be 3-dimensional "
            f"[3, 80, T], got {array.shape}"
        )

    if array.shape[
        0
    ] != 3:

        raise RuntimeError(
            "Feature channel count must be 3 "
            "(Log-Mel + phase-derived + MGD), "
            f"got shape {array.shape}"
        )

    if array.shape[
        1
    ] != 80:

        raise RuntimeError(
            "Feature Mel dimension must be 80, "
            f"got shape {array.shape}"
        )

    if array.shape[
        2
    ] <= 0:

        raise RuntimeError(
            "Feature contains zero time frames: "
            f"{array.shape}"
        )

    if not np.isfinite(
        array
    ).all():

        raise RuntimeError(
            "Feature contains NaN/Inf:\n"
            f"{feature_path}"
        )

    return array


# ============================================================
# REPLAY DATASET
# ============================================================

class ReplayDataset(
    Dataset
):
    """
    Three-channel replay dataset.

    Returned sample fields
    ----------------------
    features:
        FloatTensor [3, 80, T]

    class_label:
        0 = bona fide
        1 = spoof

    transform_label:
        0 = clean
        1 = replay (simulated / real)

    domain_label:
        0 = lecture
        1 = meeting
        2 = phone
        3 = stairway
       -1 = unknown / masked

    global_feature_id:
        Unique feature ID.

    condition:
        clean / simulated / real
    """

    def __init__(
        self,
        split,
        manifest_path=MANIFEST_PATH,
    ):

        super().__init__()

        self.split = normalize_text_value(
            split
        )

        self.manifest_path = Path(
            manifest_path
        )

        if not self.manifest_path.is_absolute():

            self.manifest_path = (
                ROOT
                / self.manifest_path
            )

        if not self.manifest_path.exists():

            raise FileNotFoundError(
                "Three-channel manifest not found:\n"
                f"{self.manifest_path}"
            )

        # low_memory=False also avoids the mixed-dtype warning
        # previously observed for recording/RIR metadata.
        full_df = pd.read_csv(
            self.manifest_path,
            low_memory=False,
        )

        # ====================================================
        # REQUIRED COLUMNS
        # ====================================================

        required_columns = {
            "global_feature_id",
            "condition",
            "source_file_id",
            "label",
            "split",
            "feature_path",
        }

        missing_columns = (
            required_columns
            - set(
                full_df.columns
            )
        )

        if missing_columns:

            raise RuntimeError(
                "manifest_three_channel.csv missing "
                "required columns: "
                f"{sorted(missing_columns)}"
            )

        # ====================================================
        # SPLIT FILTER
        # ====================================================

        split_values = (
            full_df[
                "split"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        self.df = (
            full_df[
                split_values.eq(
                    self.split
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        if self.df.empty:

            raise RuntimeError(
                "Dataset contains zero rows for split: "
                f"{split!r}"
            )

        # ====================================================
        # LABEL QC
        # ====================================================

        self.df[
            "label"
        ] = (
            pd.to_numeric(
                self.df[
                    "label"
                ],
                errors="raise",
            )
            .astype(
                np.int64
            )
        )

        invalid_class = (
            ~self.df[
                "label"
            ]
            .isin(
                [
                    0,
                    1,
                ]
            )
        )

        if invalid_class.any():

            bad = (
                self.df.loc[
                    invalid_class,
                    [
                        "global_feature_id",
                        "label",
                    ],
                ]
                .head(
                    10
                )
            )

            raise RuntimeError(
                "Invalid class labels found:\n"
                f"{bad}"
            )

        # ====================================================
        # TRANSFORM LABEL
        # ====================================================

        self.df[
            "transform_label"
        ] = (
            self.df[
                "condition"
            ]
            .apply(
                condition_to_transform_label
            )
            .astype(
                np.int64
            )
        )

        # ====================================================
        # DOMAIN LABEL
        # ====================================================

        self.df[
            "domain_label"
        ] = (
            self.df.apply(
                row_to_domain_label,
                axis=1,
            )
            .astype(
                np.int64
            )
        )

        # ====================================================
        # DOMAIN PROTOCOL QC
        # ====================================================

        valid_domain_mask = (
            self.df[
                "domain_label"
            ]
            >= 0
        )

        if valid_domain_mask.any():

            valid_domain_df = (
                self.df[
                    valid_domain_mask
                ]
                .copy()
            )

            # Every labelled domain sample must be simulated.
            valid_conditions = (
                valid_domain_df[
                    "condition"
                ]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            if not valid_conditions.eq(
                "simulated"
            ).all():

                raise RuntimeError(
                    "Domain-label protocol error: "
                    "non-simulated sample received "
                    "a valid domain label."
                )

            # Every labelled domain sample must be seen RIR.
            if "rir_split" not in valid_domain_df.columns:

                raise RuntimeError(
                    "rir_split column missing while "
                    "domain labels are enabled."
                )

            valid_rir_split = (
                valid_domain_df[
                    "rir_split"
                ]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            if not valid_rir_split.eq(
                "seen"
            ).all():

                raise RuntimeError(
                    "Domain-label protocol error: "
                    "unseen RIR received training "
                    "domain label."
                )

        # ====================================================
        # TEST UNSEEN GUARD
        # ====================================================

        if self.split == "test":

            condition_values = (
                self.df[
                    "condition"
                ]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            simulated_test = (
                self.df[
                    condition_values.eq(
                        "simulated"
                    )
                ]
            )

            if not simulated_test.empty:

                if "rir_split" not in simulated_test.columns:

                    raise RuntimeError(
                        "Test simulated samples have no "
                        "rir_split metadata."
                    )

                test_rir_split = (
                    simulated_test[
                        "rir_split"
                    ]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )

                if not test_rir_split.eq(
                    "unseen"
                ).all():

                    raise RuntimeError(
                        "Test simulated protocol error: "
                        "found non-unseen RIR."
                    )

                # Unseen test domains must remain masked.
                unseen_domain_labels = (
                    simulated_test[
                        "domain_label"
                    ]
                )

                if not unseen_domain_labels.eq(
                    UNKNOWN_DOMAIN
                ).all():

                    raise RuntimeError(
                        "Unseen test RIR received "
                        "a seen domain label."
                    )

        # ====================================================
        # PRINT SUMMARY
        # ====================================================

        self._print_summary()

    # ========================================================
    # SUMMARY
    # ========================================================

    def _print_summary(
        self,
    ):

        print()

        print(
            "=" * 75
        )

        print(
            f"REPLAY DATASET: "
            f"{self.split.upper()}"
        )

        print(
            "=" * 75
        )

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
            .value_counts(
                dropna=False
            )
        )

        print()

        print(
            "Domain label distribution:"
        )

        print(
            self.df[
                "domain_label"
            ]
            .value_counts(
                dropna=False
            )
            .sort_index()
        )

        n_valid_domain = int(
            (
                self.df[
                    "domain_label"
                ]
                >= 0
            )
            .sum()
        )

        n_unknown_domain = int(
            (
                self.df[
                    "domain_label"
                ]
                < 0
            )
            .sum()
        )

        print()

        print(
            f"Valid domain labels   : "
            f"{n_valid_domain}"
        )

        print(
            f"Unknown/masked domain : "
            f"{n_unknown_domain}"
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

        row = self.df.iloc[
            index
        ]

        features_np = load_feature_npz(
            row[
                "feature_path"
            ]
        )

        features = torch.from_numpy(
            features_np
        ).float()

        class_label = int(
            row[
                "label"
            ]
        )

        transform_label = int(
            row[
                "transform_label"
            ]
        )

        domain_label = int(
            row[
                "domain_label"
            ]
        )

        global_feature_id = str(
            row[
                "global_feature_id"
            ]
        )

        condition = str(
            row[
                "condition"
            ]
        )

        return {
            "features":
                features,

            "class_label":
                class_label,

            "transform_label":
                transform_label,

            "domain_label":
                domain_label,

            "global_feature_id":
                global_feature_id,

            "condition":
                condition,
        }


# ============================================================
# COLLATE FUNCTION
# ============================================================

def replay_collate_fn(
    batch,
):
    """
    Pad variable-length time dimension.

    Input feature:
        [3, 80, T]

    Output:
        features         [B, 3, 80, T_max]
        class_labels     [B]
        transform_labels [B]
        domain_labels    [B]
        lengths          [B]

    Metadata:
        global_feature_ids
        conditions
    """

    if len(
        batch
    ) == 0:

        raise ValueError(
            "Cannot collate empty batch."
        )

    # ========================================================
    # LENGTHS
    # ========================================================

    lengths = torch.tensor(
        [
            int(
                item[
                    "features"
                ].shape[
                    -1
                ]
            )
            for item in batch
        ],
        dtype=torch.long,
    )

    max_length = int(
        lengths.max().item()
    )

    batch_size = len(
        batch
    )

    first_feature = batch[
        0
    ][
        "features"
    ]

    if first_feature.ndim != 3:

        raise RuntimeError(
            "Expected feature shape "
            "[3, 80, T]."
        )

    channels = int(
        first_feature.shape[
            0
        ]
    )

    n_mels = int(
        first_feature.shape[
            1
        ]
    )

    # ========================================================
    # PADDED FEATURE TENSOR
    # ========================================================

    padded_features = torch.zeros(
        (
            batch_size,
            channels,
            n_mels,
            max_length,
        ),
        dtype=torch.float32,
    )

    # ========================================================
    # LABELS
    # ========================================================

    class_labels = torch.empty(
        batch_size,
        dtype=torch.long,
    )

    transform_labels = torch.empty(
        batch_size,
        dtype=torch.long,
    )

    domain_labels = torch.empty(
        batch_size,
        dtype=torch.long,
    )

    # ========================================================
    # METADATA
    # ========================================================

    global_feature_ids = []

    conditions = []

    # ========================================================
    # COPY ITEMS
    # ========================================================

    for i, item in enumerate(
        batch
    ):

        feature = item[
            "features"
        ]

        if feature.ndim != 3:

            raise RuntimeError(
                "Feature must have shape "
                "[3, 80, T]."
            )

        if int(
            feature.shape[
                0
            ]
        ) != channels:

            raise RuntimeError(
                "Channel count mismatch "
                "inside batch."
            )

        if int(
            feature.shape[
                1
            ]
        ) != n_mels:

            raise RuntimeError(
                "Mel dimension mismatch "
                "inside batch."
            )

        length = int(
            feature.shape[
                -1
            ]
        )

        padded_features[
            i,
            :,
            :,
            :length,
        ] = feature

        class_labels[
            i
        ] = int(
            item[
                "class_label"
            ]
        )

        transform_labels[
            i
        ] = int(
            item[
                "transform_label"
            ]
        )

        domain_labels[
            i
        ] = int(
            item[
                "domain_label"
            ]
        )

        global_feature_ids.append(
            str(
                item[
                    "global_feature_id"
                ]
            )
        )

        conditions.append(
            str(
                item[
                    "condition"
                ]
            )
        )

    # ========================================================
    # FINAL QC
    # ========================================================

    if not torch.isfinite(
        padded_features
    ).all():

        raise RuntimeError(
            "Non-finite feature detected "
            "after collation."
        )

    return {
        "features":
            padded_features,

        "class_labels":
            class_labels,

        "transform_labels":
            transform_labels,

        "domain_labels":
            domain_labels,

        "lengths":
            lengths,

        "global_feature_ids":
            global_feature_ids,

        "conditions":
            conditions,
    }


# ============================================================
# SELF TEST
# ============================================================

def _print_domain_mapping():

    print()

    print(
        "=" * 75
    )

    print(
        "DOMAIN MAPPING"
    )

    print(
        "=" * 75
    )

    print()

    for name, label in (
        DOMAIN_MAP.items()
    ):

        print(
            f"{label} -> {name}"
        )

    print(
        f"{UNKNOWN_DOMAIN} -> "
        "unknown / masked"
    )


def _audit_domain_split(
    dataset,
):

    df = dataset.df

    print()

    print(
        "-" * 75
    )

    print(
        f"DOMAIN AUDIT: "
        f"{dataset.split.upper()}"
    )

    print(
        "-" * 75
    )

    print()

    print(
        "Domain labels:"
    )

    print(
        df[
            "domain_label"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    valid = (
        df[
            "domain_label"
        ]
        >= 0
    )

    print()

    print(
        f"Valid domain samples : "
        f"{int(valid.sum())}"
    )

    print(
        f"Masked samples       : "
        f"{int((~valid).sum())}"
    )

    if valid.any():

        print()

        print(
            "Valid domain room distribution:"
        )

        print(
            df.loc[
                valid,
                "rir_room",
            ]
            .value_counts()
        )


def _self_test():

    print(
        "=" * 75
    )

    print(
        "REPLAY DATASET + DOMAIN LABEL SELF TEST"
    )

    print(
        "=" * 75
    )

    _print_domain_mapping()

    # ========================================================
    # BUILD DATASETS
    # ========================================================

    train_dataset = ReplayDataset(
        split="train"
    )

    validation_dataset = ReplayDataset(
        split="validation"
    )

    test_dataset = ReplayDataset(
        split="test"
    )

    # ========================================================
    # AUDIT
    # ========================================================

    _audit_domain_split(
        train_dataset
    )

    _audit_domain_split(
        validation_dataset
    )

    _audit_domain_split(
        test_dataset
    )

    # ========================================================
    # EXPECTED COUNTS
    # ========================================================

    train_valid = int(
        (
            train_dataset.df[
                "domain_label"
            ]
            >= 0
        )
        .sum()
    )

    validation_valid = int(
        (
            validation_dataset.df[
                "domain_label"
            ]
            >= 0
        )
        .sum()
    )

    test_valid = int(
        (
            test_dataset.df[
                "domain_label"
            ]
            >= 0
        )
        .sum()
    )

    if train_valid != 60:

        raise RuntimeError(
            "Expected 60 domain-labelled "
            "train simulated samples, "
            f"got {train_valid}."
        )

    if validation_valid != 45:

        raise RuntimeError(
            "Expected 45 domain-labelled "
            "validation simulated samples, "
            f"got {validation_valid}."
        )

    if test_valid != 0:

        raise RuntimeError(
            "Test unseen RIR must have zero "
            "valid seen-domain labels, "
            f"got {test_valid}."
        )

    # ========================================================
    # COLLATE TEST
    # ========================================================

    from torch.utils.data import DataLoader

    loader = DataLoader(
        train_dataset,
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
        "=" * 75
    )

    print(
        "COLLATE TEST"
    )

    print(
        "=" * 75
    )

    print()

    print(
        "Batch keys:"
    )

    print(
        list(
            batch.keys()
        )
    )

    print()

    print(
        "Feature shape:"
    )

    print(
        batch[
            "features"
        ].shape
    )

    print()

    print(
        "Class labels:"
    )

    print(
        batch[
            "class_labels"
        ]
    )

    print()

    print(
        "Transform labels:"
    )

    print(
        batch[
            "transform_labels"
        ]
    )

    print()

    print(
        "Domain labels:"
    )

    print(
        batch[
            "domain_labels"
        ]
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()

    print(
        "=" * 75
    )

    print(
        "EXPECTED DOMAIN COUNTS"
    )

    print(
        "=" * 75
    )

    print()

    print(
        f"Train valid domain      : "
        f"{train_valid} / expected 60"
    )

    print(
        f"Validation valid domain : "
        f"{validation_valid} / expected 45"
    )

    print(
        f"Test valid domain       : "
        f"{test_valid} / expected 0"
    )

    print()

    print(
        "REPLAY DATASET DOMAIN SELF TEST: PASSED"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    _self_test()