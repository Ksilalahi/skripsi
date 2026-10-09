from pathlib import Path
import argparse
import json
import math
import time

import numpy as np
import pandas as pd

import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from tqdm import tqdm

from dataset_replay import (
    ReplayDataset,
    replay_collate_fn,
)

from model_replay import ReplayResNet


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULT_ROOT = (
    ROOT
    / "results"
    / "evaluation"
)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_path(
    path,
):

    path = Path(
        path
    )

    if not path.is_absolute():

        path = (
            ROOT
            / path
        )

    return path.resolve()


# ============================================================
# SAFE JSON CONVERSION
# ============================================================

def to_jsonable(
    value,
):

    if isinstance(
        value,
        Path,
    ):

        return str(
            value
        )

    if isinstance(
        value,
        np.integer,
    ):

        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):

        value = float(
            value
        )

        if math.isnan(
            value
        ):

            return None

        if math.isinf(
            value
        ):

            return None

        return value

    if isinstance(
        value,
        np.ndarray,
    ):

        return (
            value
            .tolist()
        )

    if isinstance(
        value,
        torch.Tensor,
    ):

        if value.numel() == 1:

            return to_jsonable(
                value.item()
            )

        return (
            value
            .detach()
            .cpu()
            .tolist()
        )

    if isinstance(
        value,
        dict,
    ):

        return {
            str(
                key
            ):
            to_jsonable(
                item
            )

            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):

        return [
            to_jsonable(
                item
            )
            for item in value
        ]

    return value


# ============================================================
# CHECKPOINT CONFIG HELPERS
# ============================================================

def get_checkpoint_config(
    checkpoint,
):

    config = checkpoint.get(
        "config",
        {},
    )

    if config is None:

        config = {}

    if not isinstance(
        config,
        dict,
    ):

        config = {}

    return config


def get_validation_metrics(
    checkpoint,
):

    metrics = checkpoint.get(
        "validation_metrics",
        {},
    )

    if metrics is None:

        metrics = {}

    if not isinstance(
        metrics,
        dict,
    ):

        metrics = {}

    return metrics


# ============================================================
# MODEL TYPE DETECTION
# ============================================================

def checkpoint_has_domain_head(
    state_dict,
):

    return any(
        str(
            key
        ).startswith(
            "domain_head."
        )
        for key in state_dict.keys()
    )


def infer_n_domain(
    state_dict,
    config,
):

    # Prefer explicit checkpoint configuration.

    if (
        "n_domain"
        in config
        and
        config[
            "n_domain"
        ]
        is not None
    ):

        return int(
            config[
                "n_domain"
            ]
        )

    # Fallback:
    # find the last domain-head Linear weight.
    #
    # Current GRL model:
    #
    # domain_head.0.weight : [128, 256]
    # domain_head.2.weight : [4, 128]

    candidate_weights = []

    for key, value in state_dict.items():

        if (
            str(
                key
            ).startswith(
                "domain_head."
            )
            and
            str(
                key
            ).endswith(
                ".weight"
            )
            and
            isinstance(
                value,
                torch.Tensor,
            )
            and
            value.ndim == 2
        ):

            candidate_weights.append(
                (
                    key,
                    value,
                )
            )

    if candidate_weights:

        # The final Linear layer normally has output dimension
        # equal to number of domains.

        key, tensor = candidate_weights[
            -1
        ]

        return int(
            tensor.shape[
                0
            ]
        )

    return 4


# ============================================================
# EER
# ============================================================

def compute_eer(
    y_true,
    fake_score,
):

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    fake_score = np.asarray(
        fake_score,
        dtype=np.float64,
    )

    if y_true.size == 0:

        return (
            float(
                "nan"
            ),
            float(
                "nan"
            ),
        )

    if np.unique(
        y_true
    ).size < 2:

        return (
            float(
                "nan"
            ),
            float(
                "nan"
            ),
        )

    if not np.isfinite(
        fake_score
    ).all():

        raise RuntimeError(
            "fake_score contains NaN/Inf."
        )

    fpr, tpr, thresholds = roc_curve(
        y_true,
        fake_score,
        pos_label=1,
    )

    fnr = (
        1.0
        - tpr
    )

    index = int(
        np.nanargmin(
            np.abs(
                fpr
                - fnr
            )
        )
    )

    eer = float(
        (
            fpr[
                index
            ]
            +
            fnr[
                index
            ]
        )
        / 2.0
    )

    threshold = float(
        thresholds[
            index
        ]
    )

    return (
        eer,
        threshold,
    )


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def compute_classification_metrics(
    y_true,
    fake_score,
    decision_threshold=0.5,
    validation_threshold=None,
):

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    fake_score = np.asarray(
        fake_score,
        dtype=np.float64,
    )

    result = {
        "n_samples":
            int(
                len(
                    y_true
                )
            ),

        "n_bonafide":
            int(
                (
                    y_true
                    == 0
                )
                .sum()
            ),

        "n_spoof":
            int(
                (
                    y_true
                    == 1
                )
                .sum()
            ),
    }

    if len(
        y_true
    ) == 0:

        result.update(
            {
                "accuracy":
                    float(
                        "nan"
                    ),

                "balanced_accuracy":
                    float(
                        "nan"
                    ),

                "f1":
                    float(
                        "nan"
                    ),

                "roc_auc":
                    float(
                        "nan"
                    ),

                "eer":
                    float(
                        "nan"
                    ),

                "eer_threshold":
                    float(
                        "nan"
                    ),

                "decision_threshold":
                    float(
                        decision_threshold
                    ),

                "validation_threshold":
                    (
                        float(
                            validation_threshold
                        )
                        if validation_threshold
                        is not None
                        else float(
                            "nan"
                        )
                    ),

                "acc_at_val":
                    float(
                        "nan"
                    ),

                "bacc_at_val":
                    float(
                        "nan"
                    ),

                "far_at_val":
                    float(
                        "nan"
                    ),

                "frr_at_val":
                    float(
                        "nan"
                    ),
            }
        )

        return result

    # ========================================================
    # DEFAULT 0.5 THRESHOLD
    # ========================================================

    predictions = (
        fake_score
        >= float(
            decision_threshold
        )
    ).astype(
        np.int64
    )

    result[
        "accuracy"
    ] = float(
        accuracy_score(
            y_true,
            predictions,
        )
    )

    if np.unique(
        y_true
    ).size >= 2:

        result[
            "balanced_accuracy"
        ] = float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        )

        result[
            "roc_auc"
        ] = float(
            roc_auc_score(
                y_true,
                fake_score,
            )
        )

        (
            eer,
            eer_threshold,
        ) = compute_eer(
            y_true,
            fake_score,
        )

        result[
            "eer"
        ] = eer

        result[
            "eer_threshold"
        ] = eer_threshold

    else:

        result[
            "balanced_accuracy"
        ] = float(
            "nan"
        )

        result[
            "roc_auc"
        ] = float(
            "nan"
        )

        result[
            "eer"
        ] = float(
            "nan"
        )

        result[
            "eer_threshold"
        ] = float(
            "nan"
        )

    result[
        "f1"
    ] = float(
        f1_score(
            y_true,
            predictions,
            zero_division=0,
        )
    )

    result[
        "decision_threshold"
    ] = float(
        decision_threshold
    )

    # ========================================================
    # VALIDATION THRESHOLD TRANSFER
    # ========================================================

    if (
        validation_threshold
        is not None
        and
        np.isfinite(
            float(
                validation_threshold
            )
        )
    ):

        val_threshold = float(
            validation_threshold
        )

        val_predictions = (
            fake_score
            >= val_threshold
        ).astype(
            np.int64
        )

        result[
            "validation_threshold"
        ] = val_threshold

        result[
            "acc_at_val"
        ] = float(
            accuracy_score(
                y_true,
                val_predictions,
            )
        )

        if np.unique(
            y_true
        ).size >= 2:

            result[
                "bacc_at_val"
            ] = float(
                balanced_accuracy_score(
                    y_true,
                    val_predictions,
                )
            )

        else:

            result[
                "bacc_at_val"
            ] = float(
                "nan"
            )

        # FAR:
        # bona fide incorrectly classified as spoof

        bona_mask = (
            y_true
            == 0
        )

        if bona_mask.any():

            result[
                "far_at_val"
            ] = float(
                (
                    val_predictions[
                        bona_mask
                    ]
                    == 1
                )
                .mean()
            )

        else:

            result[
                "far_at_val"
            ] = float(
                "nan"
            )

        # FRR:
        # spoof incorrectly classified as bona fide

        spoof_mask = (
            y_true
            == 1
        )

        if spoof_mask.any():

            result[
                "frr_at_val"
            ] = float(
                (
                    val_predictions[
                        spoof_mask
                    ]
                    == 0
                )
                .mean()
            )

        else:

            result[
                "frr_at_val"
            ] = float(
                "nan"
            )

    else:

        result[
            "validation_threshold"
        ] = float(
            "nan"
        )

        result[
            "acc_at_val"
        ] = float(
            "nan"
        )

        result[
            "bacc_at_val"
        ] = float(
            "nan"
        )

        result[
            "far_at_val"
        ] = float(
            "nan"
        )

        result[
            "frr_at_val"
        ] = float(
            "nan"
        )

    return result


# ============================================================
# TRANSFORM METRICS
# ============================================================

def compute_transform_metrics(
    condition_df,
):

    if condition_df.empty:

        return {
            "n_samples":
                0,

            "expected_transform":
                float(
                    "nan"
                ),

            "mean_replay_probability":
                float(
                    "nan"
                ),

            "median_replay_probability":
                float(
                    "nan"
                ),

            "transform_accuracy":
                float(
                    "nan"
                ),
        }

    condition_name = str(
        condition_df[
            "condition"
        ].iloc[
            0
        ]
    ).strip().lower()

    if condition_name == "clean":

        expected_transform = 0

    elif condition_name in {
        "simulated",
        "real",
    }:

        expected_transform = 1

    else:

        expected_transform = None

    scores = (
        condition_df[
            "transform_replay_score"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    predictions = (
        scores
        >= 0.5
    ).astype(
        np.int64
    )

    if expected_transform is None:

        accuracy = float(
            "nan"
        )

    else:

        accuracy = float(
            (
                predictions
                == expected_transform
            )
            .mean()
        )

    return {
        "n_samples":
            int(
                len(
                    condition_df
                )
            ),

        "expected_transform":
            (
                int(
                    expected_transform
                )
                if expected_transform
                is not None
                else float(
                    "nan"
                )
            ),

        "mean_replay_probability":
            float(
                np.mean(
                    scores
                )
            ),

        "median_replay_probability":
            float(
                np.median(
                    scores
                )
            ),

        "transform_accuracy":
            accuracy,
    }


# ============================================================
# BUILD METADATA LOOKUP
# ============================================================

def build_metadata_lookup(
    dataset,
):

    df = dataset.df.copy()

    if (
        "global_feature_id"
        not in df.columns
    ):

        raise RuntimeError(
            "Dataset manifest has no "
            "global_feature_id column."
        )

    if df[
        "global_feature_id"
    ].duplicated().any():

        duplicates = (
            df[
                df[
                    "global_feature_id"
                ].duplicated(
                    keep=False
                )
            ][
                "global_feature_id"
            ]
            .head(
                10
            )
            .tolist()
        )

        raise RuntimeError(
            "Duplicate global_feature_id found "
            "in test manifest:\n"
            f"{duplicates}"
        )

    return (
        df
        .set_index(
            "global_feature_id",
            drop=False,
        )
    )


# ============================================================
# TEST PROTOCOL AUDIT
# ============================================================

def audit_test_protocol(
    dataset,
):

    df = dataset.df.copy()

    print()

    print(
        "=" * 75
    )

    print(
        "TEST PROTOCOL AUDIT"
    )

    print(
        "=" * 75
    )

    print()

    print(
        f"TEST Samples      : "
        f"{len(df)}"
    )

    print()

    print(
        "Labels:"
    )

    print(
        df[
            "label"
        ]
        .value_counts()
        .sort_index()
    )

    print()

    print(
        "Conditions:"
    )

    print(
        df[
            "condition"
        ]
        .value_counts()
    )

    condition_lower = (
        df[
            "condition"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    clean_df = (
        df[
            condition_lower.eq(
                "clean"
            )
        ]
    )

    simulated_df = (
        df[
            condition_lower.eq(
                "simulated"
            )
        ]
    )

    real_df = (
        df[
            condition_lower.eq(
                "real"
            )
        ]
    )

    print()

    print(
        f"Clean test        : "
        f"{len(clean_df)}"
    )

    print(
        f"Simulated test    : "
        f"{len(simulated_df)}"
    )

    print(
        f"Real replay test  : "
        f"{len(real_df)}"
    )

    # ========================================================
    # SIMULATED TEST MUST BE UNSEEN
    # ========================================================

    if not simulated_df.empty:

        if (
            "rir_split"
            not in simulated_df.columns
        ):

            raise RuntimeError(
                "Simulated test samples are missing "
                "rir_split metadata."
            )

        rir_split = (
            simulated_df[
                "rir_split"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        print()

        print(
            "Simulated test rir_split:"
        )

        print(
            simulated_df[
                "rir_split"
            ]
            .value_counts(
                dropna=False
            )
        )

        if not rir_split.eq(
            "unseen"
        ).all():

            raise RuntimeError(
                "TEST PROTOCOL ERROR: "
                "simulated test contains non-unseen RIR."
            )

        if (
            "domain_label"
            in simulated_df.columns
        ):

            if not (
                simulated_df[
                    "domain_label"
                ]
                == -1
            ).all():

                raise RuntimeError(
                    "TEST PROTOCOL ERROR: "
                    "unseen simulated test received "
                    "valid seen-domain labels."
                )

        print()

        print(
            "Simulated unseen guard : PASSED"
        )

    # ========================================================
    # SOURCE SPLIT INTEGRITY
    # ========================================================

    if (
        "source_file_id"
        in df.columns
    ):

        missing_source = (
            df[
                "source_file_id"
            ]
            .isna()
            .sum()
        )

        if missing_source > 0:

            raise RuntimeError(
                "Missing source_file_id in test manifest: "
                f"{missing_source}"
            )

    print(
        "Test protocol audit   : PASSED"
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_model(
    model,
    loader,
    metadata_lookup,
    device,
):

    model.eval()

    rows = []

    progress = tqdm(
        loader,
        desc="EVALUATE",
        unit="batch",
        dynamic_ncols=True,
    )

    for batch in progress:

        features = (
            batch[
                "features"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        class_labels = (
            batch[
                "class_labels"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        transform_labels = (
            batch[
                "transform_labels"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        global_feature_ids = batch[
            "global_feature_ids"
        ]

        conditions = batch[
            "conditions"
        ]

        (
            class_logits,
            transform_logits,
            embedding,
        ) = model(
            features
        )

        class_probabilities = torch.softmax(
            class_logits,
            dim=1,
        )

        transform_probabilities = torch.softmax(
            transform_logits,
            dim=1,
        )

        fake_scores = (
            class_probabilities[
                :,
                1
            ]
            .detach()
            .cpu()
            .numpy()
        )

        transform_replay_scores = (
            transform_probabilities[
                :,
                1
            ]
            .detach()
            .cpu()
            .numpy()
        )

        class_predictions = (
            class_logits
            .argmax(
                dim=1
            )
            .detach()
            .cpu()
            .numpy()
        )

        transform_predictions = (
            transform_logits
            .argmax(
                dim=1
            )
            .detach()
            .cpu()
            .numpy()
        )

        # ====================================================
        # ROW-BY-ROW METADATA
        # ====================================================

        for i, global_feature_id in enumerate(
            global_feature_ids
        ):

            global_feature_id = str(
                global_feature_id
            )

            if (
                global_feature_id
                not in metadata_lookup.index
            ):

                raise RuntimeError(
                    "global_feature_id not found "
                    "in test manifest:\n"
                    f"{global_feature_id}"
                )

            meta = metadata_lookup.loc[
                global_feature_id
            ]

            row = {
                "global_feature_id":
                    global_feature_id,

                "source_file_id":
                    str(
                        meta[
                            "source_file_id"
                        ]
                    ),

                "label":
                    int(
                        class_labels[
                            i
                        ]
                    ),

                "fake_score":
                    float(
                        fake_scores[
                            i
                        ]
                    ),

                "class_prediction":
                    int(
                        class_predictions[
                            i
                        ]
                    ),

                "transform_label":
                    int(
                        transform_labels[
                            i
                        ]
                    ),

                "transform_replay_score":
                    float(
                        transform_replay_scores[
                            i
                        ]
                    ),

                "transform_prediction":
                    int(
                        transform_predictions[
                            i
                        ]
                    ),

                "condition":
                    str(
                        conditions[
                            i
                        ]
                    ),
            }

            # =================================================
            # OPTIONAL METADATA
            # =================================================

            optional_columns = [
                "feature_id",
                "recording_id",
                "speaker_id",
                "attack_id",
                "dataset",
                "split",
                "config_id",
                "room",
                "playback_device",
                "microphone",
                "rir_id",
                "rir_room",
                "rir_split",
                "snr_db",
            ]

            for column in optional_columns:

                if column in metadata_lookup.columns:

                    value = meta[
                        column
                    ]

                    if pd.isna(
                        value
                    ):

                        row[
                            column
                        ] = None

                    else:

                        row[
                            column
                        ] = value

            rows.append(
                row
            )

    scores_df = pd.DataFrame(
        rows
    )

    if scores_df.empty:

        raise RuntimeError(
            "Evaluation generated zero scores."
        )

    if len(
        scores_df
    ) != len(
        metadata_lookup
    ):

        raise RuntimeError(
            "Score count mismatch. "
            f"Generated {len(scores_df)} scores "
            f"for {len(metadata_lookup)} test samples."
        )

    if scores_df[
        "global_feature_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate global_feature_id "
            "generated during evaluation."
        )

    if not np.isfinite(
        scores_df[
            "fake_score"
        ].to_numpy(
            dtype=np.float64
        )
    ).all():

        raise RuntimeError(
            "Generated fake_score contains NaN/Inf."
        )

    if not np.isfinite(
        scores_df[
            "transform_replay_score"
        ].to_numpy(
            dtype=np.float64
        )
    ).all():

        raise RuntimeError(
            "Generated transform score "
            "contains NaN/Inf."
        )

    return scores_df


# ============================================================
# CONDITION SUMMARY
# ============================================================

def build_condition_summary(
    scores_df,
    validation_threshold,
):

    rows = []

    groups = [
        (
            "all_test",
            scores_df,
        )
    ]

    for condition in [
        "clean",
        "simulated",
        "real",
    ]:

        subset = (
            scores_df[
                scores_df[
                    "condition"
                ]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq(
                    condition
                )
            ]
            .copy()
        )

        groups.append(
            (
                condition,
                subset,
            )
        )

    for name, subset in groups:

        metrics = compute_classification_metrics(
            y_true=
                subset[
                    "label"
                ].to_numpy(),

            fake_score=
                subset[
                    "fake_score"
                ].to_numpy(),

            decision_threshold=
                0.5,

            validation_threshold=
                validation_threshold,
        )

        row = {
            "condition":
                name,
        }

        row.update(
            metrics
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# TRANSFORM SUMMARY
# ============================================================

def build_transform_summary(
    scores_df,
):

    rows = []

    for condition in [
        "clean",
        "simulated",
        "real",
    ]:

        subset = (
            scores_df[
                scores_df[
                    "condition"
                ]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq(
                    condition
                )
            ]
            .copy()
        )

        metrics = compute_transform_metrics(
            subset
        )

        row = {
            "condition":
                condition,
        }

        row.update(
            metrics
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate three-channel replay detector "
            "on held-out test conditions."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to best_model.pt."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Unique evaluation output directory. "
            "Example: grl_seed_42_ep5"
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # ARGUMENT QC
    # ========================================================

    if args.batch_size <= 0:

        raise ValueError(
            "--batch-size must be > 0."
        )

    if args.num_workers < 0:

        raise ValueError(
            "--num-workers must be >= 0."
        )

    # ========================================================
    # DEVICE
    # ========================================================

    device = get_device()

    checkpoint_path = resolve_path(
        args.checkpoint
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if (
        "model_state_dict"
        not in checkpoint
    ):

        raise RuntimeError(
            "Checkpoint has no model_state_dict."
        )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    config = get_checkpoint_config(
        checkpoint
    )

    validation_metrics = (
        get_validation_metrics(
            checkpoint
        )
    )

    checkpoint_epoch = int(
        checkpoint.get(
            "epoch",
            -1,
        )
    )

    seed = int(
        config.get(
            "seed",
            42,
        )
    )

    validation_eer = (
        validation_metrics.get(
            "eer",
            config.get(
                "validation_eer",
                float(
                    "nan"
                ),
            ),
        )
    )

    validation_threshold = (
        validation_metrics.get(
            "eer_threshold",
            config.get(
                "validation_eer_threshold",
                float(
                    "nan"
                ),
            ),
        )
    )

    validation_eer = float(
        validation_eer
    )

    validation_threshold = float(
        validation_threshold
    )

    # ========================================================
    # DETECT MODEL TYPE
    # ========================================================

    use_domain_head = (
        checkpoint_has_domain_head(
            state_dict
        )
    )

    if use_domain_head:

        model_type = (
            "ReplayResNet18_three_channel_GRL"
        )

    else:

        model_type = (
            "ReplayResNet18_three_channel_no_GRL"
        )

    embedding_dim = int(
        config.get(
            "embedding_dim",
            256,
        )
    )

    n_transform = int(
        config.get(
            "n_transform",
            2,
        )
    )

    n_domain = infer_n_domain(
        state_dict,
        config,
    )

    grl_strength = float(
        config.get(
            "grl_strength",
            1.0,
        )
    )

    # ========================================================
    # RUN NAME
    # ========================================================

    if args.run_name is not None:

        run_name = str(
            args.run_name
        ).strip()

        if not run_name:

            raise ValueError(
                "--run-name cannot be empty."
            )

    else:

        if use_domain_head:

            run_name = (
                f"grl_seed_{seed}"
            )

        else:

            run_name = (
                f"baseline_seed_{seed}"
            )

    result_dir = (
        RESULT_ROOT
        / run_name
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # HEADER
    # ========================================================

    print(
        "=" * 75
    )

    print(
        "THREE-CHANNEL REPLAY DETECTOR EVALUATION"
    )

    print(
        "=" * 75
    )

    print()

    print(
        f"Device                 : "
        f"{device}"
    )

    print(
        f"Checkpoint             : "
        f"{checkpoint_path}"
    )

    print(
        f"Checkpoint epoch       : "
        f"{checkpoint_epoch}"
    )

    print(
        f"Seed                   : "
        f"{seed}"
    )

    print(
        f"Model type             : "
        f"{model_type}"
    )

    print(
        f"Domain head            : "
        f"{use_domain_head}"
    )

    print(
        f"Embedding dim          : "
        f"{embedding_dim}"
    )

    if use_domain_head:

        print(
            f"Domain classes         : "
            f"{n_domain}"
        )

    print(
        f"Validation EER         : "
        f"{validation_eer:.6f}"
    )

    print(
        f"Validation threshold   : "
        f"{validation_threshold:.6f}"
    )

    print(
        f"Batch size             : "
        f"{args.batch_size}"
    )

    print(
        f"Run name               : "
        f"{run_name}"
    )

    print(
        f"Result directory       : "
        f"{result_dir}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = ReplayResNet(
        n_transform=
            n_transform,

        embedding_dim=
            embedding_dim,

        n_domain=
            n_domain,

        use_domain_head=
            use_domain_head,

        grl_strength=
            grl_strength,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(
        device
    )

    model.eval()

    print()

    print(
        "Checkpoint model load : PASSED"
    )

    # ========================================================
    # TEST DATASET
    # ========================================================

    test_dataset = ReplayDataset(
        split="test"
    )

    audit_test_protocol(
        test_dataset
    )

    metadata_lookup = (
        build_metadata_lookup(
            test_dataset
        )
    )

    print()

    print(
        f"Source ID lookup       : "
        f"{len(metadata_lookup)}"
    )

    print(
        "Source lookup guard    : PASSED"
    )

    # ========================================================
    # DATALOADER
    # ========================================================

    test_loader = DataLoader(
        test_dataset,

        batch_size=
            args.batch_size,

        shuffle=False,

        num_workers=
            args.num_workers,

        collate_fn=
            replay_collate_fn,

        pin_memory=(
            device.type
            == "cuda"
        ),

        drop_last=False,
    )

    # ========================================================
    # EVALUATE
    # ========================================================

    print()

    print(
        "=" * 75
    )

    print(
        "START TEST EVALUATION"
    )

    print(
        "=" * 75
    )

    start_time = time.time()

    scores_df = evaluate_model(
        model=
            model,

        loader=
            test_loader,

        metadata_lookup=
            metadata_lookup,

        device=
            device,
    )

    elapsed = (
        time.time()
        - start_time
    )

    print()

    print(
        f"Generated scores       : "
        f"{len(scores_df)}"
    )

    print(
        f"Evaluation time        : "
        f"{elapsed:.1f} s"
    )

    print(
        "Score generation guard : PASSED"
    )

    # ========================================================
    # ADD SEED / RUN INFO TO EACH SCORE ROW
    # ========================================================

    scores_df[
        "seed"
    ] = seed

    scores_df[
        "run_name"
    ] = run_name

    scores_df[
        "model"
    ] = model_type

    # ========================================================
    # CONDITION METRICS
    # ========================================================

    condition_summary = (
        build_condition_summary(
            scores_df=
                scores_df,

            validation_threshold=
                validation_threshold,
        )
    )

    # ========================================================
    # TRANSFORM METRICS
    # ========================================================

    transform_summary = (
        build_transform_summary(
            scores_df
        )
    )

    # ========================================================
    # TEST EER FOR RUN SUMMARY
    # ========================================================

    all_test_row = (
        condition_summary[
            condition_summary[
                "condition"
            ]
            .eq(
                "all_test"
            )
        ]
        .iloc[
            0
        ]
    )

    test_eer = float(
        all_test_row[
            "eer"
        ]
    )

    # ========================================================
    # RUN SUMMARY
    # ========================================================

    run_summary = pd.DataFrame(
        [
            {
                "model":
                    model_type,

                "run_name":
                    run_name,

                "seed":
                    seed,

                "best_epoch":
                    checkpoint_epoch,

                "validation_eer":
                    validation_eer,

                "validation_threshold":
                    validation_threshold,

                "test_eer":
                    test_eer,

                "checkpoint":
                    str(
                        checkpoint_path
                    ),

                "n_test_samples":
                    int(
                        len(
                            scores_df
                        )
                    ),

                "evaluation_time_s":
                    float(
                        elapsed
                    ),
            }
        ]
    )

    # ========================================================
    # SAVE FILES
    # ========================================================

    scores_path = (
        result_dir
        / "scores.csv"
    )

    condition_summary_path = (
        result_dir
        / "condition_summary.csv"
    )

    transform_summary_path = (
        result_dir
        / "transform_summary.csv"
    )

    run_summary_path = (
        result_dir
        / "run_summary.csv"
    )

    evaluation_config_path = (
        result_dir
        / "evaluation_config.json"
    )

    scores_df.to_csv(
        scores_path,
        index=False,
    )

    condition_summary.to_csv(
        condition_summary_path,
        index=False,
    )

    transform_summary.to_csv(
        transform_summary_path,
        index=False,
    )

    run_summary.to_csv(
        run_summary_path,
        index=False,
    )

    # ========================================================
    # EVALUATION CONFIG
    # ========================================================

    evaluation_config = {
        "run_name":
            run_name,

        "model":
            model_type,

        "checkpoint":
            str(
                checkpoint_path
            ),

        "checkpoint_epoch":
            checkpoint_epoch,

        "seed":
            seed,

        "batch_size":
            int(
                args.batch_size
            ),

        "num_workers":
            int(
                args.num_workers
            ),

        "device":
            str(
                device
            ),

        "use_domain_head":
            bool(
                use_domain_head
            ),

        "embedding_dim":
            embedding_dim,

        "n_transform":
            n_transform,

        "n_domain":
            (
                n_domain
                if use_domain_head
                else None
            ),

        "grl_strength":
            (
                grl_strength
                if use_domain_head
                else None
            ),

        "validation_eer":
            validation_eer,

        "validation_eer_threshold":
            validation_threshold,

        "test_samples":
            int(
                len(
                    scores_df
                )
            ),

        "evaluation_time_s":
            float(
                elapsed
            ),

        "checkpoint_training_config":
            config,
    }

    with open(
        evaluation_config_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            to_jsonable(
                evaluation_config
            ),
            file,
            indent=2,
        )

    # ========================================================
    # PRINT CLASSIFICATION SUMMARY
    # ========================================================

    print()

    print(
        "=" * 75
    )

    print(
        "CLASSIFICATION SUMMARY"
    )

    print(
        "=" * 75
    )

    print()

    display_columns = [
        "condition",
        "n_samples",
        "n_bonafide",
        "n_spoof",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "eer",
        "eer_threshold",
        "decision_threshold",
        "validation_threshold",
        "acc_at_val",
        "bacc_at_val",
        "far_at_val",
        "frr_at_val",
    ]

    print(
        condition_summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # PRINT TRANSFORM SUMMARY
    # ========================================================

    print()

    print(
        "=" * 75
    )

    print(
        "TRANSFORMATION SUMMARY"
    )

    print(
        "=" * 75
    )

    print()

    print(
        transform_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    print()

    print(
        "=" * 75
    )

    print(
        "EVALUATION COMPLETE"
    )

    print(
        "=" * 75
    )

    print()

    print(
        "scores.csv:"
    )

    print(
        scores_path
    )

    print()

    print(
        "condition_summary.csv:"
    )

    print(
        condition_summary_path
    )

    print()

    print(
        "transform_summary.csv:"
    )

    print(
        transform_summary_path
    )

    print()

    print(
        "run_summary.csv:"
    )

    print(
        run_summary_path
    )

    print()

    print(
        "evaluation_config.json:"
    )

    print(
        evaluation_config_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()