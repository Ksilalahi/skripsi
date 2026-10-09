from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

import torch

from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset_replay import (
    ReplayDataset,
    replay_collate_fn,
)

from model_replay import ReplayResNet

from metrics import (
    classification_metrics,
    threshold_metrics,
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "replay_resnet"
    / "seed_42"
    / "best_model.pt"
)

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
# NORMALIZE TEXT
# ============================================================

def normalize_text(
    series,
):

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    checkpoint_path,
    device,
):

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    required = {
        "epoch",
        "model_state_dict",
        "validation_metrics",
        "config",
    }

    missing = (
        required
        - set(
            checkpoint.keys()
        )
    )

    if missing:

        raise RuntimeError(
            "Checkpoint missing keys: "
            f"{sorted(missing)}"
        )

    return checkpoint


# ============================================================
# BUILD MODEL
# ============================================================

def build_model_from_checkpoint(
    checkpoint,
    device,
):

    config = checkpoint[
        "config"
    ]

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

    model = ReplayResNet(
        n_transform=
            n_transform,

        embedding_dim=
            embedding_dim,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    return model


# ============================================================
# TEST PROTOCOL AUDIT
# ============================================================

def audit_test_protocol(
    dataset,
):

    df = (
        dataset.df
        .copy()
    )

    condition = normalize_text(
        df[
            "condition"
        ]
    )

    print()

    print("=" * 78)

    print(
        "TEST PROTOCOL AUDIT"
    )

    print("=" * 78)

    print()

    print(
        f"Total test samples : "
        f"{len(df)}"
    )

    print()

    print(
        "Condition distribution:"
    )

    print(
        df[
            "condition"
        ]
        .value_counts(
            dropna=False
        )
    )

    # ========================================================
    # SIMULATED
    # ========================================================

    simulated = (
        df[
            condition.eq(
                "simulated"
            )
        ]
        .copy()
    )

    print()

    print(
        f"Simulated test samples : "
        f"{len(simulated)}"
    )

    if not simulated.empty:

        if (
            "rir_split"
            not in simulated.columns
        ):

            raise RuntimeError(
                "Test simulated samples have "
                "no rir_split metadata."
            )

        print()

        print(
            "Simulated test rir_split:"
        )

        print(
            simulated[
                "rir_split"
            ]
            .value_counts(
                dropna=False
            )
        )

        rir_split = normalize_text(
            simulated[
                "rir_split"
            ]
        )

        if not rir_split.eq(
            "unseen"
        ).all():

            bad = (
                simulated[
                    ~rir_split.eq(
                        "unseen"
                    )
                ]
            )

            raise RuntimeError(
                "TEST PROTOCOL ERROR: "
                "simulated test contains "
                "non-unseen RIR samples.\n"
                f"{bad.head(10)}"
            )

        print()

        print(
            "Simulated unseen-RIR guard : PASSED"
        )

    # ========================================================
    # REAL
    # ========================================================

    real = (
        df[
            condition.eq(
                "real"
            )
        ]
    )

    print()

    print(
        f"Controlled real replay test "
        f"samples : {len(real)}"
    )

    # ========================================================
    # CLEAN
    # ========================================================

    clean = (
        df[
            condition.eq(
                "clean"
            )
        ]
    )

    print(
        f"Clean test samples           : "
        f"{len(clean)}"
    )

    return df


# ============================================================
# BUILD SOURCE ID LOOKUP
# ============================================================

def build_source_id_map(
    test_df,
):

    required_lookup_columns = {
        "global_feature_id",
        "source_file_id",
    }

    missing_lookup_columns = (
        required_lookup_columns
        - set(
            test_df.columns
        )
    )

    if missing_lookup_columns:

        raise RuntimeError(
            "Test manifest missing columns "
            "required for source tracking: "
            f"{sorted(missing_lookup_columns)}"
        )

    # ========================================================
    # NULL QC
    # ========================================================

    if test_df[
        "global_feature_id"
    ].isna().any():

        missing_count = int(
            test_df[
                "global_feature_id"
            ]
            .isna()
            .sum()
        )

        raise RuntimeError(
            "Missing global_feature_id "
            "in test manifest: "
            f"{missing_count}"
        )

    if test_df[
        "source_file_id"
    ].isna().any():

        missing_count = int(
            test_df[
                "source_file_id"
            ]
            .isna()
            .sum()
        )

        raise RuntimeError(
            "Missing source_file_id "
            "in test manifest: "
            f"{missing_count}"
        )

    # ========================================================
    # DUPLICATE FEATURE ID QC
    # ========================================================

    if test_df[
        "global_feature_id"
    ].duplicated().any():

        duplicate_count = int(
            test_df[
                "global_feature_id"
            ]
            .duplicated()
            .sum()
        )

        raise RuntimeError(
            "Duplicate global_feature_id "
            "in test manifest: "
            f"{duplicate_count}"
        )

    # ========================================================
    # LOOKUP
    # ========================================================

    source_id_map = dict(
        zip(
            test_df[
                "global_feature_id"
            ]
            .astype(str),

            test_df[
                "source_file_id"
            ]
            .astype(str),
        )
    )

    if len(
        source_id_map
    ) != len(
        test_df
    ):

        raise RuntimeError(
            "Source ID lookup size mismatch: "
            f"lookup={len(source_id_map)}, "
            f"test={len(test_df)}"
        )

    print()

    print(
        f"Source ID lookup entries : "
        f"{len(source_id_map)}"
    )

    print(
        "Source ID lookup QC      : PASSED"
    )

    return source_id_map


# ============================================================
# EVALUATE LOADER
# ============================================================

@torch.no_grad()
def evaluate_loader(
    model,
    loader,
    device,
    seed,
    source_id_map,
):

    model.eval()

    rows = []

    progress = tqdm(
        loader,
        desc="EVALUATE",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )

    for batch in progress:

        # ====================================================
        # INPUT
        # ====================================================

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
            .to(
                device,
                non_blocking=True,
            )
        )

        transform_labels = (
            batch[
                "transform_labels"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        # ====================================================
        # FORWARD
        # ====================================================

        (
            class_logits,
            transform_logits,
            embeddings,
        ) = model(
            features
        )

        # ====================================================
        # CLASS PROBABILITY
        # ====================================================

        class_probabilities = (
            torch.softmax(
                class_logits,
                dim=1,
            )
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

        # ====================================================
        # TRANSFORM PROBABILITY
        # ====================================================

        transform_probabilities = (
            torch.softmax(
                transform_logits,
                dim=1,
            )
        )

        replay_scores = (
            transform_probabilities[
                :,
                1
            ]
            .detach()
            .cpu()
            .numpy()
        )

        # ====================================================
        # LABELS
        # ====================================================

        labels = (
            class_labels
            .detach()
            .cpu()
            .numpy()
        )

        transform_y = (
            transform_labels
            .detach()
            .cpu()
            .numpy()
        )

        # ====================================================
        # NUMERICAL QC
        # ====================================================

        if not np.isfinite(
            fake_scores
        ).all():

            raise RuntimeError(
                "Non-finite fake scores detected."
            )

        if not np.isfinite(
            replay_scores
        ).all():

            raise RuntimeError(
                "Non-finite replay scores detected."
            )

        if not torch.isfinite(
            embeddings
        ).all():

            raise RuntimeError(
                "Non-finite embeddings detected."
            )

        # ====================================================
        # METADATA
        # ====================================================

        conditions = batch[
            "conditions"
        ]

        feature_ids = batch[
            "global_feature_ids"
        ]

        # ====================================================
        # PER-FILE OUTPUT
        # ====================================================

        for index in range(
            len(labels)
        ):

            feature_id = str(
                feature_ids[
                    index
                ]
            )

            # =================================================
            # SOURCE ID LOOKUP
            # =================================================

            if (
                feature_id
                not in source_id_map
            ):

                raise RuntimeError(
                    "source_file_id mapping "
                    "not found for:\n"
                    f"{feature_id}"
                )

            source_file_id = str(
                source_id_map[
                    feature_id
                ]
            )

            condition = str(
                conditions[
                    index
                ]
            )

            # =================================================
            # SCORE ROW
            # =================================================

            rows.append(
                {
                    "file_id":
                        feature_id,

                    "source_file_id":
                        source_file_id,

                    "label":
                        int(
                            labels[
                                index
                            ]
                        ),

                    "fake_score":
                        float(
                            fake_scores[
                                index
                            ]
                        ),

                    "transform_label":
                        int(
                            transform_y[
                                index
                            ]
                        ),

                    "replay_score":
                        float(
                            replay_scores[
                                index
                            ]
                        ),

                    "condition":
                        condition,

                    "seed":
                        int(
                            seed
                        ),
                }
            )

    # ========================================================
    # OUTPUT QC
    # ========================================================

    if not rows:

        raise RuntimeError(
            "Evaluation produced zero scores."
        )

    scores_df = pd.DataFrame(
        rows
    )

    required_columns = {
        "file_id",
        "source_file_id",
        "label",
        "fake_score",
        "condition",
        "seed",
    }

    missing = (
        required_columns
        - set(
            scores_df.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Evaluation scores missing "
            "required columns: "
            f"{sorted(missing)}"
        )

    if scores_df[
        "file_id"
    ].duplicated().any():

        duplicate_count = int(
            scores_df[
                "file_id"
            ]
            .duplicated()
            .sum()
        )

        raise RuntimeError(
            "Duplicate file_id detected "
            f"in evaluation scores: "
            f"{duplicate_count}"
        )

    if not np.isfinite(
        scores_df[
            "fake_score"
        ].to_numpy(
            dtype=float
        )
    ).all():

        raise RuntimeError(
            "scores.csv contains "
            "non-finite fake_score."
        )

    if not np.isfinite(
        scores_df[
            "replay_score"
        ].to_numpy(
            dtype=float
        )
    ).all():

        raise RuntimeError(
            "scores.csv contains "
            "non-finite replay_score."
        )

    return scores_df


# ============================================================
# CONDITION SUMMARY
# ============================================================

def build_condition_summary(
    scores_df,
    validation_threshold,
):

    summaries = []

    groups = [
        (
            "all_test",
            scores_df,
        )
    ]

    for condition_name in [
        "clean",
        "simulated",
        "real",
    ]:

        subset = (
            scores_df[
                normalize_text(
                    scores_df[
                        "condition"
                    ]
                )
                .eq(
                    condition_name
                )
            ]
            .copy()
        )

        if not subset.empty:

            groups.append(
                (
                    condition_name,
                    subset,
                )
            )

    # ========================================================
    # METRICS
    # ========================================================

    for name, group in groups:

        labels = (
            group[
                "label"
            ]
            .astype(int)
            .to_numpy()
        )

        scores = (
            group[
                "fake_score"
            ]
            .astype(float)
            .to_numpy()
        )

        metrics = (
            classification_metrics(
                labels,
                scores,
                threshold=0.5,
            )
        )

        transferred = (
            threshold_metrics(
                labels,
                scores,
                threshold=
                    validation_threshold,
            )
        )

        row = {
            "condition":
                name,

            **metrics,

            "validation_threshold":
                float(
                    validation_threshold
                ),

            **transferred,
        }

        summaries.append(
            row
        )

    return pd.DataFrame(
        summaries
    )


# ============================================================
# TRANSFORMATION SUMMARY
# ============================================================

def build_transform_summary(
    scores_df,
):

    rows = []

    for condition_name in [
        "clean",
        "simulated",
        "real",
    ]:

        subset = (
            scores_df[
                normalize_text(
                    scores_df[
                        "condition"
                    ]
                )
                .eq(
                    condition_name
                )
            ]
            .copy()
        )

        if subset.empty:

            continue

        transform_labels = (
            subset[
                "transform_label"
            ]
            .astype(int)
            .to_numpy()
        )

        replay_scores = (
            subset[
                "replay_score"
            ]
            .astype(float)
            .to_numpy()
        )

        replay_predictions = (
            replay_scores
            >= 0.5
        ).astype(
            np.int64
        )

        accuracy = float(
            (
                replay_predictions
                == transform_labels
            )
            .mean()
        )

        rows.append(
            {
                "condition":
                    condition_name,

                "n_samples":
                    int(
                        len(
                            subset
                        )
                    ),

                "expected_transform_label":
                    int(
                        transform_labels[
                            0
                        ]
                    ),

                "mean_replay_score":
                    float(
                        replay_scores.mean()
                    ),

                "median_replay_score":
                    float(
                        np.median(
                            replay_scores
                        )
                    ),

                "transform_accuracy":
                    accuracy,
            }
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
            "Evaluate best ReplayResNet "
            "checkpoint on clean, simulated "
            "unseen-RIR, and controlled real "
            "replay test conditions."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            DEFAULT_CHECKPOINT
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
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

    checkpoint_path = Path(
        args.checkpoint
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = get_device()

    print("=" * 78)

    print(
        "THREE-CHANNEL REPLAY RESNET EVALUATION"
    )

    print("=" * 78)

    print()

    print(
        f"Device     : {device}"
    )

    print(
        f"Checkpoint : {checkpoint_path}"
    )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint = load_checkpoint(
        checkpoint_path,
        device,
    )

    checkpoint_epoch = int(
        checkpoint[
            "epoch"
        ]
    )

    config = checkpoint[
        "config"
    ]

    seed = int(
        config.get(
            "seed",
            42,
        )
    )

    validation_metrics = (
        checkpoint[
            "validation_metrics"
        ]
    )

    validation_eer = float(
        validation_metrics.get(
            "eer",
            float("nan"),
        )
    )

    validation_threshold = float(
        validation_metrics.get(
            "eer_threshold",
            0.5,
        )
    )

    print()

    print(
        f"Checkpoint epoch      : "
        f"{checkpoint_epoch}"
    )

    print(
        f"Seed                  : "
        f"{seed}"
    )

    print(
        f"Validation EER        : "
        f"{validation_eer:.6f}"
    )

    print(
        f"Validation threshold  : "
        f"{validation_threshold:.6f}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = build_model_from_checkpoint(
        checkpoint,
        device,
    )

    # ========================================================
    # TEST DATASET
    # ========================================================

    test_dataset = ReplayDataset(
        split="test"
    )

    test_df = audit_test_protocol(
        test_dataset
    )

    # ========================================================
    # SOURCE ID LOOKUP
    # ========================================================

    source_id_map = build_source_id_map(
        test_df
    )

    # ========================================================
    # TEST LOADER
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
    # EVALUATION
    # ========================================================

    print()

    print("=" * 78)

    print(
        "START TEST EVALUATION"
    )

    print("=" * 78)

    scores_df = evaluate_loader(
        model=
            model,

        loader=
            test_loader,

        device=
            device,

        seed=
            seed,

        source_id_map=
            source_id_map,
    )

    # ========================================================
    # SCORE COUNT QC
    # ========================================================

    if len(
        scores_df
    ) != len(
        test_df
    ):

        raise RuntimeError(
            "Test score count mismatch: "
            f"scores={len(scores_df)}, "
            f"dataset={len(test_df)}"
        )

    print()

    print(
        f"Generated scores : "
        f"{len(scores_df)}"
    )

    print(
        "Score count QC   : PASSED"
    )

    # ========================================================
    # OUTPUT DIR
    # ========================================================

    run_dir = (
        RESULT_ROOT
        / f"seed_{seed}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # SCORES.CSV
    # ========================================================

    scores_path = (
        run_dir
        / "scores.csv"
    )

    scores_df.to_csv(
        scores_path,
        index=False,
    )

    # ========================================================
    # CONDITION SUMMARY
    # ========================================================

    condition_summary = (
        build_condition_summary(
            scores_df,
            validation_threshold=
                validation_threshold,
        )
    )

    condition_summary_path = (
        run_dir
        / "condition_summary.csv"
    )

    condition_summary.to_csv(
        condition_summary_path,
        index=False,
    )

    # ========================================================
    # TRANSFORM SUMMARY
    # ========================================================

    transform_summary = (
        build_transform_summary(
            scores_df
        )
    )

    transform_summary_path = (
        run_dir
        / "transform_summary.csv"
    )

    transform_summary.to_csv(
        transform_summary_path,
        index=False,
    )

    # ========================================================
    # RUN SUMMARY
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

    run_summary = pd.DataFrame(
        [
            {
                "model":
                    "ReplayResNet18_three_channel",

                "seed":
                    seed,

                "best_epoch":
                    checkpoint_epoch,

                "validation_eer":
                    validation_eer,

                "validation_eer_threshold":
                    validation_threshold,

                "test_eer":
                    float(
                        all_test_row[
                            "eer"
                        ]
                    ),

                "test_auc":
                    float(
                        all_test_row[
                            "roc_auc"
                        ]
                    ),

                "test_balanced_accuracy":
                    float(
                        all_test_row[
                            "balanced_accuracy"
                        ]
                    ),
            }
        ]
    )

    run_summary_path = (
        run_dir
        / "run_summary.csv"
    )

    run_summary.to_csv(
        run_summary_path,
        index=False,
    )

    # ========================================================
    # EVALUATION CONFIG
    # ========================================================

    eval_config = {
        "checkpoint":
            str(
                checkpoint_path
            ),

        "checkpoint_epoch":
            checkpoint_epoch,

        "seed":
            seed,

        "batch_size":
            args.batch_size,

        "num_workers":
            args.num_workers,

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
    }

    with open(
        run_dir
        / "evaluation_config.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            eval_config,
            file,
            indent=2,
        )

    # ========================================================
    # PRINT CLASSIFICATION RESULTS
    # ========================================================

    print()

    print("=" * 78)

    print(
        "CLASSIFICATION RESULTS"
    )

    print("=" * 78)

    print()

    print(
        condition_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # PRINT TRANSFORMATION RESULTS
    # ========================================================

    print()

    print("=" * 78)

    print(
        "TRANSFORMATION RESULTS"
    )

    print("=" * 78)

    print()

    print(
        transform_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print("=" * 78)

    print(
        "EVALUATION COMPLETE"
    )

    print("=" * 78)

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
        run_dir
        / "evaluation_config.json"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()