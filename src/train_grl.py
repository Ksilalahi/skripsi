from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    Subset,
    WeightedRandomSampler,
)

from tqdm import tqdm

from dataset_replay import (
    ReplayDataset,
    replay_collate_fn,
    N_DOMAIN_CLASSES,
    UNKNOWN_DOMAIN,
)

from model_replay import ReplayResNet


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_ROOT = (
    ROOT
    / "checkpoints"
    / "replay_resnet_grl"
)

RESULT_ROOT = (
    ROOT
    / "results"
    / "replay_resnet_grl"
)


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_SEED = 42

DEFAULT_BATCH_SIZE = 8

DEFAULT_EPOCHS = 20

DEFAULT_LR = 1e-3

DEFAULT_WEIGHT_DECAY = 1e-4

DEFAULT_EMBEDDING_DIM = 256

DEFAULT_N_TRANSFORM = 2

DEFAULT_TRANSFORM_LOSS_WEIGHT = 0.10

DEFAULT_DOMAIN_LOSS_WEIGHT = 0.05

DEFAULT_GRL_STRENGTH = 1.0

DEFAULT_REPLAY_SAMPLING_BOOST = 8.0

DEFAULT_GRAD_CLIP = 5.0

DEFAULT_NUM_WORKERS = 0

DEFAULT_WEIGHTING = "sqrt_inv"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(
    seed,
):

    seed = int(
        seed
    )

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    if torch.backends.cudnn.is_available():

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


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
# TEXT NORMALIZATION
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
# CONDITION -> TRANSFORM LABEL
# ============================================================

def condition_to_transform_label(
    condition,
):

    condition = (
        str(
            condition
        )
        .strip()
        .lower()
    )

    if condition == "clean":

        return 0

    if condition in {
        "simulated",
        "real",
    }:

        return 1

    raise ValueError(
        f"Unknown condition: "
        f"{condition!r}"
    )


# ============================================================
# BUILD SEEN-RIR SUBSET
# ============================================================

def build_seen_subset(
    dataset,
    name,
):

    """
    GRL protocol:

    TRAIN / VALIDATION:
        clean
        +
        simulated replay with rir_split == seen

    EXCLUDED:
        real replay
        simulated replay with unseen RIR

    Domain labels:
        valid only for simulated-seen samples
    """

    df = (
        dataset
        .df
        .copy()
    )

    required_columns = {
        "condition",
        "rir_split",
        "domain_label",
        "label",
    }

    missing_columns = (
        required_columns
        - set(
            df.columns
        )
    )

    if missing_columns:

        raise RuntimeError(
            f"{name}: missing manifest columns: "
            f"{sorted(missing_columns)}"
        )

    conditions = normalize_text(
        df[
            "condition"
        ]
    )

    rir_split = normalize_text(
        df[
            "rir_split"
        ]
    )

    clean_mask = (
        conditions.eq(
            "clean"
        )
    )

    simulated_seen_mask = (
        conditions.eq(
            "simulated"
        )
        &
        rir_split.eq(
            "seen"
        )
    )

    allowed_mask = (
        clean_mask
        |
        simulated_seen_mask
    )

    indices = (
        np.flatnonzero(
            allowed_mask
            .to_numpy()
        )
        .tolist()
    )

    if not indices:

        raise RuntimeError(
            f"{name}: no samples remain "
            "after protocol filtering."
        )

    subset = Subset(
        dataset,
        indices,
    )

    subset_df = (
        df.loc[
            allowed_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print()

    print(
        "=" * 75
    )

    print(
        f"{name.upper()} USED DISTRIBUTION"
    )

    print(
        "=" * 75
    )

    print()

    print(
        f"Samples : "
        f"{len(subset_df)}"
    )

    print()

    print(
        "Class labels:"
    )

    print(
        subset_df[
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
        subset_df[
            "condition"
        ]
        .value_counts()
    )

    simulated = (
        subset_df[
            normalize_text(
                subset_df[
                    "condition"
                ]
            )
            .eq(
                "simulated"
            )
        ]
    )

    if not simulated.empty:

        print()

        print(
            "Simulated rir_split:"
        )

        print(
            simulated[
                "rir_split"
            ]
            .value_counts(
                dropna=False
            )
        )

        if (
            "rir_room"
            in simulated.columns
        ):

            print()

            print(
                "Simulated rir_room:"
            )

            print(
                simulated[
                    "rir_room"
                ]
                .value_counts(
                    dropna=False
                )
            )

    print()

    print(
        "Domain labels:"
    )

    print(
        subset_df[
            "domain_label"
        ]
        .value_counts()
        .sort_index()
    )

    n_valid_domain = int(
        (
            subset_df[
                "domain_label"
            ]
            >= 0
        )
        .sum()
    )

    print()

    print(
        f"Valid domain samples : "
        f"{n_valid_domain}"
    )

    return (
        subset,
        subset_df,
    )


# ============================================================
# PROTOCOL GUARD
# ============================================================

def validate_seen_protocol(
    train_df,
    validation_df,
):

    def inspect(
        df,
        split_name,
    ):

        conditions = normalize_text(
            df[
                "condition"
            ]
        )

        rir_split = normalize_text(
            df[
                "rir_split"
            ]
        )

        real_mask = (
            conditions.eq(
                "real"
            )
        )

        simulated_mask = (
            conditions.eq(
                "simulated"
            )
        )

        unseen_simulated_mask = (
            simulated_mask
            &
            ~rir_split.eq(
                "seen"
            )
        )

        invalid_condition_mask = (
            ~conditions.isin(
                [
                    "clean",
                    "simulated",
                ]
            )
        )

        valid_domain_mask = (
            df[
                "domain_label"
            ]
            .astype(
                int
            )
            >= 0
        )

        # ====================================================
        # VALID DOMAIN LABELS MUST ONLY BELONG
        # TO SIMULATED-SEEN SAMPLES
        # ====================================================

        if valid_domain_mask.any():

            valid_domain_conditions = (
                conditions[
                    valid_domain_mask
                ]
            )

            if not (
                valid_domain_conditions
                .eq(
                    "simulated"
                )
                .all()
            ):

                raise RuntimeError(
                    f"{split_name}: "
                    "valid domain label assigned "
                    "to non-simulated sample."
                )

            valid_domain_rir_split = (
                rir_split[
                    valid_domain_mask
                ]
            )

            if not (
                valid_domain_rir_split
                .eq(
                    "seen"
                )
                .all()
            ):

                raise RuntimeError(
                    f"{split_name}: "
                    "valid domain label assigned "
                    "to non-seen RIR."
                )

        # ====================================================
        # CLEAN SAMPLES MUST HAVE UNKNOWN DOMAIN LABEL
        # ====================================================

        clean_mask = (
            conditions.eq(
                "clean"
            )
        )

        clean_domain_labels = (
            df.loc[
                clean_mask,
                "domain_label",
            ]
            .astype(
                int
            )
        )

        if not (
            clean_domain_labels
            .eq(
                int(
                    UNKNOWN_DOMAIN
                )
            )
            .all()
        ):

            raise RuntimeError(
                f"{split_name}: "
                "clean samples must have "
                f"domain_label={UNKNOWN_DOMAIN}."
            )

        return {
            "real":
                int(
                    real_mask.sum()
                ),

            "unseen_simulated":
                int(
                    unseen_simulated_mask.sum()
                ),

            "invalid_condition":
                int(
                    invalid_condition_mask.sum()
                ),

            "valid_domain":
                int(
                    valid_domain_mask.sum()
                ),
        }

    train_stats = inspect(
        train_df,
        "train",
    )

    validation_stats = inspect(
        validation_df,
        "validation",
    )

    print()

    print(
        "Real replay training leakage     : "
        f"{'NONE' if train_stats['real'] == 0 else train_stats['real']}"
    )

    print(
        "Simulated unseen train leakage   : "
        f"{'NONE' if train_stats['unseen_simulated'] == 0 else train_stats['unseen_simulated']}"
    )

    print(
        "Real replay validation leakage   : "
        f"{'NONE' if validation_stats['real'] == 0 else validation_stats['real']}"
    )

    print(
        "Simulated unseen val leakage     : "
        f"{'NONE' if validation_stats['unseen_simulated'] == 0 else validation_stats['unseen_simulated']}"
    )

    protocol_failed = any(
        [
            train_stats[
                "real"
            ] != 0,

            train_stats[
                "unseen_simulated"
            ] != 0,

            train_stats[
                "invalid_condition"
            ] != 0,

            validation_stats[
                "real"
            ] != 0,

            validation_stats[
                "unseen_simulated"
            ] != 0,

            validation_stats[
                "invalid_condition"
            ] != 0,
        ]
    )

    if protocol_failed:

        raise RuntimeError(
            "Seen-RIR protocol guard FAILED."
        )

    # ========================================================
    # CURRENT DATASET-SPECIFIC DOMAIN COUNTS
    # ========================================================

    if (
        train_stats[
            "valid_domain"
        ]
        != 60
    ):

        raise RuntimeError(
            "Expected 60 valid train "
            "domain samples, got "
            f"{train_stats['valid_domain']}."
        )

    if (
        validation_stats[
            "valid_domain"
        ]
        != 45
    ):

        raise RuntimeError(
            "Expected 45 valid validation "
            "domain samples, got "
            f"{validation_stats['valid_domain']}."
        )

    print(
        "Seen-RIR protocol guard          : PASSED"
    )

    print(
        "Train valid domain guard         : "
        f"PASSED ({train_stats['valid_domain']})"
    )

    print(
        "Validation valid domain guard    : "
        f"PASSED ({validation_stats['valid_domain']})"
    )


# ============================================================
# LOSS WEIGHT COMPUTATION
# ============================================================

def compute_weights_from_counts(
    counts,
    mode,
):

    counts = np.asarray(
        counts,
        dtype=np.float64,
    )

    if (
        counts.ndim
        != 1
    ):

        raise ValueError(
            "counts must be 1D."
        )

    if np.any(
        counts <= 0
    ):

        raise ValueError(
            f"Invalid class counts: "
            f"{counts}"
        )

    mode = (
        str(
            mode
        )
        .strip()
        .lower()
    )

    if mode == "none":

        weights = (
            np.ones_like(
                counts
            )
        )

    elif mode == "inverse":

        weights = (
            1.0
            / counts
        )

    elif mode == "sqrt_inv":

        weights = (
            1.0
            / np.sqrt(
                counts
            )
        )

    else:

        raise ValueError(
            f"Unknown weighting mode: "
            f"{mode}"
        )

    weights = (
        weights
        / weights.mean()
    )

    return (
        weights
        .astype(
            np.float32
        )
    )


# ============================================================
# BUILD LOSS WEIGHTS
# ============================================================

def build_loss_weights(
    train_df,
    weighting,
    device,
):

    class_counts = (
        train_df[
            "label"
        ]
        .astype(
            int
        )
        .value_counts()
        .reindex(
            [
                0,
                1,
            ],
            fill_value=0,
        )
        .to_numpy()
    )

    transform_labels = (
        train_df[
            "condition"
        ]
        .apply(
            condition_to_transform_label
        )
        .astype(
            np.int64
        )
    )

    transform_counts = (
        transform_labels
        .value_counts()
        .reindex(
            [
                0,
                1,
            ],
            fill_value=0,
        )
        .to_numpy()
    )

    class_weights_np = (
        compute_weights_from_counts(
            class_counts,
            weighting,
        )
    )

    transform_weights_np = (
        compute_weights_from_counts(
            transform_counts,
            weighting,
        )
    )

    print()

    print(
        "=" * 75
    )

    print(
        "LOSS WEIGHTS"
    )

    print(
        "=" * 75
    )

    print()

    print(
        "Classification counts:"
    )

    print(
        class_counts
    )

    print(
        "Classification weights:"
    )

    print(
        class_weights_np
    )

    print()

    print(
        "Transformation counts:"
    )

    print(
        transform_counts
    )

    print(
        "Transformation weights:"
    )

    print(
        transform_weights_np
    )

    print()

    print(
        "Domain loss weighting:"
    )

    print(
        "plain CrossEntropyLoss "
        "(no class weighting)"
    )

    class_weights = torch.tensor(
        class_weights_np,
        dtype=torch.float32,
        device=device,
    )

    transform_weights = torch.tensor(
        transform_weights_np,
        dtype=torch.float32,
        device=device,
    )

    return (
        class_weights,
        transform_weights,
    )


# ============================================================
# TRAINING SAMPLER
# ============================================================

def build_training_sampler(
    train_df,
    replay_sampling_boost,
    seed,
):

    if (
        replay_sampling_boost
        < 1.0
    ):

        raise ValueError(
            "replay_sampling_boost "
            "must be >= 1."
        )

    conditions = normalize_text(
        train_df[
            "condition"
        ]
    )

    sample_weights = np.ones(
        len(
            train_df
        ),
        dtype=np.float64,
    )

    simulated_mask = (
        conditions
        .eq(
            "simulated"
        )
        .to_numpy()
    )

    sample_weights[
        simulated_mask
    ] = float(
        replay_sampling_boost
    )

    total_weight = float(
        sample_weights.sum()
    )

    replay_weight = float(
        sample_weights[
            simulated_mask
        ]
        .sum()
    )

    expected_replay_fraction = (
        replay_weight
        / total_weight
    )

    expected_replay_draws = (
        expected_replay_fraction
        * len(
            train_df
        )
    )

    generator = torch.Generator()

    generator.manual_seed(
        int(
            seed
        )
    )

    sampler = WeightedRandomSampler(
        weights=torch.tensor(
            sample_weights,
            dtype=torch.double,
        ),

        num_samples=len(
            train_df
        ),

        replacement=True,

        generator=generator,
    )

    print()

    print(
        "=" * 75
    )

    print(
        "TRAINING SAMPLER"
    )

    print(
        "=" * 75
    )

    print()

    print(
        f"Replay sampling boost       : "
        f"{float(replay_sampling_boost):.2f}x"
    )

    print(
        f"Expected replay fraction    : "
        f"{expected_replay_fraction:.4f}"
    )

    print(
        f"Expected replay draws/epoch : "
        f"{expected_replay_draws:.1f}"
    )

    print(
        f"Total draws/epoch           : "
        f"{len(train_df)}"
    )

    print()

    print(
        "NOTE: all simulated-seen samples "
        "also carry valid room-domain labels."
    )

    return (
        sampler,
        expected_replay_fraction,
        expected_replay_draws,
    )


# ============================================================
# BALANCED ACCURACY
# ============================================================

def balanced_accuracy(
    targets,
    predictions,
):

    targets = np.asarray(
        targets,
        dtype=np.int64,
    )

    predictions = np.asarray(
        predictions,
        dtype=np.int64,
    )

    if (
        targets.size
        == 0
    ):

        return float(
            "nan"
        )

    recalls = []

    for cls in np.unique(
        targets
    ):

        mask = (
            targets
            == cls
        )

        if mask.any():

            recall = float(
                (
                    predictions[
                        mask
                    ]
                    == cls
                )
                .mean()
            )

            recalls.append(
                recall
            )

    if not recalls:

        return float(
            "nan"
        )

    return float(
        np.mean(
            recalls
        )
    )


# ============================================================
# EER
# ============================================================

def compute_eer(
    labels,
    scores,
):

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    if (
        labels.size
        == 0
    ):

        return (
            float("nan"),
            float("nan"),
        )

    if (
        np.unique(
            labels
        )
        .size
        < 2
    ):

        return (
            float("nan"),
            float("nan"),
        )

    if not (
        np.isfinite(
            scores
        )
        .all()
    ):

        raise RuntimeError(
            "EER scores contain NaN/Inf."
        )

    order = (
        np.argsort(
            -scores
        )
    )

    labels_sorted = (
        labels[
            order
        ]
    )

    scores_sorted = (
        scores[
            order
        ]
    )

    n_bonafide = float(
        (
            labels
            == 0
        )
        .sum()
    )

    n_spoof = float(
        (
            labels
            == 1
        )
        .sum()
    )

    if (
        n_bonafide
        == 0
        or
        n_spoof
        == 0
    ):

        return (
            float("nan"),
            float("nan"),
        )

    true_positive = (
        np.cumsum(
            labels_sorted
            == 1
        )
        .astype(
            np.float64
        )
    )

    false_positive = (
        np.cumsum(
            labels_sorted
            == 0
        )
        .astype(
            np.float64
        )
    )

    tpr = (
        true_positive
        / n_spoof
    )

    fpr = (
        false_positive
        / n_bonafide
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
        scores_sorted[
            index
        ]
    )

    return (
        eer,
        threshold,
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    class_criterion,
    transform_criterion,
    domain_criterion,
    transform_loss_weight,
    domain_loss_weight,
    grl_strength,
    device,
    grad_clip,
    max_batches=None,
):

    model.train()

    total_samples = 0

    total_loss_sum = 0.0

    class_loss_sum = 0.0

    transform_loss_sum = 0.0

    domain_loss_sum = 0.0

    domain_valid_samples = 0

    class_correct = 0

    transform_correct = 0

    domain_correct = 0

    all_class_true = []

    all_class_pred = []

    all_transform_true = []

    all_transform_pred = []

    all_domain_true = []

    all_domain_pred = []

    sampled_clean = 0

    sampled_simulated = 0

    if (
        max_batches
        is None
    ):

        total_batches = len(
            loader
        )

    else:

        total_batches = min(
            len(
                loader
            ),
            max_batches,
        )

    progress = tqdm(
        loader,
        total=total_batches,
        desc="TRAIN-GRL",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )

    for (
        batch_index,
        batch,
    ) in enumerate(
        progress
    ):

        if (
            max_batches
            is not None
            and
            batch_index
            >= max_batches
        ):

            break

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

        domain_labels = (
            batch[
                "domain_labels"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        batch_size = int(
            features.shape[
                0
            ]
        )

        conditions = [
            str(
                condition
            )
            .strip()
            .lower()

            for condition
            in batch[
                "conditions"
            ]
        ]

        sampled_clean += sum(
            condition
            == "clean"

            for condition
            in conditions
        )

        sampled_simulated += sum(
            condition
            == "simulated"

            for condition
            in conditions
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        (
            class_logits,
            transform_logits,
            embedding,
        ) = model(
            features
        )

        loss_class = (
            class_criterion(
                class_logits,
                class_labels,
            )
        )

        loss_transform = (
            transform_criterion(
                transform_logits,
                transform_labels,
            )
        )

        # ====================================================
        # MASKED DOMAIN LOSS
        # ====================================================

        domain_mask = (
            domain_labels
            >= 0
        )

        n_valid_domain = int(
            domain_mask
            .sum()
            .item()
        )

        if (
            n_valid_domain
            > 0
        ):

            valid_embedding = (
                embedding[
                    domain_mask
                ]
            )

            valid_domain_targets = (
                domain_labels[
                    domain_mask
                ]
            )

            domain_logits = (
                model.forward_domain(
                    valid_embedding,
                    grl_strength=
                        grl_strength,
                )
            )

            loss_domain = (
                domain_criterion(
                    domain_logits,
                    valid_domain_targets,
                )
            )

            domain_predictions = (
                domain_logits
                .argmax(
                    dim=1
                )
            )

            domain_correct += int(
                (
                    domain_predictions
                    == valid_domain_targets
                )
                .sum()
                .item()
            )

            domain_valid_samples += (
                n_valid_domain
            )

            domain_loss_sum += (
                float(
                    loss_domain.item()
                )
                * n_valid_domain
            )

            all_domain_true.extend(
                valid_domain_targets
                .detach()
                .cpu()
                .tolist()
            )

            all_domain_pred.extend(
                domain_predictions
                .detach()
                .cpu()
                .tolist()
            )

        else:

            loss_domain = torch.zeros(
                (),
                dtype=
                    loss_class.dtype,
                device=
                    device,
            )

        # ====================================================
        # TOTAL LOSS
        # ====================================================

        loss = (
            loss_class
            +
            float(
                transform_loss_weight
            )
            * loss_transform
            +
            float(
                domain_loss_weight
            )
            * loss_domain
        )

        if not (
            torch.isfinite(
                loss
            )
        ):

            raise RuntimeError(
                "Training loss became NaN/Inf."
            )

        loss.backward()

        if (
            grad_clip
            > 0
        ):

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=
                    float(
                        grad_clip
                    ),
            )

        optimizer.step()

        class_predictions = (
            class_logits
            .argmax(
                dim=1
            )
        )

        transform_predictions = (
            transform_logits
            .argmax(
                dim=1
            )
        )

        class_correct += int(
            (
                class_predictions
                == class_labels
            )
            .sum()
            .item()
        )

        transform_correct += int(
            (
                transform_predictions
                == transform_labels
            )
            .sum()
            .item()
        )

        total_samples += (
            batch_size
        )

        total_loss_sum += (
            float(
                loss.item()
            )
            * batch_size
        )

        class_loss_sum += (
            float(
                loss_class.item()
            )
            * batch_size
        )

        transform_loss_sum += (
            float(
                loss_transform.item()
            )
            * batch_size
        )

        all_class_true.extend(
            class_labels
            .detach()
            .cpu()
            .tolist()
        )

        all_class_pred.extend(
            class_predictions
            .detach()
            .cpu()
            .tolist()
        )

        all_transform_true.extend(
            transform_labels
            .detach()
            .cpu()
            .tolist()
        )

        all_transform_pred.extend(
            transform_predictions
            .detach()
            .cpu()
            .tolist()
        )

        # ====================================================
        # SAFE PROGRESS VALUES
        # ====================================================

        class_acc_display = (
            class_correct
            / total_samples
        )

        transform_acc_display = (
            transform_correct
            / total_samples
        )

        if (
            domain_valid_samples
            > 0
        ):

            domain_acc_display = (
                domain_correct
                / domain_valid_samples
            )

        else:

            domain_acc_display = float(
                "nan"
            )

        progress.set_postfix(
            loss=
                f"{loss.item():.4f}",

            cls=
                f"{loss_class.item():.4f}",

            tr=
                f"{loss_transform.item():.4f}",

            dom=(
                f"{loss_domain.item():.4f}"
                if n_valid_domain > 0
                else "-"
            ),

            cAcc=
                f"{class_acc_display:.3f}",

            tAcc=
                f"{transform_acc_display:.3f}",

            dAcc=(
                f"{domain_acc_display:.3f}"
                if np.isfinite(
                    domain_acc_display
                )
                else "-"
            ),

            sim=
                sampled_simulated,

            domN=
                domain_valid_samples,
        )

    progress.close()

    if (
        total_samples
        == 0
    ):

        raise RuntimeError(
            "Training produced zero samples."
        )

    if (
        domain_valid_samples
        > 0
    ):

        train_domain_loss = (
            domain_loss_sum
            / domain_valid_samples
        )

        train_domain_accuracy = (
            domain_correct
            / domain_valid_samples
        )

        train_domain_balanced_accuracy = (
            balanced_accuracy(
                all_domain_true,
                all_domain_pred,
            )
        )

    else:

        train_domain_loss = float(
            "nan"
        )

        train_domain_accuracy = float(
            "nan"
        )

        train_domain_balanced_accuracy = float(
            "nan"
        )

    return {
        "loss":
            total_loss_sum
            / total_samples,

        "class_loss":
            class_loss_sum
            / total_samples,

        "transform_loss":
            transform_loss_sum
            / total_samples,

        "domain_loss":
            train_domain_loss,

        "class_accuracy":
            class_correct
            / total_samples,

        "class_balanced_accuracy":
            balanced_accuracy(
                all_class_true,
                all_class_pred,
            ),

        "transform_accuracy":
            transform_correct
            / total_samples,

        "transform_balanced_accuracy":
            balanced_accuracy(
                all_transform_true,
                all_transform_pred,
            ),

        "domain_accuracy":
            train_domain_accuracy,

        "domain_balanced_accuracy":
            train_domain_balanced_accuracy,

        "domain_valid_samples":
            int(
                domain_valid_samples
            ),

        "sampled_clean":
            int(
                sampled_clean
            ),

        "sampled_simulated":
            int(
                sampled_simulated
            ),

        "n_samples":
            int(
                total_samples
            ),
    }


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    class_criterion,
    transform_criterion,
    domain_criterion,
    transform_loss_weight,
    domain_loss_weight,
    grl_strength,
    device,
    max_batches=None,
):

    model.eval()

    total_samples = 0

    total_loss_sum = 0.0

    class_loss_sum = 0.0

    transform_loss_sum = 0.0

    domain_loss_sum = 0.0

    domain_valid_samples = 0

    class_correct = 0

    transform_correct = 0

    domain_correct = 0

    all_class_true = []

    all_class_pred = []

    all_fake_scores = []

    all_transform_true = []

    all_transform_pred = []

    all_domain_true = []

    all_domain_pred = []

    if (
        max_batches
        is None
    ):

        total_batches = len(
            loader
        )

    else:

        total_batches = min(
            len(
                loader
            ),
            max_batches,
        )

    progress = tqdm(
        loader,
        total=total_batches,
        desc="VALIDATION-GRL",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )

    for (
        batch_index,
        batch,
    ) in enumerate(
        progress
    ):

        if (
            max_batches
            is not None
            and
            batch_index
            >= max_batches
        ):

            break

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

        domain_labels = (
            batch[
                "domain_labels"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        batch_size = int(
            features.shape[
                0
            ]
        )

        (
            class_logits,
            transform_logits,
            embedding,
        ) = model(
            features
        )

        loss_class = (
            class_criterion(
                class_logits,
                class_labels,
            )
        )

        loss_transform = (
            transform_criterion(
                transform_logits,
                transform_labels,
            )
        )

        domain_mask = (
            domain_labels
            >= 0
        )

        n_valid_domain = int(
            domain_mask
            .sum()
            .item()
        )

        if (
            n_valid_domain
            > 0
        ):

            valid_embedding = (
                embedding[
                    domain_mask
                ]
            )

            valid_domain_targets = (
                domain_labels[
                    domain_mask
                ]
            )

            domain_logits = (
                model.forward_domain(
                    valid_embedding,
                    grl_strength=
                        grl_strength,
                )
            )

            loss_domain = (
                domain_criterion(
                    domain_logits,
                    valid_domain_targets,
                )
            )

            domain_predictions = (
                domain_logits
                .argmax(
                    dim=1
                )
            )

            domain_correct += int(
                (
                    domain_predictions
                    == valid_domain_targets
                )
                .sum()
                .item()
            )

            domain_valid_samples += (
                n_valid_domain
            )

            domain_loss_sum += (
                float(
                    loss_domain.item()
                )
                * n_valid_domain
            )

            all_domain_true.extend(
                valid_domain_targets
                .detach()
                .cpu()
                .tolist()
            )

            all_domain_pred.extend(
                domain_predictions
                .detach()
                .cpu()
                .tolist()
            )

        else:

            loss_domain = torch.zeros(
                (),
                dtype=
                    loss_class.dtype,
                device=
                    device,
            )

        loss = (
            loss_class
            +
            float(
                transform_loss_weight
            )
            * loss_transform
            +
            float(
                domain_loss_weight
            )
            * loss_domain
        )

        if not (
            torch.isfinite(
                loss
            )
        ):

            raise RuntimeError(
                "Validation loss became NaN/Inf."
            )

        class_predictions = (
            class_logits
            .argmax(
                dim=1
            )
        )

        transform_predictions = (
            transform_logits
            .argmax(
                dim=1
            )
        )

        fake_scores = (
            torch.softmax(
                class_logits,
                dim=1,
            )[
                :,
                1
            ]
        )

        class_correct += int(
            (
                class_predictions
                == class_labels
            )
            .sum()
            .item()
        )

        transform_correct += int(
            (
                transform_predictions
                == transform_labels
            )
            .sum()
            .item()
        )

        total_samples += (
            batch_size
        )

        total_loss_sum += (
            float(
                loss.item()
            )
            * batch_size
        )

        class_loss_sum += (
            float(
                loss_class.item()
            )
            * batch_size
        )

        transform_loss_sum += (
            float(
                loss_transform.item()
            )
            * batch_size
        )

        all_class_true.extend(
            class_labels
            .detach()
            .cpu()
            .tolist()
        )

        all_class_pred.extend(
            class_predictions
            .detach()
            .cpu()
            .tolist()
        )

        all_fake_scores.extend(
            fake_scores
            .detach()
            .cpu()
            .tolist()
        )

        all_transform_true.extend(
            transform_labels
            .detach()
            .cpu()
            .tolist()
        )

        all_transform_pred.extend(
            transform_predictions
            .detach()
            .cpu()
            .tolist()
        )

        class_acc_display = (
            class_correct
            / total_samples
        )

        transform_acc_display = (
            transform_correct
            / total_samples
        )

        if (
            domain_valid_samples
            > 0
        ):

            domain_acc_display = (
                domain_correct
                / domain_valid_samples
            )

        else:

            domain_acc_display = float(
                "nan"
            )

        progress.set_postfix(
            loss=
                f"{loss.item():.4f}",

            cls=
                f"{loss_class.item():.4f}",

            tr=
                f"{loss_transform.item():.4f}",

            dom=(
                f"{loss_domain.item():.4f}"
                if n_valid_domain > 0
                else "-"
            ),

            cAcc=
                f"{class_acc_display:.3f}",

            tAcc=
                f"{transform_acc_display:.3f}",

            dAcc=(
                f"{domain_acc_display:.3f}"
                if np.isfinite(
                    domain_acc_display
                )
                else "-"
            ),

            domN=
                domain_valid_samples,
        )

    progress.close()

    if (
        total_samples
        == 0
    ):

        raise RuntimeError(
            "Validation produced zero samples."
        )

    (
        eer,
        eer_threshold,
    ) = compute_eer(
        all_class_true,
        all_fake_scores,
    )

    if (
        domain_valid_samples
        > 0
    ):

        validation_domain_loss = (
            domain_loss_sum
            / domain_valid_samples
        )

        validation_domain_accuracy = (
            domain_correct
            / domain_valid_samples
        )

        validation_domain_balanced_accuracy = (
            balanced_accuracy(
                all_domain_true,
                all_domain_pred,
            )
        )

    else:

        validation_domain_loss = float(
            "nan"
        )

        validation_domain_accuracy = float(
            "nan"
        )

        validation_domain_balanced_accuracy = float(
            "nan"
        )

    return {
        "loss":
            total_loss_sum
            / total_samples,

        "class_loss":
            class_loss_sum
            / total_samples,

        "transform_loss":
            transform_loss_sum
            / total_samples,

        "domain_loss":
            validation_domain_loss,

        "class_accuracy":
            class_correct
            / total_samples,

        "class_balanced_accuracy":
            balanced_accuracy(
                all_class_true,
                all_class_pred,
            ),

        "transform_accuracy":
            transform_correct
            / total_samples,

        "transform_balanced_accuracy":
            balanced_accuracy(
                all_transform_true,
                all_transform_pred,
            ),

        "domain_accuracy":
            validation_domain_accuracy,

        "domain_balanced_accuracy":
            validation_domain_balanced_accuracy,

        "domain_valid_samples":
            int(
                domain_valid_samples
            ),

        "eer":
            float(
                eer
            ),

        "eer_threshold":
            float(
                eer_threshold
            ),

        "n_samples":
            int(
                total_samples
            ),
    }


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    validation_metrics,
    config,
):

    checkpoint = {
        "epoch":
            int(
                epoch
            ),

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "validation_metrics":
            validation_metrics,

        "config":
            config,
    }

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# SERIALIZABLE CONFIG
# ============================================================

def build_config_dict(
    args,
    run_name,
    smoke_test,
):

    return {
        "model":
            "ReplayResNet18_three_channel_GRL",

        "seed":
            int(
                args.seed
            ),

        "epochs":
            int(
                args.epochs
            ),

        "batch_size":
            int(
                args.batch_size
            ),

        "learning_rate":
            float(
                args.lr
            ),

        "weight_decay":
            float(
                args.wd
            ),

        "embedding_dim":
            int(
                args.embedding_dim
            ),

        "n_transform":
            int(
                args.n_transform
            ),

        "n_domain":
            int(
                N_DOMAIN_CLASSES
            ),

        "use_domain_head":
            True,

        "transform_loss_weight":
            float(
                args.transform_loss_weight
            ),

        "domain_loss_weight":
            float(
                args.domain_loss_weight
            ),

        "grl_strength":
            float(
                args.grl_strength
            ),

        "weighting":
            str(
                args.weighting
            ),

        "replay_sampling_boost":
            float(
                args.replay_sampling_boost
            ),

        "grad_clip":
            float(
                args.grad_clip
            ),

        "num_workers":
            int(
                args.num_workers
            ),

        "domain_mapping":
            {
                "lecture": 0,
                "meeting": 1,
                "phone": 2,
                "stairway": 3,
            },

        "unknown_domain":
            int(
                UNKNOWN_DOMAIN
            ),

        "domain_supervision":
            "simulated_seen_rir_room_only",

        "real_replay_training":
            False,

        "simulated_unseen_training":
            False,

        "run_name":
            str(
                run_name
            ),

        "smoke_test":
            bool(
                smoke_test
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train three-channel ReplayResNet "
            "with Gradient Reversal Layer and "
            "masked seen-RIR room-domain loss."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=
            DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=
            DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=
            DEFAULT_LR,
    )

    parser.add_argument(
        "--wd",
        type=float,
        default=
            DEFAULT_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=
            DEFAULT_SEED,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=
            DEFAULT_NUM_WORKERS,
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=
            DEFAULT_EMBEDDING_DIM,
    )

    parser.add_argument(
        "--n-transform",
        type=int,
        default=
            DEFAULT_N_TRANSFORM,
    )

    parser.add_argument(
        "--transform-loss-weight",
        type=float,
        default=
            DEFAULT_TRANSFORM_LOSS_WEIGHT,
    )

    parser.add_argument(
        "--domain-loss-weight",
        type=float,
        default=
            DEFAULT_DOMAIN_LOSS_WEIGHT,
    )

    parser.add_argument(
        "--grl-strength",
        type=float,
        default=
            DEFAULT_GRL_STRENGTH,
    )

    parser.add_argument(
        "--weighting",
        choices=[
            "none",
            "inverse",
            "sqrt_inv",
        ],
        default=
            DEFAULT_WEIGHTING,
    )

    parser.add_argument(
        "--replay-sampling-boost",
        type=float,
        default=
            DEFAULT_REPLAY_SAMPLING_BOOST,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=
            DEFAULT_GRAD_CLIP,
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
    )

    args = (
        parser
        .parse_args()
    )

    # ========================================================
    # ARGUMENT QC
    # ========================================================

    if (
        args.epochs
        <= 0
    ):

        raise ValueError(
            "--epochs must be > 0."
        )

    if (
        args.batch_size
        <= 0
    ):

        raise ValueError(
            "--batch-size must be > 0."
        )

    if (
        args.lr
        <= 0
    ):

        raise ValueError(
            "--lr must be > 0."
        )

    if (
        args.wd
        < 0
    ):

        raise ValueError(
            "--wd must be >= 0."
        )

    if (
        args.transform_loss_weight
        < 0
    ):

        raise ValueError(
            "--transform-loss-weight "
            "must be >= 0."
        )

    if (
        args.domain_loss_weight
        < 0
    ):

        raise ValueError(
            "--domain-loss-weight "
            "must be >= 0."
        )

    if (
        args.grl_strength
        < 0
    ):

        raise ValueError(
            "--grl-strength "
            "must be >= 0."
        )

    if (
        args.replay_sampling_boost
        < 1.0
    ):

        raise ValueError(
            "--replay-sampling-boost "
            "must be >= 1."
        )

    if (
        args.grad_clip
        < 0
    ):

        raise ValueError(
            "--grad-clip must be >= 0."
        )

    if (
        args.max_train_batches
        is not None
        and
        args.max_train_batches
        <= 0
    ):

        raise ValueError(
            "--max-train-batches "
            "must be > 0 when provided."
        )

    if (
        args.max_val_batches
        is not None
        and
        args.max_val_batches
        <= 0
    ):

        raise ValueError(
            "--max-val-batches "
            "must be > 0 when provided."
        )

    smoke_test = (
        args.max_train_batches
        is not None
        or
        args.max_val_batches
        is not None
    )

    set_seed(
        args.seed
    )

    device = (
        get_device()
    )

    # ========================================================
    # HEADER
    # ========================================================

    print(
        "=" * 75
    )

    print(
        "THREE-CHANNEL REPLAY RESNET + GRL TRAINING"
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
        f"Seed                   : "
        f"{args.seed}"
    )

    print(
        f"Epochs                 : "
        f"{args.epochs}"
    )

    print(
        f"Batch size             : "
        f"{args.batch_size}"
    )

    print(
        f"Learning rate          : "
        f"{args.lr}"
    )

    print(
        f"Weight decay           : "
        f"{args.wd}"
    )

    print(
        f"Loss weighting         : "
        f"{args.weighting}"
    )

    print(
        f"Transform loss weight  : "
        f"{args.transform_loss_weight}"
    )

    print(
        f"Domain loss weight     : "
        f"{args.domain_loss_weight}"
    )

    print(
        f"GRL strength           : "
        f"{args.grl_strength}"
    )

    print(
        f"Domain classes         : "
        f"{N_DOMAIN_CLASSES}"
    )

    print(
        f"Replay sampling boost  : "
        f"{args.replay_sampling_boost}"
    )

    print(
        f"Gradient clip          : "
        f"{args.grad_clip}"
    )

    print(
        f"Smoke test             : "
        f"{smoke_test}"
    )

    if (
        args.max_train_batches
        is not None
    ):

        print(
            f"Max train batches      : "
            f"{args.max_train_batches}"
        )

    if (
        args.max_val_batches
        is not None
    ):

        print(
            f"Max val batches        : "
            f"{args.max_val_batches}"
        )

    # ========================================================
    # DATASETS
    # ========================================================

    full_train_dataset = ReplayDataset(
        split=
            "train"
    )

    full_validation_dataset = ReplayDataset(
        split=
            "validation"
    )

    (
        train_dataset,
        train_df,
    ) = build_seen_subset(
        full_train_dataset,
        "train",
    )

    (
        validation_dataset,
        validation_df,
    ) = build_seen_subset(
        full_validation_dataset,
        "validation",
    )

    # ========================================================
    # PROTOCOL GUARD
    # ========================================================

    validate_seen_protocol(
        train_df,
        validation_df,
    )

    # ========================================================
    # SAMPLER
    # ========================================================

    (
        train_sampler,
        expected_replay_fraction,
        expected_replay_draws,
    ) = build_training_sampler(
        train_df,
        replay_sampling_boost=
            args.replay_sampling_boost,
        seed=
            args.seed,
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,

        batch_size=
            args.batch_size,

        sampler=
            train_sampler,

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

    validation_loader = DataLoader(
        validation_dataset,

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
    # MODEL
    # ========================================================
    #
    # IMPORTANT:
    #
    # model_replay.py sekarang default:
    #
    #     use_domain_head=False
    #
    # Karena ini train_grl.py,
    # kita HARUS mengaktifkannya secara eksplisit.
    # ========================================================

    model = ReplayResNet(
        n_transform=
            args.n_transform,

        embedding_dim=
            args.embedding_dim,

        n_domain=
            N_DOMAIN_CLASSES,

        use_domain_head=True,

        grl_strength=
            args.grl_strength,
    )

    # ========================================================
    # GRL ARCHITECTURE HARD GUARD
    # ========================================================

    if not (
        model.use_domain_head
    ):

        raise RuntimeError(
            "GRL PROTOCOL ERROR: "
            "use_domain_head must be True."
        )

    if (
        model.domain_head
        is None
    ):

        raise RuntimeError(
            "GRL PROTOCOL ERROR: "
            "domain_head was not initialized."
        )

    if (
        int(
            model.n_domain
        )
        !=
        int(
            N_DOMAIN_CLASSES
        )
    ):

        raise RuntimeError(
            "GRL PROTOCOL ERROR: "
            "domain class mismatch. "
            f"model={model.n_domain}, "
            f"dataset={N_DOMAIN_CLASSES}."
        )

    model = (
        model
        .to(
            device
        )
    )

    trainable_parameters = sum(
        parameter.numel()

        for parameter
        in model.parameters()

        if parameter.requires_grad
    )

    print()

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    print(
        "Domain head          : "
        "ENABLED (GRL guard PASSED)"
    )

    # ========================================================
    # LOSS WEIGHTS
    # ========================================================

    (
        class_weights,
        transform_weights,
    ) = build_loss_weights(
        train_df,
        weighting=
            args.weighting,
        device=
            device,
    )

    class_criterion = nn.CrossEntropyLoss(
        weight=
            class_weights
    )

    transform_criterion = nn.CrossEntropyLoss(
        weight=
            transform_weights
    )

    domain_criterion = nn.CrossEntropyLoss()

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=
            args.lr,
        weight_decay=
            args.wd,
    )

    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=
                args.epochs,
        )
    )

    # ========================================================
    # RUN NAME
    # ========================================================

    if (
        args.run_name
        is not None
    ):

        run_name = (
            str(
                args.run_name
            )
            .strip()
        )

        if not run_name:

            raise ValueError(
                "--run-name cannot be empty."
            )

    elif smoke_test:

        run_name = (
            f"seed_{args.seed}"
            "_smoke"
        )

    else:

        run_name = (
            f"seed_{args.seed}"
            f"_ep{args.epochs}"
        )

    checkpoint_dir = (
        CHECKPOINT_ROOT
        / run_name
    )

    result_dir = (
        RESULT_ROOT
        / run_name
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()

    print(
        f"Run name               : "
        f"{run_name}"
    )

    print(
        f"Checkpoint directory   : "
        f"{checkpoint_dir}"
    )

    print(
        f"Result directory       : "
        f"{result_dir}"
    )

    # ========================================================
    # CONFIG
    # ========================================================

    config = (
        build_config_dict(
            args,
            run_name,
            smoke_test,
        )
    )

    config[
        "train_samples_after_filter"
    ] = int(
        len(
            train_df
        )
    )

    config[
        "validation_samples_after_filter"
    ] = int(
        len(
            validation_df
        )
    )

    config[
        "train_valid_domain_samples"
    ] = int(
        (
            train_df[
                "domain_label"
            ]
            >= 0
        )
        .sum()
    )

    config[
        "validation_valid_domain_samples"
    ] = int(
        (
            validation_df[
                "domain_label"
            ]
            >= 0
        )
        .sum()
    )

    config[
        "expected_replay_fraction"
    ] = float(
        expected_replay_fraction
    )

    config[
        "expected_replay_draws_per_epoch"
    ] = float(
        expected_replay_draws
    )

    config_path = (
        result_dir
        / "training_config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=2,
        )

    # ========================================================
    # TRAINING STATE
    # ========================================================

    history = []

    best_validation_eer = float(
        "inf"
    )

    best_epoch = -1

    # ========================================================
    # START TRAINING
    # ========================================================

    print()

    print(
        "=" * 75
    )

    print(
        "START GRL TRAINING"
    )

    print(
        "=" * 75
    )

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        epoch_start_time = (
            time.time()
        )

        learning_rate = float(
            optimizer
            .param_groups[
                0
            ][
                "lr"
            ]
        )

        print()

        print(
            f"Epoch "
            f"{epoch:03d}/"
            f"{args.epochs:03d}"
        )

        print(
            f"Learning rate: "
            f"{learning_rate:.8f}"
        )

        # ====================================================
        # TRAIN
        # ====================================================

        train_metrics = train_one_epoch(
            model=
                model,

            loader=
                train_loader,

            optimizer=
                optimizer,

            class_criterion=
                class_criterion,

            transform_criterion=
                transform_criterion,

            domain_criterion=
                domain_criterion,

            transform_loss_weight=
                args.transform_loss_weight,

            domain_loss_weight=
                args.domain_loss_weight,

            grl_strength=
                args.grl_strength,

            device=
                device,

            grad_clip=
                args.grad_clip,

            max_batches=
                args.max_train_batches,
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        validation_metrics = validate_one_epoch(
            model=
                model,

            loader=
                validation_loader,

            class_criterion=
                class_criterion,

            transform_criterion=
                transform_criterion,

            domain_criterion=
                domain_criterion,

            transform_loss_weight=
                args.transform_loss_weight,

            domain_loss_weight=
                args.domain_loss_weight,

            grl_strength=
                args.grl_strength,

            device=
                device,

            max_batches=
                args.max_val_batches,
        )

        epoch_time = (
            time.time()
            - epoch_start_time
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        print()

        print(
            "=" * 75
        )

        print(
            f"EPOCH "
            f"{epoch:03d} "
            "SUMMARY"
        )

        print(
            "=" * 75
        )

        print()

        print(
            f"Time                  : "
            f"{epoch_time:.1f} s"
        )

        print(
            f"LR used               : "
            f"{learning_rate:.8f}"
        )

        print()

        print(
            f"Sampled clean         : "
            f"{train_metrics['sampled_clean']}"
        )

        print(
            f"Sampled simulated     : "
            f"{train_metrics['sampled_simulated']}"
        )

        print(
            f"Domain-valid draws    : "
            f"{train_metrics['domain_valid_samples']}"
        )

        print()

        print(
            f"Train loss            : "
            f"{train_metrics['loss']:.6f}"
        )

        print(
            f"Train class loss      : "
            f"{train_metrics['class_loss']:.6f}"
        )

        print(
            f"Train transform loss  : "
            f"{train_metrics['transform_loss']:.6f}"
        )

        print(
            f"Train domain loss     : "
            f"{train_metrics['domain_loss']:.6f}"
        )

        print(
            f"Train class Acc       : "
            f"{train_metrics['class_accuracy']:.4f}"
        )

        print(
            "Train class BAcc      : "
            f"{train_metrics['class_balanced_accuracy']:.4f}"
        )

        print(
            "Train transform Acc   : "
            f"{train_metrics['transform_accuracy']:.4f}"
        )

        print(
            "Train transform BAcc  : "
            f"{train_metrics['transform_balanced_accuracy']:.4f}"
        )

        print(
            f"Train domain Acc      : "
            f"{train_metrics['domain_accuracy']:.4f}"
        )

        print(
            "Train domain BAcc     : "
            f"{train_metrics['domain_balanced_accuracy']:.4f}"
        )

        print()

        print(
            f"Val loss              : "
            f"{validation_metrics['loss']:.6f}"
        )

        print(
            f"Val class loss        : "
            f"{validation_metrics['class_loss']:.6f}"
        )

        print(
            "Val transform loss    : "
            f"{validation_metrics['transform_loss']:.6f}"
        )

        print(
            f"Val domain loss       : "
            f"{validation_metrics['domain_loss']:.6f}"
        )

        print(
            f"Val class Acc         : "
            f"{validation_metrics['class_accuracy']:.4f}"
        )

        print(
            "Val class BAcc        : "
            f"{validation_metrics['class_balanced_accuracy']:.4f}"
        )

        print(
            "Val transform Acc     : "
            f"{validation_metrics['transform_accuracy']:.4f}"
        )

        print(
            "Val transform BAcc    : "
            f"{validation_metrics['transform_balanced_accuracy']:.4f}"
        )

        print(
            f"Val domain Acc        : "
            f"{validation_metrics['domain_accuracy']:.4f}"
        )

        print(
            "Val domain BAcc       : "
            f"{validation_metrics['domain_balanced_accuracy']:.4f}"
        )

        print(
            "Val domain samples    : "
            f"{validation_metrics['domain_valid_samples']}"
        )

        print(
            f"Val EER               : "
            f"{validation_metrics['eer']:.6f}"
        )

        print(
            "Val EER threshold     : "
            f"{validation_metrics['eer_threshold']:.6f}"
        )

        # ====================================================
        # HISTORY
        # ====================================================

        history_row = {
            "epoch":
                int(
                    epoch
                ),

            "epoch_seconds":
                float(
                    epoch_time
                ),

            "learning_rate":
                float(
                    learning_rate
                ),

            "train_loss":
                float(
                    train_metrics[
                        "loss"
                    ]
                ),

            "train_class_loss":
                float(
                    train_metrics[
                        "class_loss"
                    ]
                ),

            "train_transform_loss":
                float(
                    train_metrics[
                        "transform_loss"
                    ]
                ),

            "train_domain_loss":
                float(
                    train_metrics[
                        "domain_loss"
                    ]
                ),

            "train_class_accuracy":
                float(
                    train_metrics[
                        "class_accuracy"
                    ]
                ),

            "train_class_balanced_accuracy":
                float(
                    train_metrics[
                        "class_balanced_accuracy"
                    ]
                ),

            "train_transform_accuracy":
                float(
                    train_metrics[
                        "transform_accuracy"
                    ]
                ),

            "train_transform_balanced_accuracy":
                float(
                    train_metrics[
                        "transform_balanced_accuracy"
                    ]
                ),

            "train_domain_accuracy":
                float(
                    train_metrics[
                        "domain_accuracy"
                    ]
                ),

            "train_domain_balanced_accuracy":
                float(
                    train_metrics[
                        "domain_balanced_accuracy"
                    ]
                ),

            "train_domain_valid_samples":
                int(
                    train_metrics[
                        "domain_valid_samples"
                    ]
                ),

            "sampled_clean":
                int(
                    train_metrics[
                        "sampled_clean"
                    ]
                ),

            "sampled_simulated":
                int(
                    train_metrics[
                        "sampled_simulated"
                    ]
                ),

            "val_loss":
                float(
                    validation_metrics[
                        "loss"
                    ]
                ),

            "val_class_loss":
                float(
                    validation_metrics[
                        "class_loss"
                    ]
                ),

            "val_transform_loss":
                float(
                    validation_metrics[
                        "transform_loss"
                    ]
                ),

            "val_domain_loss":
                float(
                    validation_metrics[
                        "domain_loss"
                    ]
                ),

            "val_class_accuracy":
                float(
                    validation_metrics[
                        "class_accuracy"
                    ]
                ),

            "val_class_balanced_accuracy":
                float(
                    validation_metrics[
                        "class_balanced_accuracy"
                    ]
                ),

            "val_transform_accuracy":
                float(
                    validation_metrics[
                        "transform_accuracy"
                    ]
                ),

            "val_transform_balanced_accuracy":
                float(
                    validation_metrics[
                        "transform_balanced_accuracy"
                    ]
                ),

            "val_domain_accuracy":
                float(
                    validation_metrics[
                        "domain_accuracy"
                    ]
                ),

            "val_domain_balanced_accuracy":
                float(
                    validation_metrics[
                        "domain_balanced_accuracy"
                    ]
                ),

            "val_domain_valid_samples":
                int(
                    validation_metrics[
                        "domain_valid_samples"
                    ]
                ),

            "val_eer":
                float(
                    validation_metrics[
                        "eer"
                    ]
                ),

            "val_eer_threshold":
                float(
                    validation_metrics[
                        "eer_threshold"
                    ]
                ),
        }

        history.append(
            history_row
        )

        (
            pd.DataFrame(
                history
            )
            .to_csv(
                result_dir
                / "training_history.csv",
                index=False,
            )
        )

        # ====================================================
        # BEST CHECKPOINT
        # ====================================================

        current_eer = float(
            validation_metrics[
                "eer"
            ]
        )

        if (
            np.isfinite(
                current_eer
            )
            and
            current_eer
            < best_validation_eer
        ):

            best_validation_eer = (
                current_eer
            )

            best_epoch = int(
                epoch
            )

            best_checkpoint_path = (
                checkpoint_dir
                / "best_model.pt"
            )

            save_checkpoint(
                path=
                    best_checkpoint_path,

                model=
                    model,

                optimizer=
                    optimizer,

                scheduler=
                    scheduler,

                epoch=
                    epoch,

                validation_metrics=
                    validation_metrics,

                config=
                    config,
            )

            print()

            print(
                "Best checkpoint updated:"
            )

            print(
                best_checkpoint_path
            )

        # ====================================================
        # SCHEDULER
        # ====================================================

        scheduler.step()

        # ====================================================
        # LAST CHECKPOINT
        # ====================================================

        last_checkpoint_path = (
            checkpoint_dir
            / "last_model.pt"
        )

        save_checkpoint(
            path=
                last_checkpoint_path,

            model=
                model,

            optimizer=
                optimizer,

            scheduler=
                scheduler,

            epoch=
                epoch,

            validation_metrics=
                validation_metrics,

            config=
                config,
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print(
        "=" * 75
    )

    print(
        "GRL TRAINING COMPLETE"
    )

    print(
        "=" * 75
    )

    print()

    print(
        f"Run name       : "
        f"{run_name}"
    )

    if (
        best_epoch
        >= 0
    ):

        print(
            f"Best epoch     : "
            f"{best_epoch}"
        )

        print(
            f"Best val EER   : "
            f"{best_validation_eer:.6f}"
        )

        print()

        print(
            "Best checkpoint:"
        )

        print(
            checkpoint_dir
            / "best_model.pt"
        )

    else:

        print(
            "Best epoch     : N/A"
        )

        print(
            "Best val EER   : N/A"
        )

        print()

        print(
            "Best checkpoint : NOT CREATED"
        )

        print(
            "Reason          : "
            "validation EER was not available."
        )

    print()

    print(
        "Last checkpoint:"
    )

    print(
        checkpoint_dir
        / "last_model.pt"
    )

    print()

    print(
        "History:"
    )

    print(
        result_dir
        / "training_history.csv"
    )

    print()

    print(
        "Config:"
    )

    print(
        config_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()