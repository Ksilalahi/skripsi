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
)

from model_replay import ReplayResNet


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
    / "replay_resnet"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "replay_resnet"
)


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4

DEFAULT_TRANSFORM_LOSS_WEIGHT = 0.10

DEFAULT_NUM_WORKERS = 0
DEFAULT_REPLAY_SAMPLING_BOOST = 8.0
DEFAULT_WEIGHTING = "sqrt_inv"
DEFAULT_GRAD_CLIP = 5.0


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
# PARAMETER COUNT
# ============================================================

def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(series):
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


# ============================================================
# BUILD SEEN-RIR SUBSET
# ============================================================

def build_seen_subset(
    dataset,
    name,
):
    """
    Baseline protocol:

        clean
            allowed

        simulated + rir_split=seen
            allowed

        real
            forbidden

        simulated + rir_split=unseen
            forbidden
    """

    df = dataset.df.copy()

    required_columns = {
        "condition",
        "rir_split",
        "split",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{name}: manifest missing columns: "
            f"{sorted(missing)}"
        )

    condition = normalize_text(
        df["condition"]
    )

    rir_split = normalize_text(
        df["rir_split"]
    )

    clean_mask = (
        condition.eq(
            "clean"
        )
    )

    simulated_seen_mask = (
        condition.eq(
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
            allowed_mask.to_numpy()
        )
        .tolist()
    )

    if not indices:
        raise RuntimeError(
            f"{name}: no samples left "
            "after seen-RIR filtering."
        )

    subset = Subset(
        dataset,
        indices,
    )

    used_df = (
        df
        .iloc[indices]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return (
        subset,
        used_df,
    )


# ============================================================
# PROTOCOL HARD GUARD
# ============================================================

def validate_seen_protocol(
    dataframe,
    name,
):
    df = dataframe.copy()

    condition = normalize_text(
        df["condition"]
    )

    rir_split = normalize_text(
        df["rir_split"]
    )

    # ========================================================
    # REAL REPLAY
    # ========================================================

    real_count = int(
        condition.eq(
            "real"
        ).sum()
    )

    if real_count > 0:
        raise RuntimeError(
            f"{name}: REAL REPLAY LEAKAGE. "
            f"Found {real_count} rows."
        )

    # ========================================================
    # SIMULATED UNSEEN
    # ========================================================

    sim_unseen_mask = (
        condition.eq(
            "simulated"
        )
        &
        rir_split.eq(
            "unseen"
        )
    )

    sim_unseen_count = int(
        sim_unseen_mask.sum()
    )

    if sim_unseen_count > 0:
        raise RuntimeError(
            f"{name}: SIMULATED UNSEEN "
            f"RIR LEAKAGE. "
            f"Found {sim_unseen_count} rows."
        )

    # ========================================================
    # SIMULATED WITHOUT SEEN LABEL
    # ========================================================

    simulated_mask = (
        condition.eq(
            "simulated"
        )
    )

    simulated_not_seen = (
        simulated_mask
        &
        ~rir_split.eq(
            "seen"
        )
    )

    if simulated_not_seen.any():
        bad = (
            df[
                simulated_not_seen
            ][
                [
                    "global_feature_id",
                    "split",
                    "condition",
                    "rir_split",
                ]
            ]
        )

        raise RuntimeError(
            f"{name}: simulated rows without "
            "rir_split=seen:\n"
            f"{bad.head(10).to_string(index=False)}"
        )

    # ========================================================
    # UNKNOWN CONDITIONS
    # ========================================================

    allowed_conditions = {
        "clean",
        "simulated",
    }

    actual_conditions = set(
        condition.unique()
    )

    unknown = (
        actual_conditions
        - allowed_conditions
    )

    if unknown:
        raise RuntimeError(
            f"{name}: unknown conditions: "
            f"{sorted(unknown)}"
        )


# ============================================================
# CONDITION -> TRANSFORM LABEL
# ============================================================

def condition_to_transform_label(
    condition,
):
    condition = (
        str(condition)
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
        f"Unknown condition: {condition}"
    )


# ============================================================
# CLASS WEIGHT COMPUTATION
# ============================================================

def compute_weights_from_counts(
    counts,
    mode="sqrt_inv",
):
    counts = np.asarray(
        counts,
        dtype=np.float64,
    )

    if counts.ndim != 1:
        raise ValueError(
            "counts must be 1D."
        )

    if np.any(
        counts <= 0
    ):
        raise ValueError(
            f"Invalid class counts: {counts}"
        )

    if mode == "none":
        weights = np.ones_like(
            counts
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
            f"Unknown weighting mode: {mode}"
        )

    weights = (
        weights
        / weights.mean()
    )

    return weights.astype(
        np.float32
    )


# ============================================================
# BUILD LOSS WEIGHTS
# ============================================================

def build_loss_weights(
    train_df,
    mode,
    device,
):
    class_counts = (
        train_df[
            "label"
        ]
        .astype(int)
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
            mode=mode,
        )
    )

    transform_weights_np = (
        compute_weights_from_counts(
            transform_counts,
            mode=mode,
        )
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

    return {
        "class_counts":
            class_counts,

        "transform_counts":
            transform_counts,

        "class_weights":
            class_weights,

        "transform_weights":
            transform_weights,
    }


# ============================================================
# MODERATE REPLAY OVERSAMPLING
# ============================================================

def build_training_sampler(
    train_df,
    replay_sampling_boost,
    seed,
):
    if replay_sampling_boost < 1.0:
        raise ValueError(
            "replay_sampling_boost must be >= 1."
        )

    conditions = (
        train_df[
            "condition"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    weights = np.ones(
        len(train_df),
        dtype=np.float64,
    )

    replay_mask = (
        conditions.eq(
            "simulated"
        )
        .to_numpy()
    )

    weights[
        replay_mask
    ] = float(
        replay_sampling_boost
    )

    total_weight = float(
        weights.sum()
    )

    expected_replay_fraction = float(
        weights[
            replay_mask
        ].sum()
        / total_weight
    )

    expected_replay_draws = float(
        expected_replay_fraction
        * len(train_df)
    )

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            weights,
            dtype=torch.double,
        ),
        num_samples=len(
            train_df
        ),
        replacement=True,
        generator=generator,
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
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.int64,
    )

    recalls = []

    for label in np.unique(
        y_true
    ):
        mask = (
            y_true
            == label
        )

        if mask.sum() == 0:
            continue

        recall = float(
            (
                y_pred[
                    mask
                ]
                == label
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
# COMPUTE EER
# ============================================================

def compute_eer(
    y_true,
    fake_scores,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    scores = np.asarray(
        fake_scores,
        dtype=np.float64,
    )

    if len(
        np.unique(
            y_true
        )
    ) < 2:
        return (
            float("nan"),
            float("nan"),
        )

    if not np.isfinite(
        scores
    ).all():
        raise RuntimeError(
            "Non-finite scores in EER."
        )

    order = np.argsort(
        -scores
    )

    y_sorted = (
        y_true[
            order
        ]
    )

    score_sorted = (
        scores[
            order
        ]
    )

    positives = float(
        (
            y_true
            == 1
        ).sum()
    )

    negatives = float(
        (
            y_true
            == 0
        ).sum()
    )

    tp = np.cumsum(
        y_sorted
        == 1
    ).astype(
        np.float64
    )

    fp = np.cumsum(
        y_sorted
        == 0
    ).astype(
        np.float64
    )

    tpr = (
        tp
        / positives
    )

    fpr = (
        fp
        / negatives
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
        score_sorted[
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
    transform_loss_weight,
    device,
    grad_clip,
    max_batches=None,
):
    model.train()

    total_loss_sum = 0.0
    class_loss_sum = 0.0
    transform_loss_sum = 0.0

    total_samples = 0

    class_correct = 0
    transform_correct = 0

    all_class_true = []
    all_class_pred = []

    all_transform_true = []
    all_transform_pred = []

    seen_conditions = {
        "clean": 0,
        "simulated": 0,
    }

    if max_batches is not None:
        total_batches = min(
            len(loader),
            max_batches,
        )
    else:
        total_batches = len(
            loader
        )

    progress = tqdm(
        loader,
        total=total_batches,
        desc="TRAIN",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )

    for batch_index, batch in enumerate(
        progress
    ):
        if (
            max_batches is not None
            and
            batch_index >= max_batches
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

        batch_size = int(
            features.shape[
                0
            ]
        )

        for condition in batch[
            "conditions"
        ]:
            condition = (
                str(condition)
                .strip()
                .lower()
            )

            if condition in seen_conditions:
                seen_conditions[
                    condition
                ] += 1

        optimizer.zero_grad(
            set_to_none=True
        )

        (
            class_logits,
            transform_logits,
            _,
        ) = model(
            features
        )

        loss_cls = (
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

        loss = (
            loss_cls
            +
            transform_loss_weight
            * loss_transform
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                "Training loss became NaN/Inf."
            )

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

        optimizer.step()

        class_pred = (
            class_logits
            .argmax(
                dim=1
            )
        )

        transform_pred = (
            transform_logits
            .argmax(
                dim=1
            )
        )

        class_correct += int(
            (
                class_pred
                == class_labels
            )
            .sum()
            .item()
        )

        transform_correct += int(
            (
                transform_pred
                == transform_labels
            )
            .sum()
            .item()
        )

        total_loss_sum += (
            float(
                loss.item()
            )
            * batch_size
        )

        class_loss_sum += (
            float(
                loss_cls.item()
            )
            * batch_size
        )

        transform_loss_sum += (
            float(
                loss_transform.item()
            )
            * batch_size
        )

        total_samples += (
            batch_size
        )

        all_class_true.extend(
            class_labels
            .detach()
            .cpu()
            .tolist()
        )

        all_class_pred.extend(
            class_pred
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
            transform_pred
            .detach()
            .cpu()
            .tolist()
        )

        running_loss = (
            total_loss_sum
            / total_samples
        )

        running_class_acc = (
            class_correct
            / total_samples
        )

        running_transform_acc = (
            transform_correct
            / total_samples
        )

        progress.set_postfix(
            loss=
                f"{running_loss:.4f}",

            cls=
                f"{loss_cls.item():.4f}",

            tr=
                f"{loss_transform.item():.4f}",

            cAcc=
                f"{running_class_acc:.3f}",

            tAcc=
                f"{running_transform_acc:.3f}",

            sim=
                seen_conditions[
                    "simulated"
                ],
        )

    progress.close()

    if total_samples == 0:
        raise RuntimeError(
            "Training produced zero samples."
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

        "sampled_clean":
            seen_conditions[
                "clean"
            ],

        "sampled_simulated":
            seen_conditions[
                "simulated"
            ],

        "samples_seen":
            total_samples,
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
    transform_loss_weight,
    device,
    max_batches=None,
):
    model.eval()

    total_loss_sum = 0.0
    class_loss_sum = 0.0
    transform_loss_sum = 0.0

    total_samples = 0

    class_correct = 0
    transform_correct = 0

    all_class_true = []
    all_class_pred = []

    all_transform_true = []
    all_transform_pred = []

    all_fake_scores = []

    if max_batches is not None:
        total_batches = min(
            len(loader),
            max_batches,
        )
    else:
        total_batches = len(
            loader
        )

    progress = tqdm(
        loader,
        total=total_batches,
        desc="VALIDATION",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )

    for batch_index, batch in enumerate(
        progress
    ):
        if (
            max_batches is not None
            and
            batch_index >= max_batches
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

        batch_size = int(
            features.shape[
                0
            ]
        )

        (
            class_logits,
            transform_logits,
            _,
        ) = model(
            features
        )

        loss_cls = (
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

        loss = (
            loss_cls
            +
            transform_loss_weight
            * loss_transform
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                "Validation loss became NaN/Inf."
            )

        probability = torch.softmax(
            class_logits,
            dim=1,
        )

        fake_score = (
            probability[
                :,
                1
            ]
        )

        class_pred = (
            class_logits
            .argmax(
                dim=1
            )
        )

        transform_pred = (
            transform_logits
            .argmax(
                dim=1
            )
        )

        class_correct += int(
            (
                class_pred
                == class_labels
            )
            .sum()
            .item()
        )

        transform_correct += int(
            (
                transform_pred
                == transform_labels
            )
            .sum()
            .item()
        )

        total_loss_sum += (
            float(
                loss.item()
            )
            * batch_size
        )

        class_loss_sum += (
            float(
                loss_cls.item()
            )
            * batch_size
        )

        transform_loss_sum += (
            float(
                loss_transform.item()
            )
            * batch_size
        )

        total_samples += (
            batch_size
        )

        all_class_true.extend(
            class_labels
            .cpu()
            .tolist()
        )

        all_class_pred.extend(
            class_pred
            .cpu()
            .tolist()
        )

        all_transform_true.extend(
            transform_labels
            .cpu()
            .tolist()
        )

        all_transform_pred.extend(
            transform_pred
            .cpu()
            .tolist()
        )

        all_fake_scores.extend(
            fake_score
            .cpu()
            .tolist()
        )

        running_loss = (
            total_loss_sum
            / total_samples
        )

        running_class_acc = (
            class_correct
            / total_samples
        )

        running_transform_acc = (
            transform_correct
            / total_samples
        )

        progress.set_postfix(
            loss=
                f"{running_loss:.4f}",

            cls=
                f"{loss_cls.item():.4f}",

            tr=
                f"{loss_transform.item():.4f}",

            cAcc=
                f"{running_class_acc:.3f}",

            tAcc=
                f"{running_transform_acc:.3f}",
        )

    progress.close()

    if total_samples == 0:
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

        "eer":
            eer,

        "eer_threshold":
            eer_threshold,

        "samples_seen":
            total_samples,
    }


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    validation_metrics,
    args,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
            (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),

        "validation_metrics":
            validation_metrics,

        "config":
            vars(args),
    }

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# PRINT DISTRIBUTION
# ============================================================

def print_distribution(
    name,
    dataframe,
):
    df = dataframe.copy()

    print()
    print(
        "=" * 75
    )

    print(
        f"{name} DISTRIBUTION"
    )

    print(
        "=" * 75
    )

    print()

    print(
        f"Samples : "
        f"{len(df)}"
    )

    print()

    print(
        "Class labels:"
    )

    print(
        df[
            "label"
        ]
        .astype(int)
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
        .sort_index()
    )

    simulated = (
        df[
            normalize_text(
                df[
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


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train baseline three-channel "
            "ReplayResNet18 with seen-RIR protocol "
            "and progress bars."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--n-transform",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--transform-loss-weight",
        type=float,
        default=DEFAULT_TRANSFORM_LOSS_WEIGHT,
    )

    parser.add_argument(
        "--weighting",
        choices=[
            "none",
            "inverse",
            "sqrt_inv",
        ],
        default=DEFAULT_WEIGHTING,
    )

    parser.add_argument(
        "--replay-sampling-boost",
        type=float,
        default=DEFAULT_REPLAY_SAMPLING_BOOST,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=DEFAULT_GRAD_CLIP,
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

    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be > 0."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be > 0."
        )

    if args.lr <= 0:
        raise ValueError(
            "--lr must be > 0."
        )

    if args.weight_decay < 0:
        raise ValueError(
            "--weight-decay must be >= 0."
        )

    if args.transform_loss_weight < 0:
        raise ValueError(
            "--transform-loss-weight must be >= 0."
        )

    if args.replay_sampling_boost < 1.0:
        raise ValueError(
            "--replay-sampling-boost must be >= 1."
        )

    if args.grad_clip < 0:
        raise ValueError(
            "--grad-clip must be >= 0."
        )

    is_smoke_test = (
        args.max_train_batches is not None
        or
        args.max_val_batches is not None
    )

    set_seed(
        args.seed
    )

    device = get_device()

    print(
        "=" * 75
    )

    print(
        "THREE-CHANNEL REPLAY RESNET TRAINING V2"
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
        f"{args.weight_decay}"
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
        f"Replay sampling boost  : "
        f"{args.replay_sampling_boost}"
    )

    print(
        f"Gradient clip          : "
        f"{args.grad_clip}"
    )

    print(
        f"Smoke test             : "
        f"{is_smoke_test}"
    )

    if args.max_train_batches is not None:
        print(
            f"Max train batches      : "
            f"{args.max_train_batches}"
        )

    if args.max_val_batches is not None:
        print(
            f"Max val batches        : "
            f"{args.max_val_batches}"
        )

    # ========================================================
    # DATASETS
    # ========================================================

    full_train_dataset = ReplayDataset(
        split="train"
    )

    full_val_dataset = ReplayDataset(
        split="validation"
    )

    # ========================================================
    # FILTER TO SEEN-RIR PROTOCOL
    # ========================================================

    (
        train_dataset,
        train_df,
    ) = build_seen_subset(
        full_train_dataset,
        name="TRAIN",
    )

    (
        val_dataset,
        val_df,
    ) = build_seen_subset(
        full_val_dataset,
        name="VALIDATION",
    )

    # ========================================================
    # HARD PROTOCOL GUARD
    # ========================================================

    validate_seen_protocol(
        train_df,
        name="TRAIN",
    )

    validate_seen_protocol(
        val_df,
        name="VALIDATION",
    )

    print_distribution(
        "TRAIN USED",
        train_df,
    )

    print_distribution(
        "VALIDATION USED",
        val_df,
    )

    print()

    print(
        "Real replay training leakage     : NONE"
    )

    print(
        "Simulated unseen train leakage   : NONE"
    )

    print(
        "Real replay validation leakage   : NONE"
    )

    print(
        "Simulated unseen val leakage     : NONE"
    )

    print(
        "Seen-RIR protocol guard          : PASSED"
    )

    # ========================================================
    # TRAINING SAMPLER
    # ========================================================

    (
        train_sampler,
        expected_replay_fraction,
        expected_replay_draws,
    ) = build_training_sampler(
        train_df=
            train_df,

        replay_sampling_boost=
            args.replay_sampling_boost,

        seed=
            args.seed,
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
        f"{args.replay_sampling_boost:.2f}x"
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

    # ========================================================
    # DATALOADERS
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

    val_loader = DataLoader(
        val_dataset,

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
    # Baseline MUST NOT instantiate the domain head.
    #
    # This prevents train.py from silently inheriting
    # use_domain_head=True from model defaults.
    # ========================================================

    model = ReplayResNet(
        n_transform=
            args.n_transform,

        embedding_dim=
            args.embedding_dim,

        use_domain_head=False,
    )

    # ========================================================
    # BASELINE ARCHITECTURE HARD GUARD
    # ========================================================

    if model.use_domain_head:
        raise RuntimeError(
            "BASELINE PROTOCOL ERROR: "
            "use_domain_head must be False."
        )

    if model.domain_head is not None:
        raise RuntimeError(
            "BASELINE PROTOCOL ERROR: "
            "domain_head must be None."
        )

    model = model.to(
        device
    )

    print()

    print(
        f"Trainable parameters : "
        f"{count_parameters(model):,}"
    )

    print(
        "Domain head            : "
        "DISABLED (baseline guard PASSED)"
    )

    # ========================================================
    # LOSS WEIGHTS
    # ========================================================

    loss_info = build_loss_weights(
        train_df=
            train_df,

        mode=
            args.weighting,

        device=
            device,
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
        loss_info[
            "class_counts"
        ]
    )

    print(
        "Classification weights:"
    )

    print(
        loss_info[
            "class_weights"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    print()

    print(
        "Transformation counts:"
    )

    print(
        loss_info[
            "transform_counts"
        ]
    )

    print(
        "Transformation weights:"
    )

    print(
        loss_info[
            "transform_weights"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    # ========================================================
    # CRITERIA
    # ========================================================

    class_criterion = nn.CrossEntropyLoss(
        weight=
            loss_info[
                "class_weights"
            ]
    )

    transform_criterion = nn.CrossEntropyLoss(
        weight=
            loss_info[
                "transform_weights"
            ]
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
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

    if is_smoke_test:
        run_name = (
            f"seed_{args.seed}"
            f"_smoke"
        )
    else:
        run_name = (
            f"seed_{args.seed}"
        )

    checkpoint_dir = (
        CHECKPOINT_DIR
        / run_name
    )

    result_dir = (
        RESULT_DIR
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
    # SAVE CONFIG
    # ========================================================

    config = vars(
        args
    ).copy()

    config[
        "model"
    ] = (
        "ReplayResNet18_three_channel_no_GRL"
    )

    config[
        "use_domain_head"
    ] = False

    config[
        "is_smoke_test"
    ] = bool(
        is_smoke_test
    )

    config[
        "run_name"
    ] = run_name

    config[
        "train_samples_after_protocol_filter"
    ] = int(
        len(train_df)
    )

    config[
        "validation_samples_after_protocol_filter"
    ] = int(
        len(val_df)
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

    best_val_eer = float(
        "inf"
    )

    best_epoch = -1

    print()

    print(
        "=" * 75
    )

    print(
        "START TRAINING"
    )

    print(
        "=" * 75
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        epoch_start = time.time()

        epoch_lr = float(
            optimizer.param_groups[
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
            f"{epoch_lr:.8f}"
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

            transform_loss_weight=
                args.transform_loss_weight,

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

        val_metrics = validate_one_epoch(
            model=
                model,

            loader=
                val_loader,

            class_criterion=
                class_criterion,

            transform_criterion=
                transform_criterion,

            transform_loss_weight=
                args.transform_loss_weight,

            device=
                device,

            max_batches=
                args.max_val_batches,
        )

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        # ====================================================
        # HISTORY
        # ====================================================

        row = {
            "epoch":
                epoch,

            "learning_rate":
                epoch_lr,

            "epoch_seconds":
                epoch_seconds,

            "train_samples_seen":
                train_metrics[
                    "samples_seen"
                ],

            "train_sampled_clean":
                train_metrics[
                    "sampled_clean"
                ],

            "train_sampled_simulated":
                train_metrics[
                    "sampled_simulated"
                ],

            "train_loss":
                train_metrics[
                    "loss"
                ],

            "train_class_loss":
                train_metrics[
                    "class_loss"
                ],

            "train_transform_loss":
                train_metrics[
                    "transform_loss"
                ],

            "train_class_accuracy":
                train_metrics[
                    "class_accuracy"
                ],

            "train_class_balanced_accuracy":
                train_metrics[
                    "class_balanced_accuracy"
                ],

            "train_transform_accuracy":
                train_metrics[
                    "transform_accuracy"
                ],

            "train_transform_balanced_accuracy":
                train_metrics[
                    "transform_balanced_accuracy"
                ],

            "val_samples_seen":
                val_metrics[
                    "samples_seen"
                ],

            "val_loss":
                val_metrics[
                    "loss"
                ],

            "val_class_loss":
                val_metrics[
                    "class_loss"
                ],

            "val_transform_loss":
                val_metrics[
                    "transform_loss"
                ],

            "val_class_accuracy":
                val_metrics[
                    "class_accuracy"
                ],

            "val_class_balanced_accuracy":
                val_metrics[
                    "class_balanced_accuracy"
                ],

            "val_transform_accuracy":
                val_metrics[
                    "transform_accuracy"
                ],

            "val_transform_balanced_accuracy":
                val_metrics[
                    "transform_balanced_accuracy"
                ],

            "val_eer":
                val_metrics[
                    "eer"
                ],

            "val_eer_threshold":
                val_metrics[
                    "eer_threshold"
                ],
        }

        history.append(
            row
        )

        history_df = pd.DataFrame(
            history
        )

        history_df.to_csv(
            result_dir
            / "training_history.csv",
            index=False,
        )

        # ====================================================
        # EPOCH SUMMARY
        # ====================================================

        print()

        print(
            "=" * 75
        )

        print(
            f"EPOCH "
            f"{epoch:03d} "
            f"SUMMARY"
        )

        print(
            "=" * 75
        )

        print()

        print(
            f"Time                  : "
            f"{epoch_seconds:.1f} s"
        )

        print(
            f"LR used               : "
            f"{epoch_lr:.8f}"
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
            f"Train class Acc       : "
            f"{train_metrics['class_accuracy']:.4f}"
        )

        print(
            f"Train class BAcc      : "
            f"{train_metrics['class_balanced_accuracy']:.4f}"
        )

        print(
            f"Train transform Acc   : "
            f"{train_metrics['transform_accuracy']:.4f}"
        )

        print(
            f"Train transform BAcc  : "
            f"{train_metrics['transform_balanced_accuracy']:.4f}"
        )

        print()

        print(
            f"Val loss              : "
            f"{val_metrics['loss']:.6f}"
        )

        print(
            f"Val class loss        : "
            f"{val_metrics['class_loss']:.6f}"
        )

        print(
            f"Val transform loss    : "
            f"{val_metrics['transform_loss']:.6f}"
        )

        print(
            f"Val class Acc         : "
            f"{val_metrics['class_accuracy']:.4f}"
        )

        print(
            f"Val class BAcc        : "
            f"{val_metrics['class_balanced_accuracy']:.4f}"
        )

        print(
            f"Val transform Acc     : "
            f"{val_metrics['transform_accuracy']:.4f}"
        )

        print(
            f"Val transform BAcc    : "
            f"{val_metrics['transform_balanced_accuracy']:.4f}"
        )

        print(
            f"Val EER               : "
            f"{val_metrics['eer']:.6f}"
        )

        print(
            f"Val EER threshold     : "
            f"{val_metrics['eer_threshold']:.6f}"
        )

        # ====================================================
        # BEST MODEL
        # ====================================================

        val_eer = float(
            val_metrics[
                "eer"
            ]
        )

        if (
            np.isfinite(
                val_eer
            )
            and
            val_eer
            < best_val_eer
        ):
            best_val_eer = (
                val_eer
            )

            best_epoch = (
                epoch
            )

            best_path = (
                checkpoint_dir
                / "best_model.pt"
            )

            save_checkpoint(
                path=
                    best_path,

                model=
                    model,

                optimizer=
                    optimizer,

                scheduler=
                    scheduler,

                epoch=
                    epoch,

                validation_metrics=
                    val_metrics,

                args=
                    args,
            )

            print()

            print(
                "Best checkpoint updated:"
            )

            print(
                best_path
            )

        # ====================================================
        # SCHEDULER
        # ====================================================

        scheduler.step()

        # ====================================================
        # LAST MODEL
        # ====================================================

        last_path = (
            checkpoint_dir
            / "last_model.pt"
        )

        save_checkpoint(
            path=
                last_path,

            model=
                model,

            optimizer=
                optimizer,

            scheduler=
                scheduler,

            epoch=
                epoch,

            validation_metrics=
                val_metrics,

            args=
                args,
        )

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print()

    print(
        "=" * 75
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 75
    )

    print()

    print(
        f"Run name       : "
        f"{run_name}"
    )

    if best_epoch >= 0:
        print(
            f"Best epoch     : "
            f"{best_epoch}"
        )

        print(
            f"Best val EER   : "
            f"{best_val_eer:.6f}"
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
            "Reason          : validation EER "
            "was not available."
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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()