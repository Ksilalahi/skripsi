from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd

import torch

from torch.utils.data import (
    DataLoader,
    Subset,
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from dataset_replay import (
    ReplayDataset,
    replay_collate_fn,
    DOMAIN_MAP,
    DOMAIN_NAMES,
    N_DOMAIN_CLASSES,
)

from model_replay import ReplayResNet


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULT_ROOT = (
    ROOT
    / "results"
    / "domain_probe"
)


# ============================================================
# DEFAULT CHECKPOINTS
# ============================================================

DEFAULT_BASELINE_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "replay_resnet"
    / "seed_42"
    / "best_model.pt"
)

DEFAULT_GRL_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "replay_resnet_grl"
    / "seed_42_ep5"
    / "best_model.pt"
)


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
# PATH
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
# CHECKPOINT HELPERS
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

        (
            _,
            tensor,
        ) = candidate_weights[
            -1
        ]

        return int(
            tensor.shape[
                0
            ]
        )

    return N_DOMAIN_CLASSES


# ============================================================
# LOAD FROZEN MODEL
# ============================================================

def load_frozen_model(
    checkpoint_path,
    device,
):

    checkpoint_path = resolve_path(
        checkpoint_path
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

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
            "Checkpoint missing model_state_dict:\n"
            f"{checkpoint_path}"
        )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    config = get_checkpoint_config(
        checkpoint
    )

    use_domain_head = (
        checkpoint_has_domain_head(
            state_dict
        )
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

    # ========================================================
    # FREEZE EVERYTHING
    # ========================================================

    for parameter in model.parameters():

        parameter.requires_grad = False

    return {
        "model":
            model,

        "checkpoint":
            checkpoint,

        "checkpoint_path":
            checkpoint_path,

        "config":
            config,

        "use_domain_head":
            use_domain_head,

        "embedding_dim":
            embedding_dim,

        "epoch":
            int(
                checkpoint.get(
                    "epoch",
                    -1,
                )
            ),
    }


# ============================================================
# DOMAIN SUBSET
# ============================================================

def build_domain_subset(
    dataset,
    split_name,
):

    df = dataset.df.copy()

    if (
        "domain_label"
        not in df.columns
    ):

        raise RuntimeError(
            "dataset has no domain_label column."
        )

    valid_mask = (
        df[
            "domain_label"
        ]
        >= 0
    )

    valid_indices = (
        np.flatnonzero(
            valid_mask.to_numpy()
        )
        .tolist()
    )

    domain_df = (
        df.loc[
            valid_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if domain_df.empty:

        raise RuntimeError(
            f"No valid domain samples in {split_name}."
        )

    # ========================================================
    # PROTOCOL GUARDS
    # ========================================================

    conditions = (
        domain_df[
            "condition"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if not conditions.eq(
        "simulated"
    ).all():

        raise RuntimeError(
            f"{split_name}: non-simulated sample "
            "received valid domain label."
        )

    rir_split = (
        domain_df[
            "rir_split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if not rir_split.eq(
        "seen"
    ).all():

        raise RuntimeError(
            f"{split_name}: non-seen RIR "
            "received valid domain label."
        )

    subset = Subset(
        dataset,
        valid_indices,
    )

    print()

    print(
        "=" * 75
    )

    print(
        f"DOMAIN PROBE DATA: "
        f"{split_name.upper()}"
    )

    print(
        "=" * 75
    )

    print()

    print(
        f"Samples : "
        f"{len(domain_df)}"
    )

    print()

    print(
        "Domain distribution:"
    )

    print(
        domain_df[
            "domain_label"
        ]
        .value_counts()
        .sort_index()
    )

    print()

    print(
        "Room distribution:"
    )

    print(
        domain_df[
            "rir_room"
        ]
        .value_counts()
    )

    return (
        subset,
        domain_df,
    )


# ============================================================
# EXTRACT EMBEDDINGS
# ============================================================

@torch.no_grad()
def extract_embeddings(
    model,
    loader,
    device,
    description,
):

    model.eval()

    embeddings = []

    labels = []

    feature_ids = []

    rooms = []

    from tqdm import tqdm

    progress = tqdm(
        loader,
        desc=description,
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

        domain_labels = (
            batch[
                "domain_labels"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        if np.any(
            domain_labels
            < 0
        ):

            raise RuntimeError(
                "Domain probe loader received "
                "masked domain label."
            )

        # ====================================================
        # FROZEN REPRESENTATION
        # ====================================================

        embedding = model.encode(
            features
        )

        embedding = (
            embedding
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        embeddings.append(
            embedding
        )

        labels.append(
            domain_labels
        )

        feature_ids.extend(
            [
                str(
                    item
                )
                for item in batch[
                    "global_feature_ids"
                ]
            ]
        )

    x = np.concatenate(
        embeddings,
        axis=0,
    )

    y = np.concatenate(
        labels,
        axis=0,
    )

    if not np.isfinite(
        x
    ).all():

        raise RuntimeError(
            "Embedding contains NaN/Inf."
        )

    if x.shape[
        0
    ] != y.shape[
        0
    ]:

        raise RuntimeError(
            "Embedding / label count mismatch."
        )

    if x.shape[
        0
    ] != len(
        feature_ids
    ):

        raise RuntimeError(
            "Embedding / feature ID count mismatch."
        )

    return (
        x,
        y,
        feature_ids,
    )


# ============================================================
# PROBE CLASSIFIER
# ============================================================

def build_probe(
    seed,
):

    """
    Linear probe:
        StandardScaler
        +
        balanced multinomial logistic regression

    Same probe configuration is used for both frozen models.
    """

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=int(
                        seed
                    ),
                    solver="lbfgs",
                ),
            ),
        ]
    )


# ============================================================
# EVALUATE PROBE
# ============================================================

def evaluate_probe(
    model_name,
    probe,
    x_train,
    y_train,
    x_validation,
    y_validation,
    validation_ids,
):

    # ========================================================
    # FIT ONLY ON TRAIN EMBEDDINGS
    # ========================================================

    probe.fit(
        x_train,
        y_train,
    )

    # ========================================================
    # TRAIN PREDICTIONS
    # ========================================================

    train_predictions = probe.predict(
        x_train
    )

    # ========================================================
    # VALIDATION PREDICTIONS
    # ========================================================

    validation_predictions = (
        probe.predict(
            x_validation
        )
    )

    validation_probabilities = (
        probe.predict_proba(
            x_validation
        )
    )

    # ========================================================
    # METRICS
    # ========================================================

    train_accuracy = float(
        accuracy_score(
            y_train,
            train_predictions,
        )
    )

    train_balanced_accuracy = float(
        balanced_accuracy_score(
            y_train,
            train_predictions,
        )
    )

    validation_accuracy = float(
        accuracy_score(
            y_validation,
            validation_predictions,
        )
    )

    validation_balanced_accuracy = float(
        balanced_accuracy_score(
            y_validation,
            validation_predictions,
        )
    )

    labels = list(
        range(
            N_DOMAIN_CLASSES
        )
    )

    matrix = confusion_matrix(
        y_validation,
        validation_predictions,
        labels=labels,
    )

    # ========================================================
    # PREDICTION TABLE
    # ========================================================

    prediction_rows = []

    for i in range(
        len(
            y_validation
        )
    ):

        row = {
            "global_feature_id":
                validation_ids[
                    i
                ],

            "true_domain_label":
                int(
                    y_validation[
                        i
                    ]
                ),

            "true_domain_name":
                DOMAIN_NAMES[
                    int(
                        y_validation[
                            i
                        ]
                    )
                ],

            "pred_domain_label":
                int(
                    validation_predictions[
                        i
                    ]
                ),

            "pred_domain_name":
                DOMAIN_NAMES[
                    int(
                        validation_predictions[
                            i
                        ]
                    )
                ],
        }

        for domain_index in labels:

            row[
                f"prob_{DOMAIN_NAMES[domain_index]}"
            ] = float(
                validation_probabilities[
                    i,
                    domain_index,
                ]
            )

        prediction_rows.append(
            row
        )

    prediction_df = pd.DataFrame(
        prediction_rows
    )

    summary = {
        "model":
            model_name,

        "train_samples":
            int(
                len(
                    y_train
                )
            ),

        "validation_samples":
            int(
                len(
                    y_validation
                )
            ),

        "train_accuracy":
            train_accuracy,

        "train_balanced_accuracy":
            train_balanced_accuracy,

        "validation_accuracy":
            validation_accuracy,

        "validation_balanced_accuracy":
            validation_balanced_accuracy,

        "chance_balanced_accuracy":
            float(
                1.0
                / N_DOMAIN_CLASSES
            ),
    }

    return (
        summary,
        matrix,
        prediction_df,
    )


# ============================================================
# CONFUSION MATRIX DATAFRAME
# ============================================================

def confusion_matrix_dataframe(
    matrix,
):

    labels = [
        DOMAIN_NAMES[
            i
        ]
        for i in range(
            N_DOMAIN_CLASSES
        )
    ]

    return pd.DataFrame(
        matrix,
        index=[
            f"true_{name}"
            for name in labels
        ],
        columns=[
            f"pred_{name}"
            for name in labels
        ],
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc room-domain probe for "
            "baseline vs GRL frozen embeddings."
        )
    )

    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        default=str(
            DEFAULT_BASELINE_CHECKPOINT
        ),
    )

    parser.add_argument(
        "--grl-checkpoint",
        type=str,
        default=str(
            DEFAULT_GRL_CHECKPOINT
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
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="seed_42",
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

    run_name = str(
        args.run_name
    ).strip()

    if not run_name:

        raise ValueError(
            "--run-name cannot be empty."
        )

    set_seed(
        args.seed
    )

    device = get_device()

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
        "POST-HOC ROOM DOMAIN PROBE"
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
        f"Batch size             : "
        f"{args.batch_size}"
    )

    print(
        f"Domain classes         : "
        f"{N_DOMAIN_CLASSES}"
    )

    print(
        f"Chance BAcc            : "
        f"{1.0 / N_DOMAIN_CLASSES:.4f}"
    )

    print(
        f"Result directory       : "
        f"{result_dir}"
    )

    # ========================================================
    # LOAD DATASET
    # ========================================================

    full_train_dataset = ReplayDataset(
        split="train"
    )

    full_validation_dataset = ReplayDataset(
        split="validation"
    )

    (
        train_domain_dataset,
        train_domain_df,
    ) = build_domain_subset(
        full_train_dataset,
        "train",
    )

    (
        validation_domain_dataset,
        validation_domain_df,
    ) = build_domain_subset(
        full_validation_dataset,
        "validation",
    )

    # ========================================================
    # EXACT EXPECTED SAMPLE COUNTS
    # ========================================================

    if len(
        train_domain_dataset
    ) != 60:

        raise RuntimeError(
            "Expected 60 train domain samples, "
            f"got {len(train_domain_dataset)}."
        )

    if len(
        validation_domain_dataset
    ) != 45:

        raise RuntimeError(
            "Expected 45 validation domain samples, "
            f"got {len(validation_domain_dataset)}."
        )

    print()

    print(
        "Domain dataset guards : PASSED"
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_loader = DataLoader(
        train_domain_dataset,

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

    validation_loader = DataLoader(
        validation_domain_dataset,

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
    # LOAD BOTH FROZEN MODELS
    # ========================================================

    baseline_info = (
        load_frozen_model(
            args.baseline_checkpoint,
            device,
        )
    )

    grl_info = (
        load_frozen_model(
            args.grl_checkpoint,
            device,
        )
    )

    print()

    print(
        "=" * 75
    )

    print(
        "CHECKPOINTS"
    )

    print(
        "=" * 75
    )

    print()

    print(
        "Baseline:"
    )

    print(
        baseline_info[
            "checkpoint_path"
        ]
    )

    print(
        f"Epoch       : "
        f"{baseline_info['epoch']}"
    )

    print(
        f"Domain head : "
        f"{baseline_info['use_domain_head']}"
    )

    print()

    print(
        "GRL:"
    )

    print(
        grl_info[
            "checkpoint_path"
        ]
    )

    print(
        f"Epoch       : "
        f"{grl_info['epoch']}"
    )

    print(
        f"Domain head : "
        f"{grl_info['use_domain_head']}"
    )

    if baseline_info[
        "use_domain_head"
    ]:

        print()

        print(
            "WARNING: baseline checkpoint "
            "contains a domain head."
        )

    if not grl_info[
        "use_domain_head"
    ]:

        raise RuntimeError(
            "GRL checkpoint does not contain "
            "a domain head."
        )

    # ========================================================
    # EXTRACT BASELINE EMBEDDINGS
    # ========================================================

    print()

    print(
        "=" * 75
    )

    print(
        "EXTRACT BASELINE EMBEDDINGS"
    )

    print(
        "=" * 75
    )

    (
        baseline_x_train,
        baseline_y_train,
        baseline_train_ids,
    ) = extract_embeddings(
        model=
            baseline_info[
                "model"
            ],

        loader=
            train_loader,

        device=
            device,

        description=
            "BASELINE TRAIN EMB",
    )

    (
        baseline_x_validation,
        baseline_y_validation,
        baseline_validation_ids,
    ) = extract_embeddings(
        model=
            baseline_info[
                "model"
            ],

        loader=
            validation_loader,

        device=
            device,

        description=
            "BASELINE VAL EMB",
    )

    # ========================================================
    # EXTRACT GRL EMBEDDINGS
    # ========================================================

    print()

    print(
        "=" * 75
    )

    print(
        "EXTRACT GRL EMBEDDINGS"
    )

    print(
        "=" * 75
    )

    (
        grl_x_train,
        grl_y_train,
        grl_train_ids,
    ) = extract_embeddings(
        model=
            grl_info[
                "model"
            ],

        loader=
            train_loader,

        device=
            device,

        description=
            "GRL TRAIN EMB",
    )

    (
        grl_x_validation,
        grl_y_validation,
        grl_validation_ids,
    ) = extract_embeddings(
        model=
            grl_info[
                "model"
            ],

        loader=
            validation_loader,

        device=
            device,

        description=
            "GRL VAL EMB",
    )

    # ========================================================
    # COMPARABILITY GUARDS
    # ========================================================

    if not np.array_equal(
        baseline_y_train,
        grl_y_train,
    ):

        raise RuntimeError(
            "Train domain labels differ "
            "between baseline and GRL extraction."
        )

    if not np.array_equal(
        baseline_y_validation,
        grl_y_validation,
    ):

        raise RuntimeError(
            "Validation domain labels differ "
            "between baseline and GRL extraction."
        )

    if (
        baseline_train_ids
        != grl_train_ids
    ):

        raise RuntimeError(
            "Train feature IDs differ "
            "between baseline and GRL."
        )

    if (
        baseline_validation_ids
        != grl_validation_ids
    ):

        raise RuntimeError(
            "Validation feature IDs differ "
            "between baseline and GRL."
        )

    print()

    print(
        "Embedding comparability guard : PASSED"
    )

    print(
        f"Baseline embedding shape       : "
        f"{baseline_x_train.shape}"
    )

    print(
        f"GRL embedding shape            : "
        f"{grl_x_train.shape}"
    )

    # ========================================================
    # BASELINE PROBE
    # ========================================================

    baseline_probe = build_probe(
        args.seed
    )

    (
        baseline_summary,
        baseline_confusion,
        baseline_predictions,
    ) = evaluate_probe(
        model_name=
            "no_grl",

        probe=
            baseline_probe,

        x_train=
            baseline_x_train,

        y_train=
            baseline_y_train,

        x_validation=
            baseline_x_validation,

        y_validation=
            baseline_y_validation,

        validation_ids=
            baseline_validation_ids,
    )

    # ========================================================
    # GRL PROBE
    # ========================================================

    grl_probe = build_probe(
        args.seed
    )

    (
        grl_summary,
        grl_confusion,
        grl_predictions,
    ) = evaluate_probe(
        model_name=
            "grl",

        probe=
            grl_probe,

        x_train=
            grl_x_train,

        y_train=
            grl_y_train,

        x_validation=
            grl_x_validation,

        y_validation=
            grl_y_validation,

        validation_ids=
            grl_validation_ids,
    )

    # ========================================================
    # SUMMARY TABLE
    # ========================================================

    summary_df = pd.DataFrame(
        [
            baseline_summary,
            grl_summary,
        ]
    )

    # ========================================================
    # DIFFERENCE
    # ========================================================

    baseline_val_bacc = float(
        baseline_summary[
            "validation_balanced_accuracy"
        ]
    )

    grl_val_bacc = float(
        grl_summary[
            "validation_balanced_accuracy"
        ]
    )

    delta_bacc = (
        grl_val_bacc
        - baseline_val_bacc
    )

    comparison_df = pd.DataFrame(
        [
            {
                "baseline_validation_bacc":
                    baseline_val_bacc,

                "grl_validation_bacc":
                    grl_val_bacc,

                "delta_grl_minus_baseline":
                    delta_bacc,

                "chance_bacc":
                    float(
                        1.0
                        / N_DOMAIN_CLASSES
                    ),
            }
        ]
    )

    # ========================================================
    # CONFUSION DATAFRAMES
    # ========================================================

    baseline_confusion_df = (
        confusion_matrix_dataframe(
            baseline_confusion
        )
    )

    grl_confusion_df = (
        confusion_matrix_dataframe(
            grl_confusion
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    summary_path = (
        result_dir
        / "domain_probe_summary.csv"
    )

    comparison_path = (
        result_dir
        / "domain_probe_comparison.csv"
    )

    baseline_confusion_path = (
        result_dir
        / "baseline_confusion_matrix.csv"
    )

    grl_confusion_path = (
        result_dir
        / "grl_confusion_matrix.csv"
    )

    baseline_predictions_path = (
        result_dir
        / "baseline_validation_predictions.csv"
    )

    grl_predictions_path = (
        result_dir
        / "grl_validation_predictions.csv"
    )

    config_path = (
        result_dir
        / "domain_probe_config.json"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    baseline_confusion_df.to_csv(
        baseline_confusion_path,
    )

    grl_confusion_df.to_csv(
        grl_confusion_path,
    )

    baseline_predictions.to_csv(
        baseline_predictions_path,
        index=False,
    )

    grl_predictions.to_csv(
        grl_predictions_path,
        index=False,
    )

    config = {
        "seed":
            int(
                args.seed
            ),

        "run_name":
            run_name,

        "baseline_checkpoint":
            str(
                baseline_info[
                    "checkpoint_path"
                ]
            ),

        "grl_checkpoint":
            str(
                grl_info[
                    "checkpoint_path"
                ]
            ),

        "baseline_epoch":
            int(
                baseline_info[
                    "epoch"
                ]
            ),

        "grl_epoch":
            int(
                grl_info[
                    "epoch"
                ]
            ),

        "train_samples":
            int(
                len(
                    baseline_y_train
                )
            ),

        "validation_samples":
            int(
                len(
                    baseline_y_validation
                )
            ),

        "embedding_dim":
            int(
                baseline_x_train.shape[
                    1
                ]
            ),

        "probe":
            (
                "StandardScaler + "
                "LogisticRegression("
                "class_weight=balanced)"
            ),

        "domain_mapping":
            DOMAIN_MAP,

        "chance_balanced_accuracy":
            float(
                1.0
                / N_DOMAIN_CLASSES
            ),
    }

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
    # PRINT RESULTS
    # ========================================================

    print()

    print(
        "=" * 75
    )

    print(
        "DOMAIN PROBE RESULTS"
    )

    print(
        "=" * 75
    )

    print()

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()

    print(
        "=" * 75
    )

    print(
        "DOMAIN PROBE COMPARISON"
    )

    print(
        "=" * 75
    )

    print()

    print(
        f"Chance BAcc             : "
        f"{1.0 / N_DOMAIN_CLASSES:.4f}"
    )

    print(
        f"Baseline Val Probe BAcc : "
        f"{baseline_val_bacc:.4f}"
    )

    print(
        f"GRL Val Probe BAcc      : "
        f"{grl_val_bacc:.4f}"
    )

    print(
        f"Delta GRL - Baseline    : "
        f"{delta_bacc:+.4f}"
    )

    print()

    print(
        "Baseline confusion matrix:"
    )

    print()

    print(
        baseline_confusion_df.to_string()
    )

    print()

    print(
        "GRL confusion matrix:"
    )

    print()

    print(
        grl_confusion_df.to_string()
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()

    print(
        "=" * 75
    )

    print(
        "DOMAIN PROBE COMPLETE"
    )

    print(
        "=" * 75
    )

    print()

    print(
        "Summary:"
    )

    print(
        summary_path
    )

    print()

    print(
        "Comparison:"
    )

    print(
        comparison_path
    )

    print()

    print(
        "Baseline confusion:"
    )

    print(
        baseline_confusion_path
    )

    print()

    print(
        "GRL confusion:"
    )

    print(
        grl_confusion_path
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