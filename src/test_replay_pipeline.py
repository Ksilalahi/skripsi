import torch

from torch.utils.data import DataLoader

from dataset_replay import (
    ReplayDataset,
    replay_collate_fn,
)

from model_replay import ReplayResNet


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "REAL DATA -> REPLAY RESNET "
        "INTEGRATION TEST"
    )
    print("=" * 75)

    print()

    print(
        f"Device : {DEVICE}"
    )

    # ========================================================
    # DATASET
    # ========================================================

    dataset = ReplayDataset(
        split="train"
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

    # ========================================================
    # REAL BATCH
    # ========================================================

    batch = next(
        iter(
            loader
        )
    )

    features = (
        batch[
            "features"
        ]
        .to(
            DEVICE
        )
    )

    class_labels = (
        batch[
            "class_labels"
        ]
        .to(
            DEVICE
        )
    )

    transform_labels = (
        batch[
            "transform_labels"
        ]
        .to(
            DEVICE
        )
    )

    # ========================================================
    # INPUT QC
    # ========================================================

    print()

    print(
        "Batch features:",
        tuple(
            features.shape
        ),
    )

    print(
        "Class labels:",
        class_labels.tolist(),
    )

    print(
        "Transform labels:",
        transform_labels.tolist(),
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

    print(
        "Input finite:",
        bool(
            torch.isfinite(
                features
            ).all()
        ),
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = ReplayResNet(
        n_transform=2,
        embedding_dim=256,
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    # ========================================================
    # FORWARD REAL DATA
    # ========================================================

    with torch.no_grad():

        (
            class_logits,
            transform_logits,
            embedding,
        ) = model(
            features
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()

    print(
        "Class logits:",
        tuple(
            class_logits.shape
        ),
    )

    print(
        "Transform logits:",
        tuple(
            transform_logits.shape
        ),
    )

    print(
        "Embedding:",
        tuple(
            embedding.shape
        ),
    )

    print()

    print(
        "Class logits finite:",
        bool(
            torch.isfinite(
                class_logits
            ).all()
        ),
    )

    print(
        "Transform logits finite:",
        bool(
            torch.isfinite(
                transform_logits
            ).all()
        ),
    )

    print(
        "Embedding finite:",
        bool(
            torch.isfinite(
                embedding
            ).all()
        ),
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

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

    print()

    print(
        "Untrained class predictions:",
        class_pred.tolist(),
    )

    print(
        "Untrained transform predictions:",
        transform_pred.tolist(),
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "Predictions are random/untrained."
    )

    # ========================================================
    # HARD ASSERTIONS
    # ========================================================

    batch_size = (
        features.shape[
            0
        ]
    )

    assert (
        features.ndim
        == 4
    )

    assert (
        features.shape[
            1
        ]
        == 3
    )

    assert (
        features.shape[
            2
        ]
        == 80
    )

    assert (
        class_logits.shape
        == (
            batch_size,
            2,
        )
    )

    assert (
        transform_logits.shape
        == (
            batch_size,
            2,
        )
    )

    assert (
        embedding.shape
        == (
            batch_size,
            256,
        )
    )

    assert torch.isfinite(
        class_logits
    ).all()

    assert torch.isfinite(
        transform_logits
    ).all()

    assert torch.isfinite(
        embedding
    ).all()

    # ========================================================
    # SUCCESS
    # ========================================================

    print()

    print("=" * 75)

    print(
        "REAL-DATA MODEL INTEGRATION: PASSED"
    )

    print("=" * 75)


if __name__ == "__main__":

    main()