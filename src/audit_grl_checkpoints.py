from pathlib import Path
import json
import torch

from model_replay import ReplayResNet


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINTS = [
    {
        "name": "GRL seed 42",
        "path": ROOT
        / "checkpoints"
        / "replay_resnet_grl"
        / "seed_42_ep5"
        / "best_model.pt",
        "expected_seed": 42,
    },
    {
        "name": "GRL seed 2026",
        "path": ROOT
        / "checkpoints"
        / "replay_resnet_grl"
        / "seed_2026_ep5"
        / "best_model.pt",
        "expected_seed": 2026,
    },
]


def get_first(checkpoint, keys, default=None):
    for key in keys:
        if key in checkpoint:
            return checkpoint[key]
    return default


def inspect_checkpoint(item):
    path = item["path"]

    print()
    print("=" * 80)
    print(item["name"])
    print("=" * 80)

    print(f"Path                  : {path}")

    if not path.exists():
        print("STATUS                : MISSING")
        return False

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    print("STATUS                : FOUND")

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            f"{item['name']}: invalid model_state_dict."
        )

    keys = list(state_dict.keys())

    has_domain_head = any(
        key.startswith("domain_head.")
        for key in keys
    )

    print(
        f"Domain-head weights   : "
        f"{'FOUND' if has_domain_head else 'NOT FOUND'}"
    )

    epoch = get_first(
        checkpoint,
        ["epoch", "best_epoch"],
        default=None,
    )

    print(f"Checkpoint epoch      : {epoch}")

    config = checkpoint.get(
        "config",
        {},
    )

    if config is None:
        config = {}

    validation_metrics = checkpoint.get(
        "validation_metrics",
        {},
    )

    if validation_metrics is None:
        validation_metrics = {}

    seed = get_first(
        checkpoint,
        ["seed"],
        default=config.get("seed"),
    )

    validation_eer = get_first(
        checkpoint,
        [
            "validation_eer",
            "val_eer",
        ],
        default=validation_metrics.get(
            "eer",
            validation_metrics.get(
                "validation_eer",
                None,
            ),
        ),
    )

    validation_threshold = get_first(
        checkpoint,
        [
            "validation_eer_threshold",
            "validation_threshold",
            "eer_threshold",
        ],
        default=validation_metrics.get(
            "eer_threshold",
            validation_metrics.get(
                "validation_eer_threshold",
                None,
            ),
        ),
    )

    use_grl = get_first(
        checkpoint,
        ["use_grl"],
        default=config.get(
            "use_grl",
            None,
        ),
    )

    use_domain_head = get_first(
        checkpoint,
        ["use_domain_head"],
        default=config.get(
            "use_domain_head",
            None,
        ),
    )

    model_name = get_first(
        checkpoint,
        ["model"],
        default=config.get(
            "model",
            None,
        ),
    )

    grl_strength = get_first(
        checkpoint,
        ["grl_strength"],
        default=config.get(
            "grl_strength",
            None,
        ),
    )

    domain_loss_weight = get_first(
        checkpoint,
        ["domain_loss_weight"],
        default=config.get(
            "domain_loss_weight",
            None,
        ),
    )

    transform_loss_weight = get_first(
        checkpoint,
        ["transform_loss_weight"],
        default=config.get(
            "transform_loss_weight",
            None,
        ),
    )

    batch_size = get_first(
        checkpoint,
        ["batch_size"],
        default=config.get(
            "batch_size",
            None,
        ),
    )

    learning_rate = get_first(
        checkpoint,
        ["lr", "learning_rate"],
        default=config.get(
            "lr",
            config.get(
                "learning_rate",
                None,
            ),
        ),
    )

    replay_sampling_boost = get_first(
        checkpoint,
        ["replay_sampling_boost"],
        default=config.get(
            "replay_sampling_boost",
            None,
        ),
    )

    weighting = get_first(
        checkpoint,
        ["weighting"],
        default=config.get(
            "weighting",
            None,
        ),
    )

    print(f"Seed                  : {seed}")
    print(f"Expected seed         : {item['expected_seed']}")
    print(f"Model name            : {model_name}")
    print(f"use_grl               : {use_grl}")
    print(f"use_domain_head       : {use_domain_head}")
    print(f"GRL strength          : {grl_strength}")
    print(f"Domain loss weight    : {domain_loss_weight}")
    print(f"Transform loss weight : {transform_loss_weight}")
    print(f"Batch size            : {batch_size}")
    print(f"Learning rate         : {learning_rate}")
    print(f"Replay sampling boost : {replay_sampling_boost}")
    print(f"Weighting             : {weighting}")
    print(f"Validation EER        : {validation_eer}")
    print(f"Validation threshold  : {validation_threshold}")

    # --------------------------------------------------------
    # Infer architecture from stored weights
    # --------------------------------------------------------

    class_head_weight = state_dict.get(
        "class_head.weight"
    )

    transform_head_weight = state_dict.get(
        "transform_head.weight"
    )

    if class_head_weight is None:
        raise RuntimeError(
            f"{item['name']}: class_head.weight missing."
        )

    if transform_head_weight is None:
        raise RuntimeError(
            f"{item['name']}: transform_head.weight missing."
        )

    embedding_dim = int(
        class_head_weight.shape[1]
    )

    n_transform = int(
        transform_head_weight.shape[0]
    )

    if has_domain_head:
        last_domain_weight = None

        for candidate in [
            "domain_head.2.weight",
            "domain_head.3.weight",
            "domain_head.1.weight",
        ]:
            if candidate in state_dict:
                last_domain_weight = state_dict[
                    candidate
                ]

        if last_domain_weight is None:
            domain_weights = [
                value
                for key, value in state_dict.items()
                if (
                    key.startswith("domain_head.")
                    and key.endswith(".weight")
                )
            ]

            if not domain_weights:
                raise RuntimeError(
                    f"{item['name']}: "
                    "domain head detected but "
                    "domain weights could not be read."
                )

            n_domain = int(
                domain_weights[-1].shape[0]
            )

        else:
            n_domain = int(
                last_domain_weight.shape[0]
            )

    else:
        n_domain = 4

    print(f"Embedding dim         : {embedding_dim}")
    print(f"Transform classes     : {n_transform}")
    print(f"Domain classes        : {n_domain}")

    # --------------------------------------------------------
    # Strict model load test
    # --------------------------------------------------------

    model = ReplayResNet(
        n_transform=n_transform,
        embedding_dim=embedding_dim,
        n_domain=n_domain,
        use_domain_head=has_domain_head,
        grl_strength=(
            float(grl_strength)
            if grl_strength is not None
            else 1.0
        ),
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(
        f"Missing keys          : "
        f"{len(missing)}"
    )

    print(
        f"Unexpected keys       : "
        f"{len(unexpected)}"
    )

    if missing:
        print("Missing key list:")
        for key in missing:
            print(f"  - {key}")

    if unexpected:
        print("Unexpected key list:")
        for key in unexpected:
            print(f"  - {key}")

    # --------------------------------------------------------
    # Hard audit rules
    # --------------------------------------------------------

    problems = []

    if seed is not None:
        if int(seed) != int(
            item["expected_seed"]
        ):
            problems.append(
                "seed mismatch"
            )

    if not has_domain_head:
        problems.append(
            "domain-head weights missing"
        )

    if (
        use_domain_head is not None
        and
        bool(use_domain_head) is not True
    ):
        problems.append(
            "use_domain_head is not True"
        )

    if missing:
        problems.append(
            "missing model keys"
        )

    if unexpected:
        problems.append(
            "unexpected model keys"
        )

    if n_domain != 4:
        problems.append(
            f"expected 4 domain classes, got {n_domain}"
        )

    if problems:
        print()
        print("AUDIT RESULT          : FAIL")

        for problem in problems:
            print(f"  - {problem}")

        return False

    print()
    print("AUDIT RESULT          : PASS")

    return True


def main():

    print("=" * 80)
    print("GRL CHECKPOINT AUDIT")
    print("=" * 80)

    results = []

    for item in CHECKPOINTS:
        result = inspect_checkpoint(
            item
        )
        results.append(
            (
                item["name"],
                result,
            )
        )

    print()
    print("=" * 80)
    print("FINAL AUDIT SUMMARY")
    print("=" * 80)

    all_pass = True

    for name, result in results:

        status = (
            "PASS"
            if result
            else "FAIL"
        )

        print(
            f"{name:<25} : {status}"
        )

        all_pass = (
            all_pass
            and result
        )

    print()

    if all_pass:
        print(
            "FINAL STATUS: "
            "BOTH GRL CHECKPOINTS ARE ARCHITECTURALLY VALID."
        )
    else:
        print(
            "FINAL STATUS: "
            "AT LEAST ONE CHECKPOINT NEEDS REVIEW."
        )


if __name__ == "__main__":
    main()