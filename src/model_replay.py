import torch
import torch.nn as nn

from torchvision.models import resnet18

from grl import grad_reverse


# ============================================================
# REPLAY RESNET WITH OPTIONAL DOMAIN-ADVERSARIAL HEAD
# ============================================================

class ReplayResNet(nn.Module):
    """
    Three-channel ResNet18 for:

    1. bona fide / spoof classification
    2. clean / replay transformation classification
    3. optional room-domain adversarial classification

    Input:
        [B, 3, 80, T]

    Standard output:
        class_logits     [B, 2]
        transform_logits [B, n_transform]
        embedding        [B, embedding_dim]

    Optional domain output:
        domain_logits    [B, n_domain]

    IMPORTANT
    ---------
    The baseline model must NOT instantiate a domain head.

    Therefore:
        use_domain_head=False

    is the safe/default configuration.

    GRL training must explicitly request:
        use_domain_head=True
    """

    def __init__(
        self,
        n_transform=2,
        embedding_dim=256,
        n_domain=4,
        use_domain_head=False,
        grl_strength=1.0,
    ):
        super().__init__()

        self.n_transform = int(n_transform)
        self.embedding_dim = int(embedding_dim)
        self.n_domain = int(n_domain)
        self.use_domain_head = bool(use_domain_head)
        self.grl_strength = float(grl_strength)

        # ====================================================
        # ARGUMENT QC
        # ====================================================

        if self.n_transform <= 0:
            raise ValueError(
                "n_transform must be > 0."
            )

        if self.embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be > 0."
            )

        if self.n_domain <= 0:
            raise ValueError(
                "n_domain must be > 0."
            )

        if self.grl_strength < 0:
            raise ValueError(
                "grl_strength must be >= 0."
            )

        # ====================================================
        # RESNET18 BACKBONE
        # ====================================================

        backbone = resnet18(
            weights=None
        )

        backbone_dim = (
            backbone
            .fc
            .in_features
        )

        backbone.fc = nn.Identity()

        self.backbone = backbone

        # ====================================================
        # EMBEDDING PROJECTION
        # ====================================================

        self.proj = nn.Sequential(
            nn.Linear(
                backbone_dim,
                self.embedding_dim,
            ),
            nn.LayerNorm(
                self.embedding_dim
            ),
            nn.ReLU(),
        )

        # ====================================================
        # CLASSIFICATION HEAD
        # ====================================================

        self.class_head = nn.Linear(
            self.embedding_dim,
            2,
        )

        # ====================================================
        # TRANSFORMATION HEAD
        # ====================================================

        self.transform_head = nn.Linear(
            self.embedding_dim,
            self.n_transform,
        )

        # ====================================================
        # OPTIONAL DOMAIN HEAD
        # ====================================================

        if self.use_domain_head:
            self.domain_head = nn.Sequential(
                nn.Linear(
                    self.embedding_dim,
                    128,
                ),
                nn.ReLU(),
                nn.Linear(
                    128,
                    self.n_domain,
                ),
            )
        else:
            self.domain_head = None

    # ========================================================
    # EMBEDDING
    # ========================================================

    def encode(
        self,
        x,
    ):
        # ====================================================
        # INPUT QC
        # ====================================================

        if x.ndim != 4:
            raise ValueError(
                "Input must have shape "
                "[B, 3, 80, T]. "
                f"Got: {tuple(x.shape)}"
            )

        if x.shape[1] != 3:
            raise ValueError(
                "Input must have exactly "
                "3 channels "
                "(Log-Mel, phase-derived, MGD). "
                f"Got: {x.shape[1]}"
            )

        if x.shape[2] != 80:
            raise ValueError(
                "Input Mel dimension must be 80. "
                f"Got: {x.shape[2]}"
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                "Input contains NaN/Inf."
            )

        # ====================================================
        # BACKBONE
        # ====================================================

        features = self.backbone(
            x
        )

        # ====================================================
        # PROJECTION
        # ====================================================

        embedding = self.proj(
            features
        )

        if not torch.isfinite(
            embedding
        ).all():
            raise RuntimeError(
                "Embedding contains NaN/Inf."
            )

        return embedding

    # ========================================================
    # STANDARD FORWARD
    # ========================================================

    def forward(
        self,
        x,
    ):
        """
        Standard forward used by baseline and evaluation.

        Returns:
            class_logits
            transform_logits
            embedding
        """

        embedding = self.encode(
            x
        )

        class_logits = self.class_head(
            embedding
        )

        transform_logits = (
            self.transform_head(
                embedding
            )
        )

        if not torch.isfinite(
            class_logits
        ).all():
            raise RuntimeError(
                "Class logits contain NaN/Inf."
            )

        if not torch.isfinite(
            transform_logits
        ).all():
            raise RuntimeError(
                "Transform logits contain NaN/Inf."
            )

        return (
            class_logits,
            transform_logits,
            embedding,
        )

    # ========================================================
    # DOMAIN FORWARD
    # ========================================================

    def forward_domain(
        self,
        embedding,
        grl_strength=None,
    ):
        """
        Compute domain logits from embedding through GRL.
        """

        if not self.use_domain_head:
            raise RuntimeError(
                "Domain head is disabled."
            )

        if self.domain_head is None:
            raise RuntimeError(
                "domain_head is not initialized."
            )

        if embedding.ndim != 2:
            raise ValueError(
                "Embedding must have shape "
                "[B, embedding_dim]. "
                f"Got: {tuple(embedding.shape)}"
            )

        if (
            embedding.shape[1]
            != self.embedding_dim
        ):
            raise ValueError(
                "Embedding dimension mismatch. "
                f"Expected {self.embedding_dim}, "
                f"got {embedding.shape[1]}."
            )

        if not torch.isfinite(
            embedding
        ).all():
            raise ValueError(
                "Embedding contains NaN/Inf."
            )

        if grl_strength is None:
            grl_strength = (
                self.grl_strength
            )

        grl_strength = float(
            grl_strength
        )

        if grl_strength < 0:
            raise ValueError(
                "grl_strength must be >= 0."
            )

        reversed_embedding = (
            grad_reverse(
                embedding,
                strength=grl_strength,
            )
        )

        domain_logits = (
            self.domain_head(
                reversed_embedding
            )
        )

        if not torch.isfinite(
            domain_logits
        ).all():
            raise RuntimeError(
                "Domain logits contain NaN/Inf."
            )

        return domain_logits

    # ========================================================
    # FULL FORWARD WITH DOMAIN
    # ========================================================

    def forward_with_domain(
        self,
        x,
        grl_strength=None,
    ):
        """
        Full DANN forward.

        Returns:
            class_logits
            transform_logits
            domain_logits
            embedding
        """

        (
            class_logits,
            transform_logits,
            embedding,
        ) = self.forward(
            x
        )

        domain_logits = (
            self.forward_domain(
                embedding,
                grl_strength=
                    grl_strength,
            )
        )

        return (
            class_logits,
            transform_logits,
            domain_logits,
            embedding,
        )


# ============================================================
# SELF TEST
# ============================================================

def _self_test():
    print(
        "=" * 75
    )

    print(
        "REPLAY RESNET + DOMAIN HEAD SELF TEST"
    )

    print(
        "=" * 75
    )

    # ========================================================
    # BASELINE MODEL TEST
    # ========================================================

    baseline_model = ReplayResNet(
        n_transform=2,
        embedding_dim=256,
        n_domain=4,
        use_domain_head=False,
        grl_strength=1.0,
    )

    baseline_parameters = sum(
        p.numel()
        for p in baseline_model.parameters()
        if p.requires_grad
    )

    print()
    print(
        "Baseline model:"
    )
    print(
        f"use_domain_head     : "
        f"{baseline_model.use_domain_head}"
    )
    print(
        f"domain_head         : "
        f"{baseline_model.domain_head}"
    )
    print(
        f"Trainable parameters: "
        f"{baseline_parameters:,}"
    )

    if baseline_model.use_domain_head:
        raise RuntimeError(
            "Baseline self-test failed: "
            "domain head must be disabled."
        )

    if baseline_model.domain_head is not None:
        raise RuntimeError(
            "Baseline self-test failed: "
            "domain_head must be None."
        )

    # ========================================================
    # GRL MODEL
    # ========================================================

    model = ReplayResNet(
        n_transform=2,
        embedding_dim=256,
        n_domain=4,
        use_domain_head=True,
        grl_strength=1.0,
    )

    model.train()

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        "GRL model:"
    )
    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    # ========================================================
    # INPUT
    # ========================================================

    x = torch.randn(
        4,
        3,
        80,
        400,
        requires_grad=True,
    )

    # ========================================================
    # STANDARD FORWARD
    # ========================================================

    (
        class_logits,
        transform_logits,
        embedding,
    ) = model(
        x
    )

    print()
    print(
        "Standard forward:"
    )
    print(
        f"class_logits     : "
        f"{tuple(class_logits.shape)}"
    )
    print(
        f"transform_logits : "
        f"{tuple(transform_logits.shape)}"
    )
    print(
        f"embedding        : "
        f"{tuple(embedding.shape)}"
    )

    # ========================================================
    # DOMAIN FORWARD
    # ========================================================

    domain_logits = (
        model.forward_domain(
            embedding,
            grl_strength=1.0,
        )
    )

    print()
    print(
        "Domain forward:"
    )
    print(
        f"domain_logits    : "
        f"{tuple(domain_logits.shape)}"
    )

    # ========================================================
    # FULL FORWARD
    # ========================================================

    (
        class_logits_full,
        transform_logits_full,
        domain_logits_full,
        embedding_full,
    ) = model.forward_with_domain(
        x,
        grl_strength=1.0,
    )

    print()
    print(
        "Full forward:"
    )
    print(
        f"class_logits     : "
        f"{tuple(class_logits_full.shape)}"
    )
    print(
        f"transform_logits : "
        f"{tuple(transform_logits_full.shape)}"
    )
    print(
        f"domain_logits    : "
        f"{tuple(domain_logits_full.shape)}"
    )
    print(
        f"embedding        : "
        f"{tuple(embedding_full.shape)}"
    )

    # ========================================================
    # EXPECTED SHAPES
    # ========================================================

    assert (
        class_logits.shape
        == (
            4,
            2,
        )
    )

    assert (
        transform_logits.shape
        == (
            4,
            2,
        )
    )

    assert (
        embedding.shape
        == (
            4,
            256,
        )
    )

    assert (
        domain_logits.shape
        == (
            4,
            4,
        )
    )

    # ========================================================
    # FINITE TEST
    # ========================================================

    assert torch.isfinite(
        class_logits
    ).all()

    assert torch.isfinite(
        transform_logits
    ).all()

    assert torch.isfinite(
        domain_logits
    ).all()

    assert torch.isfinite(
        embedding
    ).all()

    # ========================================================
    # VARIABLE TIME TEST
    # ========================================================

    x2 = torch.randn(
        2,
        3,
        80,
        257,
    )

    (
        class_logits2,
        transform_logits2,
        domain_logits2,
        embedding2,
    ) = model.forward_with_domain(
        x2
    )

    assert (
        class_logits2.shape
        == (
            2,
            2,
        )
    )

    assert (
        transform_logits2.shape
        == (
            2,
            2,
        )
    )

    assert (
        domain_logits2.shape
        == (
            2,
            4,
        )
    )

    assert (
        embedding2.shape
        == (
            2,
            256,
        )
    )

    print()
    print(
        "Variable-T test     : PASSED"
    )

    # ========================================================
    # BACKWARD / GRL TEST
    # ========================================================

    domain_targets = torch.tensor(
        [
            0,
            1,
            2,
            3,
        ],
        dtype=torch.long,
    )

    domain_loss = (
        torch.nn.functional
        .cross_entropy(
            domain_logits,
            domain_targets,
        )
    )

    domain_loss.backward()

    backbone_grad_found = False

    for parameter in (
        model
        .backbone
        .parameters()
    ):
        if parameter.grad is not None:
            if torch.isfinite(
                parameter.grad
            ).all():
                backbone_grad_found = True
                break

    if not backbone_grad_found:
        raise RuntimeError(
            "No valid gradient reached "
            "the backbone from domain loss."
        )

    print(
        "Domain backward     : PASSED"
    )

    print()
    print(
        "REPLAY RESNET DOMAIN SELF TEST: PASSED"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    _self_test()