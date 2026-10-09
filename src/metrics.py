import numpy as np

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
)


# ============================================================
# EER
# ============================================================

def compute_eer(
    labels,
    scores,
):
    """
    Parameters
    ----------
    labels:
        0 = bona fide
        1 = spoof

    scores:
        continuous spoof probability / score.
        Higher means more likely spoof.

    Returns
    -------
    eer
    threshold
    """

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    # ========================================================
    # QC
    # ========================================================

    if labels.ndim != 1:

        raise ValueError(
            "labels must be 1-dimensional."
        )

    if scores.ndim != 1:

        raise ValueError(
            "scores must be 1-dimensional."
        )

    if len(labels) != len(scores):

        raise ValueError(
            "labels and scores length mismatch."
        )

    if len(labels) == 0:

        return (
            float("nan"),
            float("nan"),
        )

    if not np.isfinite(
        scores
    ).all():

        raise ValueError(
            "scores contain NaN/Inf."
        )

    # EER requires both classes.

    if np.unique(
        labels
    ).size < 2:

        return (
            float("nan"),
            float("nan"),
        )

    # ========================================================
    # ROC
    # ========================================================

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
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
            fpr[index]
            +
            fnr[index]
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
# SAFE ROC-AUC
# ============================================================

def safe_roc_auc(
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

    if np.unique(
        labels
    ).size < 2:

        return float(
            "nan"
        )

    return float(
        roc_auc_score(
            labels,
            scores,
        )
    )


# ============================================================
# BINARY CLASSIFICATION METRICS
# ============================================================

def classification_metrics(
    labels,
    scores,
    threshold=0.5,
):

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    predictions = (
        scores
        >= threshold
    ).astype(
        np.int64
    )

    eer, eer_threshold = (
        compute_eer(
            labels,
            scores,
        )
    )

    metrics = {
        "n_samples":
            int(
                len(labels)
            ),

        "n_bonafide":
            int(
                (
                    labels
                    == 0
                )
                .sum()
            ),

        "n_spoof":
            int(
                (
                    labels
                    == 1
                )
                .sum()
            ),

        "accuracy":
            float(
                accuracy_score(
                    labels,
                    predictions,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    labels,
                    predictions,
                )
            ),

        "f1":
            float(
                f1_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),

        "roc_auc":
            safe_roc_auc(
                labels,
                scores,
            ),

        "eer":
            eer,

        "eer_threshold":
            eer_threshold,

        "decision_threshold":
            float(
                threshold
            ),
    }

    return metrics


# ============================================================
# THRESHOLD TRANSFER METRICS
# ============================================================

def threshold_metrics(
    labels,
    scores,
    threshold,
):

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    predictions = (
        scores
        >= threshold
    ).astype(
        np.int64
    )

    bona_mask = (
        labels
        == 0
    )

    spoof_mask = (
        labels
        == 1
    )

    false_accept_rate = float(
        (
            predictions[
                bona_mask
            ]
            == 1
        ).mean()
    ) if bona_mask.any() else float(
        "nan"
    )

    false_reject_rate = float(
        (
            predictions[
                spoof_mask
            ]
            == 0
        ).mean()
    ) if spoof_mask.any() else float(
        "nan"
    )

    return {
        "accuracy_at_val_threshold":
            float(
                accuracy_score(
                    labels,
                    predictions,
                )
            ),

        "balanced_accuracy_at_val_threshold":
            float(
                balanced_accuracy_score(
                    labels,
                    predictions,
                )
            ),

        "far_at_val_threshold":
            false_accept_rate,

        "frr_at_val_threshold":
            false_reject_rate,
    }