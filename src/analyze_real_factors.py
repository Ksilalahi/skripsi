from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BASELINE_SCORES = (
    ROOT
    / "results"
    / "evaluation"
    / "seed_42"
    / "scores.csv"
)

GRL_SCORES = (
    ROOT
    / "results"
    / "evaluation"
    / "grl_seed_42_ep5"
    / "scores.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "real_factor_analysis"
    / "seed_42"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FROZEN VALIDATION THRESHOLDS
# ============================================================

BASELINE_VALIDATION_THRESHOLD = 0.935573

GRL_VALIDATION_THRESHOLD = 0.757100


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

    if len(labels) == 0:

        return (
            np.nan,
            np.nan,
        )

    if np.unique(
        labels
    ).size < 2:

        return (
            np.nan,
            np.nan,
        )

    if not np.isfinite(
        scores
    ).all():

        raise RuntimeError(
            "Score contains NaN/Inf."
        )

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
# METRICS
# ============================================================

def compute_metrics(
    df,
    validation_threshold,
):

    labels = (
        df[
            "label"
        ]
        .astype(
            int
        )
        .to_numpy()
    )

    scores = (
        df[
            "fake_score"
        ]
        .astype(
            float
        )
        .to_numpy()
    )

    n_samples = int(
        len(
            df
        )
    )

    n_bonafide = int(
        (
            labels
            == 0
        )
        .sum()
    )

    n_spoof = int(
        (
            labels
            == 1
        )
        .sum()
    )

    result = {
        "n_samples":
            n_samples,

        "n_bonafide":
            n_bonafide,

        "n_spoof":
            n_spoof,
    }

    # ========================================================
    # RANKING METRICS
    # ========================================================

    if (
        n_bonafide > 0
        and
        n_spoof > 0
    ):

        (
            eer,
            eer_threshold,
        ) = compute_eer(
            labels,
            scores,
        )

        result[
            "eer"
        ] = eer

        result[
            "eer_threshold"
        ] = eer_threshold

        result[
            "roc_auc"
        ] = float(
            roc_auc_score(
                labels,
                scores,
            )
        )

    else:

        result[
            "eer"
        ] = np.nan

        result[
            "eer_threshold"
        ] = np.nan

        result[
            "roc_auc"
        ] = np.nan

    # ========================================================
    # FROZEN VALIDATION THRESHOLD
    # ========================================================

    predictions = (
        scores
        >= float(
            validation_threshold
        )
    ).astype(
        np.int64
    )

    if np.unique(
        labels
    ).size >= 2:

        result[
            "bacc_at_val_threshold"
        ] = float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        )

    else:

        result[
            "bacc_at_val_threshold"
        ] = np.nan

    # ========================================================
    # FAR
    # ========================================================

    bona_mask = (
        labels
        == 0
    )

    if bona_mask.any():

        result[
            "far_at_val_threshold"
        ] = float(
            (
                predictions[
                    bona_mask
                ]
                == 1
            )
            .mean()
        )

    else:

        result[
            "far_at_val_threshold"
        ] = np.nan

    # ========================================================
    # FRR
    # ========================================================

    spoof_mask = (
        labels
        == 1
    )

    if spoof_mask.any():

        result[
            "frr_at_val_threshold"
        ] = float(
            (
                predictions[
                    spoof_mask
                ]
                == 0
            )
            .mean()
        )

    else:

        result[
            "frr_at_val_threshold"
        ] = np.nan

    # ========================================================
    # SCORE DISTRIBUTION
    # ========================================================

    result[
        "mean_fake_score"
    ] = float(
        np.mean(
            scores
        )
    )

    result[
        "median_fake_score"
    ] = float(
        np.median(
            scores
        )
    )

    result[
        "min_fake_score"
    ] = float(
        np.min(
            scores
        )
    )

    result[
        "max_fake_score"
    ] = float(
        np.max(
            scores
        )
    )

    return result


# ============================================================
# NORMALIZE SCORE FILE
# ============================================================

def normalize_score_file(
    path,
    model_name,
):

    if not path.exists():

        raise FileNotFoundError(
            f"{model_name} scores not found:\n"
            f"{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    # ========================================================
    # SAMPLE ID NORMALIZATION
    # ========================================================

    if (
        "global_feature_id"
        in df.columns
    ):

        df[
            "sample_id"
        ] = (
            df[
                "global_feature_id"
            ]
            .astype(str)
        )

        id_source = (
            "global_feature_id"
        )

    elif (
        "file_id"
        in df.columns
    ):

        df[
            "sample_id"
        ] = (
            df[
                "file_id"
            ]
            .astype(str)
        )

        id_source = (
            "file_id"
        )

    elif (
        "feature_id"
        in df.columns
    ):

        df[
            "sample_id"
        ] = (
            df[
                "feature_id"
            ]
            .astype(str)
        )

        id_source = (
            "feature_id"
        )

    else:

        raise RuntimeError(
            f"{model_name}: no usable "
            "sample identity column found.\n"
            f"Available columns:\n"
            f"{df.columns.tolist()}"
        )

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required = {
        "sample_id",
        "source_file_id",
        "label",
        "fake_score",
        "condition",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"{model_name}: missing columns "
            f"{sorted(missing)}"
        )

    # ========================================================
    # DUPLICATE GUARD
    # ========================================================

    if df[
        "sample_id"
    ].duplicated().any():

        duplicate_ids = (
            df[
                df[
                    "sample_id"
                ].duplicated(
                    keep=False
                )
            ][
                "sample_id"
            ]
            .head(
                10
            )
            .tolist()
        )

        raise RuntimeError(
            f"{model_name}: duplicate sample IDs:\n"
            f"{duplicate_ids}"
        )

    # ========================================================
    # REAL ONLY
    # ========================================================

    condition = (
        df[
            "condition"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    real = (
        df[
            condition.eq(
                "real"
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print()

    print(
        f"{model_name}"
    )

    print(
        f"ID source      : "
        f"{id_source}"
    )

    print(
        f"All scores     : "
        f"{len(df)}"
    )

    print(
        f"Real samples   : "
        f"{len(real)}"
    )

    print(
        f"Unique real ID : "
        f"{real['sample_id'].nunique()}"
    )

    return real


# ============================================================
# MATCH BASELINE AND GRL
# ============================================================

def match_score_files(
    baseline,
    grl,
):

    baseline_ids = set(
        baseline[
            "sample_id"
        ]
        .astype(str)
    )

    grl_ids = set(
        grl[
            "sample_id"
        ]
        .astype(str)
    )

    missing_in_grl = (
        baseline_ids
        - grl_ids
    )

    missing_in_baseline = (
        grl_ids
        - baseline_ids
    )

    print()

    print(
        "=" * 75
    )

    print(
        "SAMPLE MATCHING AUDIT"
    )

    print(
        "=" * 75
    )

    print()

    print(
        f"Baseline real IDs       : "
        f"{len(baseline_ids)}"
    )

    print(
        f"GRL real IDs            : "
        f"{len(grl_ids)}"
    )

    print(
        f"Missing in GRL          : "
        f"{len(missing_in_grl)}"
    )

    print(
        f"Missing in baseline     : "
        f"{len(missing_in_baseline)}"
    )

    if missing_in_grl:

        print()

        print(
            "Example missing in GRL:"
        )

        print(
            sorted(
                missing_in_grl
            )[
                :10
            ]
        )

    if missing_in_baseline:

        print()

        print(
            "Example missing in baseline:"
        )

        print(
            sorted(
                missing_in_baseline
            )[
                :10
            ]
        )

    if (
        missing_in_grl
        or
        missing_in_baseline
    ):

        raise RuntimeError(
            "Baseline and GRL real-test "
            "sample sets are not identical."
        )

    print()

    print(
        "Matched real sample guard : PASSED"
    )

    print(
        f"Matched samples           : "
        f"{len(baseline_ids)}"
    )


# ============================================================
# METADATA TRANSFER
# ============================================================

def transfer_grl_metadata_to_baseline(
    baseline,
    grl,
):

    metadata_columns = [
        "config_id",
        "room",
        "playback_device",
        "microphone",
        "recording_id",
        "speaker_id",
        "attack_id",
    ]

    available_metadata = [
        column
        for column
        in metadata_columns
        if column
        in grl.columns
    ]

    metadata_df = (
        grl[
            [
                "sample_id",
                *available_metadata,
            ]
        ]
        .copy()
    )

    # ========================================================
    # REMOVE EXISTING METADATA BEFORE MERGE
    # ========================================================

    for column in available_metadata:

        if column in baseline.columns:

            baseline = baseline.drop(
                columns=[
                    column
                ]
            )

    baseline = baseline.merge(
        metadata_df,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    print()

    print(
        "=" * 75
    )

    print(
        "METADATA TRANSFER AUDIT"
    )

    print(
        "=" * 75
    )

    print()

    print(
        "Transferred columns:"
    )

    print(
        available_metadata
    )

    for column in [
        "config_id",
        "room",
        "playback_device",
        "microphone",
    ]:

        if column not in baseline.columns:

            continue

        missing_count = int(
            baseline[
                column
            ]
            .isna()
            .sum()
        )

        print(
            f"{column:20s}: "
            f"missing={missing_count}"
        )

        if missing_count > 0:

            raise RuntimeError(
                f"Metadata transfer failed for "
                f"{column}: "
                f"{missing_count} missing values."
            )

    print()

    print(
        "Metadata transfer guard : PASSED"
    )

    return baseline


# ============================================================
# LABEL CONSISTENCY
# ============================================================

def validate_label_consistency(
    baseline,
    grl,
):

    baseline_labels = (
        baseline[
            [
                "sample_id",
                "source_file_id",
                "label",
            ]
        ]
        .copy()
        .rename(
            columns={
                "source_file_id":
                    "baseline_source_file_id",

                "label":
                    "baseline_label",
            }
        )
    )

    grl_labels = (
        grl[
            [
                "sample_id",
                "source_file_id",
                "label",
            ]
        ]
        .copy()
        .rename(
            columns={
                "source_file_id":
                    "grl_source_file_id",

                "label":
                    "grl_label",
            }
        )
    )

    merged = baseline_labels.merge(
        grl_labels,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )

    source_mismatch = (
        merged[
            "baseline_source_file_id"
        ]
        .astype(str)
        !=
        merged[
            "grl_source_file_id"
        ]
        .astype(str)
    )

    label_mismatch = (
        merged[
            "baseline_label"
        ]
        .astype(int)
        !=
        merged[
            "grl_label"
        ]
        .astype(int)
    )

    if source_mismatch.any():

        raise RuntimeError(
            "source_file_id mismatch between "
            "baseline and GRL."
        )

    if label_mismatch.any():

        raise RuntimeError(
            "Label mismatch between "
            "baseline and GRL."
        )

    print()

    print(
        "Source/label consistency guard : PASSED"
    )


# ============================================================
# FACTOR VALUE SORT
# ============================================================

def factor_values(
    df,
    factor,
):

    values = (
        df[
            factor
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return sorted(
        values
    )


# ============================================================
# ANALYZE FACTOR
# ============================================================

def analyze_factor(
    baseline,
    grl,
    factor,
):

    if (
        factor
        not in grl.columns
    ):

        print()

        print(
            f"SKIP {factor}: "
            "column unavailable."
        )

        return None

    values = factor_values(
        grl,
        factor,
    )

    rows = []

    for value in values:

        for (
            model_name,
            df,
            threshold,
        ) in [
            (
                "no_grl",
                baseline,
                BASELINE_VALIDATION_THRESHOLD,
            ),

            (
                "grl",
                grl,
                GRL_VALIDATION_THRESHOLD,
            ),
        ]:

            subset = (
                df[
                    df[
                        factor
                    ]
                    .astype(str)
                    .eq(
                        value
                    )
                ]
                .copy()
            )

            metrics = compute_metrics(
                subset,
                threshold,
            )

            row = {
                "factor":
                    factor,

                "value":
                    value,

                "model":
                    model_name,

                "validation_threshold":
                    float(
                        threshold
                    ),
            }

            row.update(
                metrics
            )

            rows.append(
                row
            )

    table = pd.DataFrame(
        rows
    )

    # ========================================================
    # ADD DELTA EER FOR CONVENIENCE
    # ========================================================

    delta_rows = []

    for value in values:

        baseline_row = (
            table[
                (
                    table[
                        "value"
                    ]
                    == value
                )
                &
                (
                    table[
                        "model"
                    ]
                    == "no_grl"
                )
            ]
        )

        grl_row = (
            table[
                (
                    table[
                        "value"
                    ]
                    == value
                )
                &
                (
                    table[
                        "model"
                    ]
                    == "grl"
                )
            ]
        )

        if (
            len(
                baseline_row
            )
            == 1
            and
            len(
                grl_row
            )
            == 1
        ):

            baseline_eer = float(
                baseline_row[
                    "eer"
                ].iloc[
                    0
                ]
            )

            grl_eer = float(
                grl_row[
                    "eer"
                ].iloc[
                    0
                ]
            )

            delta_rows.append(
                {
                    "factor":
                        factor,

                    "value":
                        value,

                    "baseline_eer":
                        baseline_eer,

                    "grl_eer":
                        grl_eer,

                    "delta_grl_minus_baseline":
                        (
                            grl_eer
                            - baseline_eer
                        ),
                }
            )

    delta_df = pd.DataFrame(
        delta_rows
    )

    return (
        table,
        delta_df,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 75
    )

    print(
        "CONTROLLED REAL REPLAY - FACTOR ANALYSIS"
    )

    print(
        "=" * 75
    )

    # ========================================================
    # LOAD + NORMALIZE
    # ========================================================

    baseline = normalize_score_file(
        BASELINE_SCORES,
        "No-GRL",
    )

    grl = normalize_score_file(
        GRL_SCORES,
        "GRL",
    )

    # ========================================================
    # MATCH SAMPLE SET
    # ========================================================

    match_score_files(
        baseline,
        grl,
    )

    # ========================================================
    # LABEL CONSISTENCY
    # ========================================================

    validate_label_consistency(
        baseline,
        grl,
    )

    # ========================================================
    # TRANSFER METADATA
    # ========================================================

    baseline = (
        transfer_grl_metadata_to_baseline(
            baseline,
            grl,
        )
    )

    # ========================================================
    # DISTRIBUTION AUDIT
    # ========================================================

    print()

    print(
        "=" * 75
    )

    print(
        "REAL TEST DISTRIBUTION"
    )

    print(
        "=" * 75
    )

    print()

    print(
        "Labels:"
    )

    print(
        grl[
            "label"
        ]
        .value_counts()
        .sort_index()
    )

    for factor in [
        "config_id",
        "room",
        "playback_device",
        "microphone",
    ]:

        if factor in grl.columns:

            print()

            print(
                f"{factor}:"
            )

            print(
                grl[
                    factor
                ]
                .value_counts(
                    dropna=False
                )
            )

    # ========================================================
    # OVERALL REAL
    # ========================================================

    overall_rows = []

    for (
        model_name,
        df,
        threshold,
    ) in [
        (
            "no_grl",
            baseline,
            BASELINE_VALIDATION_THRESHOLD,
        ),

        (
            "grl",
            grl,
            GRL_VALIDATION_THRESHOLD,
        ),
    ]:

        metrics = compute_metrics(
            df,
            threshold,
        )

        row = {
            "model":
                model_name,

            "validation_threshold":
                float(
                    threshold
                ),
        }

        row.update(
            metrics
        )

        overall_rows.append(
            row
        )

    overall_df = pd.DataFrame(
        overall_rows
    )

    print()

    print(
        "=" * 75
    )

    print(
        "OVERALL REAL REPLAY"
    )

    print(
        "=" * 75
    )

    print()

    print(
        overall_df[
            [
                "model",
                "n_samples",
                "n_bonafide",
                "n_spoof",
                "eer",
                "eer_threshold",
                "roc_auc",
                "bacc_at_val_threshold",
                "far_at_val_threshold",
                "frr_at_val_threshold",
                "mean_fake_score",
                "median_fake_score",
            ]
        ].to_string(
            index=False
        )
    )

    overall_df.to_csv(
        OUTPUT_DIR
        / "overall_real_comparison.csv",
        index=False,
    )

    # ========================================================
    # FACTOR ANALYSIS
    # ========================================================

    factor_tables = []

    delta_tables = []

    factors = [
        "config_id",
        "room",
        "microphone",
        "playback_device",
    ]

    for factor in factors:

        result = analyze_factor(
            baseline,
            grl,
            factor,
        )

        if result is None:

            continue

        (
            table,
            delta_df,
        ) = result

        factor_tables.append(
            table
        )

        delta_tables.append(
            delta_df
        )

        print()

        print(
            "=" * 75
        )

        print(
            f"FACTOR: {factor.upper()}"
        )

        print(
            "=" * 75
        )

        print()

        print(
            table[
                [
                    "value",
                    "model",
                    "n_samples",
                    "n_bonafide",
                    "n_spoof",
                    "eer",
                    "roc_auc",
                    "bacc_at_val_threshold",
                    "far_at_val_threshold",
                    "frr_at_val_threshold",
                ]
            ].to_string(
                index=False
            )
        )

        print()

        print(
            "EER delta (GRL - No-GRL):"
        )

        print()

        print(
            delta_df.to_string(
                index=False
            )
        )

        table.to_csv(
            OUTPUT_DIR
            / f"real_by_{factor}.csv",
            index=False,
        )

        delta_df.to_csv(
            OUTPUT_DIR
            / f"real_by_{factor}_delta.csv",
            index=False,
        )

    # ========================================================
    # COMBINED TABLES
    # ========================================================

    if factor_tables:

        combined = pd.concat(
            factor_tables,
            ignore_index=True,
        )

        combined.to_csv(
            OUTPUT_DIR
            / "real_factor_summary.csv",
            index=False,
        )

    if delta_tables:

        combined_delta = pd.concat(
            delta_tables,
            ignore_index=True,
        )

        combined_delta.to_csv(
            OUTPUT_DIR
            / "real_factor_delta_summary.csv",
            index=False,
        )

        print()

        print(
            "=" * 75
        )

        print(
            "COMBINED EER DELTA SUMMARY"
        )

        print(
            "=" * 75
        )

        print()

        print(
            combined_delta.to_string(
                index=False
            )
        )

    # ========================================================
    # SAVE MATCHED PAIR TABLE
    # ========================================================

    paired_baseline = (
        baseline[
            [
                "sample_id",
                "source_file_id",
                "label",
                "fake_score",
                "config_id",
                "room",
                "microphone",
                "playback_device",
            ]
        ]
        .copy()
        .rename(
            columns={
                "fake_score":
                    "baseline_fake_score"
            }
        )
    )

    paired_grl = (
        grl[
            [
                "sample_id",
                "fake_score",
            ]
        ]
        .copy()
        .rename(
            columns={
                "fake_score":
                    "grl_fake_score"
            }
        )
    )

    paired = paired_baseline.merge(
        paired_grl,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )

    paired[
        "score_delta_grl_minus_baseline"
    ] = (
        paired[
            "grl_fake_score"
        ]
        -
        paired[
            "baseline_fake_score"
        ]
    )

    paired.to_csv(
        OUTPUT_DIR
        / "paired_real_scores.csv",
        index=False,
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print(
        "=" * 75
    )

    print(
        "REAL FACTOR ANALYSIS COMPLETE"
    )

    print(
        "=" * 75
    )

    print()

    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()