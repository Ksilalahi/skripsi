from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SIM_MANIFEST = (
    ROOT
    / "manifests"
    / "manifest_replay_sim.csv"
)

THREE_CHANNEL_MANIFEST = (
    ROOT
    / "manifests"
    / "manifest_three_channel.csv"
)

SOURCE_MANIFEST = (
    ROOT
    / "manifests"
    / "source_manifest.csv"
)


# ============================================================
# POSSIBLE RIR / DOMAIN COLUMNS
# ============================================================

RIR_ID_CANDIDATES = [
    "rir_id",
    "rir_file",
    "rir_path",
    "rir_name",
    "rir",
]

RIR_GROUP_CANDIDATES = [
    "rir_split",
    "rir_group",
    "rir_domain",
    "rir_seen",
    "seen_unseen",
    "domain_split",
]

ROOM_CANDIDATES = [
    "room",
    "room_type",
    "rir_room",
    "environment",
]

SOURCE_ID_CANDIDATES = [
    "source_file_id",
    "source_id",
    "file_id",
]

SPLIT_CANDIDATES = [
    "split",
    "dataset_split",
]


# ============================================================
# HELPERS
# ============================================================

def find_existing_columns(
    df,
    candidates,
):

    return [
        column
        for column in candidates
        if column in df.columns
    ]


def print_header(
    title,
):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    print()


def load_manifest(
    path,
    name,
):

    if not path.exists():

        raise FileNotFoundError(
            f"{name} tidak ditemukan:\n"
            f"{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    if df.empty:

        raise RuntimeError(
            f"{name} kosong."
        )

    return df


def normalize_text_series(
    series,
):

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


# ============================================================
# COLUMN AUDIT
# ============================================================

def audit_columns(
    df,
    name,
):

    print_header(
        f"{name} - COLUMN AUDIT"
    )

    print(
        f"Rows    : {len(df)}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    print()

    for column in df.columns:

        print(
            f"- {column}"
        )


# ============================================================
# BASIC DISTRIBUTION
# ============================================================

def audit_basic_distribution(
    df,
    name,
):

    print_header(
        f"{name} - BASIC DISTRIBUTION"
    )

    if "condition" in df.columns:

        print(
            "Condition:"
        )

        print(
            df[
                "condition"
            ]
            .value_counts(
                dropna=False
            )
        )

        print()

    if "split" in df.columns:

        print(
            "Split:"
        )

        print(
            df[
                "split"
            ]
            .value_counts(
                dropna=False
            )
        )

        print()

    if "label" in df.columns:

        print(
            "Label:"
        )

        print(
            df[
                "label"
            ]
            .value_counts(
                dropna=False
            )
        )

        print()


# ============================================================
# RIR METADATA AUDIT
# ============================================================

def audit_rir_metadata(
    df,
    name,
):

    print_header(
        f"{name} - RIR METADATA AUDIT"
    )

    rir_id_columns = (
        find_existing_columns(
            df,
            RIR_ID_CANDIDATES,
        )
    )

    rir_group_columns = (
        find_existing_columns(
            df,
            RIR_GROUP_CANDIDATES,
        )
    )

    room_columns = (
        find_existing_columns(
            df,
            ROOM_CANDIDATES,
        )
    )

    print(
        "RIR identity columns:"
    )

    print(
        rir_id_columns
        if rir_id_columns
        else "NONE"
    )

    print()

    print(
        "RIR seen/unseen candidate columns:"
    )

    print(
        rir_group_columns
        if rir_group_columns
        else "NONE"
    )

    print()

    print(
        "Room/environment candidate columns:"
    )

    print(
        room_columns
        if room_columns
        else "NONE"
    )

    # ========================================================
    # PRINT VALUES
    # ========================================================

    all_columns = (
        rir_id_columns
        + rir_group_columns
        + room_columns
    )

    for column in all_columns:

        print()

        print(
            f"--- {column} ---"
        )

        nunique = (
            df[
                column
            ]
            .nunique(
                dropna=False
            )
        )

        print(
            f"Unique values: {nunique}"
        )

        counts = (
            df[
                column
            ]
            .value_counts(
                dropna=False
            )
            .head(
                30
            )
        )

        print(
            counts
        )


# ============================================================
# SEARCH SEEN / UNSEEN TEXT
# ============================================================

def search_seen_unseen_text(
    df,
    name,
):

    print_header(
        f"{name} - SEARCH 'SEEN' / 'UNSEEN'"
    )

    found_any = False

    for column in df.columns:

        if not (
            pd.api.types.is_object_dtype(
                df[column]
            )
            or
            pd.api.types.is_string_dtype(
                df[column]
            )
        ):

            continue

        values = (
            normalize_text_series(
                df[column]
            )
        )

        mask = (
            values.str.contains(
                "seen",
                na=False,
            )
            |
            values.str.contains(
                "unseen",
                na=False,
            )
        )

        if mask.any():

            found_any = True

            print(
                f"Column: {column}"
            )

            print(
                df.loc[
                    mask,
                    column
                ]
                .value_counts(
                    dropna=False
                )
                .head(
                    30
                )
            )

            print()

    if not found_any:

        print(
            "No explicit 'seen/unseen' text found."
        )


# ============================================================
# SIMULATED REPLAY SPLIT CROSS-TAB
# ============================================================

def audit_simulated_split(
    sim_df,
):

    print_header(
        "SIMULATED REPLAY - SPLIT CROSS TAB"
    )

    if "split" not in sim_df.columns:

        print(
            "Column 'split' tidak ditemukan."
        )

        return

    print(
        "Simulation rows by source split:"
    )

    print(
        sim_df[
            "split"
        ]
        .value_counts(
            dropna=False
        )
    )

    # ========================================================
    # CROSS-TAB WITH RIR METADATA
    # ========================================================

    candidates = (
        find_existing_columns(
            sim_df,
            RIR_GROUP_CANDIDATES
            + ROOM_CANDIDATES
        )
    )

    if not candidates:

        print()

        print(
            "Tidak ada RIR-group / room metadata "
            "yang bisa di-cross-tab dengan split."
        )

        return

    for column in candidates:

        print()

        print(
            f"split x {column}"
        )

        print(
            pd.crosstab(
                sim_df[
                    "split"
                ],
                sim_df[
                    column
                ],
                dropna=False,
            )
        )


# ============================================================
# SOURCE DERIVATIVE LEAKAGE AUDIT
# ============================================================

def audit_source_leakage(
    df,
    name,
):

    print_header(
        f"{name} - SOURCE DERIVATIVE SPLIT LEAKAGE"
    )

    source_columns = (
        find_existing_columns(
            df,
            SOURCE_ID_CANDIDATES,
        )
    )

    split_columns = (
        find_existing_columns(
            df,
            SPLIT_CANDIDATES,
        )
    )

    if not source_columns:

        print(
            "Tidak ada source ID column."
        )

        return

    if not split_columns:

        print(
            "Tidak ada split column."
        )

        return

    source_column = (
        source_columns[
            0
        ]
    )

    split_column = (
        split_columns[
            0
        ]
    )

    temp = (
        df[
            [
                source_column,
                split_column,
            ]
        ]
        .dropna()
        .copy()
    )

    per_source_splits = (
        temp
        .groupby(
            source_column
        )[
            split_column
        ]
        .nunique()
    )

    leaking_sources = (
        per_source_splits[
            per_source_splits
            > 1
        ]
    )

    print(
        f"Source column : {source_column}"
    )

    print(
        f"Split column  : {split_column}"
    )

    print(
        f"Unique source : "
        f"{temp[source_column].nunique()}"
    )

    print(
        f"Leak sources  : "
        f"{len(leaking_sources)}"
    )

    if len(
        leaking_sources
    ) > 0:

        print()

        print(
            "First leaking sources:"
        )

        print(
            leaking_sources.head(
                20
            )
        )

    else:

        print()

        print(
            "SOURCE-SPLIT LEAKAGE: NONE"
        )


# ============================================================
# THREE-CHANNEL SIMULATED AUDIT
# ============================================================

def audit_three_channel_simulated(
    df,
):

    print_header(
        "THREE-CHANNEL - SIMULATED ROW AUDIT"
    )

    if "condition" not in df.columns:

        print(
            "condition column missing."
        )

        return

    sim = (
        df[
            normalize_text_series(
                df[
                    "condition"
                ]
            )
            .eq(
                "simulated"
            )
        ]
        .copy()
    )

    print(
        f"Simulated rows: {len(sim)}"
    )

    if sim.empty:

        return

    if "split" in sim.columns:

        print()

        print(
            "Simulated split distribution:"
        )

        print(
            sim[
                "split"
            ]
            .value_counts(
                dropna=False
            )
        )

    # Print useful metadata columns only
    useful_candidates = [
        "global_feature_id",
        "feature_id",
        "source_file_id",
        "split",
        "condition",
        "label",
        "rir_id",
        "rir_file",
        "rir_path",
        "rir_split",
        "rir_group",
        "rir_seen",
        "room",
        "room_type",
        "environment",
    ]

    useful = [
        c
        for c in useful_candidates
        if c in sim.columns
    ]

    print()

    print(
        "Available simulated metadata:"
    )

    print(
        useful
    )

    if useful:

        print()

        print(
            "First 20 simulated rows:"
        )

        print(
            sim[
                useful
            ]
            .head(
                20
            )
            .to_string(
                index=False
            )
        )


# ============================================================
# SUMMARY DECISION
# ============================================================

def protocol_summary(
    sim_df,
    three_df,
):

    print_header(
        "PROTOCOL AUDIT SUMMARY"
    )

    sim_rir_group_columns = (
        find_existing_columns(
            sim_df,
            RIR_GROUP_CANDIDATES,
        )
    )

    three_rir_group_columns = (
        find_existing_columns(
            three_df,
            RIR_GROUP_CANDIDATES,
        )
    )

    sim_rir_identity_columns = (
        find_existing_columns(
            sim_df,
            RIR_ID_CANDIDATES,
        )
    )

    three_rir_identity_columns = (
        find_existing_columns(
            three_df,
            RIR_ID_CANDIDATES,
        )
    )

    print(
        "manifest_replay_sim.csv:"
    )

    print(
        "  explicit RIR seen/unseen metadata :",
        (
            "YES"
            if sim_rir_group_columns
            else "NO"
        ),
    )

    print(
        "  RIR identity metadata             :",
        (
            "YES"
            if sim_rir_identity_columns
            else "NO"
        ),
    )

    print()

    print(
        "manifest_three_channel.csv:"
    )

    print(
        "  explicit RIR seen/unseen metadata :",
        (
            "YES"
            if three_rir_group_columns
            else "NO"
        ),
    )

    print(
        "  RIR identity metadata             :",
        (
            "YES"
            if three_rir_identity_columns
            else "NO"
        ),
    )

    print()

    if (
        sim_rir_group_columns
        and
        three_rir_group_columns
    ):

        print(
            "STATUS: explicit seen/unseen metadata "
            "tersedia sampai final feature manifest."
        )

    elif (
        sim_rir_group_columns
        and
        not three_rir_group_columns
    ):

        print(
            "STATUS: seen/unseen metadata tersedia "
            "di simulation manifest tetapi HILANG "
            "saat feature integration."
        )

        print(
            "NEXT: propagate metadata ke "
            "manifest_three_channel.csv."
        )

    elif (
        sim_rir_identity_columns
        and
        not sim_rir_group_columns
    ):

        print(
            "STATUS: RIR identity tersedia tetapi "
            "belum ada explicit seen/unseen label."
        )

        print(
            "NEXT: derive seen/unseen dari daftar RIR "
            "yang sudah ditetapkan pada Tahap 2."
        )

    else:

        print(
            "STATUS: metadata belum cukup untuk "
            "memverifikasi simulated seen/unseen."
        )

        print(
            "NEXT: simulation manifest perlu ditelusuri "
            "lebih lanjut sebelum final training."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "REPLAY PROTOCOL AUDIT"
    )

    print(
        "Goal:"
    )

    print(
        "1. Verify simulated seen/unseen metadata."
    )

    print(
        "2. Verify source-level split integrity."
    )

    print(
        "3. Verify metadata survives into "
        "three-channel manifest."
    )

    # ========================================================
    # LOAD
    # ========================================================

    sim_df = load_manifest(
        SIM_MANIFEST,
        "manifest_replay_sim.csv",
    )

    three_df = load_manifest(
        THREE_CHANNEL_MANIFEST,
        "manifest_three_channel.csv",
    )

    source_df = load_manifest(
        SOURCE_MANIFEST,
        "source_manifest.csv",
    )

    # ========================================================
    # SIMULATION MANIFEST
    # ========================================================

    audit_columns(
        sim_df,
        "SIMULATION MANIFEST",
    )

    audit_basic_distribution(
        sim_df,
        "SIMULATION MANIFEST",
    )

    audit_rir_metadata(
        sim_df,
        "SIMULATION MANIFEST",
    )

    search_seen_unseen_text(
        sim_df,
        "SIMULATION MANIFEST",
    )

    audit_simulated_split(
        sim_df
    )

    audit_source_leakage(
        sim_df,
        "SIMULATION MANIFEST",
    )

    # ========================================================
    # THREE-CHANNEL
    # ========================================================

    audit_basic_distribution(
        three_df,
        "THREE-CHANNEL MANIFEST",
    )

    audit_rir_metadata(
        three_df,
        "THREE-CHANNEL MANIFEST",
    )

    search_seen_unseen_text(
        three_df,
        "THREE-CHANNEL MANIFEST",
    )

    audit_three_channel_simulated(
        three_df
    )

    audit_source_leakage(
        three_df,
        "THREE-CHANNEL MANIFEST",
    )

    # ========================================================
    # SOURCE MANIFEST
    # ========================================================

    audit_source_leakage(
        source_df,
        "SOURCE MANIFEST",
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    protocol_summary(
        sim_df,
        three_df,
    )

    print()

    print("=" * 78)

    print(
        "REPLAY PROTOCOL AUDIT COMPLETE"
    )

    print("=" * 78)


if __name__ == "__main__":

    main()