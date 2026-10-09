"""Frozen-embedding room-domain probe for ReplayResNet18 V1 (no retraining).

Copy to src/domain_probe_multiseed.py, then execute from project root:
 python src/domain_probe_multiseed.py --seed 2026
 python src/domain_probe_multiseed.py --seed 2027
 python src/domain_probe_multiseed.py --seed 42

Data: only simulated + seen RIR, train split -> train logistic probe;
validation split -> independent probe evaluation. Compare No-GRL / GRL
using identical samples and probe settings. Do not use domain_head logits.
"""
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from dataset_replay import ReplayDataset, replay_collate_fn
from model_replay import ReplayResNet

ROOT = Path(__file__).resolve().parents[1]


def select_domain_examples(split):
    dataset = ReplayDataset(split=split)
    df = dataset.df.reset_index(drop=True).copy()
    needed = {"condition", "rir_split", "domain_label"}
    if missing := needed - set(df.columns):
        raise ValueError(f"Dataset {split} kehilangan kolom {missing}")
    cond = df["condition"].astype(str).str.lower().str.strip()
    rs = df["rir_split"].astype(str).str.lower().str.strip()
    dom = pd.to_numeric(df["domain_label"], errors="coerce")
    mask = cond.eq("simulated") & rs.eq("seen") & dom.ge(0)
    ids = np.flatnonzero(mask.to_numpy()).tolist()
    selected = df.iloc[ids].reset_index(drop=True)
    if len(selected) < 4:
        raise ValueError(f"{split}: terlalu sedikit data domain, n={len(selected)}")
    if selected["domain_label"].nunique() != 4:
        raise ValueError(f"{split}: diharapkan 4 kelas domain, ditemukan {selected['domain_label'].unique()}")
    return Subset(dataset, ids), selected


def load_frozen_model(file, use_grl):
    if not file.is_file():
        raise FileNotFoundError(file)
    # Checkpoints produced locally by train.py/train_grl.py are trusted input.
    ckpt = torch.load(file, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    dim = int(ckpt.get("config", {}).get("embedding_dim", 256))
    model = ReplayResNet(
        n_transform=2, embedding_dim=dim, n_domain=4,
        use_domain_head=use_grl, grl_strength=1.0,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"  Checkpoint OK: {file} | epoch={ckpt.get('epoch', '?')} | dim={dim}")
    return model


@torch.inference_mode()
def extract_embeddings(model, subset, batch_size):
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=replay_collate_fn)
    vectors, domains = [], []
    for batch in loader:
        x = batch["features"].to("cpu")
        dom = batch["domain_labels"].cpu().numpy().astype(int)
        embedding = model.encode(x).detach().cpu().numpy()
        vectors.append(embedding)
        domains.append(dom)
    return np.concatenate(vectors), np.concatenate(domains)


def run_probe(seed, args):
    basen = args.baseline or ROOT / "checkpoints" / "replay_resnet" / f"seed_{seed}" / "best_model.pt"
    grln = args.grl or ROOT / "checkpoints" / "replay_resnet_grl" / f"seed_{seed}_ep5" / "best_model.pt"
    train_subset, train_df = select_domain_examples("train")
    val_subset, val_df = select_domain_examples("validation")
    print(f"Domain samples: train={len(train_df)}, validation={len(val_df)}")
    print("Train domain counts:", train_df['domain_label'].value_counts().sort_index().to_dict())
    print("Validation domain counts:", val_df['domain_label'].value_counts().sort_index().to_dict())
    if "source_file_id" in train_df.columns and "source_file_id" in val_df.columns:
        both = set(train_df['source_file_id'].astype(str)) & set(val_df['source_file_id'].astype(str))
        if both:
            raise ValueError(f"Source leakage train/validation: {sorted(both)[:5]}")
        print(f"Unique source IDs: train={train_df['source_file_id'].nunique()}, "
              f"validation={val_df['source_file_id'].nunique()} (disjoint)")

    rows = []
    out = args.out or ROOT / "results" / "domain_probe_multiseed"
    out.mkdir(parents=True, exist_ok=True)
    for method, file, grl in (("No-GRL", Path(basen), False), ("GRL", Path(grln), True)):
        model = load_frozen_model(file, grl)
        xtrain, ytrain = extract_embeddings(model, train_subset, args.batch_size)
        xval, yval = extract_embeddings(model, val_subset, args.batch_size)
        expected_train = train_df['domain_label'].to_numpy(dtype=int)
        expected_val = val_df['domain_label'].to_numpy(dtype=int)
        if not np.array_equal(ytrain, expected_train) or not np.array_equal(yval, expected_val):
            raise RuntimeError(f"Label/order mismatch pada {method}")
        if not np.isfinite(xtrain).all() or not np.isfinite(xval).all():
            raise RuntimeError(f"Embedding mengandung NaN/Inf untuk {method}")
        probe = make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=3000, solver="lbfgs", random_state=args.probe_seed
        ))
        probe.fit(xtrain, ytrain)
        pred = probe.predict(xval)
        acc = float(accuracy_score(yval, pred))
        bacc = float(balanced_accuracy_score(yval, pred))
        matrix = confusion_matrix(yval, pred, labels=[0,1,2,3])
        pd.DataFrame(matrix, index=[f"true_{i}" for i in range(4)],
                     columns=[f"pred_{i}" for i in range(4)]).to_csv(
                         out / f"confusion_seed_{seed}_{'grl' if grl else 'no_grl'}.csv")
        rows.append(dict(seed=seed, method=method, train_n=len(ytrain), validation_n=len(yval),
                         n_train_sources=train_df['source_file_id'].nunique() if 'source_file_id' in train_df else None,
                         n_validation_sources=val_df['source_file_id'].nunique() if 'source_file_id' in val_df else None,
                         probe_accuracy=acc, probe_balanced_accuracy=bacc,
                         chance_balanced_accuracy=0.25, checkpoint=str(file)))
        print(f"{method}: accuracy={acc:.4f} | balanced_accuracy={bacc:.4f} | "
              f"chance BAcc=0.2500")
    pd.DataFrame(rows).to_csv(out / f"domain_probe_seed_{seed}.csv", index=False)
    print(f"Saved: {out / f'domain_probe_seed_{seed}.csv'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=[42,2026,2027], required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--probe-seed", type=int, default=2026)
    p.add_argument("--baseline", type=Path, help="Override baseline checkpoint")
    p.add_argument("--grl", type=Path, help="Override GRL checkpoint")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.seed == 2026:
        candidate = ROOT / 'checkpoints' / 'replay_resnet' / 'seed_2026_fixed' / 'best_model.pt'
        if candidate.exists() and args.baseline is None:
            args.baseline = candidate
    run_probe(args.seed, args)

if __name__ == "__main__":
    main()