"""
Prepare a tiny dataset for the DeniMS end-to-end smoke test.

This script makes the full pipeline runnable on a machine without the full
FragHub corpus. It looks for an already-filtered FragHub parquet file, and if
only the raw parquet is present it falls back to running `pixi run prep-data`
to materialise the filtered version. It then deterministically subsamples
1100 rows (1000 train + 100 test) and writes:

    tests/e2e/data/tiny.parquet
    tests/e2e/data/tiny_smiles_dict.pt
    tests/e2e/data/tiny_smiles_canonical.txt
    tests/e2e/data/splits_tiny_random.pkl

All paths are anchored to the repository root so the script can be invoked
from any directory (it uses its own location to locate the repo root).
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TINY_DIR = REPO_ROOT / "tests" / "e2e" / "data"
FRAGHUB_DIR = REPO_ROOT / "Preprocessing" / "fraghub"
FILTERED_PARQUET = FRAGHUB_DIR / "fraghub_filtered.parquet"
RAW_PARQUET = FRAGHUB_DIR / "fraghub.parquet"
ZENODO_RECORD = "19060052"
TINY_PARQUET = DEFAULT_TINY_DIR / "tiny.parquet"
TINY_SMILES_TXT = DEFAULT_TINY_DIR / "tiny_smiles_canonical.txt"
TINY_SMILES_DICT = DEFAULT_TINY_DIR / "tiny_smiles_dict.pt"
TINY_SPLITS = DEFAULT_TINY_DIR / "splits_tiny_random.pkl"
# Per spec: 1000 train + 100 test. We allocate a small non-empty validation
# split (5 rows) because the encoder's training loop unconditionally calls
# `next(iter(val_loader))` per epoch and an empty val set raises StopIteration.
TINY_NUM_TRAIN = 1000
TINY_NUM_VAL = 5
TINY_NUM_TEST = 100
TINY_TOTAL = TINY_NUM_TRAIN + TINY_NUM_VAL + TINY_NUM_TEST  # 1105
TINY_SEED = 42


def _print(msg: str) -> None:
    print(f"[prepare_tiny_data] {msg}", flush=True)


def ensure_filtered_parquet(verbose: bool = True) -> Path:
    """Make sure `fraghub_filtered.parquet` exists, running prep-data if needed."""
    if FILTERED_PARQUET.exists():
        if verbose:
            _print(f"Found filtered parquet: {FILTERED_PARQUET}")
        return FILTERED_PARQUET

    if not RAW_PARQUET.exists():
        sys.stderr.write(
            "\nERROR: Could not find FragHub data.\n"
            f"  Looked for: {FILTERED_PARQUET}\n"
            f"  Looked for: {RAW_PARQUET}\n\n"
            "To get started, download `FragHub_filtered.parquet` from Zenodo "
            f"record {ZENODO_RECORD} "
            f"(https://zenodo.org/records/{ZENODO_RECORD}) and place it at:\n"
            f"  {FILTERED_PARQUET}\n\n"
            "If you instead have the *raw* `fraghub.parquet` (pre-filtering), "
            "place it at:\n"
            f"  {RAW_PARQUET}\n"
            "and the script will run `pixi run prep-data` automatically.\n\n"
            "After providing the data, re-run `pixi run test-prepare-data` "
            "and then `pixi run test-e2e`.\n"
        )
        raise FileNotFoundError(str(RAW_PARQUET))

    _print(f"Filtered parquet missing; running pixi run prep-data on {RAW_PARQUET}.")
    cmd = [
        "pixi", "run", "prep-data",
        "-input_parquet", str(RAW_PARQUET),
        "-generate_graph_dict",
        "-split_type", "random",
        "-val_fraction", "0.0",
        "-test_fraction", "0.05",
    ]
    _print("Executing: " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    if not FILTERED_PARQUET.exists():
        raise RuntimeError(
            f"prep-data did not produce expected file: {FILTERED_PARQUET}"
        )
    return FILTERED_PARQUET


def select_tiny_rows(filtered_parquet: Path, n_total: int, seed: int) -> pd.DataFrame:
    """Deterministically pick `n_total` rows covering `n_total` unique SMILES."""
    _print(f"Loading filtered parquet: {filtered_parquet}")
    df = pq.read_table(filtered_parquet, use_threads=True).to_pandas()
    _print(f"  Loaded shape: {df.shape}")

    required_cols = {"smiles"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Filtered parquet is missing required columns: {sorted(missing)}"
        )

    # Keep only rows that have valid SMILES and at least one fragment-formula
    # entry (the encoder needs a non-empty spectrum).
    has_smiles = df["smiles"].notna() & (df["smiles"].astype(str).str.len() > 0)
    if "cleaned_fragment_formulas_str" in df.columns:
        has_spectrum = df["cleaned_fragment_formulas_str"].notna()
        keep = has_smiles & has_spectrum
    else:
        keep = has_smiles
    df = df.loc[keep].reset_index(drop=True)
    _print(f"  After validity filter: {len(df)} rows")

    if len(df) < n_total:
        raise ValueError(
            f"Filtered parquet has only {len(df)} valid rows; need at least "
            f"{n_total}. Aborting tiny dataset construction."
        )

    # Deduplicate by SMILES so each unique molecule appears at most once.
    # Then shuffle deterministically and take the first n_total.
    unique_df = df.drop_duplicates(subset=["smiles"], keep="first").reset_index(drop=True)
    if len(unique_df) < n_total:
        raise ValueError(
            f"Filtered parquet has only {len(unique_df)} unique SMILES; need "
            f"at least {n_total}. Aborting."
        )

    sampled = unique_df.sample(n=n_total, random_state=seed).reset_index(drop=True)
    _print(f"  Selected {len(sampled)} unique-SMILES rows (seed={seed}).")
    return sampled


def write_tiny_parquet(tiny_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiny_df.to_parquet(out_path)
    _print(f"Wrote tiny parquet: {out_path}")


def run_graph_dict(tiny_parquet: Path, dict_path: Path) -> None:
    cmd = [
        "pixi", "run", "graph-dict",
        "-input_parquet", str(tiny_parquet),
        "-output_dict", str(dict_path),
    ]
    _print("Executing: " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    if not dict_path.exists():
        raise RuntimeError(f"graph-dict did not produce {dict_path}")


def write_splits(tiny_df: pd.DataFrame, splits_path: Path,
                 num_train: int, num_val: int, num_test: int) -> None:
    """Write the split pickle. Uses the row order of `tiny_df` directly."""
    if num_train + num_val + num_test > len(tiny_df):
        raise ValueError(
            f"Requested {num_train} train + {num_val} val + {num_test} test "
            f"rows, but tiny parquet only has {len(tiny_df)} rows."
        )
    smiles_list = tiny_df["smiles"].astype(str).tolist()
    train_smiles = smiles_list[:num_train]
    val_smiles = smiles_list[num_train:num_train + num_val]
    test_smiles = smiles_list[num_train + num_val:num_train + num_val + num_test]

    splits = {"train": train_smiles, "val": val_smiles, "test": test_smiles}
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_path, "wb") as fh:
        pickle.dump(splits, fh)
    _print(
        f"Wrote splits: {splits_path} "
        f"(train={len(train_smiles)}, val={len(val_smiles)}, "
        f"test={len(test_smiles)})"
    )


def write_canonical_smiles(tiny_df: pd.DataFrame, out_path: Path) -> None:
    """Write the canonical SMILES list in the same order as the tiny parquet."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for smi in tiny_df["smiles"].astype(str).tolist():
            fh.write(f"{smi}\n")
    _print(f"Wrote canonical SMILES: {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-train", type=int, default=TINY_NUM_TRAIN)
    parser.add_argument("--num-val", type=int, default=TINY_NUM_VAL)
    parser.add_argument("--num-test", type=int, default=TINY_NUM_TEST)
    parser.add_argument("--seed", type=int, default=TINY_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_TINY_DIR)
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    tiny_parquet = out_dir / "tiny.parquet"
    tiny_smiles_dict = out_dir / "tiny_smiles_dict.pt"
    tiny_smiles_txt = out_dir / "tiny_smiles_canonical.txt"
    tiny_splits = out_dir / "splits_tiny_random.pkl"

    filtered = ensure_filtered_parquet()
    n_total = args.num_train + args.num_val + args.num_test
    tiny_df = select_tiny_rows(filtered, n_total=n_total, seed=args.seed)

    write_tiny_parquet(tiny_df, tiny_parquet)
    write_canonical_smiles(tiny_df, tiny_smiles_txt)
    run_graph_dict(tiny_parquet, tiny_smiles_dict)
    write_splits(tiny_df, tiny_splits, args.num_train, args.num_val, args.num_test)

    print()
    print("Tiny dataset created:")
    for p in [tiny_parquet, tiny_smiles_txt, tiny_smiles_dict, tiny_splits]:
        print(f"  - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
