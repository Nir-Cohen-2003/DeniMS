#!/usr/bin/env python
"""
Build a DeniMS *training* parquet from an MSP / MGF / MSPEC spectral library.

This script uses HRMS_utils (without modifying it) to:
  1. Read & annotate the input spectral library via the same
     ``process_spectral_library`` function that the
     ``build-spectral-library`` CLI wraps (PubChem enrichment is skipped —
     only entries that already carry SMILES / InChI are kept).
  2. Remap the HRMS_utils 12-element formula arrays
     (``[H, C, N, O, F, Na, P, S, Cl, K, Br, I]``) down to the DeniMS
     9-element order ``[H, C, N, O, F, S, Cl, Br, I]`` and emit the exact
     training schema described in DATA_FORMATS.md §2.

Output columns: smiles, precursor_type, collision_energy_NCE,
clean_spectrum_formula_array, spectral_information_score.

Run with the HRMS_utils pixi environment on PATH, e.g.:
    pixi run -e scripts python denims_converters/build_training_parquet.py <input> -o out.parquet
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from time import perf_counter

import polars as pl

# Importing hrms_utils also registers the ``mass_decomposition`` Polars
# expression namespace (via hrms_core), which the library uses internally.
from hrms_utils.formats.spectral_library import process_spectral_library
from hrms_utils.formula_annotation.element_table import ELEMENT_SYMBOLS

# RDKit is a transitive dependency of parallel_rdkit (an HRMS_utils dep) and is
# used only for the SMILES-level filters required by DATA_FORMATS.md §2.5.
from rdkit import Chem


# ---------------------------------------------------------------------------
# Element-order bridge: HRMS_utils (12, mass-ordered) -> DeniMS (9)
# ---------------------------------------------------------------------------
# HRMS_utils order (by increasing monoisotopic mass):
#   [H, C, N, O, F, Na, P, S, Cl, K, Br, I]
# DeniMS order (DATA_FORMATS.md §0):
#   [H, C, N, O, F, S, Cl, Br, I]
HRMS_INDEX = {sym: i for i, sym in enumerate(ELEMENT_SYMBOLS)}
DENIMS_ELEMENTS = ["H", "C", "N", "O", "F", "S", "Cl", "Br", "I"]
# For each DeniMS element, the index it occupies in the HRMS 12-vector.
REMAP_HRMS_TO_DENIMS = [HRMS_INDEX[sym] for sym in DENIMS_ELEMENTS]
# HRMS elements that DeniMS cannot represent (must be zero for a row to survive).
UNSUPPORTED_HRMS_INDICES = [
    i for i, sym in enumerate(ELEMENT_SYMBOLS) if sym not in DENIMS_ELEMENTS
]

# DeniMS-allowed heavy atoms (SMILES may contain only these + H).
ALLOWED_ATOMIC_NUMBERS = {  # H is allowed as hydrogen, not counted as heavy
    6,   # C
    7,   # N
    8,   # O
    9,   # F
    16,  # S
    17,  # Cl
    35,  # Br
    53,  # I
}


def collect_library_files(path: Path) -> list[Path]:
    """Collect supported spectral-library files from a file or directory path."""
    valid_suffixes = {".msp", ".mspec", ".mgf", ".MSP", ".MSPEC", ".MGF"}
    if path.is_file():
        assert path.suffix in valid_suffixes, (
            f"File {path} does not have a valid library suffix: {path.suffix}"
        )
        return [path]
    if path.is_dir():
        files = [f for f in path.iterdir() if f.is_file() and f.suffix in valid_suffixes]
        assert len(files) > 0, f"No library files found in directory: {path}"
        return sorted(files)
    raise ValueError(f"Path does not exist or is not a file/directory: {path}")


def smiles_is_denims_compatible(smiles: str, max_heavy_atoms: int) -> bool:
    """Return True iff ``smiles`` parses, has < max_heavy_atoms heavy atoms,
    and every atom is in {C, N, O, F, S, Cl, Br, I} (plus H)."""
    if smiles is None:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    if mol.GetNumHeavyAtoms() >= max_heavy_atoms:
        return False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in ALLOWED_ATOMIC_NUMBERS:
            return False
    return True


def build_valid_smiles_set(smiles_series: pl.Series, max_heavy_atoms: int) -> set[str]:
    """Evaluate the unique SMILES once and return the subset that passes the
    DeniMS compatibility filter. Iterating over *unique* SMILES (not over every
    spectrum) keeps this O(molecules) rather than O(spectra)."""
    unique = smiles_series.drop_nulls().unique().to_list()
    return {s for s in unique if smiles_is_denims_compatible(s, max_heavy_atoms)}


def remap_formula_arrays_expr(col: str) -> pl.Expr:
    """Polars expression that remaps a ``List[Array[Int32, 12]]`` column to the
    DeniMS ``List[Array[Int32, 9]]`` order, picking the 9 supported elements."""
    return (
        pl.col(col)
        .list.eval(
            pl.concat_list([pl.element().arr.get(i) for i in REMAP_HRMS_TO_DENIMS])
            .list.to_array(len(DENIMS_ELEMENTS))
        )
        .alias(col)
    )


def has_unsupported_elements_expr(col: str) -> pl.Expr:
    """True when any peak in the 12-vector list has a non-zero count for an
    element DeniMS cannot represent (Na, P, K)."""
    # Sum the unsupported element counts per peak with explicit addition
    # (``pl.sum_horizontal`` is not usable inside ``list.eval``).
    unsupported_sum = pl.element().arr.get(UNSUPPORTED_HRMS_INDICES[0])
    for i in UNSUPPORTED_HRMS_INDICES[1:]:
        unsupported_sum = unsupported_sum + pl.element().arr.get(i)
    return pl.col(col).list.eval(unsupported_sum).list.max() > 0


def build_training_parquet(
    input_path: Path,
    output_path: Path,
    library_parquet: Path | None,
    raw_fragment_tolerance_ppm: float,
    normalized_fragment_tolerance_ppm: float,
    molecular_ion_tolerance_ppm: float,
    min_explained_intensity: float,
    deduplicate: bool,
    clean_identifiers: bool,
    max_heavy_atoms: int,
    min_peaks: int,
    max_peaks: int,
    min_nce: float,
    max_nce: float,
    log_path: Path | None,
) -> None:
    t0 = perf_counter()

    # --- Step 1: produce the annotated HRMS_utils library parquet ----------
    if library_parquet is None:
        library_files = collect_library_files(input_path)
        if input_path.is_file():
            library_parquet = input_path.with_suffix(".parquet")
            inchikey_changes_path = input_path.with_suffix(".inchikey_changes.csv")
        else:
            library_parquet = input_path / f"{input_path.name}.parquet"
            inchikey_changes_path = input_path / f"{input_path.name}.inchikey_changes.csv"

        if log_path is None:
            log_path = library_parquet.with_suffix(".log")
        # Pass a file-like object as ``logger``: the current API accepts either
        # a logging.Logger or a TextIO, and older builds do ``print(msg,
        # file=logger)``, so a file handle is the common denominator.
        logger = open(log_path, "w")

        print(f"[1/2] Building annotated library parquet -> {library_parquet}")

        # Build the call adaptively so the script works against both the
        # current ``process_spectral_library`` API (writes to ``output_path``
        # and returns a LazyFrame) and older installed builds that lack some
        # kwargs and return an in-memory DataFrame. PubChem is always skipped
        # (``pubchem_path=None``): only SMILES/InChI-bearing entries are kept.
        accepted = set(inspect.signature(process_spectral_library).parameters)
        kwargs: dict = dict(
            files=list(library_files),
            raw_fragment_tolerance_ppm=raw_fragment_tolerance_ppm,
            normalized_fragment_tolerance_ppm=normalized_fragment_tolerance_ppm,
            molecular_ion_tolerance_ppm=molecular_ion_tolerance_ppm,
            pubchem_path=None,
            min_explained_intensity=min_explained_intensity,
            logger=logger,
        )
        if "deduplicate" in accepted:
            kwargs["deduplicate"] = deduplicate
        if "clean_identifiers" in accepted:
            kwargs["clean_identifiers"] = clean_identifiers
        if "inchikey_changes_path" in accepted:
            kwargs["inchikey_changes_path"] = inchikey_changes_path
        if "output_path" in accepted:
            kwargs["output_path"] = library_parquet

        result = process_spectral_library(**kwargs)

        # Older builds return an in-memory DataFrame and never write to disk;
        # persist it ourselves so step 2 can scan a stable path.
        if "output_path" not in accepted:
            if isinstance(result, pl.LazyFrame):
                result = result.collect()
            result.write_parquet(library_parquet)
        logger.close()
    else:
        print(f"[1/2] Using existing library parquet -> {library_parquet}")

    # --- Step 2: transform to the DeniMS training schema ------------------
    print(f"[2/2] Transforming to DeniMS training schema -> {output_path}")
    lf = pl.scan_parquet(library_parquet)

    # Peak count is the length of the (still 12-wide) per-peak formula list.
    lf = lf.with_columns(
        pl.col("cleaned_fragment_formulas").list.len().alias("num_clean_peaks")
    )

    # Cheap, vectorized filters first. The unsupported-elements check must run
    # on the original 12-wide column (Na/P/K indices exist only there), so it
    # is applied BEFORE the remap to the 9-wide DeniMS order below.
    lf = lf.filter(
        pl.col("smiles").is_not_null()
        & pl.col("precursor_type").is_in(["[M+H]+", "[M-H]-"])
        & pl.col("collision_energy_NCE").is_not_null()
        & (pl.col("num_clean_peaks") > min_peaks)
        & (pl.col("num_clean_peaks") < max_peaks)
        & (pl.col("collision_energy_NCE") > min_nce)
        & (pl.col("collision_energy_NCE") < max_nce)
        & ~has_unsupported_elements_expr("cleaned_fragment_formulas")
    )

    # Now remap the 12-wide per-peak formula list to the DeniMS 9-element order
    # and rename to the exact training column name.
    lf = lf.with_columns(
        remap_formula_arrays_expr("cleaned_fragment_formulas").alias(
            "clean_spectrum_formula_array"
        )
    ).drop("cleaned_fragment_formulas", "num_clean_peaks")

    df = lf.collect()

    # SMILES-level filter (DATA_FORMATS.md §2.5): only the 8 allowed heavy
    # elements and < max_heavy_atoms heavy atoms. Evaluated over unique SMILES.
    valid_smiles = build_valid_smiles_set(df["smiles"], max_heavy_atoms)
    df = df.filter(pl.col("smiles").is_in(list(valid_smiles)))

    # Spectral information score is optional; default to 1.0 when absent.
    if "spectral_information_score" not in df.columns:
        df = df.with_columns(pl.lit(1.0).alias("spectral_information_score"))
    else:
        df = df.with_columns(
            pl.col("spectral_information_score").fill_null(1.0).cast(pl.Float64)
        )

    out = df.select(
        pl.col("smiles").cast(pl.Utf8),
        pl.col("precursor_type").cast(pl.Utf8),
        pl.col("collision_energy_NCE").cast(pl.Float64),
        pl.col("clean_spectrum_formula_array"),
        pl.col("spectral_information_score"),
    )

    out.write_parquet(output_path)

    print(f"Wrote {out.height} training spectra ({out['smiles'].n_unique()} molecules) "
          f"to {output_path} in {perf_counter() - t0:.2f}s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a DeniMS training parquet from an MSP/MGF/MSPEC spectral "
            "library using HRMS_utils."
        )
    )
    p.add_argument("input_path", type=Path,
                   help="Spectral library file (.msp/.mspec/.mgf) or a directory of them.")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output DeniMS training parquet (default: <input>_denims_training.parquet).")
    p.add_argument("--library-parquet", type=Path, default=None,
                   help="Skip step 1 and transform this existing HRMS_utils library parquet instead.")

    # Pass-through to process_spectral_library (annotation accuracy).
    p.add_argument("--raw-fragment-tolerance-ppm", type=float, default=10.0)
    p.add_argument("--normalized-fragment-tolerance-ppm", type=float, default=5.0)
    p.add_argument("--molecular-ion-tolerance-ppm", type=float, default=5.0)
    p.add_argument("--min-explained-intensity", type=float, default=0.0,
                   help="Min explained intensity during library annotation (default 0.0 = keep all).")
    p.add_argument("--deduplicate", action="store_true",
                   help="Run pairwise spectrum deduplication (off by default; training wants broad coverage).")
    p.add_argument("--no-clean-identifiers", action="store_true",
                   help="Skip MS-Ready identifier standardization (not recommended).")
    p.add_argument("--log-file", type=Path, default=None)

    # DeniMS §2.5 filtering knobs.
    p.add_argument("--max-heavy-atoms", type=int, default=30,
                   help="Drop SMILES with >= this many heavy atoms (DeniMS §2.5).")
    p.add_argument("--min-peaks", type=int, default=3,
                   help="Min number of clean peaks (DeniMS §2.5: > 2).")
    p.add_argument("--max-peaks", type=int, default=127,
                   help="Max number of clean peaks, inclusive (DeniMS §2.5: < 128).")
    p.add_argument("--min-nce", type=float, default=5.0,
                   help="Min collision_energy_NCE (DeniMS §2.5: > 4).")
    p.add_argument("--max-nce", type=float, default=300.0,
                   help="Max collision_energy_NCE (DeniMS §2.5: < 300).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path.resolve()
    assert input_path.exists(), f"Input path does not exist: {input_path}"

    if args.output is not None:
        output_path = args.output.resolve()
    elif args.library_parquet is not None:
        output_path = args.library_parquet.resolve().with_suffix("").parent / "denims_training.parquet"
    elif input_path.is_file():
        output_path = input_path.with_suffix(".denims_training.parquet")
    else:
        output_path = input_path / f"{input_path.name}.denims_training.parquet"

    build_training_parquet(
        input_path=input_path,
        output_path=output_path,
        library_parquet=args.library_parquet.resolve() if args.library_parquet else None,
        raw_fragment_tolerance_ppm=args.raw_fragment_tolerance_ppm,
        normalized_fragment_tolerance_ppm=args.normalized_fragment_tolerance_ppm,
        molecular_ion_tolerance_ppm=args.molecular_ion_tolerance_ppm,
        min_explained_intensity=args.min_explained_intensity,
        deduplicate=args.deduplicate,
        clean_identifiers=not args.no_clean_identifiers,
        max_heavy_atoms=args.max_heavy_atoms,
        min_peaks=args.min_peaks,
        max_peaks=args.max_peaks,
        min_nce=args.min_nce,
        max_nce=args.max_nce,
        log_path=args.log_file.resolve() if args.log_file else None,
    )


if __name__ == "__main__":
    main()