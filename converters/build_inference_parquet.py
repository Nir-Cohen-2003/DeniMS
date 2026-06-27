#!/usr/bin/env python
"""
Build a DeniMS *inference* parquet from mzML files.

Pipeline (uses HRMS_utils without modifying it):
  1. Read mzML via the Rust ``read_mzml_files`` reader (``hrms_core.io_mzml``).
  2. For every MS/MS spectrum, decompose the precursor m/z into candidate
     precursor formulas. When MS1 spectra are available, the candidate bounds
     are tightened with ``deduce_isotopic_pattern`` (permissive isotopic
     defaults); otherwise permissive default bounds are used. Na, P and K are
     always forced to 0 because DeniMS only supports 9 elements.
  3. For each candidate precursor formula, annotate the fragment peaks with
     ``clean_and_normalize_spectrum`` and compute the explained intensity
     (sum of cleaned intensities / sum of raw intensities).
  4. Keep the candidate(s) with maximal explained intensity per spectrum; if
     there is a tie, keep all tied candidates (each emitted as its own
     Compound_index).
  5. Remap the HRMS_utils 12-element formula vectors to the DeniMS 9-element
     order ``[H, C, N, O, F, S, Cl, Br, I]`` and write the inference schema
     from DATA_FORMATS.md §3.

Output columns: Compound_index, precursor_formula, formulas,
collision_energy_NCE, precursor_type.

Run with the HRMS_utils pixi environment on PATH, e.g.:
    pixi run -e scripts python denims_converters/build_inference_parquet.py <mzml> -o out.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import polars as pl

# Importing hrms_utils registers the ``mass_decomposition`` Polars namespace.
from hrms_utils.formula_annotation.element_table import (
    DEFAULT_MAX_BOUND,
    ELEMENT_SYMBOLS,
)
from hrms_utils.hrms_core import CalibratedIsotopicModel, read_mzml_files


# ---------------------------------------------------------------------------
# Element-order bridge: HRMS_utils (12, mass-ordered) -> DeniMS (9)
# ---------------------------------------------------------------------------
HRMS_INDEX = {sym: i for i, sym in enumerate(ELEMENT_SYMBOLS)}
DENIMS_ELEMENTS = ["H", "C", "N", "O", "F", "S", "Cl", "Br", "I"]
REMAP_HRMS_TO_DENIMS = [HRMS_INDEX[sym] for sym in DENIMS_ELEMENTS]
UNSUPPORTED_HRMS_INDICES = [
    i for i, sym in enumerate(ELEMENT_SYMBOLS) if sym not in DENIMS_ELEMENTS
]
HRMS_N = len(ELEMENT_SYMBOLS)  # 12


def permissive_max_bounds(
    max_h: int, max_c: int, max_n: int, max_o: int, max_f: int,
    max_s: int, max_cl: int, max_br: int, max_i: int,
) -> list[int]:
    """Build the 12-element max-bounds vector with Na/P/K forced to 0."""
    bounds = dict(DEFAULT_MAX_BOUND)
    bounds.update({
        "H": max_h, "C": max_c, "N": max_n, "O": max_o, "F": max_f,
        "S": max_s, "Cl": max_cl, "Br": max_br, "I": max_i,
        "Na": 0, "P": 0, "K": 0,
    })
    return [bounds[sym] for sym in ELEMENT_SYMBOLS]


def collision_energy_to_nce_expr(
    ce_col: str, unit_col: str, precursor_mz_col: str, override: str
) -> pl.Expr:
    """Normalize collision energy to NCE.

    NCE = eV * 500 / precursor_mz  (the HRMS_utils convention, see
    ``spectral_library._extract_collision_energy_values``).
    """
    ce = pl.col(ce_col)
    mz = pl.col(precursor_mz_col)
    if override == "nce":
        return ce
    if override == "ev":
        return (ce * 500.0 / mz).abs()
    # auto: infer from the unit string.
    unit = pl.col(unit_col).str.to_lowercase()
    is_ev = unit.str.contains("ev|volt", literal=False).fill_null(False)
    return pl.when(is_ev).then((ce * 500.0 / mz).abs()).otherwise(ce)


def precursor_type_from_polarity(polarity: str | None, override: str) -> str:
    if override != "auto":
        return override
    if polarity is None:
        return "[M+H]+"
    return "[M+H]+" if polarity.lower().startswith("pos") else "[M-H]-"


def build_inference_parquet(
    input_path: Path,
    output_path: Path,
    precursor_tolerance_ppm: float,
    raw_fragment_tolerance_ppm: float,
    normalized_fragment_tolerance_ppm: float,
    min_dbe: float,
    max_dbe: float,
    dbe_mode: str,
    water_absorption: bool,
    max_bounds: list[int],
    use_isotopic_bounds: bool,
    ms1_mass_tolerance_ppm: float,
    isotopic_mass_tolerance_ppm: float,
    minimum_intensity: float,
    isotopic_model: CalibratedIsotopicModel,
    max_candidates: int,
    collision_energy_unit: str,
    precursor_type_override: str,
) -> None:
    t0 = perf_counter()

    # --- Step 1: read mzML ------------------------------------------------
    if input_path.is_file():
        mzml_files = [str(input_path)]
    elif input_path.is_dir():
        mzml_files = sorted(str(p) for p in input_path.iterdir()
                            if p.suffix.lower() == ".mzml")
        assert mzml_files, f"No .mzml files in directory: {input_path}"
    else:
        raise ValueError(f"Path does not exist: {input_path}")

    print(f"[1/5] Reading {len(mzml_files)} mzML file(s)...")
    dfs = read_mzml_files(mzml_files)
    dfs = [
        df.with_columns(pl.lit(i, dtype=pl.Int32).alias("source_file_idx"))
        for i, df in enumerate(dfs)
    ]
    df = pl.concat(dfs, how="diagonal_relaxed")
    print(f"      {df.height} spectra loaded")

    # --- Step 2: keep MS/MS, normalize CE & precursor type ----------------
    ms2 = df.filter(pl.col("ms_level") == 2).filter(
        pl.col("precursor_mz").is_not_null()
        & pl.col("collision_energy").is_not_null()
        & (pl.col("mz").list.len() > 0)
    ).with_row_index("row_id", offset=1)

    if ms2.is_empty():
        raise ValueError("No MS/MS spectra with precursor m/z and collision energy found.")

    ms2 = ms2.with_columns(
        collision_energy_to_nce_expr(
            "collision_energy", "collision_energy_unit", "precursor_mz",
            collision_energy_unit,
        ).alias("collision_energy_NCE"),
    ).with_columns(
        pl.col("polarity").map_elements(
            lambda p: precursor_type_from_polarity(p, precursor_type_override),
            return_dtype=pl.Utf8,
        ).alias("precursor_type")
    )

    # --- Step 3: pair each MS2 with the nearest preceding MS1 -------------
    ms1 = df.filter(pl.col("ms_level") == 1).select(
        ["source_file_idx", "scan_time", "mz", "intensity"]
    ).rename({"mz": "ms1_mz", "intensity": "ms1_intensity"})

    if use_isotopic_bounds and not ms1.is_empty():
        ms2 = ms2.sort(["source_file_idx", "scan_time"])
        ms1 = ms1.sort(["source_file_idx", "scan_time"])
        ms2 = ms2.join_asof(
            ms1, on="scan_time", by="source_file_idx", strategy="backward"
        )
    else:
        ms2 = ms2.with_columns(
            pl.lit(None, dtype=pl.List(pl.Float64)).alias("ms1_mz"),
            pl.lit(None, dtype=pl.List(pl.Float64)).alias("ms1_intensity"),
        )

    # --- Step 4: per-row elemental bounds ---------------------------------
    # Default bounds (permissive, Na/P/K = 0).
    default_min = [0] * HRMS_N
    default_max = max_bounds

    min_bounds_expr = pl.concat_list(
        [pl.lit(0, dtype=pl.Int32) for _ in range(HRMS_N)]
    ).list.to_array(HRMS_N).alias("min_bounds")
    max_bounds_expr = pl.concat_list(
        [pl.lit(int(v), dtype=pl.Int32) for v in default_max]
    ).list.to_array(HRMS_N).alias("max_bounds")

    ms2 = ms2.with_columns([min_bounds_expr, max_bounds_expr])

    has_ms1 = pl.col("ms1_mz").is_not_null() & pl.col("ms1_intensity").is_not_null()
    if use_isotopic_bounds:
        # Deduce isotopic bounds (C, S, Cl, Br) from the paired MS1 cluster.
        # Pass max_bounds with P=0 so the base bounds already exclude P.
        iso_max_kwargs = {
            sym: 0 if sym in ("Na", "P", "K") else val
            for sym, val in zip(ELEMENT_SYMBOLS, default_max)
        }
        ms2 = ms2.with_columns(
            pl.when(has_ms1)
            .then(
                pl.col("precursor_mz").mass_decomposition.deduce_isotopic_pattern(
                    pl.col("ms1_mz"),
                    pl.col("ms1_intensity"),
                    ms1_mass_tolerance_ppm=ms1_mass_tolerance_ppm,
                    isotopic_mass_tolerance_ppm=isotopic_mass_tolerance_ppm,
                    minimum_intensity=minimum_intensity,
                    model=isotopic_model,
                    max_bounds=iso_max_kwargs,
                )
            )
            .otherwise(pl.lit(None))
            .alias("isotopic_bounds")
        )

        # Fold the deduced 24-vector into per-element min/max, coalescing to the
        # default bounds when no MS1 was available. Na/P/K are forced to 0.
        ib = pl.col("isotopic_bounds")  # Array(Int32, 24), nullable
        ib_list = ib.arr.to_list()  # List(Int32), nullable
        min_parts, max_parts = [], []
        for i in range(HRMS_N):
            min_parts.append(pl.coalesce([ib_list.list.get(i), pl.lit(0, dtype=pl.Int32)]))
            if i in UNSUPPORTED_HRMS_INDICES:
                max_parts.append(pl.lit(0, dtype=pl.Int32))
            else:
                max_parts.append(
                    pl.coalesce([ib_list.list.get(i + HRMS_N), pl.lit(int(default_max[i]), dtype=pl.Int32)])
                )
        ms2 = ms2.with_columns(
            pl.concat_list(min_parts).list.to_array(HRMS_N).alias("min_bounds"),
            pl.concat_list(max_parts).list.to_array(HRMS_N).alias("max_bounds"),
        ).drop("isotopic_bounds")

    # --- Step 5: decompose precursor m/z into candidate formulas ----------
    print("[2/5] Decomposing precursor m/z into candidate formulas...")
    ms2 = ms2.with_columns(
        pl.struct([
            pl.col("precursor_mz").alias("mass"),
            pl.col("min_bounds"),
            pl.col("max_bounds"),
        ]).mass_decomposition.decompose_mass_with_bounds(
            tolerance_ppm=precursor_tolerance_ppm,
            min_dbe=min_dbe,
            max_dbe=max_dbe,
            dbe_mode=dbe_mode,
        ).alias("candidates")
    ).with_columns(
        pl.col("candidates").struct.field("formulas").alias("cand_formula"),
        pl.col("candidates").struct.field("errors_ppm").alias("cand_error_ppm"),
    ).drop("candidates")

    # Explode to one row per (spectrum, candidate) and cap by error.
    ms2 = ms2.filter(pl.col("cand_formula").is_not_null())
    ms2 = ms2.explode(["cand_formula", "cand_error_ppm"]).filter(
        pl.col("cand_formula").is_not_null()
    )
    if max_candidates > 0:
        ms2 = ms2.sort(["row_id", "cand_error_ppm"]).with_columns(
            pl.int_range(pl.len(), dtype=pl.Int32).over("row_id").alias("cand_rank")
        ).filter(pl.col("cand_rank") < max_candidates).drop("cand_rank")

    if ms2.is_empty():
        raise ValueError("No precursor formula candidates could be decomposed.")

    # --- Step 6: annotate fragments per candidate & score -----------------
    print("[3/5] Annotating fragment peaks per candidate precursor formula...")
    ms2 = ms2.with_columns(
        pl.struct([
            pl.col("mz").alias("mz"),
            pl.col("intensity").alias("intensities"),
            pl.col("cand_formula").alias("precursor_formula"),
        ]).mass_decomposition.clean_and_normalize_spectrum(
            raw_fragment_tolerance_ppm=raw_fragment_tolerance_ppm,
            normalized_fragment_tolerance_ppm=normalized_fragment_tolerance_ppm,
            min_dbe=min_dbe,
            max_dbe=max_dbe,
            dbe_mode=dbe_mode,
            water_absorption=water_absorption,
        ).alias("cleaned")
    ).with_columns(
        pl.col("intensity").list.sum().alias("raw_intensity_sum"),
        pl.col("cleaned").struct.field("formulas").alias("frag_formula_12"),
        pl.col("cleaned").struct.field("intensities").list.sum().alias("clean_intensity_sum"),
    ).with_columns(
        (pl.col("clean_intensity_sum") / pl.col("raw_intensity_sum"))
        .alias("explained_intensity")
    )

    # --- Step 7: select maximal explained intensity (ties kept) -----------
    print("[4/5] Selecting best precursor formula by explained intensity...")
    ms2 = ms2.with_columns(
        pl.col("explained_intensity").max().over("row_id").alias("max_explained")
    ).filter(
        pl.col("explained_intensity") == pl.col("max_explained")
    ).filter(
        pl.col("frag_formula_12").list.len().ge(1)
    )

    # --- Step 8: remap 12-vec -> 9-vec and write --------------------------
    print("[5/5] Remapping to DeniMS 9-element order and writing parquet...")
    out = ms2.with_columns(
        # precursor formula: Array[Int32, 12] -> Array[Int32, 9]
        pl.concat_list(
            [pl.col("cand_formula").arr.get(i) for i in REMAP_HRMS_TO_DENIMS]
        ).list.to_array(len(DENIMS_ELEMENTS)).alias("precursor_formula"),
        # fragment formulas: List[Array[Int32, 12]] -> List[Array[Int32, 9]]
        pl.col("frag_formula_12").list.eval(
            pl.concat_list(
                [pl.element().arr.get(i) for i in REMAP_HRMS_TO_DENIMS]
            ).list.to_array(len(DENIMS_ELEMENTS))
        ).alias("formulas"),
    ).with_columns(
        pl.int_range(1, pl.len() + 1).alias("Compound_index")
    ).select(
        "Compound_index",
        "precursor_formula",
        "formulas",
        "collision_energy_NCE",
        "precursor_type",
    )

    # §3.6 validation: 1 <= len(formulas) <= 128.
    out = out.filter(
        (pl.col("formulas").list.len() >= 1)
        & (pl.col("formulas").list.len() <= 128)
        & pl.col("collision_energy_NCE").is_not_null()
        & pl.col("collision_energy_NCE").is_finite()
    )

    out.write_parquet(output_path)
    print(f"Wrote {out.height} inference rows "
          f"({out['Compound_index'].n_unique()} compound interpretations) "
          f"to {output_path} in {perf_counter() - t0:.2f}s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a DeniMS inference parquet from mzML files using HRMS_utils."
    )
    p.add_argument("input_path", type=Path,
                   help="mzML file or directory of .mzml files.")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output DeniMS inference parquet (default: <input>_denims_inference.parquet).")

    # --- Mass accuracy ---
    g = p.add_argument_group("mass accuracy")
    g.add_argument("--precursor-tolerance-ppm", type=float, default=10.0,
                   help="Mass tolerance for precursor formula decomposition (ppm).")
    g.add_argument("--raw-fragment-tolerance-ppm", type=float, default=15.0,
                   help="Initial fragment mass tolerance for clean_and_normalize_spectrum (ppm, permissive).")
    g.add_argument("--normalized-fragment-tolerance-ppm", type=float, default=10.0,
                   help="Post-normalization fragment mass tolerance (ppm, permissive).")

    # --- DBE ---
    g = p.add_argument_group("degree of unsaturation (permissive defaults)")
    g.add_argument("--min-dbe", type=float, default=-10.0)
    g.add_argument("--max-dbe", type=float, default=100.0)
    g.add_argument("--dbe-mode", type=str, default="any", choices=["any", "integer", "half_integer"])

    g.add_argument("--no-water-absorption", action="store_true",
                   help="Disable water absorption in fragment annotation (allowed by default).")

    # --- Precursor decomposition bounds (permissive) ---
    g = p.add_argument_group("precursor elemental bounds (permissive defaults; Na/P/K forced to 0)")
    g.add_argument("--max-h", type=int, default=200)
    g.add_argument("--max-c", type=int, default=100)
    g.add_argument("--max-n", type=int, default=30)
    g.add_argument("--max-o", type=int, default=30)
    g.add_argument("--max-f", type=int, default=50)
    g.add_argument("--max-s", type=int, default=10)
    g.add_argument("--max-cl", type=int, default=10)
    g.add_argument("--max-br", type=int, default=10)
    g.add_argument("--max-i", type=int, default=10)
    g.add_argument("--max-candidates", type=int, default=50,
                   help="Cap candidate precursor formulas per spectrum (by smallest error). 0 = no cap.")

    # --- Isotopic pattern (permissive defaults) ---
    g = p.add_argument_group("isotopic pattern (permissive defaults)")
    g.add_argument("--no-isotopic-bounds", action="store_true",
                   help="Disable MS1-based isotopic bound tightening (use permissive default bounds).")
    g.add_argument("--ms1-mass-tolerance-ppm", type=float, default=10.0,
                   help="Tolerance for matching the precursor in the MS1 spectrum (ppm).")
    g.add_argument("--isotopic-mass-tolerance-ppm", type=float, default=10.0,
                   help="Tolerance for matching isotopic peaks (ppm, permissive).")
    g.add_argument("--minimum-intensity", type=float, default=1e3,
                   help="Minimum MS1 peak intensity to consider (permissive: low threshold).")
    g.add_argument("--isotopic-alpha", type=float, default=1.32762)
    g.add_argument("--isotopic-beta", type=float, default=0.981853)
    g.add_argument("--isotopic-offset", type=float, default=0.0)
    g.add_argument("--hetero-lower-a", type=float, default=2.0,
                   help="Lower-band amplitude (permissive: smaller than calibrated 5.5).")
    g.add_argument("--hetero-lower-b", type=float, default=-0.2)
    g.add_argument("--hetero-upper-a", type=float, default=25.0,
                   help="Upper-band amplitude (permissive: larger than calibrated 16.7).")
    g.add_argument("--hetero-upper-b", type=float, default=-0.29)

    # --- Collision energy & precursor type ---
    g = p.add_argument_group("collision energy / precursor type")
    g.add_argument("--collision-energy-unit", type=str, default="auto",
                   choices=["auto", "nce", "ev"],
                   help="Unit of the mzML collision energy. 'auto' infers from the unit string; "
                        "eV is converted to NCE via NCE = eV*500/precursor_mz.")
    g.add_argument("--precursor-type", type=str, default="auto",
                   choices=["auto", "[M+H]+", "[M-H]-"],
                   help="Force a precursor type instead of inferring from polarity.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path.resolve()
    assert input_path.exists(), f"Input path does not exist: {input_path}"

    if args.output is not None:
        output_path = args.output.resolve()
    elif input_path.is_file():
        output_path = input_path.with_suffix(".denims_inference.parquet")
    else:
        output_path = input_path / f"{input_path.name}.denims_inference.parquet"

    max_bounds = permissive_max_bounds(
        args.max_h, args.max_c, args.max_n, args.max_o, args.max_f,
        args.max_s, args.max_cl, args.max_br, args.max_i,
    )

    isotopic_model = CalibratedIsotopicModel(
        alpha=args.isotopic_alpha,
        beta=args.isotopic_beta,
        offset=args.isotopic_offset,
        hetero_lower_a=args.hetero_lower_a,
        hetero_lower_b=args.hetero_lower_b,
        hetero_upper_a=args.hetero_upper_a,
        hetero_upper_b=args.hetero_upper_b,
    )

    build_inference_parquet(
        input_path=input_path,
        output_path=output_path,
        precursor_tolerance_ppm=args.precursor_tolerance_ppm,
        raw_fragment_tolerance_ppm=args.raw_fragment_tolerance_ppm,
        normalized_fragment_tolerance_ppm=args.normalized_fragment_tolerance_ppm,
        min_dbe=args.min_dbe,
        max_dbe=args.max_dbe,
        dbe_mode=args.dbe_mode,
        water_absorption=not args.no_water_absorption,
        max_bounds=max_bounds,
        use_isotopic_bounds=not args.no_isotopic_bounds,
        ms1_mass_tolerance_ppm=args.ms1_mass_tolerance_ppm,
        isotopic_mass_tolerance_ppm=args.isotopic_mass_tolerance_ppm,
        minimum_intensity=args.minimum_intensity,
        isotopic_model=isotopic_model,
        max_candidates=args.max_candidates,
        collision_energy_unit=args.collision_energy_unit,
        precursor_type_override=args.precursor_type,
    )


if __name__ == "__main__":
    main()