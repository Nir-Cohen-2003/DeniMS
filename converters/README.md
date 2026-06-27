# DeniMS Converters

Two standalone scripts that convert raw MS data into the parquet formats
DeniMS consumes (see `DATA_FORMATS.md`). They use the **HRMS_utils** library
as-is — they do not modify it.

| Script | Input | Output | DeniMS path |
|---|---|---|---|
| `build_training_parquet.py` | MSP / MGF / MSPEC spectral library (with SMILES/InChI) | `<name>_denims_training.parquet` | Training (§2) |
| `build_inference_parquet.py` | mzML file(s) | `<name>_denims_inference.parquet` | Inference (§3) |

Both scripts are plain Python CLIs. Run them with the HRMS_utils pixi
environment on `PATH`, e.g.:

```bash
pixi run -e scripts python denims_converters/build_training_parquet.py library.msp -o train.parquet
pixi run -e scripts python denims_converters/build_inference_parquet.py run.mzML -o infer.parquet
```

> **Element-order bridge (critical).** HRMS_utils uses a **12-element** formula
> vector ordered by increasing monoisotopic mass:
> `[H, C, N, O, F, Na, P, S, Cl, K, Br, I]`. DeniMS uses a **9-element**
> vector in a different order: `[H, C, N, O, F, S, Cl, Br, I]`. Both scripts
> remap the 12-vector to the 9-vector by selecting the supported elements in
> the DeniMS order, and drop any row/peak that carries an unsupported element
> (`Na`, `P`, `K`). The remap is done **after** annotation, so HRMS_utils'
> native 12-element engine does all the mass-decomposition work.

---

## 1. `build_training_parquet.py` — training data

### What it does

1. Calls `hrms_utils.formats.spectral_library.process_spectral_library` on the
   input file(s) — the same function the `build-spectral-library` CLI wraps.
   PubChem enrichment is **skipped** (`pubchem_path=None`): only entries that
   already carry SMILES or InChI survive.
2. Reads the resulting annotated library parquet and applies the DeniMS §2.5
   filters:
   - `smiles` not null
   - `precursor_type` ∈ `{[M+H]+, [M-H]-}`
   - `3 < num_clean_peaks < 128`
   - `4 < collision_energy_NCE < 300`
   - SMILES has `< 30` heavy atoms (RDKit)
   - SMILES contains only `{C, N, O, F, S, Cl, Br, I}` (+ H)
   - no peak carries `Na`, `P`, or `K`
3. Remaps the per-peak 12-element formula list to the DeniMS 9-element order
   and writes the training schema.

### Output columns

| Column | Type |
|---|---|
| `smiles` | `str` (canonical) |
| `precursor_type` | `str` ∈ `{[M+H]+, [M-H]-}` |
| `collision_energy_NCE` | `float64` |
| `clean_spectrum_formula_array` | `list[array[int32, 9]]` — one 9-vec per peak |
| `spectral_information_score` | `float64` (defaults to 1.0 if absent) |

### CLI arguments

```
build_training_parquet.py <input_path> [-o OUTPUT] [--library-parquet PATH]
```

| Argument | Default | Description |
|---|---|---|
| `input_path` (positional) | — | `.msp`/`.mspec`/`.mgf` file or directory of them. |
| `-o`, `--output` | `<input>_denims_training.parquet` | Output DeniMS training parquet. |
| `--library-parquet` | `None` | Skip step 1 and transform this existing HRMS_utils library parquet instead. |
| `--raw-fragment-tolerance-ppm` | `10.0` | Pass-through to `process_spectral_library`. |
| `--normalized-fragment-tolerance-ppm` | `5.0` | Pass-through. |
| `--molecular-ion-tolerance-ppm` | `5.0` | Pass-through. |
| `--min-explained-intensity` | `0.0` | Min explained intensity during library annotation (0 = keep all). |
| `--deduplicate` | off | Run pairwise spectrum deduplication (off by default; training wants broad coverage). |
| `--no-clean-identifiers` | off | Skip MS-Ready identifier standardization (not recommended). |
| `--log-file` | `<library_parquet>.log` | Execution log path. |
| `--max-heavy-atoms` | `30` | Drop SMILES with `>=` this many heavy atoms (§2.5). |
| `--min-peaks` | `3` | Min number of clean peaks (§2.5: `> 2`). |
| `--max-peaks` | `127` | Max number of clean peaks, inclusive (§2.5: `< 128`). |
| `--min-nce` | `5.0` | Min `collision_energy_NCE` (§2.5: `> 4`). |
| `--max-nce` | `300.0` | Max `collision_energy_NCE` (§2.5: `< 300`). |

### Notes

- The script is adaptive to the installed `process_spectral_library` signature:
  it only passes kwargs the installed version accepts, and persists the result
  itself when the installed build does not write to `output_path`.
- The SMILES-level filter is evaluated over **unique** SMILES (O(molecules),
  not O(spectra)) using RDKit, which is a transitive dependency of
  `parallel_rdkit` (an HRMS_utils dependency).

---

## 2. `build_inference_parquet.py` — inference data

### What it does

1. Reads mzML via the Rust `hrms_core.io_mzml.read_mzml_files` reader
   (`from hrms_utils.hrms_core import read_mzml_files`).
2. Keeps MS/MS spectra (`ms_level == 2`) that have a precursor m/z, a
   collision energy, and at least one peak.
3. Normalizes collision energy to NCE. eV is converted via
   `NCE = eV * 500 / precursor_mz` (the HRMS_utils convention); `auto` infers
   the unit from the mzML `collision_energy_unit` string.
4. Infers `precursor_type` from polarity (`positive → [M+H]+`,
   `negative → [M-H]-`), or uses `--precursor-type` to force one.
5. For each MS/MS spectrum, decomposes the precursor m/z into candidate
   precursor formulas using
   `pl.col.mass_decomposition.decompose_mass_with_bounds`, with permissive
   elemental bounds (Na/P/K forced to 0). When MS1 spectra are available and
   `--no-isotopic-bounds` is not set, the bounds for C, S, Cl, Br are tightened
   with `deduce_isotopic_pattern` (permissive isotopic defaults).
6. For each candidate precursor formula, annotates the fragment peaks with
   `clean_and_normalize_spectrum` and computes the explained intensity
   (`sum(cleaned intensities) / sum(raw intensities)`).
7. Selects the candidate(s) with **maximal explained intensity** per spectrum.
   If there is a tie, **all tied candidates are kept** — each emitted as its
   own `Compound_index`.
8. Remaps the 12-element precursor and per-peak formula vectors to the DeniMS
   9-element order and writes the inference schema.

### Output columns

| Column | Type |
|---|---|
| `Compound_index` | `int64` — one per kept (spectrum, tied-candidate) |
| `precursor_formula` | `array[int32, 9]` |
| `formulas` | `list[array[int32, 9]]` — one 9-vec per peak |
| `collision_energy_NCE` | `float64` |
| `precursor_type` | `str` ∈ `{[M+H]+, [M-H]-}` |

### CLI arguments

```
build_inference_parquet.py <input_path> [-o OUTPUT]
```

#### Mass accuracy

| Argument | Default | Description |
|---|---|---|
| `--precursor-tolerance-ppm` | `10.0` | Mass tolerance for precursor formula decomposition (ppm). |
| `--raw-fragment-tolerance-ppm` | `15.0` | Initial fragment mass tolerance for `clean_and_normalize_spectrum` (ppm, **permissive**). |
| `--normalized-fragment-tolerance-ppm` | `10.0` | Post-normalization fragment mass tolerance (ppm, **permissive**). |

#### Degree of unsaturation (permissive defaults)

| Argument | Default | Description |
|---|---|---|
| `--min-dbe` | `-10.0` | Min DBE. |
| `--max-dbe` | `100.0` | Max DBE. |
| `--dbe-mode` | `any` | `any` / `integer` / `half_integer`. |
| `--no-water-absorption` | off | Disable water absorption in fragment annotation (allowed by default). |

#### Precursor elemental bounds (permissive; Na/P/K forced to 0)

| Argument | Default |
|---|---|
| `--max-h` | `200` |
| `--max-c` | `100` |
| `--max-n` | `30` |
| `--max-o` | `30` |
| `--max-f` | `50` |
| `--max-s` | `10` |
| `--max-cl` | `10` |
| `--max-br` | `10` |
| `--max-i` | `10` |
| `--max-candidates` | `50` (cap candidate precursor formulas per spectrum by smallest error; `0` = no cap) |

#### Isotopic pattern (permissive defaults)

| Argument | Default | Description |
|---|---|---|
| `--no-isotopic-bounds` | off | Disable MS1-based isotopic bound tightening (use permissive default bounds). |
| `--ms1-mass-tolerance-ppm` | `10.0` | Tolerance for matching the precursor in the MS1 spectrum (ppm). |
| `--isotopic-mass-tolerance-ppm` | `10.0` | Tolerance for matching isotopic peaks (ppm, **permissive**). |
| `--minimum-intensity` | `1e3` | Minimum MS1 peak intensity to consider (**permissive**: low threshold). |
| `--isotopic-alpha` | `1.32762` | Calibrated model α. |
| `--isotopic-beta` | `0.981853` | Calibrated model β. |
| `--isotopic-offset` | `0.0` | Calibrated model offset. |
| `--hetero-lower-a` | `2.0` | Lower-band amplitude (**permissive**: smaller than calibrated `5.5`). |
| `--hetero-lower-b` | `-0.2` | Lower-band slope. |
| `--hetero-upper-a` | `25.0` | Upper-band amplitude (**permissive**: larger than calibrated `16.7`). |
| `--hetero-upper-b` | `-0.29` | Upper-band slope. |

#### Collision energy / precursor type

| Argument | Default | Description |
|---|---|---|
| `--collision-energy-unit` | `auto` | `auto` / `nce` / `ev`. `auto` infers from the unit string; eV → NCE via `NCE = eV*500/precursor_mz`. |
| `--precursor-type` | `auto` | `auto` / `[M+H]+` / `[M-H]-`. Force a precursor type instead of inferring from polarity. |

### Isotopic pattern error estimation (overview)

When MS1 spectra are paired with each MS/MS scan, `deduce_isotopic_pattern`
fits the observed isotope-cluster spacings of the precursor against a
calibrated heteroscedastic model and returns per-element lower/upper count
bounds for the elements whose isotopes produce a measurable M+2 signal
(`C`, `S`, `Cl`, `Br`). Conceptually:

- The expected isotopic spacing for a precursor of mass `m` is modeled as a
  linear function of `m` whose slope and intercept are `alpha`/`beta` (the
  centroid of the expected M+1/M+2 envelope) plus an `offset`.
- The tolerated deviation around that expectation is a **heteroscedastic
  band** — two lines (`lower` and `upper`) whose amplitudes
  (`hetero_lower_a`, `hetero_upper_a`) and slopes
  (`hetero_lower_b`, `hetero_upper_b`) widen with mass. Peaks falling inside
  the band are accepted as isotopes; the count of `C`/`S`/`Cl`/`Br` is then
  bounded by inverting the observed M+2 intensity ratio against each element's
  known isotopic abundance.
- The defaults here deliberately **widen** the band (smaller lower amplitude,
  larger upper amplitude) and lower the minimum-intensity gate so that
  borderline or noisy isotope clusters are still used to tighten bounds,
  rather than discarding the precursor outright. This is what makes the
  isotopic parameters "permissive".

The model is exposed as `CalibratedIsotopicModel` in
`hrms_utils.hrms_core`; the bounds it returns are folded into the
`min_bounds`/`max_bounds` passed to `decompose_mass_with_bounds`, with
`Na`/`P`/`K` forced to 0 so only DeniMS-representable elements can appear.

### Notes

- MS1 pairing uses `join_asof` on `scan_time` within the same source file
  (backward strategy), so each MS/MS gets the nearest preceding MS1 cluster.
- When no MS1 is available (or `--no-isotopic-bounds`), the permissive default
  bounds are used for every spectrum.
- The "maximal explained intensity, ties kept" rule is implemented as a Polars
  window-max filter: `explained_intensity == max(explained_intensity).over(row_id)`.
  Each surviving candidate becomes its own `Compound_index`, so a tied spectrum
  yields multiple rows — DeniMS groups by `Compound_index` and treats each as a
  separate compound interpretation.

---

## 3. Dependencies

These scripts import only packages already provided by HRMS_utils' pixi
environments:

- `polars` (HRMS_utils core dependency)
- `hrms_utils` (this repo)
- `rdkit` (transitive via `parallel_rdkit`)

No additional dependencies are required. If you place these scripts in a
separate project, ensure `hrms_utils` and `parallel_rdkit` are installed
(e.g. via the HRMS_utils pixi environment) so that `polars`, `rdkit`, and the
compiled `hrms_core` extension are importable.