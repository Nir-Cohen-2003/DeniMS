"""
End-to-end smoke test orchestrator for DeniMS.

Runs the full MS2Mol training pipeline on a tiny dataset using the existing
pixi tasks:

    1. pixi run test-prepare-data   (writes tests/e2e/data/*)
    2. pixi run train-encoder        (writes checkpoints/<run_name>/*.pth)
    3. pixi run pretrain-diffusion   (writes MS_diffusion/src/checkpoints/e2e_g2mol/)
    4. pixi run finetune-diffusion   (writes MS_diffusion/src/checkpoints/e2e_ms2mol/)
    5. pixi run test-inference       (writes MS_diffusion/outputs/...)

Uses small models and 2 epochs, so it is *not* scientifically meaningful; the
goal is to verify that the pipeline (data prep -> encoder pretraining ->
graph2mol -> ms2mol -> inference) runs end-to-end without errors.

If a step fails the orchestrator prints the failed command, leaves a clear
log of the most recent output dir, and exits with a non-zero status.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
E2E_DATA_DIR = REPO_ROOT / "tests" / "e2e" / "data"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
DIFFUSION_DIR = REPO_ROOT / "MS_diffusion" / "src"
DIFF_CKPT_DIR = DIFFUSION_DIR / "checkpoints"
# Hydra's chdir + run.dir="../outputs/..." (relative to MS_diffusion/src/)
# means diffusion outputs land under MS_diffusion/outputs/, not
# MS_diffusion/src/outputs/.
DIFF_OUTPUTS_DIR = REPO_ROOT / "MS_diffusion" / "outputs"
ENCODER_CKPT_GLOB = str(CHECKPOINTS_DIR / "*" / "*e2e_test*.pth")
# Hydra's chdir=True rewrites the working directory to
# MS_diffusion/outputs/<date>/<time>-<name>/, so the diffusion checkpoints
# end up under that directory rather than next to MS_diffusion/src/. We
# discover them by glob after each stage.


def _find_latest_diffusion_ckpt(name: str) -> Path:
    """Find the most recently-written `last*.ckpt` under Hydra's output dir
    for a given `general.name`.

    Preference order:
      1. The plain `last.ckpt` from the newest matching run (this is the
         earliest version PyTorch Lightning writes and resumes correctly
         from a fresh `ModelCheckpoint` callback).
      2. Any versioned `last-*.ckpt` from the newest run (e.g. `last-v1.ckpt`).
    """
    # Hydra nests outputs in <date>/<time>-<name>, where <time> is HH-MM-SS.
    run_dirs = sorted(
        [d for d in DIFF_OUTPUTS_DIR.glob(f"*/[0-9][0-9]*-*[0-9][0-9]-{name}") if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        ckpt_dir = run_dir / "checkpoints" / name
        # Prefer the unversioned `last.ckpt` because resuming from a
        # versioned file (e.g. `last-v1.ckpt`) interacts badly with
        # Pytorch Lightning's ModelCheckpoint state tracking when the new
        # run uses save_top_k=-1: the callback won't fire and won't save
        # a new checkpoint.
        plain = ckpt_dir / "last.ckpt"
        if plain.exists():
            return plain
        # Fall back to the most recently written versioned file.
        versioned = sorted(
            ckpt_dir.glob("last*.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if versioned:
            return versioned[0]
    raise FileNotFoundError(
        f"Could not find diffusion checkpoint for {name!r} under "
        f"{DIFF_OUTPUTS_DIR}. Looked for 'last*.ckpt' inside any "
        f"MS_diffusion/outputs/*/*-{name}/checkpoints/{name}/."
    )


def _print(msg: str) -> None:
    print(f"[test-e2e] {msg}", flush=True)


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    """Run a command, streaming its output, and return its exit code."""
    cwd_str = str(cwd) if cwd is not None else None
    _print(">>> " + " ".join(cmd) + (f"   (cwd={cwd_str})" if cwd_str else ""))
    start = time.time()
    proc = subprocess.run(cmd, cwd=cwd_str)
    elapsed = time.time() - start
    _print(f"<<< exit={proc.returncode} elapsed={elapsed:.1f}s")
    return proc.returncode


def _pixi_run(task: str, args: list[str] | None = None, cwd: Path | None = None) -> int:
    cmd = ["pixi", "run", task]
    if args:
        cmd.extend(args)
    return _run(cmd, cwd=cwd)


# ---------------------------------------------------------------------------
# Step 0: clean prior e2e artifacts
# ---------------------------------------------------------------------------
def clean_artifacts() -> None:
    _print("Cleaning prior e2e artifacts...")

    # Any encoder checkpoints whose directory/run-name contains e2e_test
    for ckpt in glob.glob(ENCODER_CKPT_GLOB):
        _print(f"  removing encoder ckpt: {ckpt}")
        try:
            os.remove(ckpt)
        except OSError as exc:
            _print(f"  WARNING: could not remove {ckpt}: {exc}")

    # Encoder run dirs that contained only e2e_test runs
    for run_dir in CHECKPOINTS_DIR.glob("*"):
        if not run_dir.is_dir():
            continue
        if "e2e_test" in run_dir.name.lower() and not any(
            run_dir.glob("*.pth")
        ):
            _print(f"  removing empty run dir: {run_dir}")
            try:
                run_dir.rmdir()
            except OSError:
                pass

    # Diffusion checkpoints land under Hydra's output dir, not next to
    # MS_diffusion/src/. Remove both the (likely-absent) src/ side dirs and
    # any Hydra output trees whose name contains "e2e_".
    for d in [DIFF_CKPT_DIR / "e2e_g2mol",
              DIFF_CKPT_DIR / "e2e_ms2mol",
              DIFF_CKPT_DIR / "e2e_test"]:
        if d.exists():
            _print(f"  removing diffusion ckpt dir: {d}")
            shutil.rmtree(d, ignore_errors=True)

    if DIFF_OUTPUTS_DIR.exists():
        for child in DIFF_OUTPUTS_DIR.iterdir():
            if child.is_dir() and child.name.startswith("e2e_"):
                _print(f"  removing diffusion outputs: {child}")
                shutil.rmtree(child, ignore_errors=True)
            elif child.is_dir():
                # Hydra nests outputs in <date>/<time>-<name>; prune e2e leaves
                for sub in child.iterdir():
                    if sub.is_dir() and "e2e_" in sub.name:
                        _print(f"  removing diffusion outputs: {sub}")
                        shutil.rmtree(sub, ignore_errors=True)

    # Drop any cached LMDB datasets created from prior e2e runs.
    diff_data_dir = REPO_ROOT / "MS_diffusion" / "data"
    if diff_data_dir.exists():
        for child in diff_data_dir.iterdir():
            if not child.is_dir():
                continue
            for sub in child.iterdir():
                if sub.is_dir() and "e2e_" in sub.name:
                    _print(f"  removing cached diffusion data: {sub}")
                    shutil.rmtree(sub, ignore_errors=True)

    # Tiny dataset is left in place (it's small and deterministic).

    _print("Clean-up done.")


# ---------------------------------------------------------------------------
# Step 1: prepare tiny data via the pixi task
# ---------------------------------------------------------------------------
def step_prepare_data() -> Path:
    _print("Step 1/5: prepare tiny dataset (pixi run test-prepare-data)")
    rc = _pixi_run("test-prepare-data")
    if rc != 0:
        _print(f"ERROR: test-prepare-data failed with exit code {rc}.")
        sys.exit(rc)

    if not E2E_DATA_DIR.exists():
        _print(f"ERROR: {E2E_DATA_DIR} not found after test-prepare-data.")
        sys.exit(1)
    _print(f"  tiny data dir: {E2E_DATA_DIR}")
    return E2E_DATA_DIR


# ---------------------------------------------------------------------------
# Step 2: train encoder
# ---------------------------------------------------------------------------
def step_train_encoder(data_dir: Path) -> Path:
    _print("Step 2/5: train encoder (pixi run train-encoder)")
    cmd_args = [
        "-mode", "contrastive",
        "-data_path", "tests/e2e/data/tiny.parquet",
        "-smiles_path", "tests/e2e/data/tiny_smiles_dict.pt",
        "-split_path", "tests/e2e/data/splits_tiny_random.pkl",
        "-batch_size", "16",
        "-epochs", "2",
        "-warmsteps", "50",
        "-hidden_dim", "256",
        "-num_transformer_layers", "2",
        "-nhead", "4",
        "-ordered_sub_batch_size", "8",
        "-trainable_temperature",
        "-lr", "4e-4",
        "-wd", "5e-4",
        "-epochs_cp", "1",
        "-temp_cp", "0",
        "-comment", "e2e_test",
    ]
    rc = _pixi_run("train-encoder", cmd_args, cwd=REPO_ROOT)
    if rc != 0:
        _print(f"ERROR: train-encoder failed with exit code {rc}.")
        sys.exit(rc)

    ckpts = sorted(glob.glob(ENCODER_CKPT_GLOB), key=os.path.getmtime)
    if not ckpts:
        _print(
            f"ERROR: no encoder checkpoint matching {ENCODER_CKPT_GLOB} "
            "after train-encoder."
        )
        sys.exit(1)
    latest = Path(ckpts[-1])
    _print(f"  encoder checkpoint: {latest}")
    return latest


# ---------------------------------------------------------------------------
# Step 3: pretrain graph2mol diffusion
# ---------------------------------------------------------------------------
def step_pretrain_diffusion(encoder_ckpt: Path) -> Path:
    _print("Step 3/5: pretrain graph2mol diffusion (pixi run pretrain-diffusion)")
    encoder_rel = os.path.relpath(encoder_ckpt, REPO_ROOT)
    cmd_args = [
        "conditioning.embeddings_type=mol2emb",
        f"conditioning.embedding_model_path={encoder_rel}",
        "conditioning.ms_data_path=tests/e2e/data/tiny.parquet",
        "conditioning.graph_dict_path=tests/e2e/data/tiny_smiles_dict.pt",
        "conditioning.splitting_path=tests/e2e/data/splits_tiny_random.pkl",
        "train.finetune_ms_encoder=False",
        "train.n_epochs=2",
        "train.batch_size=8",
        # The diffusion dataloader uses persistent_workers=True, which
        # requires num_workers > 0.
        "train.num_workers=1",
        "model.n_layers=2",
        "model.diffusion_steps=50",
        "model.hidden_dims.dx=64",
        "model.hidden_dims.de=32",
        "model.hidden_dims.dy=64",
        "model.hidden_dims.dim_ffX=64",
        "model.hidden_dims.dim_ffE=32",
        "model.hidden_dims.dim_ffy=64",
        "model.hidden_mlp_dims.X=64",
        "model.hidden_mlp_dims.E=32",
        "model.hidden_mlp_dims.y=64",
        "general.gpus=0",
        "general.check_val_every_n_epochs=1",
        "general.name=e2e_g2mol",
        "general.samples_to_generate=10",
        "general.samples_to_save=2",
        "general.chains_to_save=1",
        # Disable wandb: the default value of general.wandb is the string
        # "disabled", which is truthy and triggers setup_wandb -> wandb.login
        # -> UsageError when no API key is configured. An empty string is
        # falsy and makes the if-check in on_fit_start skip the wandb setup.
        "general.wandb=",
    ]
    rc = _pixi_run("pretrain-diffusion", cmd_args, cwd=REPO_ROOT)
    if rc != 0:
        _print(f"ERROR: pretrain-diffusion failed with exit code {rc}.")
        sys.exit(rc)

    try:
        g2mol_ckpt = _find_latest_diffusion_ckpt("e2e_g2mol")
    except FileNotFoundError as exc:
        _print(f"ERROR: {exc}")
        sys.exit(1)
    _print(f"  g2mol checkpoint: {g2mol_ckpt}")
    return g2mol_ckpt


# ---------------------------------------------------------------------------
# Step 4: finetune ms2mol diffusion
# ---------------------------------------------------------------------------
def step_finetune_diffusion(encoder_ckpt: Path, g2mol_ckpt: Path) -> Path:
    _print("Step 4/5: finetune ms2mol diffusion (pixi run finetune-diffusion)")
    encoder_rel = os.path.relpath(encoder_ckpt, REPO_ROOT)
    # Hydra's chdir=True moves the cwd into MS_diffusion/outputs/.../, so we
    # pass the g2mol checkpoint as an absolute path. main.py hands it
    # verbatim to trainer.fit(ckpt_path=...).
    g2mol_abs = str(g2mol_ckpt.resolve())
    cmd_args = [
        "conditioning.embeddings_type=ms2emb",
        f"conditioning.embedding_model_path={encoder_rel}",
        "conditioning.ms_data_path=tests/e2e/data/tiny.parquet",
        "conditioning.graph_dict_path=tests/e2e/data/tiny_smiles_dict.pt",
        "conditioning.splitting_path=tests/e2e/data/splits_tiny_random.pkl",
        f"general.resume={g2mol_abs}",
        "train.finetune_ms_encoder=True",
        "general.name=e2e_ms2mol",
        "train.n_epochs=2",
        "train.batch_size=8",
        "train.num_workers=1",
        "model.n_layers=2",
        "model.diffusion_steps=50",
        "model.hidden_dims.dx=64",
        "model.hidden_dims.de=32",
        "model.hidden_dims.dy=64",
        "model.hidden_dims.dim_ffX=64",
        "model.hidden_dims.dim_ffE=32",
        "model.hidden_dims.dim_ffy=64",
        "model.hidden_mlp_dims.X=64",
        "model.hidden_mlp_dims.E=32",
        "model.hidden_mlp_dims.y=64",
        "general.gpus=0",
        "general.check_val_every_n_epochs=1",
        "general.samples_to_generate=10",
        "general.samples_to_save=2",
        "general.chains_to_save=1",
        "general.wandb=",
    ]
    rc = _pixi_run("finetune-diffusion", cmd_args, cwd=REPO_ROOT)
    if rc != 0:
        _print(f"ERROR: finetune-diffusion failed with exit code {rc}.")
        sys.exit(rc)

    try:
        ms2mol_ckpt = _find_latest_diffusion_ckpt("e2e_ms2mol")
    except FileNotFoundError as exc:
        _print(f"ERROR: {exc}")
        sys.exit(1)
    _print(f"  ms2mol checkpoint: {ms2mol_ckpt}")
    return ms2mol_ckpt


# ---------------------------------------------------------------------------
# Step 5: test inference
# ---------------------------------------------------------------------------
def step_test_inference(encoder_ckpt: Path, ms2mol_ckpt: Path) -> Path:
    _print("Step 5/5: test inference (pixi run test-inference)")
    encoder_rel = os.path.relpath(encoder_ckpt, REPO_ROOT)
    # See step_finetune_diffusion for why we use absolute paths.
    ms2mol_abs = str(ms2mol_ckpt.resolve())
    cmd_args = [
        "conditioning.embeddings_type=ms2emb",
        f"conditioning.embedding_model_path={encoder_rel}",
        "conditioning.ms_data_path=tests/e2e/data/tiny.parquet",
        "conditioning.graph_dict_path=tests/e2e/data/tiny_smiles_dict.pt",
        "conditioning.splitting_path=tests/e2e/data/splits_tiny_random.pkl",
        f"general.test_only={ms2mol_abs}",
        "general.name=e2e_test",
        "general.samples_to_generate=all",
        "general.test_iterations=1",
        "general.number_chain_steps=10",
        "general.final_model_samples_to_generate=8",
        "general.final_model_samples_to_save=4",
        "general.final_model_chains_to_save=2",
        "train.finetune_ms_encoder=True",
        "model.n_layers=2",
        "model.diffusion_steps=50",
        "model.hidden_dims.dx=64",
        "model.hidden_dims.de=32",
        "model.hidden_dims.dy=64",
        "model.hidden_dims.dim_ffX=64",
        "model.hidden_dims.dim_ffE=32",
        "model.hidden_dims.dim_ffy=64",
        "model.hidden_mlp_dims.X=64",
        "model.hidden_mlp_dims.E=32",
        "model.hidden_mlp_dims.y=64",
        "general.gpus=0",
        "general.wandb=",
    ]
    rc = _pixi_run("test-inference", cmd_args, cwd=REPO_ROOT)
    if rc != 0:
        _print(f"ERROR: test-inference failed with exit code {rc}.")
        sys.exit(rc)

    # Hydra run dir: MS_diffusion/outputs/<date>/<time>-e2e_test/
    run_dirs = sorted(
        DIFF_OUTPUTS_DIR.glob(f"*/[0-9][0-9]-[0-9][0-9]-[0-9][0-9]-e2e_test"),
        key=lambda p: p.stat().st_mtime,
    ) if DIFF_OUTPUTS_DIR.exists() else []
    # Some Hydra versions use the format <date>/<time>-<name>
    if not run_dirs and DIFF_OUTPUTS_DIR.exists():
        run_dirs = sorted(
            (p for p in DIFF_OUTPUTS_DIR.glob("*/*-e2e_test") if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )
    if not run_dirs:
        _print(
            "WARNING: Could not locate Hydra run dir for e2e_test. "
            f"Searched under {DIFF_OUTPUTS_DIR}."
        )
        return DIFF_OUTPUTS_DIR
    out_dir = run_dirs[-1]
    _print(f"  inference output dir: {out_dir}")
    smiles_files = sorted(out_dir.glob("*.txt")) + sorted(out_dir.glob("*.csv"))
    if smiles_files:
        _print("  generated SMILES files:")
        for p in smiles_files:
            _print(f"    - {p}")
    else:
        _print("  (no .txt or .csv files found in the inference output dir)")
    return out_dir


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-clean", action="store_true",
        help="Skip removing prior e2e artifacts (useful for debugging).",
    )
    args = parser.parse_args(argv)

    _print("DeniMS end-to-end smoke test")
    _print(f"  repo root: {REPO_ROOT}")

    if not args.skip_clean:
        clean_artifacts()

    data_dir = step_prepare_data()
    encoder_ckpt = step_train_encoder(data_dir)
    g2mol_ckpt = step_pretrain_diffusion(encoder_ckpt)
    ms2mol_ckpt = step_finetune_diffusion(encoder_ckpt, g2mol_ckpt)
    out_dir = step_test_inference(encoder_ckpt, ms2mol_ckpt)

    _print("")
    _print("=" * 60)
    _print("END-TO-END SMOKE TEST PASSED")
    _print("=" * 60)
    _print(f"  Tiny data dir:     {E2E_DATA_DIR}")
    _print(f"  Encoder ckpt:      {encoder_ckpt}")
    _print(f"  Graph2Mol ckpt:    {g2mol_ckpt}")
    _print(f"  MS2Mol ckpt:       {ms2mol_ckpt}")
    _print(f"  Inference output:  {out_dir}")
    _print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
