# DeniMS - *De-novo* Identification of Mass Spectra

This repository accompanies the paper: [Paper](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15000101/v1)

![Overview Figure](Overview%20Figure.jpeg)

## Installation

This project is managed with [pixi](https://pixi.sh). To install the necessary environments, follow these steps:

1. Install pixi (if you haven't already):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

For other installation methods, see the [pixi documentation](https://pixi.sh/latest/#installation).

2. Clone the repository and enter it:

```bash
git clone https://github.com/Milo-group/DeniMS.git
cd DeniMS
```

3. Install all environments:

```bash
pixi install --all
```

The workspace defines two environments:

- **`gpu`** (default, python 3.10) — the CUDA training/inference stack (PyTorch, torch-geometric, rdkit, graph-tool, ...). Used for encoder/diffusion training and model application.
- **`data-prep`** (python 3.12) — a lightweight CPU environment for the data converters (`hrms_utils`, polars, rdkit). Used to turn raw MS data into the parquet files DeniMS consumes.

Run a task in a specific environment with `pixi run -e <env> <task>`. Without `-e`, the default (`gpu`) environment is used.

## Usage

Use `apply-model` to run the trained DeniMS model on experimental MS data.

### Data preparation

DeniMS consumes **Apache Parquet** files whose exact schemas are specified in [`DATA_FORMATS.md`](DATA_FORMATS.md). There are two paths:

- **Training** (`<name>_filtered.parquet`): one row per MS/MS spectrum, grouped by `smiles`. Requires SMILES, `precursor_type`, `collision_energy_NCE`, and `clean_spectrum_formula_array` (a list of 9-element element-count vectors per peak).
- **Inference** (`experimental.parquet`): one row per MS/MS spectrum, grouped by `Compound_index`. Requires `precursor_formula` and `formulas` (9-element element-count vectors), plus `collision_energy_NCE`. No SMILES needed.

> **Element order (critical).** Every formula is a fixed-length 9-element integer vector in the order `[H, C, N, O, F, S, Cl, Br, I]` — *not* Hill order. See `DATA_FORMATS.md` §0.

#### Converters (raw data → DeniMS parquet)

The `converters/` package contains two standalone scripts that turn raw MS data into the parquet formats above, using the [HRMS_utils](https://github.com/Nir-Cohen-2003/HRMS_utils) library without modifying it. See [`converters/README.md`](converters/README.md) for the full option reference.

| Script | Input | Output | DeniMS path |
|---|---|---|---|
| `converters/build_training_parquet.py` | MSP / MGF / MSPEC spectral library (with SMILES/InChI) | `<name>_denims_training.parquet` | Training (§2) |
| `converters/build_inference_parquet.py` | mzML file(s) | `<name>_denims_inference.parquet` | Inference (§3) |

Both remap HRMS_utils' 12-element formula vector (`[H, C, N, O, F, Na, P, S, Cl, K, Br, I]`) down to DeniMS' 9-element order, dropping any row/peak carrying an unsupported element (`Na`, `P`, `K`).

They are exposed as pixi tasks in the `data-prep` environment:

```bash
# Training data: spectral library -> DeniMS training parquet
pixi run -e data-prep convert-training ./sample_data/example.msp \
    -o Preprocessing/fraghub/fraghub_denims_training.parquet

# Inference data: mzML -> DeniMS inference parquet
pixi run -e data-prep convert-inference ./sample_data/example.mzML \
    -o Preprocessing/experimental/experimental.parquet
```

#### End-to-end data pipelines (combined tasks)

Two combined tasks chain a converter (data-prep env) with the existing preprocessing/inference step (gpu env) in sequence, matching their inputs and outputs through a shared intermediate parquet path.

> **Why shell chaining instead of pixi `depends-on`.** `hrms_utils` requires `python >=3.12` (data-prep env) while the GPU stack is pinned to `python 3.10` (gpu env), and pixi `depends-on` only works *within a single environment*. No single environment can hold both stacks, so the combined tasks shell out to `pixi run -e data-prep ...` then `pixi run -e gpu ...`.

**Training pipeline** — spectral library → training parquet → filtered parquet + graph dict + splits:

```bash
# Uses the conventional input Preprocessing/fraghub/library.msp by default.
# Override the input library with DENIMS_LIB_INPUT:
#   DENIMS_LIB_INPUT=path/to/library.msp pixi run prepare-training
pixi run prepare-training
```

This runs `convert-training` (data-prep) producing `Preprocessing/fraghub/fraghub_denims_training.parquet`, then `prep-data` (gpu) which filters it and writes `fraghub_denims_training_filtered.parquet`, the `*_smiles_dict.pt` graph dictionary, and `splits_*_random.pkl` next to it.

**Inference pipeline** — mzML → inference parquet → model predictions:

```bash
# Override the mzML input with DENIMS_MZML_INPUT:
#   DENIMS_MZML_INPUT=path/to/run.mzML pixi run prepare-inference --model_checkpoint [ckpt.ckpt]
# Extra args are forwarded to apply-model (the second step):
pixi run prepare-inference --model_checkpoint [path_to_model_ckpt.ckpt] --num_repeats 50
```

This runs `convert-inference` (data-prep) producing `Preprocessing/experimental/experimental.parquet`, then `apply-model` (gpu) on it.

For custom intermediate paths or non-default converter options, run the two steps individually via `convert-training`/`convert-inference` and `prep-data`/`apply-model`.

#### Manual preprocessing (existing tasks)

To preprocess an already-built training parquet from the repository root:

```bash
pixi run prep-data -input_parquet Preprocessing/fraghub/fraghub.parquet \
                   -generate_graph_dict -split_type random
```

To generate a graph dictionary from an already-filtered parquet:

```bash
pixi run graph-dict -input_parquet Preprocessing/fraghub/fraghub_filtered.parquet \
                    -output_dict Preprocessing/fraghub/smiles_dict_fraghub.pt
```

To calculate formulas from a raw MS file you can also use our `HRMS_utils` repository (wrapped by the converters above). SIRIUS or other formula annotation programs can also be used. See `Preprocessing/experimental/experimental.csv` for a human-readable reference of the inference schema.

### Model application

To run our model, you can use the notebook `MS_diffusion/src/apply_model.ipynb`.

You can also run experimental inference directly from the command line:

```bash
pixi run apply-model \
  --model_checkpoint [path_to_model_ckpt.ckpt] \
  --experimental_parquet ../../Preprocessing/experimental/experimental.parquet \
  --output_dir ./inference_results \
  --num_repeats 50 \
```

Our trained models (Fraghub_contrastive_random.ckpt, Fraghub_FP_random.ckpt) can be downloaded from [Zenodo](https://zenodo.org/records/19060052).

#### Running `apply-model` as an ensemble

`apply-model` supports generating molecules using **multiple diffusion checkpoints** and aggregating the results into an ensemble:

```bash
pixi run apply-model \
  --experimental_parquet ../../Preprocessing/experimental/experimental.parquet \
  --output_dir ./inference_ensemble \
  --ensemble_models_dir [path_to_model_ckpt.ckpt] \
  --repeats_per_model 25
```

In ensemble mode, the function:

- Runs model for each checkpoint and write results into a subdirectory.
- Merges all generated molecules and SMILES per compound across models.
- Writes ensemble summary files in `output_dir`, including:
  - `inference_summary_ensemble.txt`
  - `all_compounds_smiles_ensemble.txt`
  - `top3_smiles_per_compound_ensemble.csv`


## End-to-end smoke test

A single command that exercises the full MS2Mol pipeline (data prep → encoder
pretraining → graph2mol diffusion pretraining → ms2mol finetuning → test
inference) on a tiny dataset:

```bash
pixi run test-e2e
```

What `test-e2e` does:

1. Runs `pixi run test-prepare-data` to materialise a tiny dataset
   (1000 train + 5 val + 100 test spectra, 1105 rows total) under
   `tests/e2e/data/`.
2. Trains a small contrastive MS encoder (2 epochs).
3. Pretrains a small graph2mol diffusion model (2 epochs, no MS conditioning).
4. Finetunes a small ms2mol diffusion model (2 epochs, MS conditioning) on top
   of the graph2mol checkpoint.
5. Runs test-time inference on the 100 test spectra.

The dataset construction step requires the original FragHub parquet. The
script will look for, in order:

- `Preprocessing/fraghub/fraghub_filtered.parquet`
- `Preprocessing/fraghub/fraghub.parquet`

If neither file is present, the script prints a clear error telling you to
download `FragHub_filtered.parquet` from
[Zenodo record 19060052](https://zenodo.org/records/19060052) and place it at
`Preprocessing/fraghub/fraghub_filtered.parquet`. (If you have the *raw*
`fraghub.parquet` from before filtering, place it at
`Preprocessing/fraghub/fraghub.parquet` and the script will run `pixi run
prep-data` automatically.) Once the data is in place, re-run `pixi run
test-e2e`.

> **Note:** the smoke test uses tiny models (encoder `hidden_dim=256`,
> 2 transformer layers; diffusion `n_layers=2`, `diffusion_steps=50`) and
> only 2 epochs per stage. It is **not** scientifically meaningful — its
> purpose is to verify that every step of the pipeline runs end-to-end on
> your machine before you commit to a full retraining run.

If you want to rebuild the tiny dataset without re-running the rest of the
pipeline (e.g. after changing the seed), use:

```bash
pixi run test-prepare-data
```

To skip the cleanup of previous e2e artifacts (useful when iterating on a
failure), pass `--skip-clean`:

```bash
pixi run test-e2e --skip-clean
```

## Retrain a model

### Preprocessing

We process high-resolution MS datasets using a standardized preparation pipeline that filters invalid entries, annotates fragment-ion formulas, and associates each spectrum with the relevant metadata. The initial processing steps follow our previous work, available in the following repository: https://github.com/Nir-Cohen-2003/HRMS_utils.

The final preprocessing scripts used in this project are provided in the `Preprocessing/` folder, and the raw-data → parquet converters in `converters/` (see [Data preparation](#data-preparation) above).

A fully integrated pipeline with step-by-step explanations **will be added soon**.

In the meantime, the preprocessed FragHub Parquet file and the corresponding molecular graph dictionary (FragHub_filtered_smiles_dict.pt, FragHub_filtered.parquet) can be downloaded from [Zenodo](https://zenodo.org/records/19060052).

### Stage 1: Encoder pretraining

Train the MS spectra encoder with respect to molecular structures. The pretrained contrastive and FP_prediction encoderes (Contrastive_FragHub_random.pth, FP_FragHub_random.pth) can also be downloaded from [Zenodo](https://zenodo.org/records/19060052).

#### Basic Training Example

```bash
pixi run train-encoder \
    -mode contrastive \
    -data_path Preprocessing/fraghub/fraghub_filtered.parquet \
    -smiles_path Preprocessing/fraghub/smiles_dict_fraghub.pt \
    -split_path Preprocessing/fraghub/splits_fraghub_random.pkl \
    -trainable_temperature \
    -comment "contrastive_fraghub"
```

#### Training Modes

- **`contrastive`**: InfoNCE contrastive loss (default)
- **`fp`**: MSE loss to molecular fingerprints

#### Key Arguments

- `-mode`: Training mode (`contrastive`, `fp`, or `mixed`)

**Training:**
- `-batch_size`: Batch size (default: 512)
- `-epochs`: Number of training epochs (default: 500)
- `-lr`: Learning rate (default: 4e-4)
- `-trainable_temperature`: Make temperature trainable

**Data:**
- `-data_path`: Path to processed parquet file
- `-smiles_path`: Path to SMILES graph dictionary
- `-split_path`: Path to predefined splits (uses random split if not provided)

#### Evaluation

Evaluate a trained model:

```bash
pixi run eval-encoder \
    -cp_name [cp_name] \
    -mode contrastive \
    -data_path Preprocessing/fraghub/fraghub_filtered.parquet \
    -smiles_path Preprocessing/fraghub/smiles_dict_fraghub.pt \
    -split_path Preprocessing/fraghub/splits_fraghub_random.pkl \
    -total_samples 512
```

**Evaluation Output:**

The evaluation computes metrics between MS spectrum embeddings and molecular graph embeddings:

- **Pairwise matching accuracy**: Measures how often the model correctly matches MS spectra to their corresponding molecular structures (bidirectional top-1 accuracy)
- **Mean Absolute Error (MAE)**: Average L1 distance between MS and molecular embeddings in the shared embedding space
- **Cosine Similarity**: Average cosine similarity between aligned MS and molecular embedding pairs

Additionally, a **t-SNE visualization** is generated (unless `-no_plot` is specified) showing the 2D projection of both MS and molecular embeddings, with corresponding pairs highlighted. The plot is saved to `analysis_outputs/` with a timestamp.

### Diffusion Model Training

The diffusion model generates molecular graphs conditioned on MS embeddings from the contrastive model. Stages 2-3 consist of pretraining, finetuning, inference, and post-analysis.
This diffusion stage is adapted from [DiGress](https://github.com/cvignac/DiGress).

#### Stage 2: Graph2Mol Pretraining

First, pretrain the diffusion model on molecular graphs without MS conditioning (graph2mol). For example, to run a diffusion model based on graph embeddings from the contrastive model, run:

```bash
pixi run pretrain-diffusion \
    conditioning.embeddings_type=mol2emb \
    conditioning.embedding_model_path=[contrastive_cp_path] \
    train.finetune_ms_encoder=False
```

By default, it uses the FragHub dataset, but you can modify the configuration files in `MS_diffusion/configs/` to work with other datasets.

**Conditioning Types:**
- `ms2emb`: MS spectra → embeddings (from contrastive model) - used for MS2Mol inference
- `ms2fp`: MS spectra → molecular fingerprints
- `mol2emb`: Molecular graphs → embeddings - used for Graph2Mol pretraining
- `mol2fp`: Molecular graphs → fingerprints
- `null`: No conditioning (unconditional generation)

**Key Configuration Files:**
- `configs/conditioning/conditioning_default.yaml`: Conditioning settings
- `configs/general/general_default.yaml`: Training settings, GPU and wandb configuration
- `configs/train/train_default.yaml`: Learning rate, epochs, batch size

#### Stage 3a: MS2Mol Finetuning

Finetune the pretrained Graph2Mol model and pretrained MS encoder to enable MS-to-molecule generation:

```bash
pixi run finetune-diffusion \
    conditioning.embeddings_type=ms2emb \
    conditioning.embedding_model_path=[contrastive_cp_path] \
    general.resume=[graph2mol_cp_path] \
    train.finetune_ms_encoder=True \
    train.lr=0.0001 
```

Note: Change `embeddings_type` from `mol2emb` to `ms2emb` to switch from molecular graph conditioning to MS spectrum conditioning. For FP-based model, change mol2fp to ms2fp.

#### Stage 3b: MS2Mol Inference

Run inference on the test set using the finetuned MS2Mol model:

```bash
pixi run test-inference \
    conditioning.embeddings_type=ms2emb \
    conditioning.embedding_model_path=[contrastive_cp_path] \
    general.test_only=[ms2mol_cp_path] \ 
    general.samples_to_generate='all' \
    train.finetune_ms_encoder=True  
```

This generates molecular structures for each test MS spectrum. By default, the model generates 50 candidate molecules per spectrum and computes metrics using a statistical approach. The inference results are saved in an output folder (typically in `MS_diffusion/outputs/`).

Note that train.finetune_ms_encoder should be true just if the checkpoint provide fineruned MS encoder weights. When relying on freezed MS encoder weights, use False.

#### Stage 3c: Post-Analysis of Inference Results

Analyze the inference results using the provided Jupyter notebook:

```bash
pixi run post-analysis
```

Then in the notebook:

1. Run the first three cells to load necessary functions and data
2. In the fourth cell, specify the inference output folder path from Stage 3b
3. Execute the remaining cells to generate comprehensive evaluation metrics and visualizations

The notebook computes detailed metrics and generates plots, which are saved to `MS_diffusion/analysis_outputs/`.
