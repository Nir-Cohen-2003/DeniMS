"""
Inference function for experimental MS data.
Processes experimental.parquet file, groups by Compound_index, and runs inference.
"""

import os
import sys
import pathlib
import torch
from datetime import datetime
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from collections import defaultdict, Counter
import copy
import re

_ms_diffusion_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ms_diffusion_dir not in sys.path:
    sys.path.insert(0, _ms_diffusion_dir)

_ms2mol_root = pathlib.Path(os.path.realpath(__file__)).parents[2]
if str(_ms2mol_root) not in sys.path:
    sys.path.insert(0, str(_ms2mol_root))

from diffusion_model_ms import DiscreteEdgesDenoisingDiffusion
import utils
from diffusion.extra_features import ExtraFeatures
from diffusion.extra_features_molecular import ExtraMolecularFeatures
from dataloaders import one_hot_encode_precursor, one_hot_encode_energy, positional_encoding, elements
from model import Contrastive_model
from analysis.rdkit_functions import build_molecule_with_partial_charges, fix_aromatic_smiles
from rdkit import Chem
from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscreteEdges
from metrics.molecular_metrics import SamplingMolecularMetricsEdges
from analysis.visualization import MolecularVisualization
from pytorch_lightning import Trainer


HRMS_ELEMENT_ORDER = ("H", "C", "N", "O", "F", "Na", "P", "S", "Cl", "K", "Br", "I")
DIFFUSION_PAD_NODES = 29


def checkpoint_label(checkpoint_path):
    """Return a filesystem-safe label from a checkpoint path."""
    if not checkpoint_path:
        return "unknown_model"
    name = os.path.basename(checkpoint_path)
    if name.endswith(".ckpt"):
        name = name[:-5]
    name = re.sub(r"[^\w.\-]+", "_", name).strip("._")
    return name or "unknown_model"


def _parse_formula_array(precursor_formula):
    if isinstance(precursor_formula, str):
        import ast
        text = precursor_formula.strip()
        try:
            precursor_formula = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            inner = text.strip("[]")
            precursor_formula = np.fromstring(inner, sep=" ")

    return np.asarray(precursor_formula, dtype=int).flatten()


def normalize_precursor_formula_counts(precursor_formula):
    """
    Convert a precursor formula vector into element-symbol counts.

    Supports:
    - DeniMS 9-element order: H, C, N, O, F, S, Cl, Br, I
    - HRMS 12-element order: H, C, N, O, F, Na, P, S, Cl, K, Br, I
    """
    formula = _parse_formula_array(precursor_formula)

    if len(formula) == len(elements):
        return {elements[i]: int(formula[i]) for i in range(len(elements))}

    if len(formula) == len(HRMS_ELEMENT_ORDER):
        counts = {sym: 0 for sym in elements}
        for i, sym in enumerate(HRMS_ELEMENT_ORDER):
            if sym in counts:
                counts[sym] += int(formula[i])
        return counts

    raise ValueError(
        f"precursor_formula length {len(formula)} not supported; "
        f"expected {len(elements)} (DeniMS) or {len(HRMS_ELEMENT_ORDER)} (HRMS)"
    )


def parse_precursor_formula_to_nodes(precursor_formula, atom_decoder, remove_h=True):
    """Convert a precursor elemental formula into a one-hot node feature matrix."""
    counts = normalize_precursor_formula_counts(precursor_formula)
    atom_encoder = {sym: i for i, sym in enumerate(atom_decoder)}

    atom_types = []
    for sym in elements:
        if remove_h and sym == "H":
            continue
        count = counts.get(sym, 0)
        if count <= 0:
            continue
        if sym not in atom_encoder:
            raise ValueError(f"Element {sym} from precursor_formula is not in atom_decoder")
        atom_types.extend([atom_encoder[sym]] * count)

    for sym in atom_decoder:
        if sym in elements or (remove_h and sym == "H"):
            continue
        count = counts.get(sym, 0)
        if count > 0:
            atom_types.extend([atom_encoder[sym]] * count)

    n_nodes = len(atom_types)
    if n_nodes == 0:
        raise ValueError(f"precursor_formula resulted in 0 nodes: {precursor_formula}")

    X = torch.zeros(n_nodes, len(atom_decoder))
    for i, atom_type in enumerate(atom_types):
        X[i, atom_type] = 1.0

    return X, n_nodes


def pad_node_features_for_diffusion(X, target_n=DIFFUSION_PAD_NODES):
    """Zero-pad node features along the node dimension to target_n."""
    if X.dim() == 2:
        n_nodes, feat_dim = X.shape
        if n_nodes >= target_n:
            return X
        pad = torch.zeros(target_n - n_nodes, feat_dim, dtype=X.dtype, device=X.device)
        return torch.cat([X, pad], dim=0)

    if X.dim() == 3:
        batch, n_nodes, feat_dim = X.shape
        if n_nodes >= target_n:
            return X
        pad = torch.zeros(batch, target_n - n_nodes, feat_dim, dtype=X.dtype, device=X.device)
        return torch.cat([X, pad], dim=1)

    raise ValueError(f"Expected 2D or 3D node features, got {X.dim()}D")


def process_spectra_to_embeddings(spectra_data, ms_encoder_model, device, max_peaks=128):
    """Convert a list of experimental spectra into a single MS embedding."""
    pos_encoding = positional_encoding()
    padded_tensor_template = torch.zeros((max_peaks, 144), dtype=torch.float32)
    sos_list, formula_array_list, mask_list = [], [], []

    for spec in spectra_data:
        precursor_type = spec.get('precursor_type', '[M+H]+')
        if not isinstance(precursor_type, str) or pd.isna(precursor_type):
            precursor_type = '[M+H]+'
        precursor_onehot = one_hot_encode_precursor(precursor_type)
        energy_onehot = one_hot_encode_energy(int(spec['collision_energy_NCE']))
        sos = torch.cat([precursor_onehot, energy_onehot], dim=0).view(1, -1)
        sos_list.append(sos)

        formulas = spec['formulas']
        if isinstance(formulas, str):
            import ast
            formulas = ast.literal_eval(formulas)

        total_dim = len(elements) * 16
        peak_encodings = []
        for peak_formula in formulas:
            if isinstance(peak_formula, str):
                import ast
                peak_formula = ast.literal_eval(peak_formula)
            peak_formula = np.array(peak_formula)
            tensor = torch.zeros(total_dim)
            start_idx = 0
            for idx in range(len(elements)):
                value = int(peak_formula[idx]) if idx < len(peak_formula) else 0
                tensor[start_idx:start_idx + 16] += pos_encoding.encode(value)
                start_idx += 16
            peak_encodings.append(tensor)

        if len(peak_encodings) == 0:
            peak_encodings = [torch.zeros(total_dim)]

        array = torch.stack(peak_encodings)
        n = array.shape[0]
        padded_tensor = padded_tensor_template.clone()
        padded_tensor[:n] = array
        mask = torch.ones(max_peaks + 1, dtype=torch.bool)
        mask[:n + 1] = 0
        formula_array_list.append(padded_tensor)
        mask_list.append(mask)

    sos_batch = torch.stack(sos_list, dim=0).to(device)
    formula_array_batch = torch.stack(formula_array_list, dim=0).to(device)
    mask_batch = torch.stack(mask_list, dim=0).to(device)

    ms_encoder_model.eval()
    with torch.no_grad():
        ms_emb = ms_encoder_model.ms_encoder(sos_batch, formula_array_batch, mask=mask_batch).float()
        ms_emb = ms_emb / (ms_emb.norm(dim=1, keepdim=True) + 1e-8)
        emb = torch.mean(ms_emb, dim=0, keepdim=True)
        emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)

    return emb


def aggregate_ensemble_from_existing_runs(root_dir, output_dir=None):
    """Aggregate SMILES outputs from previous ensemble inference runs on disk."""
    output_dir = output_dir or root_dir
    if not os.path.isdir(root_dir):
        raise ValueError(f"Root directory '{root_dir}' does not exist")

    def _compound_smiles_paths(search_dir):
        paths = []
        if not os.path.isdir(search_dir):
            return paths
        for fname in os.listdir(search_dir):
            fpath = os.path.join(search_dir, fname)
            if fname.startswith("compound_") and fname.endswith("_smiles.txt") and os.path.isfile(fpath):
                paths.append(fpath)
            elif os.path.isdir(fpath):
                for nested in os.listdir(fpath):
                    npath = os.path.join(fpath, nested)
                    if nested.startswith("compound_") and nested.endswith("_smiles.txt") and os.path.isfile(npath):
                        paths.append(npath)
        return paths

    model_dirs = [
        os.path.join(root_dir, d) for d in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    if len(model_dirs) == 0:
        raise ValueError(f"No model subdirectories found in '{root_dir}'")

    compound_to_smiles = defaultdict(list)
    for mdir in model_dirs:
        for fpath in _compound_smiles_paths(mdir):
            with open(fpath, "r") as f:
                lines = [line.strip() for line in f.readlines()]
            if not lines:
                continue
            compound_index = None
            for line in lines:
                if line.startswith("Compound_index:"):
                    compound_index = int(line.split(":", 1)[1].strip())
                    break
            if compound_index is None:
                continue
            for line in lines:
                if line.startswith("Repeat_") and ":" in line:
                    _, val = line.split(":", 1)
                    compound_to_smiles[compound_index].append(val.strip())

    rows = []
    for compound_index, smiles_list in sorted(compound_to_smiles.items()):
        valid_smiles = [s for s in smiles_list if s not in (None, "N/A")]
        total_generated = len(smiles_list)
        if valid_smiles:
            for rank, (smiles, count) in enumerate(Counter(valid_smiles).most_common(5), start=1):
                rows.append({"compound_index": compound_index, "rank": rank, "smiles": smiles,
                            "count": count, "total_generated": total_generated})
        else:
            rows.append({"compound_index": compound_index, "rank": 1, "smiles": "N/A",
                        "count": 0, "total_generated": total_generated})

    if not rows:
        raise ValueError(f"No SMILES data found under '{root_dir}'")

    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "top5_smiles_per_compound_ensemble.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Top-5 ensemble SMILES per compound saved to {out_csv}")
    return pd.DataFrame(rows)


def run_inference_experimental(
    experimental_parquet_path,
    model_checkpoint_path=None,
    encoder_checkpoint_path=None,
    output_dir="./inference_results",
    num_repeats=25,
    batch_size=32,
    device=None,
    ensemble_model_checkpoints=None,
    repeats_per_model=25,
    ensemble_models_dir=None,
):
    """Run diffusion-based molecular structure inference on experimental MS data."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_suffix = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    base_output_dir = output_dir
    output_dir = os.path.join(base_output_dir, run_suffix)

    if ensemble_models_dir and (not ensemble_model_checkpoints or len(ensemble_model_checkpoints) == 0):
        if not os.path.isdir(ensemble_models_dir):
            raise ValueError(f"ensemble_models_dir '{ensemble_models_dir}' does not exist")
        ckpt_files = [os.path.join(ensemble_models_dir, f) for f in sorted(os.listdir(ensemble_models_dir)) if f.endswith(".ckpt")]
        if len(ckpt_files) == 0:
            raise ValueError(f"No .ckpt files found in ensemble_models_dir '{ensemble_models_dir}'")
        ensemble_model_checkpoints = ckpt_files

    df = pq.read_table(experimental_parquet_path, use_threads=True).to_pandas()
    compounds = {idx: group.to_dict("records") for idx, group in df.groupby("Compound_index")}

    if ensemble_model_checkpoints and len(ensemble_model_checkpoints) > 0:
        model_ckpts = list(ensemble_model_checkpoints)
        if model_checkpoint_path and model_checkpoint_path not in model_ckpts:
            model_ckpts.insert(0, model_checkpoint_path)

        all_runs_results = []
        for ckpt in model_ckpts:
            member_output_dir = os.path.join(output_dir, checkpoint_label(ckpt))
            member_results = run_inference_experimental(
                model_checkpoint_path=ckpt, experimental_parquet_path=experimental_parquet_path,
                encoder_checkpoint_path=encoder_checkpoint_path, output_dir=member_output_dir,
                num_repeats=repeats_per_model, batch_size=batch_size, device=device,
                ensemble_model_checkpoints=None, repeats_per_model=repeats_per_model,
            )
            all_runs_results.append(member_results)

        if not all_runs_results:
            return []

        base_results = all_runs_results[0]
        index_to_pos = {res["compound_index"]: pos for pos, res in enumerate(base_results)}
        combined_results = []

        for compound_index, base_pos in index_to_pos.items():
            combined_res = copy.deepcopy(base_results[base_pos])
            combined_molecules = list(combined_res.get("molecules", []))
            combined_smiles = list(combined_res.get("smiles", []))
            total_success = combined_res.get("num_successful_repeats", 0)

            for member_results in all_runs_results[1:]:
                member_pos = next((pos for pos, res in enumerate(member_results) if res["compound_index"] == compound_index), None)
                if member_pos is None:
                    raise ValueError(f"compound_index {compound_index} missing in ensemble member results")
                member_res = member_results[member_pos]
                combined_molecules.extend(member_res.get("molecules", []))
                combined_smiles.extend(member_res.get("smiles", []))
                total_success += member_res.get("num_successful_repeats", 0)

            combined_res["molecules"] = combined_molecules
            combined_res["smiles"] = combined_smiles
            combined_res["num_successful_repeats"] = total_success
            combined_results.append(combined_res)

        os.makedirs(output_dir, exist_ok=True)
        total_repeats = repeats_per_model * len(model_ckpts)

        with open(os.path.join(output_dir, "inference_summary_ensemble.txt"), "w") as f:
            f.write("Ensemble Inference Summary\n" + "=" * 50 + "\n\n")
            f.write(f"Total compounds: {len(combined_results)}\n")
            f.write(f"Models in ensemble: {len(model_ckpts)}\n")
            f.write("Model checkpoints:\n")
            for ckpt in model_ckpts:
                f.write(f"  - {checkpoint_label(ckpt)}: {ckpt}\n")
            f.write(f"Repeats per model: {repeats_per_model}\n")
            f.write(f"Total repeats per compound: {total_repeats}\n\n")
            for result in combined_results:
                valid_smiles = [s for s in result.get("smiles", []) if s is not None]
                f.write(f"Compound_index {result['compound_index']}:\n")
                f.write(f"  Nodes (non-H): {result['n_nodes']}\n")
                f.write(f"  Spectra: {result['num_spectra']}\n")
                f.write(f"  Successful repeats: {result['num_successful_repeats']}/{total_repeats}\n")
                f.write(f"  Valid SMILES: {len(valid_smiles)}/{len(result.get('smiles', []))}\n\n")

        with open(os.path.join(output_dir, "all_compounds_smiles_ensemble.txt"), "w") as f:
            f.write("All Generated SMILES (Ensemble)\n" + "=" * 50 + "\n\n")
            for result in combined_results:
                f.write(f"Compound_index: {result['compound_index']}\n")
                f.write(f"Precursor_formula: {result['precursor_formula']}\n")
                for i, smiles in enumerate(result.get("smiles", [])):
                    f.write(f"  Repeat_{i+1}: {smiles if smiles is not None else 'N/A'}\n")
                f.write("\n")

        top5_rows = []
        for res in combined_results:
            smiles_list = [s for s in res.get("smiles", []) if s not in (None, "N/A")]
            if smiles_list:
                total = len(smiles_list)
                for rank, (smi, cnt) in enumerate(Counter(smiles_list).most_common(5), start=1):
                    top5_rows.append({"compound_index": res["compound_index"], "rank": rank,
                                    "smiles": smi, "count": cnt, "total_generated": total})

        if top5_rows:
            pd.DataFrame(top5_rows).to_csv(os.path.join(output_dir, "top5_smiles_per_compound_ensemble.csv"), index=False)

        return combined_results

    checkpoint = torch.load(model_checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint['hyper_parameters']['cfg']
    if cfg is None:
        raise ValueError("Could not find config in checkpoint")

    dataset_infos = checkpoint['hyper_parameters'].get('dataset_infos', None)
    if dataset_infos is None:
        raise ValueError(
            "dataset_infos not found in checkpoint hyper_parameters. "
            "Please ensure models were trained with dataset_infos passed to the Lightning module."
        )

    default_edge_dist = torch.tensor([9.0163e-01, 4.7308e-02, 6.2059e-03, 1.2201e-04, 4.4735e-02])
    dataset_infos.edge_types = default_edge_dist

    train_smiles = []
    extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos, embeddings=True)

    ms_encoder_model = None
    if getattr(cfg.train, "finetune_ms_encoder", False):
        model_state_dict = checkpoint.get('state_dict', {})
        encoder_keys_in_model = [k for k in model_state_dict.keys() if k.startswith('ms_encoder_model.')]

        if len(encoder_keys_in_model) > 0:
            encoder_state_dict = {key[len('ms_encoder_model.'):]: model_state_dict[key] for key in encoder_keys_in_model}
            encoder_ckpt = checkpoint
        else:
            if encoder_checkpoint_path is None:
                encoder_checkpoint_path = getattr(cfg.conditioning, "embedding_model_path", None)
            if encoder_checkpoint_path is None:
                raise ValueError("finetune_ms_encoder=True but no encoder weights found")
            if not os.path.isabs(encoder_checkpoint_path):
                encoder_checkpoint_path = os.path.join(str(_ms2mol_root), encoder_checkpoint_path)
            if not os.path.exists(encoder_checkpoint_path):
                raise ValueError(f"encoder_checkpoint_path not found: {encoder_checkpoint_path}")
            encoder_ckpt = torch.load(encoder_checkpoint_path, map_location=device, weights_only=False)
            if "model" not in encoder_ckpt:
                raise KeyError(f"Checkpoint at {encoder_checkpoint_path} does not contain a 'model' state_dict")
            encoder_state_dict = encoder_ckpt["model"]

        emb_type = getattr(cfg.conditioning, "embeddings_type", None)
        if emb_type == "ms2fp":
            output_dim = encoder_state_dict["out_mlp.2.bias"].shape[0]
        else:
            output_dim = encoder_state_dict["ms_encoder.proj"].shape[1]

        is_graph = any("graph_encoder" in k for k in encoder_state_dict.keys())
        fp_pred = any("out_mlp" in k for k in encoder_state_dict.keys())
        trainable_temperature = any("inv_temperature" in k for k in encoder_state_dict.keys())
        temperature = encoder_ckpt.get("temperature", 15.0)

        ms_encoder_model = Contrastive_model(
            hidden_dim=512, max_len=129, num_transformer_layers=3, nhead=8,
            embeddings_dim=output_dim, dropout=0.1, input_dropout=0.1,
            fp_length=output_dim if emb_type == "ms2fp" else 2048,
            graph=is_graph, fp_pred=fp_pred, initial_temperature=temperature,
            trainable_temperature=trainable_temperature,
        ).to(device)
        ms_encoder_model.load_state_dict(encoder_state_dict, strict=False)
        ms_encoder_model.eval()

    train_metrics = TrainMolecularMetricsDiscreteEdges(dataset_infos)
    sampling_metrics = SamplingMolecularMetricsEdges(
        dataset_infos, train_smiles,
        compute_mces=getattr(cfg.general, 'compute_mces', True),
        mces_timeout_sec=getattr(cfg.general, 'mces_timeout_sec', 120))
    visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

    finetune_encoder = getattr(cfg.train, "finetune_ms_encoder", False)
    ms_dataframe_for_model = pd.DataFrame([{'precursor_type': '[M+H]+', 'collision_energy_NCE': 0,
                                           'clean_spectrum_formula_array': []}]) if finetune_encoder else None
    ms_graph_dict_for_model = {'dummy_smiles': []} if finetune_encoder else None

    model = DiscreteEdgesDenoisingDiffusion(
        cfg=cfg, dataset_infos=dataset_infos, train_metrics=train_metrics,
        sampling_metrics=sampling_metrics, visualization_tools=visualization_tools,
        extra_features=extra_features, domain_features=domain_features,
        ms_dataframe=ms_dataframe_for_model, ms_graph_dict=ms_graph_dict_for_model,
    )

    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.to(device)
    model.eval()

    accelerator = 'gpu' if (isinstance(device, torch.device) and device.type == 'cuda') or (isinstance(device, str) and device == 'cuda') or torch.cuda.is_available() else 'cpu'
    inference_trainer = Trainer(accelerator=accelerator, devices=1, logger=False, enable_checkpointing=False,
                                enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
                                limit_train_batches=0, limit_val_batches=0, limit_test_batches=0)
    model.trainer = inference_trainer

    if not hasattr(inference_trainer, "strategy") or inference_trainer.strategy is None:
        class MinimalStrategy:
            def __init__(self):
                self.is_global_zero = True

        inference_trainer.strategy = MinimalStrategy()

    model_ckpts = [model_checkpoint_path]
    model_labels = [checkpoint_label(p) for p in model_ckpts]
    per_model_repeats = num_repeats
    total_repeats = per_model_repeats * len(model_ckpts)

    ensemble_state_dicts = []
    expected_emb_dim = getattr(dataset_infos, "embeddings_dims", None)
    emb_type = getattr(cfg.conditioning, "embeddings_type", None)

    for ckpt_path in model_ckpts:
        ckpt_i = checkpoint if ckpt_path == model_checkpoint_path else torch.load(ckpt_path, map_location=device, weights_only=False)
        if "state_dict" not in ckpt_i:
            raise ValueError(f"Checkpoint {ckpt_path} has no 'state_dict' key")

        state_dict_i = ckpt_i["state_dict"]
        encoder_keys = [k for k in state_dict_i.keys() if k.startswith("ms_encoder_model.")]

        if encoder_keys:
            if emb_type == "ms2fp":
                emb_dim_i = state_dict_i.get("ms_encoder_model.out_mlp.2.bias", torch.tensor([])).shape[0] if "ms_encoder_model.out_mlp.2.bias" in state_dict_i else None
            else:
                emb_dim_i = state_dict_i.get("ms_encoder_model.ms_encoder.proj.weight", torch.tensor([])).shape[1] if "ms_encoder_model.ms_encoder.proj.weight" in state_dict_i else None

            if expected_emb_dim is not None and emb_dim_i is not None and emb_dim_i != expected_emb_dim:
                raise ValueError(f"Ensemble checkpoint {ckpt_path} has encoder output dim {emb_dim_i}, but expected {expected_emb_dim}")

        ensemble_state_dicts.append(state_dict_i)

    model.load_state_dict(ensemble_state_dicts[0], strict=False)
    model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    per_model_outputs = {label: [] for label in model_labels}

    for compound_idx, spectra_list in tqdm(compounds.items(), desc="Processing compounds"):
        precursor_formula = spectra_list[0]['precursor_formula']
        X_init, n_nodes = parse_precursor_formula_to_nodes(
            precursor_formula, dataset_infos.atom_decoder, remove_h=cfg.dataset.remove_h
        )
        X_init = pad_node_features_for_diffusion(X_init)

        if ms_encoder_model is None:
            raise ValueError("MS encoder is required for inference")

        embeddings = process_spectra_to_embeddings(spectra_list, ms_encoder_model, device)
        if embeddings.shape[0] != 1:
            raise ValueError(f"Embeddings batch size must be 1, got {embeddings.shape[0]}")
        if embeddings.dim() != 2:
            raise ValueError(f"Embeddings must be 2D, got {embeddings.dim()}D")

        expected_emb_dim = getattr(dataset_infos, 'embeddings_dims', None)
        if expected_emb_dim is not None and embeddings.shape[1] != expected_emb_dim:
            raise ValueError(f"Embedding dimension mismatch: got {embeddings.shape[1]}, expected {expected_emb_dim}")

        X_dense = X_init.unsqueeze(0).to(device)
        embeddings = embeddings.to(device)

        compound_smiles_by_model = {}
        compound_molecules_by_model = {}

        for m_idx, state_dict in enumerate(ensemble_state_dicts):
            model_label = model_labels[m_idx]
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()

            X_dense_batch = X_dense.repeat(per_model_repeats, 1, 1)
            embeddings_batch = embeddings.repeat(per_model_repeats, 1)
            target_smiles_batch = [f"compound_{compound_idx}_{model_label}_repeat_{i}" for i in range(per_model_repeats)]

            dense_data_batch = utils.PlaceHolder(X=X_dense_batch, E=None, y=None)
            data_list = [(dense_data_batch, embeddings_batch, target_smiles_batch)]

            molecule_list, _ = model.sample_batch(batch_id=0, batch_size=per_model_repeats, keep_chain=0,
                                                  number_chain_steps=0, save_final=0, num_nodes=None, data=data_list)

            smiles_list = []
            atom_decoder = dataset_infos.atom_decoder

            for molecule in molecule_list:
                atom_types, edge_types = molecule
                atom_types = atom_types.cpu() if isinstance(atom_types, torch.Tensor) else torch.tensor(atom_types, dtype=torch.long)
                edge_types = edge_types.cpu() if isinstance(edge_types, torch.Tensor) else torch.tensor(edge_types, dtype=torch.long)

                valid_mask = (atom_types >= 0) & (atom_types < len(atom_decoder))
                if not valid_mask.all():
                    valid_indices = torch.where(valid_mask)[0]
                    if len(valid_indices) == 0:
                        smiles_list.append(None)
                        continue
                    atom_types = atom_types[valid_indices]
                    edge_types = edge_types[valid_indices][:, valid_indices]

                mol = build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder=atom_decoder)
                if mol is None:
                    smiles_list.append(None)
                    continue

                smiles = Chem.MolToSmiles(mol)
                smiles = fix_aromatic_smiles(smiles)
                smiles_list.append(smiles if smiles and "." not in smiles else None)

            compound_smiles_by_model[model_label] = smiles_list
            compound_molecules_by_model[model_label] = molecule_list

            model_results_dir = os.path.join(output_dir, model_label)
            os.makedirs(model_results_dir, exist_ok=True)

            with open(os.path.join(model_results_dir, f"compound_{compound_idx}_molecules.txt"), 'w') as f:
                f.write(f"Model: {model_label}\n")
                f.write(f"Model_checkpoint: {model_ckpts[m_idx]}\n")
                f.write(f"Compound_index: {compound_idx}\nPrecursor_formula: {precursor_formula}\n")
                f.write(f"Number of nodes (non-H): {n_nodes}\nNumber of spectra: {len(spectra_list)}\n")
                f.write(f"Number of successful repeats: {len(molecule_list)}\n\n")
                for i, (molecule, smiles) in enumerate(zip(molecule_list, smiles_list)):
                    f.write(f"=== Repeat {i+1} ===\nSMILES: {smiles if smiles is not None else 'N/A'}\n")
                    atom_types, edge_types = molecule
                    f.write(f"N={atom_types.shape[0]}\nX: \n")
                    f.write(" ".join(map(str, atom_types.tolist())) + "\nE: \n")
                    for bond_list in edge_types:
                        if hasattr(bond_list, "tolist"):
                            row_vals = bond_list.tolist()
                        else:
                            row_vals = list(bond_list)
                        f.write(" ".join(str(int(v)) for v in row_vals) + "\n")
                    f.write("\n")

            with open(os.path.join(model_results_dir, f"compound_{compound_idx}_smiles.txt"), 'w') as f:
                f.write(f"Model: {model_label}\n")
                f.write(f"Model_checkpoint: {model_ckpts[m_idx]}\n")
                f.write(f"Compound_index: {compound_idx}\nPrecursor_formula: {precursor_formula}\n")
                f.write(f"Number of successful repeats: {len(molecule_list)}\n\n")
                for i, smiles in enumerate(smiles_list):
                    f.write(f"Repeat_{i+1}: {smiles if smiles is not None else 'N/A'}\n")

            per_model_outputs[model_label].append({
                'compound_index': compound_idx,
                'n_nodes': n_nodes,
                'precursor_formula': precursor_formula,
                'num_spectra': len(spectra_list),
                'num_successful_repeats': len(molecule_list),
                'molecules': molecule_list,
                'smiles': smiles_list,
                'model_label': model_label,
                'model_checkpoint': model_ckpts[m_idx],
            })

        all_molecules = []
        smiles_list = []
        for model_label in model_labels:
            all_molecules.extend(compound_molecules_by_model[model_label])
            smiles_list.extend(compound_smiles_by_model[model_label])

        compound_results = {
            'compound_index': compound_idx, 'n_nodes': n_nodes, 'precursor_formula': precursor_formula,
            'num_spectra': len(spectra_list), 'num_successful_repeats': len(all_molecules),
            'molecules': all_molecules, 'smiles': smiles_list,
            'models': model_labels,
        }
        all_results.append(compound_results)

    for model_label in model_labels:
        model_results_dir = os.path.join(output_dir, model_label)
        os.makedirs(model_results_dir, exist_ok=True)
        model_results = per_model_outputs[model_label]

        with open(os.path.join(model_results_dir, "inference_summary.txt"), 'w') as f:
            f.write("Inference Summary\n" + "=" * 50 + "\n\n")
            f.write(f"Model: {model_label}\n")
            f.write(f"Model_checkpoint: {model_ckpts[model_labels.index(model_label)]}\n")
            f.write(f"Total compounds: {len(model_results)}\n")
            f.write(f"Total repeats per compound: {per_model_repeats}\n\n")
            for result in model_results:
                valid_smiles = [s for s in result.get('smiles', []) if s is not None]
                f.write(f"Compound_index {result['compound_index']}:\n")
                f.write(f"  Nodes (non-H): {result['n_nodes']}\n  Spectra: {result['num_spectra']}\n")
                f.write(f"  Successful repeats: {result['num_successful_repeats']}/{per_model_repeats}\n")
                f.write(f"  Valid SMILES: {len(valid_smiles)}/{len(result.get('smiles', []))}\n\n")

        with open(os.path.join(model_results_dir, "all_compounds_smiles.txt"), 'w') as f:
            f.write("All Generated SMILES\n" + "=" * 50 + "\n\n")
            f.write(f"Model: {model_label}\n")
            f.write(f"Model_checkpoint: {model_ckpts[model_labels.index(model_label)]}\n\n")
            for result in model_results:
                f.write(f"Compound_index: {result['compound_index']}\n")
                f.write(f"Precursor_formula: {result['precursor_formula']}\n")
                for i, smiles in enumerate(result.get('smiles', [])):
                    f.write(f"  Repeat_{i+1}: {smiles if smiles is not None else 'N/A'}\n")
                f.write("\n")

        top5_rows = []
        for res in model_results:
            smiles_list = [s for s in res.get("smiles", []) if s not in (None, "N/A")]
            if smiles_list:
                total = len(smiles_list)
                for rank, (smi, cnt) in enumerate(Counter(smiles_list).most_common(5), start=1):
                    top5_rows.append({
                        "model": model_label,
                        "compound_index": res["compound_index"],
                        "rank": rank,
                        "smiles": smi,
                        "count": cnt,
                        "total_generated": total,
                    })

        if top5_rows:
            pd.DataFrame(top5_rows).to_csv(os.path.join(model_results_dir, "top5_smiles_per_compound.csv"), index=False)

    with open(os.path.join(output_dir, "inference_summary.txt"), 'w') as f:
        f.write("Inference Summary\n" + "=" * 50 + "\n\n")
        f.write(f"Total compounds: {len(all_results)}\n")
        f.write(f"Models: {', '.join(model_labels)}\n")
        f.write(f"Total repeats per compound (all models): {total_repeats}\n\n")
        for result in all_results:
            valid_smiles = [s for s in result.get('smiles', []) if s is not None]
            f.write(f"Compound_index {result['compound_index']}:\n")
            f.write(f"  Nodes (non-H): {result['n_nodes']}\n  Spectra: {result['num_spectra']}\n")
            f.write(f"  Successful repeats: {result['num_successful_repeats']}/{total_repeats}\n")
            f.write(f"  Valid SMILES: {len(valid_smiles)}/{len(result.get('smiles', []))}\n\n")

    with open(os.path.join(output_dir, "all_compounds_smiles.txt"), 'w') as f:
        f.write("All Generated SMILES\n" + "=" * 50 + "\n\n")
        f.write(f"Models: {', '.join(model_labels)}\n\n")
        for result in all_results:
            f.write(f"Compound_index: {result['compound_index']}\n")
            f.write(f"Precursor_formula: {result['precursor_formula']}\n")
            for i, smiles in enumerate(result.get('smiles', [])):
                f.write(f"  Repeat_{i+1}: {smiles if smiles is not None else 'N/A'}\n")
            f.write("\n")

    top5_rows = []
    for res in all_results:
        smiles_list = [s for s in res.get("smiles", []) if s not in (None, "N/A")]
        if smiles_list:
            total = len(smiles_list)
            for rank, (smi, cnt) in enumerate(Counter(smiles_list).most_common(5), start=1):
                top5_rows.append({"compound_index": res["compound_index"], "rank": rank, "smiles": smi,
                                "count": cnt, "total_generated": total})

    if top5_rows:
        pd.DataFrame(top5_rows).to_csv(os.path.join(output_dir, "top5_smiles_per_compound.csv"), index=False)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on experimental MS data")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        help="Path to trained diffusion model checkpoint (single-model mode, or optionally included in ensemble)",
    )
    parser.add_argument("--experimental_parquet", type=str, default="/gpfs0/bgu-anatm/users/harniky/ms2mol/Preprocessing/experimental/experimental.parquet", help="Path to experimental.parquet file")
    parser.add_argument("--encoder_checkpoint", type=str, default=None, help="Path to MS encoder checkpoint (if finetune_ms_encoder was used)")
    parser.add_argument("--output_dir", type=str, default="./inference_results", help="Directory to save results")
    parser.add_argument("--num_repeats", type=int, default=100, help="Number of inference repeats per compound")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument(
        "--ensemble_model_checkpoints",
        type=str,
        nargs="+",
        default=None,
        help="List of diffusion model checkpoints to use as an ensemble",
    )
    parser.add_argument(
        "--repeats_per_model",
        type=int,
        default=25,
        help="Number of inference repeats per compound for each ensemble member",
    )
    parser.add_argument(
        "--ensemble_models_dir",
        type=str,
        default=None,
        help="Directory containing multiple .ckpt files to use as an ensemble (optional alternative to --ensemble_model_checkpoints)",
    )

    args = parser.parse_args()

    if args.model_checkpoint is None and not args.ensemble_model_checkpoints and args.ensemble_models_dir is None:
        parser.error("You must provide either --model_checkpoint or --ensemble_model_checkpoints or --ensemble_models_dir")

    run_inference_experimental(
        experimental_parquet_path=args.experimental_parquet,
        model_checkpoint_path=args.model_checkpoint,
        encoder_checkpoint_path=args.encoder_checkpoint,
        output_dir=args.output_dir,
        num_repeats=args.num_repeats,
        batch_size=args.batch_size,
        device=torch.device(args.device) if args.device else None,
        ensemble_model_checkpoints=args.ensemble_model_checkpoints,
        repeats_per_model=args.repeats_per_model,
        ensemble_models_dir=args.ensemble_models_dir,
    )
