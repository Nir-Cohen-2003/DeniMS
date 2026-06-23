import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import pytorch_lightning as pl
import time
import wandb
import os
import random
import pathlib
import sys

from tqdm import tqdm

from models.transformer_model import GraphTransformer
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete, MarginalUniformEdgesTransition
from diffusion import diffusion_utils
from metrics.train_metrics import TrainLossDiscreteEdges
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
import utils

_ms2mol_root = pathlib.Path(os.path.realpath(__file__)).parents[2]
if str(_ms2mol_root) not in sys.path:
    sys.path.insert(0, str(_ms2mol_root))

from model import Contrastive_model
from dataloaders import (
    one_hot_encode_precursor, one_hot_encode_energy, 
    positional_encoding, elements
)

class DiscreteEdgesDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features, ms_dataframe=None, ms_graph_dict=None):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist
        self.embeddings_dims = dataset_infos.embeddings_dims
        self.max_nodes = len(dataset_infos.n_nodes)

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.len_data = len(sampling_metrics.train_smiles)

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscreteEdges(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features
        self.ms_dataframe = ms_dataframe
        self.ms_graph_dict = ms_graph_dict

        self.model = GraphTransformer(
            n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        )

        # Optional ms2mol encoder for online embedding & finetuning
        self.finetune_ms_encoder = getattr(cfg.train, "finetune_ms_encoder", False)
        self.embeddings_type = getattr(cfg.conditioning, "embeddings_type", None)
        self.ms_encoder_model = None

        if self.finetune_ms_encoder:
            if Contrastive_model is None:
                raise ImportError("Could not import Contrastive_model from ms2mol.model for MS encoder finetuning.")
            if one_hot_encode_precursor is None or one_hot_encode_energy is None:
                raise ImportError("Could not import MS data processing functions from ms2mol.dataloaders.")
            if self.ms_dataframe is None or self.ms_graph_dict is None:
                raise ValueError("ms_dataframe and ms_graph_dict must be provided when finetune_ms_encoder=True.")
            
            # Initialize positional encoding for MS feature processing
            self.ms_positional_encoding = positional_encoding()
            self.ms_max_peaks = 128 
            self.ms_padded_tensor_template = torch.zeros((self.ms_max_peaks, 144), dtype=torch.float32)


            #load the MS encoder
            encoder_ckpt_path = getattr(cfg.conditioning, "embedding_model_path", None)
            encoder_ckpt_path_clean = None
            if encoder_ckpt_path is not None:
                encoder_ckpt_path_clean = str(encoder_ckpt_path).strip()

            load_from_ckpt = (encoder_ckpt_path_clean is not None and encoder_ckpt_path_clean.lower() not in ("none", ""))

            if load_from_ckpt:

                if not os.path.isabs(encoder_ckpt_path_clean):
                    encoder_ckpt_path_clean = os.path.join(str(_ms2mol_root), encoder_ckpt_path_clean)

                checkpoint = torch.load(
                    encoder_ckpt_path_clean,
                    map_location=torch.device("cpu"),
                    weights_only=False,
                )

                if "model" not in checkpoint:
                        raise KeyError(
                            f"Checkpoint at {encoder_ckpt_path_clean} does not contain a 'model' state_dict."
                        )

                state_dict = checkpoint["model"]

                if self.embeddings_type == "ms2fp":
                    output_dim = state_dict["out_mlp.2.bias"].shape[0]
                    hidden_dim = state_dict["ms_encoder.proj"].shape[0]
                else:
                    # proj has shape [hidden_dim, embeddings_dim]
                    proj = state_dict["ms_encoder.proj"]
                    hidden_dim = proj.shape[0]
                    output_dim = proj.shape[1]

                is_graph = any("graph_encoder" in k for k in state_dict.keys())
                fp_pred = any("out_mlp" in k for k in state_dict.keys())
                trainable_temperature = any("inv_temperature" in k for k in state_dict.keys())

                # Infer number of transformer layers from the state_dict so the
                # reconstructed encoder matches the saved weights.
                layer_keys = [k for k in state_dict.keys()
                              if k.startswith("ms_encoder.transformer_encoder.layers.")]
                num_layers = 0
                for k in layer_keys:
                    try:
                        idx = int(k.split("ms_encoder.transformer_encoder.layers.")[1].split(".")[0])
                    except (ValueError, IndexError):
                        continue
                    if idx + 1 > num_layers:
                        num_layers = idx + 1

                # nhead is not encoded in the state_dict shapes; default to 8
                # (matches the historical hardcoded value and the train.py
                # default). For non-default configurations callers should
                # ensure the encoder was trained with a compatible nhead.
                nhead = 8

                self.ms_encoder_model = Contrastive_model(
                    hidden_dim=hidden_dim,
                    max_len=129,
                    num_transformer_layers=num_layers,
                    nhead=nhead,
                    embeddings_dim=output_dim,
                    dropout=0.1,
                    input_dropout=0.1,
                    fp_length=output_dim if self.embeddings_type == "ms2fp" else 2048,
                    graph=is_graph,
                    fp_pred=fp_pred,
                    initial_temperature=checkpoint.get("temperature", 15.0),
                    trainable_temperature=trainable_temperature,
                )
                self.ms_encoder_model.load_state_dict(state_dict, strict=False)

            else:
                output_dim = int(self.embeddings_dims)
                is_graph = getattr(cfg.conditioning, "ms_encoder_graph", False)
                fp_pred = self.embeddings_type == "ms2fp"

                initial_temperature = getattr(cfg.conditioning, "ms_encoder_initial_temperature", 30.0)
                trainable_temperature = getattr(cfg.conditioning, "ms_encoder_trainable_temperature", False)

                self.ms_encoder_model = Contrastive_model(
                    hidden_dim=512,
                    max_len=129,
                    num_transformer_layers=3,
                    nhead=8,
                    embeddings_dim=output_dim,
                    dropout=0.1,
                    input_dropout=0.1,
                    fp_length=output_dim if self.embeddings_type == "ms2fp" else 2048,
                    graph=is_graph,
                    fp_pred=fp_pred,
                    initial_temperature=initial_temperature,
                    trainable_temperature=trainable_temperature,
                )

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        node_types = self.dataset_info.node_types.float()
        x_marginals = node_types / torch.sum(node_types)

        edge_types = self.dataset_info.edge_types.float()
        e_marginals = edge_types / torch.sum(edge_types)
        print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
        self.transition_model = MarginalUniformEdgesTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                            y_classes=self.ydim_output)
        self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                            y=torch.ones(self.ydim_output) / self.ydim_output)

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics', 'ms_features', 'ms_dataframe', 'ms_graph_dict'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

    def on_load_checkpoint(self, checkpoint):

        if 'state_dict' not in checkpoint:
            return

        state_dict = checkpoint['state_dict']
        ms_encoder_keys_in_checkpoint = [
            k for k in state_dict.keys() if k.startswith('ms_encoder_model.')
        ]

        if 'optimizer_states' in checkpoint and len(checkpoint['optimizer_states']) > 0:
            opt_state = checkpoint['optimizer_states'][0]
            saved_groups = len(opt_state.get('param_groups', []))

            expected_groups = 1
            if self.finetune_ms_encoder and self.ms_encoder_model is not None:
                expected_groups = 2

            if saved_groups != expected_groups:
                checkpoint['optimizer_states'] = []
                checkpoint['lr_schedulers'] = []
                return

            if self.ms_encoder_model is not None and len(ms_encoder_keys_in_checkpoint) == 0:
                checkpoint['optimizer_states'] = []
                checkpoint['lr_schedulers'] = []
            elif self.ms_encoder_model is None and len(ms_encoder_keys_in_checkpoint) > 0:
                checkpoint['optimizer_states'] = []
                checkpoint['lr_schedulers'] = []
            else:
                saved_param_counts = [len(g.get('params', [])) for g in opt_state.get('param_groups', [])]
                encoder_params_count = sum(1 for name, _ in self.named_parameters() if name.startswith("ms_encoder_model.") and _.requires_grad)
                other_params_count = sum(1 for name, _ in self.named_parameters() if not name.startswith("ms_encoder_model.") and _.requires_grad)
                
                if expected_groups == 1:
                    expected_param_counts = [other_params_count + encoder_params_count]
                else:
                    expected_param_counts = [other_params_count, encoder_params_count]
                
                if len(saved_param_counts) != len(expected_param_counts) or any(sc != ec for sc, ec in zip(saved_param_counts, expected_param_counts)):
                    checkpoint['optimizer_states'] = []
                    checkpoint['lr_schedulers'] = []

    def load_state_dict(self, state_dict, strict=True):

        ms_encoder_keys_in_checkpoint = [k for k in state_dict.keys() if k.startswith('ms_encoder_model.')]
        
        if self.ms_encoder_model is not None:
            # Current model has ms_encoder_model
            if len(ms_encoder_keys_in_checkpoint) == 0:
                # Old checkpoint doesn't have ms_encoder_model - load everything else
                # The ms_encoder_model will keep its initialization from __init__ (loaded from encoder checkpoint)
                filtered_checkpoint = {k: v for k, v in state_dict.items() 
                                      if not k.startswith('ms_encoder_model.')}
                # Load without ms_encoder_model keys, use strict=False to allow missing keys
                return super().load_state_dict(filtered_checkpoint, strict=False)
            else:
                # Checkpoint has ms_encoder_model keys - normal loading
                return super().load_state_dict(state_dict, strict=False)
        else:
            # Current model doesn't have ms_encoder_model (finetune_ms_encoder=False)
            if len(ms_encoder_keys_in_checkpoint) > 0:
                # Checkpoint has ms_encoder keys but current model doesn't - remove them
                filtered_checkpoint = {k: v for k, v in state_dict.items() 
                                      if not k.startswith('ms_encoder_model.')}
                return super().load_state_dict(filtered_checkpoint, strict=False)
            else:
                # Normal loading
                return super().load_state_dict(state_dict, strict=False)

    def _process_ms_row(self, row_idx):

        row = self.ms_dataframe.iloc[row_idx]
        precursor_type = one_hot_encode_precursor(row['precursor_type'])
        collision_energy_nce = one_hot_encode_energy(int(row['collision_energy_NCE']))
        spectrum = row['clean_spectrum_formula_array']
        
        if 'spectral_information_score' in row:
            information_score = torch.tensor([row['spectral_information_score']], dtype=torch.float32)
        else:
            information_score = torch.tensor([1.0], dtype=torch.float32)
        
        sos = torch.cat([precursor_type, collision_energy_nce], dim=0).view(1, -1)
        assert sos.shape == (1, 13), f"SOS shape should be (1, 13), got {sos.shape}"
        

        spectrum = np.array(spectrum)
        
        total_dim = len(elements) * 16
        results = []
        for arr in spectrum:
            tensor = torch.zeros(total_dim)
            start_idx = 0
            for idx in range(len(elements)):
                value = arr[idx]
                tensor[start_idx:start_idx + 16] += self.ms_positional_encoding.encode(value)
                start_idx += 16
            results.append(tensor)
        array = torch.stack(results)
        
        n = array.shape[0]
        padded_tensor = self.ms_padded_tensor_template.clone()
        padded_tensor[:n] = array
        
        mask = torch.ones(self.ms_max_peaks + 1, dtype=torch.bool)
        mask[:n + 1] = 0
        
        return sos, padded_tensor, mask, information_score
    
    def _get_ms_features_for_smiles(self, smiles_list):

        sos_list = []
        formula_array_list = []
        mask_list = []
        num_spectra_list = []
        
        for smi in smiles_list:
            if smi not in self.ms_graph_dict:
                raise KeyError(f"SMILES {smi} not found in ms_graph_dict.")
            
            indices = self.ms_graph_dict[smi]

            for row_idx in indices:
                sos, formula_array, mask, _ = self._process_ms_row(row_idx)
                sos_list.append(sos)
                formula_array_list.append(formula_array)
                mask_list.append(mask)
            
            num_spectra_list.append(len(indices))
        

        sos_batch = torch.stack(sos_list, dim=0)  


        formula_array_batch = torch.stack(formula_array_list, dim=0)  
        mask_batch = torch.stack(mask_list, dim=0)  

        print (f"formula_array_batch shape: {formula_array_batch.shape}")
        print (f"mask_batch shape: {mask_batch.shape}")
        
        return sos_batch, formula_array_batch, mask_batch, num_spectra_list

    def _get_embeddings(self, data):

        if self.finetune_ms_encoder and self.ms_encoder_model is not None:
            
            if not hasattr(data, 'smiles') or data.smiles is None:
                raise ValueError("finetune_ms_encoder=True but data.smiles not found in batch.")
            
            smiles_list = data.smiles if isinstance(data.smiles, list) else [data.smiles]
            sos_batch, formula_array_batch, mask_batch, num_spectra_list = \
                self._get_ms_features_for_smiles(smiles_list)
            
            # Move to device
            device = self.device
            sos_batch = sos_batch.to(device)
            formula_array_batch = formula_array_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            embeddings = []
            start_idx = 0
            
            for num_spec in num_spectra_list:
                end_idx = start_idx + num_spec
                
                sos = sos_batch[start_idx:end_idx]  # Should be (N_spec, 1, 13)
                formula_array = formula_array_batch[start_idx:end_idx]  # (N_spec, max_peaks, 144)
                mask = mask_batch[start_idx:end_idx]  # (N_spec, max_peaks+1)
               
                ms_emb = self.ms_encoder_model.ms_encoder(sos, formula_array, mask=mask).float()  # (N_spec, D)
                ms_emb = ms_emb / (ms_emb.norm(dim=1, keepdim=True) + 1e-8)
                emb = torch.mean(ms_emb, dim=0, keepdim=True)  # (1, D)
                emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
                
                embeddings.append(emb)
                start_idx = end_idx
            
            return torch.cat(embeddings, dim=0)  # (batch_size, emb_dim)
        
        # Fallback: use precomputed embeddings
        if not hasattr(data, 'embedding') or data.embedding is None:
            raise ValueError(
                "finetune_ms_encoder=False but data.embedding is None. "
                "Either enable finetune_ms_encoder=True or provide precomputed embeddings in the dataset."
            )
        return data.embedding

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return

        opt = self.optimizers()
        current_lr = opt.param_groups[0]['lr']

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, data.atom_attr)
        dense_data = dense_data.mask(node_mask)
        E = dense_data.E
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        embeddings = self._get_embeddings(data)
        extra_data = self.compute_extra_data(noisy_data, embeddings)
        atom_attr = None

        pred = self.forward(noisy_data, extra_data, node_mask, atom_attr)

        loss = self.train_loss(masked_pred_E=pred.E,
                               true_E=E,
                               log=i % self.log_every_steps == 0)

        self.train_metrics(masked_pred_E=pred.E, true_E=E,
                           log=i % self.log_every_steps == 0)

        self.pbar.update()

        return {'loss': loss}

    def configure_optimizers(self):

        base_lr = self.cfg.train.lr
        ms_encoder_lr = getattr(self.cfg.train, "ms_encoder_lr", None) or base_lr
        weight_decay = self.cfg.train.weight_decay

        # If there is no encoder attached, fall back to a single parameter group.
        if not self.finetune_ms_encoder or self.ms_encoder_model is None:
            return torch.optim.AdamW(
                self.parameters(),
                lr=base_lr,
                amsgrad=True,
                weight_decay=weight_decay,
            )

        # Split parameters into diffusion vs encoder groups.
        encoder_params = []
        other_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("ms_encoder_model."):
                encoder_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {"params": other_params, "lr": base_lr},
            {"params": encoder_params, "lr": ms_encoder_lr},
        ]

        return torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            amsgrad=True,
            weight_decay=weight_decay,
        )

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Size of the input features", self.Xdim, self.Edim, self.ydim)
        if self.cfg.general.wandb:
            utils.setup_wandb(self.cfg)


    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()
        self.pbar = tqdm(total=int(self.len_data/self.cfg.train.batch_size) + 1, desc="epoch progress")

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}:"
                      f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
                      f" -- {time.time() - self.start_epoch_time:.1f}s ")
        epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: {epoch_bond_metrics}")
        print(torch.cuda.memory_summary())

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.sampling_metrics.reset()
        self.val_smiles = []
        self.val_data = []

    def validation_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, data.atom_attr)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        embeddings = self._get_embeddings(data)
        extra_data = self.compute_extra_data(noisy_data, embeddings)

        pred = self.forward(noisy_data, extra_data, node_mask)
        pred.X = dense_data.X
        pred.y = data.y

        nll = self.compute_val_loss(
            pred,
            noisy_data,
            dense_data.X,
            dense_data.E,
            data.y,
            node_mask,
            test=False,
            atom_attr=None,
            smiles=data.smiles,
            embeddings=embeddings,
        )
        
        self.val_data.append([dense_data, embeddings, data.smiles])
        self.val_smiles.extend(data.smiles)
            
        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(), self.val_X_kl.compute() * self.T, self.val_E_kl.compute() * self.T,
                   self.val_X_logp.compute(), self.val_E_logp.compute()]
        if wandb.run:
            wandb.log({"val/epoch_NLL": metrics[0],
                       "val/X_kl": metrics[1],
                       "val/E_kl": metrics[2],
                       "val/X_logp": metrics[3],
                       "val/E_logp": metrics[4]}, commit=False)

        self.print(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Atom type KL {metrics[1] :.2f} -- ",
                   f"Val Edge type KL: {metrics[2] :.2f}")

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)
        val_E_logp = metrics[4]
        self.log("val/E_logp", val_E_logp, sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))

        if (self.val_counter % self.cfg.general.sample_every_val == 0 and self.val_counter != 0):

            start = time.time()
            # Handle "all" option for samples_to_generate
            if self.cfg.general.samples_to_generate == "all":
                total_samples = sum(len(batch_data[2]) for batch_data in self.val_data)
                samples_left_to_generate = total_samples
                self.print(f"Generating samples for all {total_samples} validation samples")
            else:
                samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save
            bs = self.cfg.train.batch_size

            molecules = []
            targets = []

            ident = 0
            while samples_left_to_generate > 0:

                if (ident + 1) > len(self.val_data):
                    self.print("sampled the entire validation data")
                    break
                
                to_generate = min(samples_left_to_generate, bs, len(self.val_data[ident][2]))
                to_save = min(samples_left_to_save, bs, len(self.val_data[ident][2]))
                chains_save = min(chains_left_to_save, bs, len(self.val_data[ident][2]))

                molecule_list, target_smiles = self.sample_batch(batch_id=ident, 
                                                    batch_size=to_generate, 
                                                    num_nodes=None,
                                                    save_final=to_save,
                                                    keep_chain=chains_save,
                                                    number_chain_steps=self.number_chain_steps,
                                                    data = self.val_data) 

                molecules.extend(molecule_list)
                targets.extend(target_smiles)

                ident += 1

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save

            self.print("Computing sampling metrics...")
            self.sampling_metrics.forward(molecules, self.name, self.current_epoch, val_counter=-1, test=False,
                                          local_rank=self.local_rank, targets = targets)
            self.print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
            print("Validation epoch end ends...")

        self.val_counter += 1

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        self.test_data=[]
        self.test_smiles=[]


    def test_step(self, data, i):

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, data.atom_attr)

        batch_size = dense_data.X.shape[0]  
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        embeddings = self._get_embeddings(data)
        extra_data = self.compute_extra_data(noisy_data, embeddings)

        pred = self.forward(noisy_data, extra_data, node_mask)
        pred.X = dense_data.X
        pred.y = data.y

        nll = self.compute_val_loss(
            pred,
            noisy_data,
            dense_data.X,
            dense_data.E,
            data.y,
            node_mask,
            test=True,
            atom_attr=None,
            smiles=data.smiles,
            embeddings=embeddings,
        )

        self.test_data.append([dense_data, embeddings, data.smiles])
        self.test_smiles.extend(data.smiles)
            
        return {'loss': nll}

    @torch.enable_grad()
    @torch.inference_mode(False)

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(), self.test_X_kl.compute(), self.test_E_kl.compute(),
                   self.test_X_logp.compute(), self.test_E_logp.compute()]

        self.print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- ",
                   f"Test Edge type KL: {metrics[2] :.2f}")

        test_nll = metrics[0]

        self.print(f'Test loss: {test_nll :.4f}')

        smiles_summary = []
        identity_summary = []
        tanimoto_summary = []
        MCES_summary = []

        # Get number of test iterations from config, default to 25
        num_test_iterations = getattr(self.cfg.general, 'test_iterations', 25)
        self.print(f"Running {num_test_iterations} test iterations...")

        for itr in range(num_test_iterations):

            # Handle "all" option for samples_to_generate
            if self.cfg.general.samples_to_generate == "all":
                total_samples = sum(len(batch_data[2]) for batch_data in self.test_data)
                samples_left_to_generate = total_samples
                if itr == 0:
                    self.print(f"Generating samples for all {total_samples} test samples")
            else:
                samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save
            bs = self.cfg.train.batch_size

            molecules = []
            targets = []

            ident = 0
            while samples_left_to_generate > 0:

                if (ident + 1) > len(self.test_data):
                    self.print("sampled the entire test data")
                    break
                
                to_generate = min(samples_left_to_generate, bs, len(self.test_data[ident][2]))
                to_save = min(samples_left_to_save, bs, len(self.test_data[ident][2]))
                chains_save = min(chains_left_to_save, bs, len(self.test_data[ident][2]))

                molecule_list, target_smiles = self.sample_batch(batch_id=ident, 
                                                    batch_size=to_generate, 
                                                    num_nodes=None,
                                                    save_final=to_save,
                                                    keep_chain=chains_save,
                                                    number_chain_steps=self.number_chain_steps,
                                                    data = self.test_data) 

                molecules.extend(molecule_list)
                targets.extend(target_smiles)

                ident += 1

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save

            self.print("Saving the generated graphs")

            filename = f'generated_samples{itr+1}.txt'
            with open(filename, 'w') as f:
                for item in molecules:
                    f.write(f"N={item[0].shape[0]}\n")
                    atoms = item[0].tolist()
                    f.write("X: \n")
                    for at in atoms:
                        f.write(f"{at} ")
                    f.write("\n")
                    f.write("E: \n")
                    for bond_list in item[1]:
                        for bond in bond_list:
                            f.write(f"{bond} ")
                        f.write("\n")
                    f.write("\n")

            self.print("Generated graphs Saved. Computing sampling metrics...")
            all_smiles, identical_list, tanimoto_list, MCES_list = self.sampling_metrics(molecules, self.name, self.current_epoch, self.val_counter, test=True, local_rank=self.local_rank, targets = targets)

            filename = f'final_smiles{itr+1}.txt'
            with open(filename, 'w') as fp:
                for idx in range(len(all_smiles)):
                    # write each item on a new line
                    fp.write(f"{targets[idx]} {all_smiles[idx]} {identical_list[idx]} {tanimoto_list[idx]} {MCES_list[idx]}\n")# % smiles)

            print (tanimoto_list)
            print (MCES_list)

            smiles_summary.append(all_smiles)
            identity_summary.append(identical_list)
            tanimoto_summary.append(tanimoto_list)
            MCES_summary.append(MCES_list)

        self.print("Done testing.")

        top1_matches, any_matches, top10_tanimoto_vals, top10_MCES_vals = evaluate_smiles(smiles_summary, target_smiles, identity_summary, tanimoto_summary, MCES_summary)
        print(f"top1 - {sum(top1_matches) / len(top1_matches)}")
        print(f"top10 - {sum(any_matches) / len(any_matches)}")
        print(f"top_tanimoto - {sum(top10_tanimoto_vals) / len(top10_tanimoto_vals)}")
        # Guard against zero-length lists: an undertrained e2e-smoke-test
        # model can produce zero valid SMILES per test spectrum, which would
        # otherwise trigger a ZeroDivisionError on the MCES mean.
        valid_mces = [i for i in top10_MCES_vals if i < 1000]
        if valid_mces:
            print(f"top_MCES - {sum(valid_mces) / len(valid_mces)}")
        else:
            print("top_MCES - N/A (no valid molecules)")
        print (top1_matches)
        print (any_matches)
        print (top10_tanimoto_vals)
        print (top10_MCES_vals)

    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')

        return diffusion_utils.sum_except_batch(kl_distance_X) + \
               diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, node_mask, atom_attr=None, smiles = None, embeddings = None):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.transition_model.X_classes).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.transition_model.E_classes).float()
        y0 = sampled0.y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {
            'X_t': sampled_0.X,
            'E_t': sampled_0.E,
            'y_t': sampled_0.y,
            'node_mask': node_mask,
            't': torch.zeros(X0.shape[0], 1).type_as(y0),
        }
        extra_data = self.compute_extra_data(noisy_data, embeddings)
        pred0 = self.forward(noisy_data, extra_data, node_mask, atom_attr)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, node_mask, frags = None):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.transition_model.X_classes)
        E_t = F.one_hot(sampled_t.E, num_classes=self.transition_model.E_classes)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False, atom_attr = None, smiles = None, embeddings = None):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, node_mask, atom_attr, smiles, embeddings)

        loss_term_0 = self.val_X_logp(X * prob0.X[:,:,:11].log()) + self.val_E_logp(E * prob0.E.log())

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch

        if wandb.run:
            wandb.log({"kl prior": kl_prior.mean(),
                       "Estimator loss terms": loss_all_t.mean(),
                       "log_pn": log_pN.mean(),
                       "loss_term_0": loss_term_0,
                       'batch_test_nll' if test else 'val_nll': nll}, commit=False)
        return nll

    def forward(self, noisy_data, extra_data, node_mask, atom_attr = None):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        if atom_attr != None:
            X = torch.cat((X, atom_attr), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)


    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None, data = None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if data == None:
            if num_nodes is None:
                n_nodes = self.node_dist.sample_n(batch_size, self.device)
            elif type(num_nodes) == int:
                n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
            else:
                assert isinstance(num_nodes, torch.Tensor)
                n_nodes = num_nodes

        else:
            X = data[batch_id][0].X[:batch_size]
            max_nodes = X.shape[1]
            n_nodes = X.any(dim=2).sum(dim=1)

        n_max = torch.max(n_nodes).item()

        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        E, y = z_T.E, z_T.y

        embeddings = data[batch_id][1][:batch_size]
        target_smiles = data[batch_id][2][:batch_size]

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        # Chain tensors are only needed when we actually keep intermediate steps
        if keep_chain > 0 and number_chain_steps > 0:
            chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
            chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

            chain_X = torch.zeros(chain_X_size)
            chain_E = torch.zeros(chain_E_size)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        with tqdm(total=self.T, desc="diffusion steps generation") as pbar:         

            for s_int in reversed(range(0, self.T)):
                s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
                t_array = s_array + 1
                s_norm = s_array / self.T
                t_norm = t_array / self.T   

                sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask, embeddings = embeddings) 
                
                X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

                # Save the first keep_chain graphs (only if we requested chains)
                if keep_chain > 0 and number_chain_steps > 0:
                    write_index = (s_int * number_chain_steps) // self.T
                    if 0 <= write_index < chain_X.size(0):
                        chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
                        chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

                pbar.update()

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Prepare the chain for saving
        if keep_chain > 0 and number_chain_steps > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []

        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        # Visualize chains
        if self.visualization_tools is not None and keep_chain > 0 and number_chain_steps > 0:
            self.print('Visualizing chains...')
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)       # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                         f'epoch{self.current_epoch}/'
                                                         f'chains/molecule_{batch_id + i}')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(result_path,
                                                                 chain_X[:, i, :].numpy(),
                                                                 chain_E[:, i, :].numpy())
                self.print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)
            self.print('\nVisualizing molecules...')

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list, target_smiles

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask, atom_attr = None, embeddings = None):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape

        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data, embeddings)

        pred = self.forward(noisy_data, extra_data, node_mask, atom_attr)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X[:,:,:11].unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.transition_model.X_classes).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.transition_model.E_classes).float()
        
        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_t, E=E_s, y=y_t)
        out_discrete = utils.PlaceHolder(X=X_t, E=E_s, y=y_t)

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

    def compute_extra_data(self, noisy_data, embeddings = None):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        # Add Gaussian noise to embeddings if enabled
        if embeddings is not None and hasattr(self.cfg.train, 'add_embedding_noise') and self.cfg.train.add_embedding_noise:
            noise_variance = getattr(self.cfg.train, 'embedding_noise_variance', 0.01)
            noise = torch.randn_like(embeddings) * (noise_variance ** 0.5)
            embeddings = embeddings + noise

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data, embeddings)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)


from collections import Counter

def evaluate_smiles(smiles_summary, target_smiles, identity_summary, tanimoto_summary, MCES_summary):
    transposed = list(zip(*smiles_summary))  # Transpose to get 256 lists of 10 values
    transposed_identity = list(zip(*identity_summary))
    transposed_tanimoto = list(zip(*tanimoto_summary))
    transposed_MCES = list(zip(*MCES_summary))

    top1_matches = []
    any_matches = []
    top10_tanimoto_vals = []
    top10_MCES_vals = []

    for i, smiles_list in enumerate(transposed):
        identity_list = transposed_identity[i]
        tanimoto_list = transposed_tanimoto[i]
        mces_list = transposed_MCES[i]
        
        clean_list = [s for s in smiles_list if s is not None]  # Remove None values
        most_common = Counter(clean_list).most_common(1)[0][0] if len (clean_list) > 0 else None
        most_common_idx = smiles_list.index(most_common)

        top1_match = identity_list[most_common_idx]  # Check Top-1
        any_match = max(identity_list)   # Check if any matches
        top10_tanimoto = max(tanimoto_list)
        top10_MCES = min(mces_list)

        top1_matches.append(top1_match)
        any_matches.append(any_match)
        top10_tanimoto_vals.append(top10_tanimoto)
        top10_MCES_vals.append(top10_MCES)

    return top1_matches, any_matches, top10_tanimoto_vals, top10_MCES_vals  # Lists of 256 boolean values


