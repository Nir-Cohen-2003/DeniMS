import os
import pathlib
base_path_ms2mol = str(pathlib.Path(os.path.realpath(__file__)).parents[3])

import warnings
warnings.filterwarnings(
    "ignore",
    message="to-Python converter for std::pair<double, double> already registered"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*weights_only=False.*"
)

import torch
import torch.nn.functional as F
from torch import nn
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT

from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.utils import subgraph

import utils
from datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos
from analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges
from analysis.rdkit_functions import compute_molecular_metrics

from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import AllChem

import sys
sys.path.append ("../..")

from model import Contrastive_model
from evaluation_utils import batch_graphs_to_padded_data
from dataloaders import load_data

class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data

from datasets.lmdb_utils import open_env, _dumps, _loads, write_meta, read_meta, LMDB_META_KEY

class MSDatasetLMDB(Dataset):
    """
    Like your MSDataset but reads/writes single samples in LMDB.
    Keys are zero-padded integers: b'%08d'
    """
    def __init__(self, stage, data, root, remove_h: bool,
                 embeddings_type=None, model_path=None, transform=None, pre_transform=None, pre_filter=None,
                 lmdb_subdir="lmdb"):

        super().__init__(root=root,
                         transform=transform,
                         pre_transform=pre_transform,
                         pre_filter=pre_filter)
                         
        self.stage = stage
        self.remove_h = remove_h
        self.embeddings_type = embeddings_type
        self.model_path = model_path
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        # split index like before
        self.file_idx = 0 if stage == "train" else (1 if stage == "val" else 2)
        
        # LMDB path: <root>/<split>.lmdb/
        split_name = ["train", "val", "test"][self.file_idx]
        self.lmdb_path = os.path.join(root, lmdb_subdir, f"{split_name}.lmdb")
        os.makedirs(os.path.dirname(self.lmdb_path), exist_ok=True)

        if not os.path.exists(self.lmdb_path):
            self.dataloader = data[self.file_idx]
            self.emb_dict = data[-1]
            self._build_lmdb()
        else:
            self.dataloader = None
            self.emb_dict = {}

        self.env = open_env(self.lmdb_path, readonly=True)
        with self.env.begin(write=False) as txn:
            meta = read_meta(txn)
        self._length = int(meta["length"])

    def len(self):
        return self._length

    def get(self, idx):
        with self.env.begin(write=False) as txn:
            key = f"{idx:08d}".encode()
            raw = txn.get(key)
            if raw is None:
                raise IndexError(f"sample {idx} not found in LMDB")
            data = _loads(raw)
        if self.transform:
            data = self.transform(data)
        return data

    # -------------------- building --------------------

    def _build_lmdb(self):

        from evaluation_utils import batch_graphs_to_padded_data
        # everything below mirrors your current .process() logic,
        # except we stream into LMDB per-sample instead of collating + torch.save
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")

        if self.embeddings_type in ["ms2emb", "mol2emb", "ms2fp"]:

            if self.model_path != None:
                checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=False)

                if self.embeddings_type == "ms2fp":
                    output_dim = checkpoint['model']['out_mlp.2.bias'].shape[0]

                else:
                    # proj has shape [hidden_dim, embeddings_dim]; use both dims
                    proj = checkpoint['model']['ms_encoder.proj']
                    hidden_dim = proj.shape[0]
                    output_dim = proj.shape[1]

                is_graph = any("graph_encoder" in submodule for submodule in checkpoint['model'].keys())
                fp_pred = any("out_mlp" in submodule for submodule in checkpoint['model'].keys())
                trainable_temperature = any("inv_temperature" in submodule for submodule in checkpoint['model'].keys())

                # Infer number of transformer layers from the state_dict so the
                # reconstructed encoder matches the saved weights.
                layer_keys = [k for k in checkpoint['model'].keys()
                              if k.startswith('ms_encoder.transformer_encoder.layers.')]
                num_layers = 0
                for k in layer_keys:
                    try:
                        idx = int(k.split('ms_encoder.transformer_encoder.layers.')[1].split('.')[0])
                    except (ValueError, IndexError):
                        continue
                    if idx + 1 > num_layers:
                        num_layers = idx + 1

                # Default nhead to 8 (matches historical hardcoded value and the
                # default in train.py). nhead is not encoded in the state_dict
                # shapes, so we rely on a sensible default that divides
                # hidden_dim for typical configurations.
                nhead = 8

                self.model = Contrastive_model(
                    hidden_dim=hidden_dim,
                    num_transformer_layers=num_layers,
                    embeddings_dim=output_dim,
                    graph=is_graph,
                    fp_pred=fp_pred,
                    trainable_temperature=trainable_temperature,
                    nhead=nhead,
                ).to("cuda")
                self.model.load_state_dict(checkpoint['model'])
                self.model.eval()

            else:
                raise Exception("No embedding model path were given")

        if self.embeddings_type in ["ms2fp"]:
            sigmoid = nn.Sigmoid()

        env = open_env(self.lmdb_path, readonly=False)
        count = 0
        with env.begin(write=True) as txn:
            batch_it = iter(self.dataloader)
            batches_num = int(len(batch_it))
            for i in tqdm(range(batches_num), desc=f"building {self.stage} LMDB"):

                if (i%100) == 0: 
                    print (f"Processing batch {i} / {batches_num}")
                    
                sos, formula_array, mask, smiles, entropies = next(batch_it)

                mol = Chem.MolFromSmiles(smiles[0])
                N = mol.GetNumAtoms()

                types = {'B':0,'C':1,'N':2,'O':3,'F':4,'Si':5,'P':6,'S':7,'Cl':8,'Br':9,'I':10,'H':11}
                bonds = {Chem.rdchem.BondType.SINGLE:0, Chem.rdchem.BondType.DOUBLE:1,
                         Chem.rdchem.BondType.TRIPLE:2, Chem.rdchem.BondType.AROMATIC:3}

                type_idx = [types[a.GetSymbol()] for a in mol.GetAtoms()]

                charges_list, Hs_list = [], []
                for a in mol.GetAtoms():
                    ch = a.GetFormalCharge()
                    if ch < -4 or ch > 4:
                        # skip invalid, continue to next sample
                        continue
                    charges_list.append(int(ch + 4))
                    h = a.GetNumExplicitHs()
                    if h > 4:
                        continue
                    Hs_list.append(h)

                row, col, edge_type = [], [], []
                for b in mol.GetBonds():
                    s, e = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                    row += [s, e]; col += [e, s]
                    edge_type += 2 * [bonds[b.GetBondType()] + 1]
                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_type = torch.tensor(edge_type, dtype=torch.long)
                edge_attr = F.one_hot(edge_type, num_classes=5).to(torch.float)

                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]

                x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
                charges = F.one_hot(torch.tensor(charges_list), num_classes=9).float()
                Hs = F.one_hot(torch.tensor(Hs_list), num_classes=5).float()
                atom_attr = torch.cat((charges, Hs), dim=1)
                y = torch.zeros((1, 0), dtype=torch.float)

                if self.remove_h:
                    type_idx_t = torch.tensor(type_idx).long()
                    to_keep = type_idx_t <= 11
                    edge_index, edge_attr = subgraph(to_keep, edge_index, edge_attr, relabel_nodes=True,
                                                     num_nodes=len(to_keep))
                    x = x[to_keep][:, :-1]
                    
                embedding = None
                if self.embeddings_type is not None:

                    with torch.no_grad():
                        if self.embeddings_type == "ms2emb":
                            ms_embeddings = self.model.ms_encoder(sos, formula_array, mask)
                            embedding = (ms_embeddings / ms_embeddings.norm(dim=1, keepdim=True)).to('cpu').detach()

                                                        
                            embedding_try = torch.mean(embedding, dim=0).unsqueeze(0)

                            print ("Embedding try shape:", embedding_try.shape)

                            eps = 1e-8
                            raw_weights = (entropies + eps)
                            weights = raw_weights / (raw_weights.sum())
                            embedding = torch.sum(embedding * weights, dim=0).float()
                            embedding = embedding.unsqueeze(0)

                            print ("Embedding shape:", embedding.shape)
                            print ("****")

                            # embedding = torch.mean(embedding, dim=0).unsqueeze(0)
                            
                            embedding = embedding / embedding.norm(dim=1, keepdim=True)

                            # embedding = (ms_embeddings / ms_embeddings.norm(dim=1, keepdim=True)).to('cpu').detach()
                            # embedding = torch.mean(embedding, dim=0).unsqueeze(0)
                            # embedding = embedding / embedding.norm(dim=1, keepdim=True)

                        elif self.embeddings_type == "ms2fp":
                            ms_embeddings = self.model.ms_encoder(sos, formula_array, mask)
                            preds = torch.mean(self.model.out_mlp(ms_embeddings), dim=0)
                            embedding = torch.sigmoid(preds).cpu().detach().unsqueeze(0)

                        elif self.embeddings_type == "mol2emb":
                            try:
                                graph = batch_graphs_to_padded_data([self.emb_dict[smiles[0]][0]])
                                ge = self.model.graph_encoder(graph.X, graph.E, graph.y, graph.node_mask)
                                embedding = (ge / ge.norm(dim=1, keepdim=True)).cpu().detach()

                            except:
                                print(f"Failed to get mol2emb for {smiles[0]}, skipping...")
                                continue    

                        elif self.embeddings_type == "mol2fp":
                            from rdkit.Chem import AllChem
                            import numpy as np
                            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                            embedding = torch.tensor(np.array(fp, dtype=np.float32)).unsqueeze(0)
                        else:
                            raise ValueError(f"Unknown embeddings_type {self.embeddings_type}")

                data = Data(
                    x=x, atom_attr=atom_attr,
                    edge_index=edge_index, edge_attr=edge_attr,
                    y=y, idx=count, smiles=Chem.MolToSmiles(mol)
                )
                if embedding is not None:
                    data.embedding = embedding

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                # write single sample
                key = f"{count:08d}".encode()
                txn.put(key, _dumps(data))
                count += 1

            # meta
            write_meta(txn, {
                "length": count,
                "remove_h": bool(self.remove_h),
                "embeddings_type": self.embeddings_type,
                "stage": self.stage,
            })
        env.sync()
        env.close()

class MSDataModule(MolecularDataModule):
    def __init__(self, cfg):
        
        root_path = os.path.join(base_path_ms2mol, "MS_diffusion/data", cfg.dataset.name)

        if cfg.conditioning.load_subdata_dir != None:
            root_path = os.path.join(root_path, cfg.conditioning.load_subdata_dir)
            os.makedirs(root_path, exist_ok=True)
       
        self.remove_h = cfg.dataset.remove_h

        if cfg.conditioning.splitting_path == None:

            dataloader = load_data(os.path.join(base_path_ms2mol, cfg.conditioning.ms_data_path),
                                os.path.join(base_path_ms2mol, cfg.conditioning.graph_dict_path), 
                                batch_size = 1, 
                                shuffle_train = False,
                                batch = True)
        else:
            print("Using predefined splitting")
            dataloader = load_data(os.path.join(base_path_ms2mol, cfg.conditioning.ms_data_path),
                    os.path.join(base_path_ms2mol, cfg.conditioning.graph_dict_path), 
                    "predefined",
                    os.path.join(base_path_ms2mol, cfg.conditioning.splitting_path),
                    batch_size = 1, 
                    shuffle_train = False,
                    batch = True)

        if cfg.conditioning.embeddings_type in ["ms2emb", "mol2emb", "ms2fp"] and cfg.conditioning.embedding_model_path is not None:
            model_path = os.path.join(base_path_ms2mol, cfg.conditioning.embedding_model_path)
        else:
            model_path = None

        datasets = {'train': MSDataset(stage='train', data = dataloader, root=root_path, remove_h=cfg.dataset.remove_h,
                                       embeddings_type = cfg.conditioning.embeddings_type, model_path = model_path, transform=RemoveYTransform()),
                      'val': MSDataset(stage='val', data = dataloader, root=root_path, remove_h=cfg.dataset.remove_h,
                                       embeddings_type = cfg.conditioning.embeddings_type, model_path = model_path, transform=RemoveYTransform()),
                     'test': MSDataset(stage='test', data = dataloader, root=root_path, remove_h=cfg.dataset.remove_h,
                                       embeddings_type = cfg.conditioning.embeddings_type, model_path = model_path, transform=RemoveYTransform())}
        super().__init__(cfg, datasets)

class MSDataModule_lmdb(MolecularDataModule):
    def __init__(self, cfg):
        import os, pathlib
        base_path_ms2mol = str(pathlib.Path(os.path.realpath(__file__)).parents[3])

        root_path = os.path.join(base_path_ms2mol, "MS_diffusion/data", cfg.dataset.name)
        if cfg.conditioning.load_subdata_dir is not None:
            root_path = os.path.join(root_path, cfg.conditioning.load_subdata_dir)
            os.makedirs(root_path, exist_ok=True)

        # Check if all LMDB files already exist
        lmdb_dir = os.path.join(root_path, "lmdb")
        train_lmdb = os.path.join(lmdb_dir, "train.lmdb")
        val_lmdb = os.path.join(lmdb_dir, "val.lmdb")
        test_lmdb = os.path.join(lmdb_dir, "test.lmdb")
        
        all_lmdb_exist = (os.path.exists(train_lmdb) and 
                         os.path.exists(val_lmdb) and 
                         os.path.exists(test_lmdb))
        
        if all_lmdb_exist:
            print("All LMDB files already exist. Skipping raw data loading.")
            dataloader = [[], [], [], {}]  # [train, val, test, emb_dict]
        else:
            if cfg.conditioning.splitting_path == None:
                dataloader = load_data(os.path.join(base_path_ms2mol, cfg.conditioning.ms_data_path),
                                    os.path.join(base_path_ms2mol, cfg.conditioning.graph_dict_path),
                                    "random",
                                    None, 
                                    batch_size = 1, 
                                    shuffle_train = False,
                                    batch = True)
            else:
                print("Using predefined splitting")
                dataloader = load_data(os.path.join(base_path_ms2mol, cfg.conditioning.ms_data_path),
                        os.path.join(base_path_ms2mol, cfg.conditioning.graph_dict_path), 
                        "predefined",
                        os.path.join(base_path_ms2mol, cfg.conditioning.splitting_path),
                        batch_size = 1, 
                        shuffle_train = False,
                        batch = True)


        if cfg.conditioning.embeddings_type in ["ms2emb", "mol2emb", "ms2fp"] and cfg.conditioning.embedding_model_path is not None:
            model_path = os.path.join(base_path_ms2mol, cfg.conditioning.embedding_model_path)
        else:
            model_path = None

        datasets = {
            "train": MSDatasetLMDB("train", dataloader, root=root_path, remove_h=cfg.dataset.remove_h,
                                   embeddings_type=cfg.conditioning.embeddings_type, model_path=model_path,
                                   transform=RemoveYTransform()),
            "val":   MSDatasetLMDB("val",   dataloader, root=root_path, remove_h=cfg.dataset.remove_h,
                                   embeddings_type=cfg.conditioning.embeddings_type, model_path=model_path,
                                   transform=RemoveYTransform()),
            "test":  MSDatasetLMDB("test",  dataloader, root=root_path, remove_h=cfg.dataset.remove_h,
                                   embeddings_type=cfg.conditioning.embeddings_type, model_path=model_path,
                                   transform=RemoveYTransform())
        }
        super().__init__(cfg, datasets)


class MSinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.remove_h = cfg.dataset.remove_h
        self.need_to_strip = False  # to indicate whether we need to ignore one output from the model
        self.name = 'MS'

        if self.remove_h:
            self.atom_encoder = {'B': 0, 'C': 1, 'N': 2, 'O': 3, 'F':4, 'Si':5, 'P':6, 'S':7, 'Cl':8, 'Br':9, 'I':10}
            self.atom_decoder = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']
            self.num_atom_types = 11
            self.valencies = [3, 4, 3, 2, 1, 2, 3, 2, 1, 1, 1]
            self.atom_weights = {0: 11, 1: 12, 2: 14, 3: 16, 4:19, 5:28, 6:31, 7:32, 8:35.5, 9:80, 10:127}
            self.max_n_nodes = 30
            self.max_weight = 1000

        else:
            self.atom_encoder = {'B': 0, 'C': 1, 'N': 2, 'O': 3, 'F':4, 'Si':5, 'P':6, 'S':7, 'Cl':8, 'Br':9, 'I':10, 'H':11}
            self.num_atom_types = 12
            self.valencies = [3, 4, 3, 2, 1, 2, 3, 2, 1, 1, 1, 1]
            self.atom_weights = {0: 11, 1: 12, 2: 14, 3: 16, 4:19, 5:28, 6:31, 7:32, 8:35.5, 9:80, 10:127, 11:1}
            self.max_n_nodes = 150

        self.n_nodes = datamodule.node_counts() + 1e-6
        self.n_nodes = self.n_nodes/sum(self.n_nodes)

        self.node_types = datamodule.node_types() 
        self.edge_types = datamodule.edge_counts() 

        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
        self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
        self.valency_distribution[0: 6] = torch.tensor([2.6071e-06, 0.163, 0.352, 0.320, 0.16313, 0.00073])


def get_train_smiles(cfg, train_dataloader, dataset_infos, evaluate_dataset=False, source = False):

    if evaluate_dataset:
        assert dataset_infos is not None, "If wanting to evaluate dataset, need to pass dataset_infos"

    remove_h = cfg.dataset.remove_h
    atom_decoder = dataset_infos.atom_decoder
    root_path = os.path.join(base_path_ms2mol, "MS_diffusion/data", cfg.dataset.name)

    if cfg.conditioning.load_subdata_dir != None:
        root_path = os.path.join(root_path, cfg.conditioning.load_subdata_dir)

    smiles_file_name = 'train_smiles_no_h.npy' if remove_h else 'train_smiles_h.npy'
    smiles_path = os.path.join(root_path, smiles_file_name)
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.load(smiles_path)
    else:
        print("Computing dataset smiles...")
        train_smiles = compute_MS_smiles(atom_decoder, train_dataloader, remove_h, source)
        np.save(smiles_path, np.array(train_smiles))

    if evaluate_dataset:
        train_dataloader = train_dataloader
        all_molecules = []
        for i, data in enumerate(train_dataloader):
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])

        print("Evaluating the dataset -- number of molecules to evaluate", len(all_molecules))
        metrics = compute_molecular_metrics(molecule_list=all_molecules, train_smiles=train_smiles,
                                            dataset_info=dataset_infos)
        print(metrics[0])

    return train_smiles


def compute_MS_smiles(atom_decoder, train_dataloader, remove_h, source):
    '''

    :param dataset_name: MS or MS_second_half
    :return:
    '''
    print(f"\tConverting MS dataset to SMILES for remove_h={remove_h}...")

    mols_smiles = []
    len_train = len(train_dataloader)
    invalid = 0
    disconnected = 0
    for i, data in tqdm(enumerate(train_dataloader), total = len(train_dataloader)):

        if source:
            mols_smiles.extend(data.smiles)
            continue

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, data.atom_attr)
        dense_data = dense_data.mask(node_mask, collapse=True)
        X, E = dense_data.X, dense_data.E 
        charges = dense_data.charges if hasattr(dense_data, 'charges') else None
        Hs = dense_data.Hs if hasattr(dense_data, 'Hs') else None

        n_nodes = [int(torch.sum((X != -1)[j, :])) for j in range(X.size(0))]

        molecule_list = []
        for k in range(X.size(0)):
            n = n_nodes[k]
            atom_types = X[k, :n].cpu()
            edge_types = E[k, :n, :n].cpu()
            if charges != None:
                charges_types = charges[k, :n].cpu() - 4
                Hs_types = Hs[k, :n].cpu()
                molecule_list.append([atom_types, edge_types, charges_types, Hs_types])
    
            else:
                molecule_list.append([atom_types, edge_types, None, None])

        for l, molecule in enumerate(molecule_list):

            mol = build_molecule_with_partial_charges(molecule[0], molecule[1], atom_decoder, molecule[2], molecule[3])
            smile = mol2smiles(mol)
            if smile is not None:

                mols_smiles.append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                if len(mol_frags) > 1:
                    print("Disconnected molecule", mol, mol_frags)
                    disconnected += 1
            else:
                print("Invalid molecule obtained.")
                invalid += 1
    
    print("Number of invalid molecules", invalid)
    print("Number of disconnected molecules", disconnected)

    return mols_smiles
