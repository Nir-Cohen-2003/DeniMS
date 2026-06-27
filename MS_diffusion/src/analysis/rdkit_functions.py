import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError

import numpy as np
import torch
import re
import wandb

import networkx as nx

from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.DataStructs import TanimotoSimilarity
    from rdkit.Chem import AllChem
    print("Found rdkit, all good")
except ModuleNotFoundError as e:
    use_rdkit = False
    from warnings import warn
    warn("Didn't find rdkit, this will fail")
    assert use_rdkit, "Didn't find rdkit"

from myopic_mces import MCES
import pulp
solver = pulp.listSolvers(onlyAvailable=True)[0]

MCES_TIMEOUT_SEC = 120
MCES_TIMEOUT_FALLBACK = 1000

_mces_executor = None


def _mces_compute_pair(smiles1, smiles2):
    import pulp
    from myopic_mces import MCES
    mces_solver = pulp.listSolvers(onlyAvailable=True)[0]
    return float(MCES(
        smiles1, smiles2,
        solver=mces_solver,
        threshold=100,
        always_stronger_bound=False,
        solver_options=dict(msg=0),
    )[1])


def _reset_mces_executor():
    global _mces_executor
    if _mces_executor is not None:
        _mces_executor.shutdown(wait=False, cancel_futures=True)
        _mces_executor = None


def compute_mces_with_timeout(smiles1, smiles2, timeout_sec=MCES_TIMEOUT_SEC):
    if timeout_sec is None:
        try:
            return _mces_compute_pair(smiles1, smiles2)
        except Exception:
            return MCES_TIMEOUT_FALLBACK

    global _mces_executor
    if _mces_executor is None:
        _mces_executor = ProcessPoolExecutor(
            max_workers=1, mp_context=mp.get_context("spawn"))
    try:
        future = _mces_executor.submit(_mces_compute_pair, smiles1, smiles2)
        return future.result(timeout=timeout_sec)
    except FuturesTimeoutError:
        _reset_mces_executor()
        _mces_executor = ProcessPoolExecutor(
            max_workers=1, mp_context=mp.get_context("spawn"))
        return MCES_TIMEOUT_FALLBACK
    except Exception:
        return MCES_TIMEOUT_FALLBACK


from utils import *


allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'B': 3, 'Al': 3, 'Si': 4, 'P': [3, 5],
                 'S': 4, 'Cl': 1, 'As': 3, 'Br': 1, 'I': 1, 'Hg': [1, 2], 'Bi': [3, 5], 'Se': [2, 4, 6]}
bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
ATOM_VALENCY_2 = {2: 3, 3: 2, 6:3, 7:2}

def fix_aromatic_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol:  # If already valid, return it as is
        return smiles
    
    # If invalid, attempt to fix aromatic atoms
    fixed_smiles = None
    for i in range(len(smiles)):
        if smiles[i].islower() and smiles[i].isalpha() and smiles[i] != 'c':  # Identifying aromatic atoms
            modified_smiles = smiles[:i] + f'[{smiles[i]}H]' + smiles[i+1:]
            mol = Chem.MolFromSmiles(modified_smiles)
            
            if mol != None:
                fixed_smiles = modified_smiles  # Return canonical form
                break
    
    return fixed_smiles if fixed_smiles else None


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, targets=None, compute_mces=True, mces_timeout_sec=MCES_TIMEOUT_SEC):
        self.atom_decoder = dataset_info.atom_decoder
        self.dataset_info = dataset_info
        self.compute_mces = compute_mces
        self.mces_timeout_sec = mces_timeout_sec
        self.dataset_smiles_list = targets

    def compute_validity(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        valid = []
        num_components = []
        all_smiles = []
        for graph in generated:
            atom_types, edge_types = graph
            # mol = build_molecule(atom_types, edge_types, self.dataset_info.atom_decoder)
            mol = build_molecule_with_partial_charges(atom_types, edge_types)
            smiles = Chem.MolToSmiles(mol)
            smiles = fix_aromatic_smiles(smiles)
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                num_components.append(len(mol_frags))
            except:
                pass
            if smiles is not None and "." not in smiles:
                valid.append(True)
                all_smiles.append(smiles)

            else:
                valid.append(False)
                all_smiles.append(None)

        return valid, len([i for i in valid if i]) / len(generated), np.array(num_components), all_smiles

    def calculate_smiles_similarity(self, smiles1, smiles2, generated):

        if len(smiles1) != len(smiles2):
            raise ValueError("Both SMILES lists must have the same length.")

        identical_list = []
        tanimoto_list = []
        MCES_list = []
        valid_pairs = 0

        for idx in tqdm(range(len(smiles2)), desc="Calculating SMILES similarity"):
            if smiles2[idx] is None:
                identical_list.append(0)
                tanimoto_list.append(0)
                MCES_list.append(1000)
                continue

            G_1 = smiles_to_graph(smiles1[idx])
            G_2 = [generated[idx][0], generated[idx][1]]

            nx_1 = convert_to_nx_graph(G_1[0], G_1[1])
            nx_2 = convert_to_nx_graph(G_2[0], G_2[1])

            if nx.is_isomorphic(nx_1, nx_2):
                identical_list.append(1)
                tanimoto_list.append(1)
                MCES_list.append(0)
                valid_pairs += 1
                continue

            else:
                identical_list.append(0)

            mol1 = Chem.MolFromSmiles(smiles1[idx])
            mol2 = Chem.MolFromSmiles(smiles2[idx])

            if mol1 is None or mol2 is None:
                tanimoto_list.append(0)
                MCES_list.append(1000)
                valid_pairs += 1
                continue

            # Calculate Tanimoto similarity
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
            tanimoto_list.append(TanimotoSimilarity(fp1, fp2))

            if self.compute_mces:
                MCES_list.append(compute_mces_with_timeout(
                    smiles1[idx], smiles2[idx], timeout_sec=self.mces_timeout_sec))
            else:
                MCES_list.append(1000)
            valid_pairs += 1

        # Calculate results
        identical_ratio = sum(identical_list) / valid_pairs if valid_pairs > 0 else 0
        average_tanimoto = sum(tanimoto_list) / valid_pairs if valid_pairs > 0 else 0
        valid_MCES = [mc for mc in MCES_list if mc < 1000]
        average_MCES = sum(valid_MCES) / len (valid_MCES) if len (valid_MCES) > 0 else 0

        return [identical_ratio, average_tanimoto, average_MCES, identical_list, tanimoto_list, MCES_list]

    def compute_uniqueness(self, valid):
        """ valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        if self.dataset_smiles_list is None:
            print("Dataset smiles is None, novelty computation skipped")
            return 1, 1
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def compute_relaxed_validity(self, generated):
        valid = []
        for graph in generated:
            atom_types, edge_types = graph
            mol = build_molecule_with_partial_charges(atom_types, edge_types, self.dataset_info.atom_decoder)
            smiles = mol2smiles(mol)
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    smiles = mol2smiles(largest_mol)
                    valid.append(smiles)
                except:
                    pass
        return valid, len(valid) / len(generated)

    def evaluate(self, generated):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        valid, validity, num_components, all_smiles = self.compute_validity(generated)

        identity, tanimoto, MCES, identical_list, tanimoto_list, MCES_list = self.calculate_smiles_similarity (self.dataset_smiles_list, all_smiles, generated)
        nc_mu = num_components.mean() if len(num_components) > 0 else 0
        nc_min = num_components.min() if len(num_components) > 0 else 0
        nc_max = num_components.max() if len(num_components) > 0 else 0
        print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        print(f"Identity over {len([i for i in valid if i])} valid molecules: {identity * 100 :.2f}%")
        print(f"Tanimoto over {len([i for i in valid if i])} valid molecules: {tanimoto * 100 :.2f}%")
        valid_samples = len([i for i in valid if i])
        close_match = len([i for i in tanimoto_list if i > 0.675]) / valid_samples if valid_samples > 0 else 0
        meaningful_match = len([i for i in tanimoto_list if i > 0.4]) / valid_samples if valid_samples > 0 else 0
        print(f"close match over {len([i for i in valid if i])} valid molecules: {close_match * 100 :.2f}%")
        print(f"meaningful match over {len([i for i in valid if i])} valid molecules: {meaningful_match * 100 :.2f}%")
        print(f"Tanimoto over {len([i for i in valid if i])} valid molecules: {tanimoto * 100 :.2f}%")
        print(f"MCES over {len([i for i in valid if i])} valid molecules: {MCES :.2f}")
        print(f"Number of connected components of {len(generated)} molecules: min:{nc_min:.2f} mean:{nc_mu:.2f} max:{nc_max:.2f}")

        relaxed_valid, relaxed_validity = self.compute_relaxed_validity(generated)
        print(f"Relaxed validity over {len(generated)} molecules: {relaxed_validity * 100 :.2f}%")
        if relaxed_validity > 0:
            unique, uniqueness = self.compute_uniqueness(relaxed_valid)
            print(f"Uniqueness over {len(relaxed_valid)} valid molecules: {uniqueness * 100 :.2f}%")

            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                print(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
            else:
                novelty = -1.0
        else:
            novelty = -1.0
            uniqueness = 0.0
            unique = []
        return ([validity, relaxed_validity, uniqueness, novelty, identity, tanimoto, MCES], unique,
                dict(nc_min=nc_min, nc_max=nc_max, nc_mu=nc_mu), identical_list, tanimoto_list, MCES_list, all_smiles)

def mol2smiles(mol):
    return Chem.MolToSmiles(mol)


def build_molecule(atom_types, edge_types, atom_decoder, verbose=False):
    if verbose:
        print("building new molecule")

    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])

    edge_types = torch.triu(edge_types)
    all_bonds = torch.nonzero(edge_types)
    for i, bond in enumerate(all_bonds):
        if bond[0].item() != bond[1].item():
            mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[edge_types[bond[0], bond[1]].item()])
            if verbose:
                print("bond added:", bond[0].item(), bond[1].item(), edge_types[bond[0], bond[1]].item(),
                      bond_dict[edge_types[bond[0], bond[1]].item()] )
    return mol

def convert_to_nx_graph(atoms, bonds):
    G = nx.Graph()
    num_nodes = len(atoms)
    
    for node in range(num_nodes):
        G.add_node(node, atom_type=int(atoms[node]))
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if bonds[i, j] > 0:  # If there is a bond
                G.add_edge(i, j, bond_type=int(bonds[i, j]))
    
    return G

def smiles_to_graph(smiles):
    types = {'B': 0, 'C': 1, 'N': 2, 'O': 3, 'F':4, 'Si':5, 'P':6, 'S':7, 'Cl':8, 'Br':9, 'I':10, 'H':11}
    bonds = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    num_atoms = mol.GetNumAtoms()
    atom_types = torch.tensor([types.get(mol.GetAtomWithIdx(i).GetSymbol(), -1) for i in range(num_atoms)], dtype=torch.int32)
    bond_matrix = torch.zeros((num_atoms, num_atoms), dtype=torch.int32)
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bonds.get(bond.GetBondType(), -1)
        bond_matrix[i, j] = bond_type
        bond_matrix[j, i] = bond_type
    
    return [atom_types, bond_matrix]

def fix_valence_issues(mol, verbose=False):
    while True:
        flag, atomid_valence = check_valency(mol)
        if verbose:
            print("Valence check:", flag, atomid_valence)
        
        if flag:
            break  # Valid molecule
        
        assert len(atomid_valence) == 2
        idx = atomid_valence[0]
        v = atomid_valence[1]
        an = mol.GetAtomWithIdx(idx).GetAtomicNum()
        
        if verbose:
            print("Fixing valence for atom:", idx, "Atomic num:", an, "Valence:", v)
        
        if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
            mol.GetAtomWithIdx(idx).SetFormalCharge(1)
        else:
            # Additional valence correction strategies can be added here
            break  # If no fix is found, exit to avoid infinite loops



def build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder =  ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I'],
                                        charges=None, Hs=None, verbose=False):
    if verbose:
        print("\nBuilding new molecule")
    
    mol = Chem.RWMol()
    for idx, atom in enumerate(atom_types):
        a = Chem.Atom(atom_decoder[atom.item()])
        if charges is not None:
            a.SetFormalCharge(charges[idx])
        if Hs is not None:
            a.SetNumExplicitHs(Hs[idx])
        mol.AddAtom(a)
        if verbose:
            print("Atom added:", atom.item(), atom_decoder[atom.item()])
    
    edge_types = torch.triu(edge_types)
    all_bonds = torch.nonzero(edge_types)
    
    for bond in all_bonds:
        if bond[0].item() != bond[1].item():
            mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[edge_types[bond[0], bond[1]].item()])
            if verbose:
                print("Bond added:", bond[0].item(), bond[1].item(), edge_types[bond[0], bond[1]].item())
    
    # If no charges are provided, attempt to iteratively fix valence issues
    if charges is None:
        fix_valence_issues(mol, verbose=verbose)
    
    
    return mol


# Functions from GDSS
def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def correct_mol(m):
    # xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = m

    #####
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            check_idx = 0
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                type = int(b.GetBondType())
                queue.append((b.GetIdx(), type, b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                if type == 12:
                    check_idx += 1
            queue.sort(key=lambda tup: tup[1], reverse=True)

            if queue[-1][1] == 12:
                return None, no_correct
            elif len(queue) > 0:
                start = queue[check_idx][2]
                end = queue[check_idx][3]
                t = queue[check_idx][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_dict[t])
    return mol, no_correct


def valid_mol_can_with_seg(m, largest_connected_comp=True):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol

use_rdkit = True

def check_stability(atom_types, edge_types, dataset_info, debug=False,atom_decoder=None):
    if atom_decoder is None:
        atom_decoder = dataset_info.atom_decoder

    n_bonds = np.zeros(len(atom_types), dtype='int')

    for i in range(len(atom_types)):
        for j in range(i + 1, len(atom_types)):
            n_bonds[i] += abs((edge_types[i, j] + edge_types[j, i])/2)
            n_bonds[j] += abs((edge_types[i, j] + edge_types[j, i])/2)
    n_stable_bonds = 0
    for atom_type, atom_n_bond in zip(atom_types, n_bonds):
        possible_bonds = allowed_bonds[atom_decoder[atom_type]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == atom_n_bond
        else:
            is_stable = atom_n_bond in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type], atom_n_bond))
        n_stable_bonds += int(is_stable)

    molecule_stable = n_stable_bonds == len(atom_types)
    return molecule_stable, n_stable_bonds, len(atom_types)


def compute_molecular_metrics(molecule_list, targets, dataset_info, compute_mces=True, mces_timeout_sec=MCES_TIMEOUT_SEC):
    """ molecule_list: (dict) """

    if not dataset_info.remove_h:
        print(f'Analyzing molecule stability...')

        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0
        n_molecules = len(molecule_list)

        for i, mol in enumerate(molecule_list):
            atom_types, edge_types = mol

            validity_results = check_stability(atom_types, edge_types, dataset_info)

            molecule_stable += int(validity_results[0])
            nr_stable_bonds += int(validity_results[1])
            n_atoms += int(validity_results[2])

        # Validity
        fraction_mol_stable = molecule_stable / float(n_molecules)
        fraction_atm_stable = nr_stable_bonds / float(n_atoms)
        validity_dict = {'mol_stable': fraction_mol_stable, 'atm_stable': fraction_atm_stable}
        if wandb.run:
            wandb.log(validity_dict)
    else:
        validity_dict = {'mol_stable': -1, 'atm_stable': -1}

    metrics = BasicMolecularMetrics(
        dataset_info, targets, compute_mces=compute_mces, mces_timeout_sec=mces_timeout_sec)
    rdkit_metrics = metrics.evaluate(molecule_list)
    all_smiles = rdkit_metrics[-1]
    identical_list = rdkit_metrics[-4]
    tanimoto_list = rdkit_metrics[-3]
    mces_list = rdkit_metrics[-2]
    if wandb.run:
        nc = rdkit_metrics[-5]
        dic = {'Validity': rdkit_metrics[0][0], 'Relaxed Validity': rdkit_metrics[0][1],
               'Uniqueness': rdkit_metrics[0][2], 'Novelty': rdkit_metrics[0][3],
               'Identity': rdkit_metrics[0][4], 'Tanimoto': rdkit_metrics[0][5],
               'nc_max': nc['nc_max'], 'nc_mu': nc['nc_mu'], 'MCES': rdkit_metrics[0][6]}
        wandb.log(dic)

    return validity_dict, rdkit_metrics, all_smiles, identical_list, tanimoto_list, mces_list
