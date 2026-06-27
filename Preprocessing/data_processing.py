
import argparse
import os
import pickle
import random
import re

import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto

from generate_graph_dict import generate_graph_dict

# Try to import MCES splitting, but make it optional
try:
    from mces_splitting import split_dataset_lower_bound_only
    MCES_AVAILABLE = True
except ImportError:
    MCES_AVAILABLE = False


RDLogger.DisableLog("rdApp.warning")


ALLOWED_ATOMS = {"C", "N", "O", "F", "S", "Cl", "Br", "I"}

desired_columns = [
    "instrument_type",
    "collision_energy_NCE",
    "collision_energy_ev",
    "collision_energy_list",
    "collision_energy_mean",
    "smiles",
    "precursor_type",
    "precursor_mz",
    "molecular_formula",
    "molecular_formula_array",
    "cleaned_normalized_mz",
    "cleaned_normalized_intensity",
    "cleaned_fragment_formulas_str",
    "clean_spectrum_formula_array",
    "spectral_information_score"
]
    


def formulas_to_arrays(formulas):
    """
    Convert a list/array of molecular formula strings to a list of numpy arrays
    in the order: H, C, N, O, F, P, S, Cl, Br, I.
    """
    # Define the element order and a lookup
    elements = ["H", "C", "N", "O", "F", "S", "Cl", "Br", "I"]
    idx = {el: i for i, el in enumerate(elements)}

    # Pattern: element symbol = capital letter + optional lowercase
    # followed by an optional integer count
    pattern = re.compile(r"([A-Z][a-z]?)(\d*)")

    result = []

    for formula in formulas:
        counts = np.zeros(len(elements), dtype=int)

        for symbol, num in pattern.findall(formula):
            if symbol not in idx:
                return np.zeros(len(elements), dtype=int)

            n = int(num) if num else 1
            counts[idx[symbol]] += n

        result.append(counts)

    return result


def get_num_atoms(smi: str, fallback: int = 1000) -> int:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return fallback
        return mol.GetNumAtoms()
    except Exception:
        return fallback


def only_allowed_atoms(smiles: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return False
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in ALLOWED_ATOMS:
                return False
        return True
    except Exception:
        return False



def filter_df(df):
    print("\n" + "=" * 60)
    print("Step 1/3: Deriving spectrum features")
    print("=" * 60)

    # Enable tqdm integration with pandas
    try:
        from tqdm.auto import tqdm as _tqdm

        _tqdm.pandas()
        use_tqdm = True
    except Exception:
        use_tqdm = False

    # Derive clean_spectrum_formula_array (if not already present) and num_clean_peaks.
    # Pre-computed arrays are the preferred input; formula-string parsing is a
    # fallback that runs only when the array column is missing.
    if "clean_spectrum_formula_array" in df.columns:
        print("  'clean_spectrum_formula_array' already present -- using pre-computed arrays.")
        if "num_clean_peaks" not in df.columns:
            if use_tqdm:
                _tqdm.pandas(desc="Counting clean peaks per spectrum")
                df["num_clean_peaks"] = df["clean_spectrum_formula_array"].progress_apply(len)
            else:
                df["num_clean_peaks"] = df["clean_spectrum_formula_array"].apply(len)
    else:
        if "cleaned_fragment_formulas_str" not in df.columns:
            raise ValueError(
                "Neither 'clean_spectrum_formula_array' (pre-computed 9-element count "
                "vectors) nor 'cleaned_fragment_formulas_str' (list of formula strings) "
                "is present in the parquet. Supply one of them."
            )
        print("  'clean_spectrum_formula_array' missing -- deriving from 'cleaned_fragment_formulas_str'.")
        if use_tqdm:
            _tqdm.pandas(desc="Converting fragment formulas to arrays")
            df["clean_spectrum_formula_array"] = df[
                "cleaned_fragment_formulas_str"
            ].progress_apply(formulas_to_arrays)

            _tqdm.pandas(desc="Counting clean peaks per spectrum")
            df["num_clean_peaks"] = df["cleaned_fragment_formulas_str"].progress_apply(len)
        else:
            df["clean_spectrum_formula_array"] = df[
                "cleaned_fragment_formulas_str"
            ].apply(formulas_to_arrays)
            df["num_clean_peaks"] = df["cleaned_fragment_formulas_str"].apply(len)

    print("\n" + "=" * 60)
    print("Step 2/3: Applying basic filters")
    print("=" * 60)
    print("Conditions:")
    print("  - SMILES not null")
    print("  - Precursor type in {[M+H]+, [M-H]-}")
    print("  - 2 < num_clean_peaks < 128")
    print("  - 4 < collision_energy_NCE < 300")

    print(f"\nRows before basic filters: {len(df)}")
        
    initial_count = len(df)
    df = df[
        (df["smiles"].notna())
        & (df["precursor_type"].isin(["[M+H]+", "[M-H]-"]))
        & (df["num_clean_peaks"] > 2)
        & (df["num_clean_peaks"] < 128)
        & (df["collision_energy_NCE"] > 4)
        & (df["collision_energy_NCE"] < 300)
    ]
    
    after_basic = len(df)
    print(f"  After basic filters: {after_basic} rows ({initial_count - after_basic} removed)")

    print("\n" + "=" * 60)
    print("Step 3/3: Molecular filters")
    print("=" * 60)
    print("Computing atom counts and allowed-atom flags...")

    if use_tqdm:
        _tqdm.pandas(desc="Computing number of atoms per SMILES")
        df["number_of_atoms"] = df["smiles"].progress_apply(get_num_atoms)

        _tqdm.pandas(desc="Checking allowed atom types in SMILES")
        df["only_allowed_atoms"] = df["smiles"].progress_apply(only_allowed_atoms)
    else:
        df["number_of_atoms"] = df["smiles"].apply(get_num_atoms)
        df["only_allowed_atoms"] = df["smiles"].apply(only_allowed_atoms)

    print("\nApplying molecular filters:")
    print("  - number_of_atoms < 30")
    print("  - only_allowed_atoms == True")
    
    before_molecular = len(df)
    df = df[
        (df["number_of_atoms"] < 30)
        & (df["only_allowed_atoms"] == True)
    ]
    after_molecular = len(df)
    print(f"  After molecular filters: {after_molecular} rows ({before_molecular - after_molecular} removed)")

    print("\n" + "="*60)
    print("Filtering summary")
    print("="*60)
    print(f"Final shape: {df.shape}")
    print(f"Final unique SMILES: {len(df['smiles'].unique())}")
    print("="*60 + "\n")
    return df


def write_canonical_smiles_from_filt(df_filt, smiles_out):

    smiles_list = df_filt["smiles"].unique().tolist()
    random.shuffle(smiles_list)

    with open(smiles_out, "w") as f:
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            f.write(f"{canonical_smiles}\n")
    print(f"Wrote canonical SMILES to: {smiles_out}")



def load_parquet_with_column_filter(parquet_path, print_schema=False):
    """
    Load a parquet file, checking schema and filtering to only load desired columns that exist.
    """
    print(f"Loading parquet from: {parquet_path}")
    pf = pq.ParquetFile(parquet_path)
    
    if print_schema:
        print("\nParquet schema:")
        print(pf.schema)
    
    try:
        # Read just the first row to get column names (very fast)
        sample_table = pq.read_table(parquet_path, columns=None)
        schema_column_names = list(sample_table.column_names)
    except Exception as e:
        try:
            arrow_schema = pf.schema_arrow if hasattr(pf, 'schema_arrow') else pf.schema.to_arrow_schema()
            schema_column_names = list(arrow_schema.names)
        except:
            # Last resort: extract from Parquet schema fields (may miss nested columns)
            schema_column_names = [field.name for field in pf.schema]
    
    # Filter to only columns that exist in the schema
    columns_to_load = [col for col in desired_columns if col in schema_column_names]
    missing_columns = [col for col in desired_columns if col not in schema_column_names]
    
    if missing_columns:
        print(f"\nWarning: The following columns are not in the schema and will be skipped:")
        for col in missing_columns:
            print(f"  - {col}")
    
    print(f"\nLoading {len(columns_to_load)} columns: {columns_to_load}")
    
    # Load only the specified columns
    table = pq.read_table(parquet_path, columns=columns_to_load)
    df = table.to_pandas()
    print(f"Loaded dataframe shape: {df.shape}")
    
    return df


def random_split_smiles(smiles_list, val_fraction=0.05, test_fraction=0.05, seed=None):
    """
    Randomly split SMILES list into train, validation, and test sets.
    """
    print(f"\nPerforming random split...")
    print(f"  Total SMILES: {len(smiles_list)}")
    print(f"  Validation fraction: {val_fraction}")
    print(f"  Test fraction: {test_fraction}")
    if seed is not None:
        random.seed(seed)
    
    smiles_copy = smiles_list.copy()
    random.shuffle(smiles_copy)
    
    n_total = len(smiles_copy)
    n_val = int(n_total * val_fraction)
    n_test = int(n_total * test_fraction)
    n_train = n_total - n_val - n_test
    
    train_set = smiles_copy[:n_train]
    validation_set = smiles_copy[n_train:n_train + n_val]
    test_set = smiles_copy[n_train + n_val:]
    
    return train_set, validation_set, test_set


def mces_split_smiles(
    smiles_list,
    val_fraction=0.05,
    test_fraction=0.05,
    initial_distinction_threshold=5,
    min_distinction_threshold=5,
    threshold_step=-1,
    mces_matrix_save_path=None,
    use_saved_mces_matrix_path=None,
):
    """
    Split SMILES list using MCES (Maximum Common Edge Subgraph) based splitting.
    """
    print(f"\nPerforming MCES-based split...")
    print(f"  Total SMILES: {len(smiles_list)}")
    print(f"  Validation fraction: {val_fraction}")
    print(f"  Test fraction: {test_fraction}")

    if use_saved_mces_matrix_path is not None:
        print(f"  Using saved MCES matrix from: {use_saved_mces_matrix_path}")
    elif mces_matrix_save_path is not None:
        print(f"  Saving MCES matrix to: {mces_matrix_save_path}")

    if not MCES_AVAILABLE:
        raise ImportError(
            "MCES splitting requires mces_splitting module. "
            "Please install it or use random splitting instead."
        )

    if use_saved_mces_matrix_path is not None:
        train_set, validation_set, test_set, threshold = split_dataset_lower_bound_only(
        smiles_list.copy(),
        validation_fraction=val_fraction,
        test_fraction=test_fraction,
        initial_distinction_threshold=initial_distinction_threshold,
        min_distinction_threshold=min_distinction_threshold,
        threshold_step=threshold_step,
        use_saved_mces_matrix_path = use_saved_mces_matrix_path
    )
    else:
        train_set, validation_set, test_set, threshold = split_dataset_lower_bound_only(
        smiles_list.copy(),
        validation_fraction=val_fraction,
        test_fraction=test_fraction,
        initial_distinction_threshold=initial_distinction_threshold,
        min_distinction_threshold=min_distinction_threshold,
        threshold_step=threshold_step,
        mces_matrix_save_path=mces_matrix_save_path
    )

    
    return train_set, validation_set, test_set


def create_and_save_splits(
    smiles_list,
    split_type,
    output_path,
    val_fraction=0.05,
    test_fraction=0.05,
    seed=None,
    mces_kwargs=None,
):

    if split_type.lower() == "random":
        train_set, validation_set, test_set = random_split_smiles(
            smiles_list, val_fraction, test_fraction, seed
        )
    elif split_type.upper() == "MCES":
        if mces_kwargs is None:
            mces_kwargs = {}
        train_set, validation_set, test_set = mces_split_smiles(
            smiles_list, val_fraction, test_fraction, **mces_kwargs
        )
    else:
        raise ValueError(f"Unknown split type: {split_type}. Use 'random' or 'MCES'")
    
    split_dict = {"train": train_set, "val": validation_set, "test": test_set}
    
    with open(output_path, "wb") as f:
        pickle.dump(split_dict, f)
    
    print(f"Saved splits to: {output_path}")
    print(f"Train: {len(train_set)}, Val: {len(validation_set)}, Test: {len(test_set)}")



def parse_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-input_parquet", required=True, type=str, help="Path to input parquet file")
    parser.add_argument("-generate_graph_dict", action="store_true", help="Generate graph dictionary after filtering")
    
    # Splitting arguments
    parser.add_argument("-split_type", type=str, choices=["random", "MCES", "None"], default="random",
                        help="Type of splitting: 'random' or 'MCES' or 'None' (default: random)")
    parser.add_argument("-val_fraction", type=float, default=0.05,
                        help="Fraction of data for validation set (default: 0.05)")
    parser.add_argument("-test_fraction", type=float, default=0.05,
                        help="Fraction of data for test set (default: 0.05)")
    parser.add_argument("-random_seed", type=int, default=None,
                        help="Random seed for random splitting (default: None)")
    
    # MCES-specific arguments
    parser.add_argument("-mces_threshold", type=int, default=5,
                        help="MCES splitting threshold (default: 5)")
    parser.add_argument("-mces_matrix_save_path", type=str, default=None,
                        help="Path to save MCES matrix (default: None)")
    parser.add_argument("-mces_matrix_load_path", type=str, default=None,
                        help="Path to load saved MCES matrix (default: None)")

    return parser.parse_args()


def main():
    args = parse_args()

    input_parquet = args.input_parquet
    if not os.path.exists(input_parquet):
        print(f"Input parquet not found at {input_parquet}; aborting.")
        return

    # Set default output paths based on input directory if not provided
    input_dir = os.path.dirname(os.path.abspath(input_parquet))
    input_parquet_name = os.path.basename(input_parquet).replace(".parquet", "")

    output_parquet = os.path.join(input_dir, input_parquet_name + "_filtered.parquet")
    output_smiles = os.path.join(input_dir, input_parquet_name + "_filtered_smiles_canonical.txt")

    # Load parquet with column filtering
    df = load_parquet_with_column_filter(input_parquet)
    df_filt = filter_df(df.copy(deep=True))

    print(f"Saving filtered dataframe to: {output_parquet}")
    df_filt.to_parquet(output_parquet)

    # Canonical SMILES from filtered dataframe
    write_canonical_smiles_from_filt(df_filt, output_smiles)

    # Run external graph dict generation (if provided)
    if args.generate_graph_dict:
        output_dict = os.path.join(input_dir, input_parquet_name + "_smiles_dict.pt")
        generate_graph_dict(
            input_parquet=output_parquet,
            output_dict=output_dict,
            input_smiles=output_smiles,
        )
    
    # Create splits if requested
    if args.split_type != "None":

        smiles_list = df_filt["smiles"].unique().tolist()
        
        split_name = input_parquet_name
        splits_output = os.path.join(input_dir, f"splits_{split_name}_{args.split_type}.pkl")
        
        if args.split_type.lower() == "random":
            train_set, validation_set, test_set = random_split_smiles(
                smiles_list, args.val_fraction, args.test_fraction, args.random_seed
            )
        elif args.split_type.upper() == "MCES":
            train_set, validation_set, test_set = mces_split_smiles(
                smiles_list, args.val_fraction, args.test_fraction,
                initial_distinction_threshold=args.mces_threshold,
                min_distinction_threshold=args.mces_threshold,
                threshold_step=-1,
                mces_matrix_save_path=args.mces_matrix_save_path,
                use_saved_mces_matrix_path=args.mces_matrix_load_path
            )
        else:
            raise ValueError(f"Unknown split type: {args.split_type}. Use 'random' or 'MCES'")
        
        split_dict = {"train": train_set, "val": validation_set, "test": test_set}
        
        print(f"\nSaving splits to: {splits_output}")
        with open(splits_output, "wb") as f:
            pickle.dump(split_dict, f)
        
        print("\n" + "="*60)
        print("Splitting complete!")
        print(f"Final split sizes:")
        print(f"  Train: {len(train_set)} SMILES")
        print(f"  Validation: {len(validation_set)} SMILES")
        print(f"  Test: {len(test_set)} SMILES")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()


