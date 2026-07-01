import pandas as pd
from biopandas.pdb import PandasPdb
import tarfile
import os
from definitions import ROOT_DIR

EXTRACTED_PDB_PATH = os.path.join(ROOT_DIR, "pdb", "extracted")


def _load_pdb_file(filepath: str, model_index: int = 1) -> pd.DataFrame:
    pdbb = PandasPdb().read_pdb(filepath)
    model = pdbb.get_model(model_index)
    atom_df = model.df["ATOM"]
    if len(atom_df) == 0:
        raise ValueError(f"No model found for index: {model_index}")

    return atom_df


def _load_tar_file(filepath: str, out_dir: str) -> pd.DataFrame:
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            if not name.endswith(".pdb"):
                continue
            os.remove(os.path.join(out_dir, name))
    else:
        os.makedirs(out_dir, exists_ok=True)

    with tarfile.open(filepath) as file:
        file.extractall(out_dir)
        print(f"Extracted {filepath} to {out_dir}")

    bundles = []
    for name in os.listdir(out_dir):
        if not name.endswith(".pdb"):
            continue

        df = _load_pdb_file(os.path.join(out_dir, name))
        bundles.append(df)

    if not bundles:
        raise FileNotFoundError(f"No PDB files found in {filepath}")

    return pd.concat(bundles, ignore_index=True)


class PDB:
    def __init__(self, filepath: str, model_index: int = 1):
        if not os.path.isfile(filepath):
            raise FileExistsError(f"File {filepath} does not exist")

        if not filepath.endswith(".pdb") and not filepath.endswith(".tar.gz"):
            raise ValueError("Formats accepted: {.pdb, .tar.gz}")

        self.filepath = filepath
        self.model_index = model_index

        self.dataframe: pd.DataFrame | None = None

        if filepath.endswith(".pdb"):
            self.dataframe = _load_pdb_file(filepath)

        if filepath.endswith(".tar.gz"):
            self.dataframe = _load_tar_file(
                filepath, out_dir=os.path.join(ROOT_DIR, "pdb", "extracted")
            )

        if self.dataframe is None:
            raise Exception("Failed to load PDB file")

        self.process_dataframe()

        print(f"Loaded PDB at {filepath}")
        print(f"Dataframe columns {self.dataframe.columns}")
        print(f"Dataframe shape {self.dataframe.shape}")

    def process_dataframe(self):
        df_copy = self.dataframe.copy()
        df_copy = df_copy.drop(
            columns=[
                "record_name",
                "atom_number",
                "blank_1",
                "blank_2",
                "blank_3",
                "occupancy",
                "b_factor",
                "blank_4",
                "segment_id",
                "charge",
                "line_idx",
                "model_id",
            ]
        )

        if "alt_loc" in df_copy.columns:
            df_copy = df_copy[df_copy["alt_loc"].isin(["", "A"])]

        df_copy["node_id"] = (
            df_copy["chain_id"].astype(str)
            + ":"
            + df_copy["residue_number"].astype(str)
            + df_copy["insertion"].astype(str).str.strip()
        )

        self.dataframe = df_copy

    def filter_atoms(self, atom: str = "CA") -> pd.DataFrame:
        df_copy = self.dataframe.copy()
        if atom == "CB":
            mask_cb = df_copy["atom_name"] == "CB"
            mask_gly_ca = (df_copy["residue_name"] == "GLY") & (
                df_copy["atom_name"] == "CA"
            )
            df_copy = df_copy.loc[mask_cb | mask_gly_ca]
        else:
            df_copy = df_copy.loc[df_copy["atom_name"] == atom]
        df_copy = df_copy.drop_duplicates(
            subset=["chain_id", "residue_number", "insertion"]
        )
        return df_copy.reset_index(drop=True)
