import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from importlib import reload
import sys
sys.path.append("../..") # LC-NICER/
import medi_ai
import medi_ai.call as call
import pandas as pd
import copy
import yaml
from datetime import datetime
from pathlib import Path
import pickle
import numpy as np
import polars as pl
from pathlib import Path
import shutil


def train():
    task_settings = yaml.load(open("tasks/hi_train.yaml", "r"), Loader=yaml.FullLoader)

    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    now_str = ""
    output_dir = Path("output") / task_settings["Task_Name"]
    print("=> Output directory:", output_dir)

    # Load Data
    load_state = dict(Step="LoadData", Datasets=copy.deepcopy(task_settings["Datasets"]))

    # Preprocessing
    prep_state = call.run_step("Preprocessing", task_settings, load_state, output_dir)

    # Various Mask
    varm_state = call.run_step("VariousMask", task_settings, prep_state, output_dir)

    # Pre Segmentation
    seg_state = call.run_step("PreSegmentation", task_settings, varm_state, output_dir)

    # SV Level
    ## Feature Extraction
    sv_fe_state = call.run_step("FeatureExtraction#SV", task_settings, seg_state, output_dir)

    ## DL Feature Extraction
    sv_dl_state = call.run_step("DeepLearningFeatureExtraction#SV", task_settings, seg_state, output_dir)

    # Merge
    def merge_datasets(fe_state, dl_state):
        step_name = "Merge"
        new_state = dict(Step=step_name, Datasets={})
        out_dir = output_dir / step_name
        out_dir.mkdir(parents=True, exist_ok=True)
        

        for suffix, state in [("Rad", fe_state), ("DL", dl_state)]:
            for key, value in state["Datasets"].items():
                new_key = f"{key}#{suffix}"
                value = value.copy()
                new_path = out_dir / Path(value["path"]).name.replace(".csv", f"#{suffix}.csv")

                shutil.copy(value["path"], new_path)  # Copy the file

                value["path"] = str(new_path)
                new_state["Datasets"][new_key] = value

                src_dir = output_dir / task_settings["SubRegionClustering"]["src_dir"] / key
                dst_dir = output_dir / step_name / new_key
                if dst_dir.exists():
                    if dst_dir.is_symlink():
                        dst_dir.unlink()
                    else:
                        shutil.rmtree(dst_dir)
                else:
                    dst_dir.symlink_to(src_dir.resolve(), target_is_directory=True)

                
        for key in fe_state["Datasets"]:
            assert key in dl_state["Datasets"], f"Key {key} not found in DL state"
            new_key = f"{key}#Rad+DL"

            fe_df = pl.read_csv(fe_state["Datasets"][key]["path"])
            dl_df = pl.read_csv(dl_state["Datasets"][key]["path"])
            
            assert fe_df["ID"].equals(dl_df["ID"]), "ID not equal"
            assert fe_df["sv_id"].equals(dl_df["sv_id"]), "SV ID not equal"

            feat_col_idx = dl_df.columns.index("#EC#") + 1
            dl_feat_df = dl_df.select(dl_df.columns[feat_col_idx:])

            new_df = fe_df.hstack(dl_feat_df)
            new_path = out_dir / f"{new_key}.csv"
            new_df.write_csv(new_path)

            new_state["Datasets"][new_key] = fe_state["Datasets"][key].copy()
            new_state["Datasets"][new_key]["path"] = str(new_path)

            src_dir = output_dir / task_settings["SubRegionClustering"]["src_dir"] / key
            dst_dir = output_dir / step_name / new_key
            if dst_dir.exists():
                if dst_dir.is_symlink():
                    dst_dir.unlink()
                else:
                    shutil.rmtree(dst_dir)
            else:
                dst_dir.symlink_to(src_dir.resolve(), target_is_directory=True)

        return new_state
    merge_state = merge_datasets(sv_fe_state, sv_dl_state)

    # Normalization
    norm_state = call.run_step("Normalization#01", task_settings, merge_state, output_dir)

    # SubRegionClustering
    norm_state["image_src_dir"] = "Merge"
    sr_state = call.run_step("SubRegionClustering", task_settings, norm_state, output_dir)
    
    # SV Level
    ## Feature Extraction
    sr_fe_state = call.run_step("FeatureExtraction#SR", task_settings, sr_state, output_dir)

    # SV Level
    ## DL Feature Extraction
    sr_dl_state = call.run_step("DeepLearningFeatureExtraction#SR", task_settings, sr_state, output_dir)

    # Merge SR
    def merge_datasets_SR(fe_state, dl_state):
        step_name = "MergeSR"
        new_state = dict(Step=step_name, Datasets={})
        out_dir = output_dir / step_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for suffix, state in [("Rad", fe_state), ("DL", dl_state)]:
            for key, value in state["Datasets"].items():
                new_key = f"{key}#{suffix}"
                value = value.copy()
                new_path = out_dir / Path(value["path"]).name.replace(".csv", f"#{suffix}.csv")

                shutil.copy(value["path"], new_path)  # Copy the file

                value["path"] = str(new_path)
                new_state["Datasets"][new_key] = value

                
        for key in fe_state["Datasets"]:
            assert key in dl_state["Datasets"], f"Key {key} not found in DL state"
            new_key = f"{key}#Rad+DL"

            fe_df = pl.read_csv(fe_state["Datasets"][key]["path"])
            dl_df = pl.read_csv(dl_state["Datasets"][key]["path"])
            
            assert fe_df["ID"].equals(dl_df["ID"]), "ID not equal"
            assert fe_df["sr_id"].equals(dl_df["sr_id"]), "SV ID not equal"

            feat_col_idx = dl_df.columns.index("#EC#") + 1
            dl_feat_df = dl_df.select(dl_df.columns[feat_col_idx:])

            new_df = fe_df.hstack(dl_feat_df)
            new_path = out_dir / f"{new_key}.csv"
            new_df.write_csv(new_path)

            new_state["Datasets"][new_key] = fe_state["Datasets"][key].copy()
            new_state["Datasets"][new_key]["path"] = str(new_path)


        return new_state

    mergeSR_state = merge_datasets_SR(sr_fe_state, sr_dl_state)

    # Tumor Heterogeneity
    th_state = call.run_step("TumorHeterogeneity", task_settings, mergeSR_state, output_dir)

    og_df = pd.read_csv('data/train/data.csv', dtype={"ID": str})
    for ds in th_state["Datasets"].values():
        path = ds["path"]
        df = pd.read_csv(path, dtype={"ID": str})
        merged_df = og_df.merge(df, on="ID", how="left", indicator=True)
        merged_df.insert(1, "OK", merged_df.pop("_merge").astype(str))
        merged_df["OK"] = merged_df["OK"].apply(lambda x: "OK" if x == "both" else "fillna")
        
        merged_df.fillna(0.0, inplace=True)
        merged_df["#EC#"] = None

        dst_csv_path = output_dir / "TumorHeterogeneityFillNA" / Path(path).name
        dst_csv_path.parent.mkdir(exist_ok=True, parents=True)
        merged_df.to_csv(dst_csv_path, index=False)

    pkl_root = Path("./pkl")
    pkl_dir = pkl_root / "Normalization#01"
    for src_path in (output_dir / "Normalization#01").glob("*Rad+DL.pkl"):
        dst_path = pkl_dir / src_path.name
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src=src_path, dst=dst_path)

    pkl_dir = pkl_root / "SubRegionClustering"
    for src_path in (output_dir / "SubRegionClustering").rglob("*Rad+DL.pkl"):
        dst_path = pkl_dir / src_path.name
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src=src_path, dst=dst_path)

    pkl_dir = pkl_root / "TumorHeterogeneity"
    for src_dir in (output_dir / "TumorHeterogeneity").glob("*Rad+DL#Rad+DL"):
        if src_dir.is_dir():
            dst_dir = pkl_dir / src_dir.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src_path in src_dir.glob("*.pkl"):
                dst_path = dst_dir / src_path.name
                shutil.copy2(src=src_path, dst=dst_path)



if __name__ == '__main__':
    train()
