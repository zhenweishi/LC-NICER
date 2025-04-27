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
from sklearn.decomposition import PCA
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def train_PCA():
    task_settings = yaml.load(open("tasks/dl_train_PCA.yaml", "r"), Loader=yaml.FullLoader)
    output_dir = Path("output") / "dl_train_PCA"
    print("=> Output directory:", output_dir)

    load_state = dict(Step="LoadData", Datasets=copy.deepcopy(task_settings["Datasets"]))
    crop_state = call.run_step("Cropper", task_settings, load_state, output_dir)
    dl_state = call.run_step("DeepLearningFeatureExtraction", task_settings, crop_state, output_dir)
    print(dl_state)

    df = pd.read_csv(dl_state["Datasets"]["lung"]["path"], dtype={"ID": str})
    
    post_df = df.query("`phase` == 'post'").copy().reset_index(drop=True)
    pre_df = df.query("`phase` == 'pre'").copy().reset_index(drop=True)

    feat_col_idx = df.columns.get_loc("#EC#") + 1
    pre_feat = pre_df.iloc[:, feat_col_idx:].values
    post_feat = post_df.iloc[:, feat_col_idx:].values
    delta_feat = (pre_feat - post_feat) / (pre_feat + 1e-6)

    delta_df = pre_df.copy()
    delta_df.iloc[:, feat_col_idx:] = delta_feat
    delta_df["ID"] = delta_df["ID"].apply(lambda x: x.replace("pre", "delta"))
    delta_df["phase"] = "delta"

    dfs = {"pre": pre_df, "post": post_df, "delta": delta_df}
    for phase, df in dfs.items():
        print(phase)
        pca = PCA(n_components=16, random_state=42)
        scaler = StandardScaler()

        feat_col_idx = df.columns.get_loc("#EC#") + 1
        feat = df.iloc[:, feat_col_idx:].values

        feat = scaler.fit_transform(feat)
        pca_feat = pd.DataFrame(pca.fit_transform(feat))
        pca_feat.rename(columns={i: f"DL_PCA_{i+1}" for i in range(pca_feat.shape[-1])}, inplace=True)
        pca_df = df.iloc[:, :feat_col_idx].copy()
        pca_df = pd.concat([pca_df, pca_feat], axis=1)

        pkl_root = Path("./pkl")
        pkl_dir = pkl_root / "DeepLearningFeatureExtraction#Tumor"
        pkl_dir.mkdir(exist_ok=True, parents=True)
        pca_model_path = pkl_dir / f"dl_{phase}_PCA.pkl"
        norm_model_path = pkl_dir / f"dl_{phase}_norm.pkl"
        with open(pca_model_path, "wb") as f:
            pickle.dump(pca, f)
        with open(norm_model_path, "wb") as f:
            pickle.dump(scaler, f)
        

if __name__ == '__main__':
    train_PCA()