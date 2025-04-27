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
import pandas as pd
import copy
import yaml
from datetime import datetime
from pathlib import Path
import pickle
import numpy as np
from collections import OrderedDict
import hi
import shutil
import monai
from monai.bundle import ConfigParser
from medi_ai.workflow import run_workflow


def rad_dl_hi_feat(dataset_name):
    task_settings = yaml.load(open(f"tasks/rad_dl_hi_feat_{dataset_name}.yaml", "r"), Loader=yaml.FullLoader)
    output_dir = Path("output") / task_settings["Task_Name"]
    print("=> Output directory:", output_dir)

    # =============== HI ===============
    load_state = dict(Step="LoadData", Datasets=copy.deepcopy(task_settings["Datasets"]))
    prep_state = medi_ai.call.run_step("Preprocessing", task_settings, load_state, output_dir)
    varm_state = medi_ai.call.run_step("VariousMask", task_settings, prep_state, output_dir)
    seg_state = medi_ai.call.run_step("PreSegmentation", task_settings, varm_state, output_dir)
    sv_fe_state = medi_ai.call.run_step("FeatureExtraction#SV", task_settings, seg_state, output_dir)
    sv_dl_state = medi_ai.call.run_step("DeepLearningFeatureExtraction#SV", task_settings, seg_state, output_dir)
    sv_ff_state = medi_ai.call.run_step(
        "FeatureFusion#SV-Rad+DL", 
        task_settings, 
        dict(Datasets=pd.DataFrame([sv_fe_state, sv_dl_state])["Datasets"].values.tolist()), 
        output_dir)
    sv_ff_norm_state = medi_ai.call.run_step("Normalization#01", task_settings, sv_ff_state, output_dir)
    sv_ff_norm_state["image_src_dir"] = seg_state["Step"]
    sr_state = medi_ai.call.run_step("SubRegionClustering", task_settings, sv_ff_norm_state, output_dir)
    sr_fe_state = medi_ai.call.run_step("FeatureExtraction#SR", task_settings, sr_state, output_dir)
    sr_dl_state = medi_ai.call.run_step("DeepLearningFeatureExtraction#SR", task_settings, sr_state, output_dir)
    sr_ff_state = medi_ai.call.run_step(
        "FeatureFusion#SR-Rad+DL", 
        task_settings, 
        dict(Datasets=pd.DataFrame([sr_fe_state, sr_dl_state])["Datasets"].values.tolist()), 
        output_dir)
    th_state = medi_ai.call.run_step("TumorHeterogeneity", task_settings, sr_ff_state, output_dir)
    og_df = pd.read_csv(load_state["Datasets"]["lung"]["path"], dtype={"ID": str})
    for ds in th_state["Datasets"].values():
        path = ds["path"]
        df = pd.read_csv(path, dtype={"ID": str})
        merged_df = og_df.merge(df, on="ID", how="left", indicator=True)
        merged_df.insert(1, "OK", merged_df.pop("_merge").astype(str))
        merged_df["OK"] = merged_df["OK"].apply(lambda x: "OK" if x == "both" else "fillna")
        
        merged_df.fillna(0.0, inplace=True)
        merged_df["#EC#"] = None

        for i in range(1, 6):
            merged_df.rename(columns={f"SR_{i}": f"HI_{i}"}, inplace=True)

        dst_csv_path = output_dir / "TumorHeterogeneityFillNA" / Path(path).name
        dst_csv_path.parent.mkdir(exist_ok=True, parents=True)
        merged_df.to_csv(dst_csv_path, index=False)

    hi_dfs = {
        "peritumor3": pd.read_csv(f"{output_dir}/TumorHeterogeneityFillNA/peritumor#3#lung.csv"),
        "peritumor5": pd.read_csv(f"{output_dir}/TumorHeterogeneityFillNA/peritumor#5#lung.csv"),
        "peritumor7": pd.read_csv(f"{output_dir}/TumorHeterogeneityFillNA/peritumor#7#lung.csv"),
        "tumor": pd.read_csv(f"{output_dir}/TumorHeterogeneityFillNA/tumor#lung.csv"),
    }

    for type_ in hi_dfs.keys():
        hi_cols = hi_dfs[type_].columns[hi_dfs[type_].columns.get_loc("#EC#")+1:]
        df = hi_dfs[type_].copy()
        df["ID2"] = df["ID"].apply(lambda x: x.replace("pre", "xxx").replace("post", "xxx"))

        # calculate delta
        post_df = df.query("`phase` == 'post'").copy().reset_index(drop=True)
        pre_df = df.query("`phase` == 'pre'").copy().reset_index(drop=True)
        assert post_df[["ID2"]].equals(pre_df[["ID2"]])

        pre_feat = pre_df.loc[:, hi_cols].values
        post_feat = post_df.loc[:, hi_cols].values
        delta_feat = (pre_feat - post_feat) / (pre_feat + 1e-6)

        delta_df = pre_df.copy()
        delta_df.loc[:, hi_cols] = delta_feat
        delta_df["ID"] = delta_df["ID"].apply(lambda x: x.replace("pre", "delta"))
        delta_df["phase"] = "delta"

        df = pd.concat([
            pre_df,
            post_df,
            delta_df
        ], axis=0).reset_index(drop=True)
        df.drop(columns=["ID2"], inplace=True)
        
        # df = df[["ID", "ID2", "center", "phase", "#EC#"] + feat_df.columns.tolist()]
        hi_dfs[type_] = df.copy()

    for type_ in hi_dfs.keys():
        (output_dir / "FinalFeature").mkdir(exist_ok=True, parents=True)
        hi_dfs[type_].to_csv(f"{output_dir}/FinalFeature/HI_{type_}.csv", index=False)

    # =============== DL PCA ===============
    tu_crop_state = medi_ai.call.run_step("Cropper", task_settings, load_state, output_dir)
    tu_dl_state = medi_ai.call.run_step("DeepLearningFeatureExtraction#Tumor", task_settings, tu_crop_state, output_dir)

    dl_768_df = pd.read_csv(tu_dl_state["Datasets"]["lung"]["path"])

    df = dl_768_df.copy()
    cols = df.columns[df.columns.get_loc("#EC#")+1:]
    df["ID2"] = df["ID"].apply(lambda x: x.replace("pre", "xxx").replace("post", "xxx"))

    # calculate delta
    post_df = df.query("`phase` == 'post'").copy().reset_index(drop=True)
    pre_df = df.query("`phase` == 'pre'").copy().reset_index(drop=True)
    assert post_df[["ID2"]].equals(pre_df[["ID2"]])

    pre_feat = pre_df.loc[:, cols].values
    post_feat = post_df.loc[:, cols].values
    delta_feat = (pre_feat - post_feat) / (pre_feat + 1e-6)

    delta_df = pre_df.copy()
    delta_df.loc[:, cols] = delta_feat
    delta_df["ID"] = delta_df["ID"].apply(lambda x: x.replace("pre", "delta"))
    delta_df["phase"] = "delta"

    df = pd.concat([
        pre_df,
        post_df,
        delta_df
    ], axis=0).reset_index(drop=True)
    df.drop(columns=["ID2"], inplace=True)

    dl_768_df = df.copy()
    dl_768_df.to_csv(f"{output_dir}/FinalFeature/DL_768.csv", index=False)

    import pickle
    dl_pca_df = []
    for phase in ["pre", "post", "delta"]:
        df = dl_768_df.query(f"`phase` == '{phase}'").copy().reset_index(drop=True)
        cols = df.columns[df.columns.get_loc("#EC#")+1:]
        feat = df.loc[:, cols].values
        scaler = pickle.load(open(f"pkl/DeepLearningFeatureExtraction#Tumor/dl_{phase}_norm.pkl", "rb"))
        feat = scaler.transform(feat)
        pca = pickle.load(open(f"pkl/DeepLearningFeatureExtraction#Tumor/dl_{phase}_PCA.pkl", "rb"))
        feat_pca = pca.transform(feat)
        pca_df = pd.DataFrame(feat_pca)
        pca_df.rename(columns={i: f"DL_PCA_{i+1}" for i in range(pca_df.shape[-1])}, inplace=True)
        df = pd.concat([df, pca_df], axis=1)
        df.drop(columns=cols, inplace=True)
        dl_pca_df.append(df)
    dl_pca_df = pd.concat(dl_pca_df, axis=0).reset_index(drop=True)
    dl_pca_df.to_csv(f"{output_dir}/FinalFeature/DL_PCA.csv", index=False)


    # =============== Rad ===============
    tu_fe_state = medi_ai.call.run_step("FeatureExtraction#Tumor", task_settings, load_state, output_dir)
    rad_df = pd.read_csv(f"{output_dir}/FeatureExtraction#Tumor/lung.csv")

    df = rad_df.copy()
    cols = df.columns[df.columns.get_loc("#EC#")+1:]
    df["ID2"] = df["ID"].apply(lambda x: x.replace("pre", "xxx").replace("post", "xxx"))

    # calculate delta
    post_df = df.query("`phase` == 'post'").copy().reset_index(drop=True)
    pre_df = df.query("`phase` == 'pre'").copy().reset_index(drop=True)
    assert post_df[["ID2"]].equals(pre_df[["ID2"]])

    pre_feat = pre_df.loc[:, cols].values
    post_feat = post_df.loc[:, cols].values
    delta_feat = (pre_feat - post_feat) / (pre_feat + 1e-6)

    delta_df = pre_df.copy()
    delta_df.loc[:, cols] = delta_feat
    delta_df["ID"] = delta_df["ID"].apply(lambda x: x.replace("pre", "delta"))
    delta_df["phase"] = "delta"

    df = pd.concat([
        pre_df,
        post_df,
        delta_df
    ], axis=0).reset_index(drop=True)
    df.drop(columns=["ID2"], inplace=True)
    rad_df = df.copy()
    rad_df.to_csv(f"{output_dir}/FinalFeature/Rad.csv", index=False)

def prepare_data():
    output_dir = Path("data") / "__AUTO__"
    for dataset_name in ["train", "val"]:
        res_dir = Path("output") / f"rad_dl_hi_feat_{dataset_name}" / "FinalFeature"
        for type_ in ["Rad", "DL_PCA", "HI_peritumor3", "HI_peritumor5", "HI_peritumor7", "HI_tumor"]:
            df = pd.read_csv(res_dir / f"{type_}.csv")
            for phase in ["pre", "post", "delta"]:
                phase_df = df.query(f"`phase` == '{phase}'").copy().reset_index(drop=True)
                dst_path = output_dir / f"{phase}/{type_}/{dataset_name}.csv"
                dst_path.parent.mkdir(exist_ok=True, parents=True)
                phase_df.to_csv(dst_path, index=False)

def feature_selection():
    output_dir = Path("output") / "LC-NICER_train/feature_selection"
    for phase in ["pre", "post", "delta"]:
        for type_ in ["Rad", "DL_PCA", "HI_peritumor3", "HI_peritumor5", "HI_peritumor7", "HI_tumor"]:
            task_name = f"FeatureSelection#{phase}#{type_}"
            task_settings = {
                "Datasets": {
                    "train": {
                        "type": "training",
                        "path": f"data/__AUTO__/{phase}/{type_}/train.csv"
                    },
                    "val": {
                        "type": "validation",
                        "path": f"data/__AUTO__/{phase}/{type_}/val.csv"
                    }
                },
                task_name: OrderedDict([
                    ("variance_threshold_selection", {"var_thred": 0.75}),
                    ("correlation_selection", {"method": "spearman", "threshold": 0.75}),
                    ("univariate_statistical_test", {"test": "mannwhitneyu", "p_value_cutoff": 0.05}),
                    ("lasso_selection", {"cv": "5"}),
                ]) 
            }
            load_state = dict(Step="LoadData", Datasets=copy.deepcopy(task_settings["Datasets"]))
            fs_state = medi_ai.call.run_step(task_name, task_settings, load_state, output_dir)

def run_merge_feature():
    fs_root = Path("output/LC-NICER_train/feature_selection")
    for phase in ["pre", "post", "delta"]:
        output_dir = Path("output/LC-NICER_train/feature_merge") / phase
        output_dir.mkdir(parents=True, exist_ok=True)

        for table_name in ["train", "val"]:
            # ============ HI ============
            df_list = []
            for type_ in ["Rad", "DL_PCA", "HI_peritumor3", "HI_peritumor5", "HI_peritumor7", "HI_tumor"]:
                if type_ not in ["HI_peritumor3", "HI_peritumor5", "HI_peritumor7", "HI_tumor"]:
                    continue
                df = pd.read_csv(fs_root / f"FeatureSelection#{phase}#{type_}/{table_name}/final.csv")

                cols = df.columns.tolist()
                feat_col_idx = cols.index("#EC#")
                cols = cols[feat_col_idx + 1:]
                df = df.rename(columns={col: f"{phase}#{type_}#{col}" for col in cols})
                df_list.append(df)
        
            for i in range(1, len(df_list)):
                assert (df_list[0]["ID"] == df_list[i]["ID"]).all()
                assert (df_list[0]["Label"] == df_list[i]["Label"]).all()

                cols = df_list[i].columns.tolist()
                feat_col_idx = cols.index("#EC#")
                cols = cols[feat_col_idx + 1:]
                df_list[i] = df_list[i][cols]

            df = pd.concat(df_list, axis=1)
            dst_csv_path = output_dir / "HI" / f"{table_name}.csv"
            dst_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(dst_csv_path, index=False)
            print(f"=> {dst_csv_path} saved")

            # ============ Rad ============
            src_csv_path = fs_root / f"FeatureSelection#{phase}#Rad/{table_name}/final.csv"
            dst_csv_path = output_dir / "Rad" / f"{table_name}.csv"
            dst_csv_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_csv_path, dst_csv_path)
            print(f"=> {dst_csv_path} saved")

            # ============ DL ============
            src_csv_path = fs_root / f"FeatureSelection#{phase}#DL_PCA/{table_name}/final.csv"
            dst_csv_path = output_dir / "DL" / f"{table_name}.csv"
            dst_csv_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_csv_path, dst_csv_path)
            print(f"=> {dst_csv_path} saved")

def auto_run():
    output_dir = Path(f"output/LC-NICER_train")
    config = ConfigParser.load_config_files([f"tasks/LC-NICER_train.yaml"])
    parser = ConfigParser(config)
    args = parser.get_parsed_content()
    run_workflow(task_settings=args, inputs={}, output_dir=output_dir)

    for path in output_dir.iterdir():
        if not path.name.startswith("Workflow"):
            continue
        cdir = path / "Compose"
        norm_dir = cdir / "Normalization"
        norm_pkl = norm_dir / "train.pkl"
        model_dir = cdir / "ModelSelection"
        model_pkl = model_dir.rglob("model.pkl")
        # print(norm_pkl, next(model_pkl))
        name = path.name.replace("Workflow_", "")
        dst_norm_pkl = Path("pkl/LC-NICER/") / name / "norm.pkl"
        dst_model_pkl = Path("pkl/LC-NICER/") / name / "model.pkl"
        dst_norm_pkl.parent.mkdir(exist_ok=True, parents=True)
        dst_model_pkl.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(norm_pkl, dst_norm_pkl)
        shutil.copy(next(model_pkl), dst_model_pkl)

    Path("pkl/Feature").mkdir(parents=True, exist_ok=True)
    all_cols = {}
    for phase in ["pre", "post", "delta"]:
        for type_ in ["Rad", "DL_PCA", "HI_tumor", "HI_peritumor3", "HI_peritumor5", "HI_peritumor7"]:
            df = pd.read_csv(output_dir / f"feature_selection/FeatureSelection#{phase}#{type_}/train/final.csv")
            cols = df.columns[df.columns.get_loc("#EC#")+1:]
            all_cols[(phase, type_)] = cols.tolist()
    with open("pkl/Feature/feature_selection.pkl", "wb") as f:
        pickle.dump(all_cols, f)
    
    feat_cols = {} 
    for phase in ["pre", "post", "delta"]:
        for type_ in ["Rad", "DL", "HI"]:
            df = pd.read_csv(output_dir / f"feature_merge/{phase}/{type_}/train.csv")
            cols = df.columns[df.columns.get_loc("#EC#")+1:]
            feat_cols[(phase, type_)] = cols.tolist()
    with open("pkl/Feature/feature_merge.pkl", "wb") as f:
        pickle.dump(feat_cols, f)

def main():
    print("=> Generate HI")
    hi.train()

    print("=> Predict Rad, DL, and HI features")
    rad_dl_hi_feat("train")
    rad_dl_hi_feat("val")

    print("=> Prepare data for training")
    prepare_data()

    print("=> Feature selection")
    feature_selection()

    print("=> Merge features")
    run_merge_feature()

    print("=> Auto run LC-NICER training")
    auto_run()

    
if __name__ == "__main__":
    main()