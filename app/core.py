import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
from pathlib import Path

CWD = Path(__file__).parent
ROOT = CWD.parent
sys.path.append(str(ROOT))

import medi_ai
import pandas as pd
import copy
import yaml
from datetime import datetime
import pickle
import numpy as np
import hashlib
import shutil
import SimpleITK as sitk


YAML_PATH = CWD / "task.yaml"
PKL_DIR = ROOT / "examples" / "test" / "pkl"
CACHE_ROOT = CWD / "cache" 

def get_file_md5(file_path):
    """
    Calculate MD5 hash value of file
    
    Args:
        file_path: File path
        
    Returns:
        str: MD5 hash value
    """
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read in chunks, suitable for large files
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def split_df(df):
    pre_df = df.query("`phase` == 'pre'").copy().reset_index(drop=True)
    post_df = df.query("`phase` == 'post'").copy().reset_index(drop=True)
    delta_df = df.query("`phase` == 'delta'").copy().reset_index(drop=True)
    return pre_df, post_df, delta_df

def feature_selection(df, cols):
    df = df[["ID", "PID", "#EC#"] + cols].copy()
    new_cols = df.columns[df.columns.get_loc("#EC#")+1:].tolist()
    assert len(new_cols) == len(cols), f"new_cols: {new_cols}, cols: {cols}"
    return df


def prepare_task(pre_image_path, pre_mask_path, post_image_path, post_mask_path):
    pre_image_md5 = get_file_md5(pre_image_path)
    pre_mask_md5 = get_file_md5(pre_mask_path)
    post_image_md5 = get_file_md5(post_image_path)
    post_mask_md5 = get_file_md5(post_mask_path)

    task_root = CACHE_ROOT / f"{pre_image_md5[:4]}_{pre_mask_md5[:4]}_{post_image_md5[:4]}_{post_mask_md5[:4]}"
    print("=> Task root:", task_root)

    data_dir = task_root / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    (data_dir / "image").mkdir(parents=True, exist_ok=True)
    (data_dir / "mask").mkdir(parents=True, exist_ok=True)

    data_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([
        dict(ID="pre_X", PID="X", phase="pre", image_path=f"image/{pre_image_md5}.nii.gz", mask_path=f"mask/{pre_mask_md5}.nii.gz"),
        dict(ID="post_X", PID="X", phase="post", image_path=f"image/{post_image_md5}.nii.gz", mask_path=f"mask/{post_mask_md5}.nii.gz"),
    ])
    df.to_csv(data_dir / "data.csv", index=False)
    try:
        Path(data_dir / f"image/{pre_image_md5}.nii.gz").symlink_to(pre_image_path)
        Path(data_dir / f"mask/{pre_mask_md5}.nii.gz").symlink_to(pre_mask_path)
        Path(data_dir / f"image/{post_image_md5}.nii.gz").symlink_to(post_image_path)
        Path(data_dir / f"mask/{post_mask_md5}.nii.gz").symlink_to(post_mask_path)
    except OSError:
        shutil.copy(pre_image_path, data_dir / f"image/{pre_image_md5}.nii.gz")
        shutil.copy(pre_mask_path, data_dir / f"mask/{pre_mask_md5}.nii.gz")
        shutil.copy(post_image_path, data_dir / f"image/{post_image_md5}.nii.gz")
        shutil.copy(post_mask_path, data_dir / f"mask/{post_mask_md5}.nii.gz")

    output_dir = task_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    task_settings = yaml.load(open(YAML_PATH, "r"), Loader=yaml.FullLoader)
    task_settings["Datasets"]["lung"] = {
        "type": "testing",
        "path": str((data_dir / "data.csv").resolve()),
    }
    task_settings["FeatureExtraction#SV"]["params"] = task_settings["FeatureExtraction#SV"]["params"].format(ROOT=ROOT)
    task_settings["DeepLearningFeatureExtraction#SV"]["mae3d_lung_CT"]["root"] = task_settings["DeepLearningFeatureExtraction#SV"]["mae3d_lung_CT"]["root"].format(ROOT=ROOT)
    task_settings["FeatureExtraction#SR"]["params"] = task_settings["FeatureExtraction#SR"]["params"].format(ROOT=ROOT)
    task_settings["DeepLearningFeatureExtraction#SR"]["mae3d_lung_CT"]["root"] = task_settings["DeepLearningFeatureExtraction#SR"]["mae3d_lung_CT"]["root"].format(ROOT=ROOT)
    task_settings["FeatureExtraction#Tumor"]["params"] = task_settings["FeatureExtraction#Tumor"]["params"].format(ROOT=ROOT)
    task_settings["DeepLearningFeatureExtraction#Tumor"]["mae3d_lung_CT"]["root"] = task_settings["DeepLearningFeatureExtraction#Tumor"]["mae3d_lung_CT"]["root"].format(ROOT=ROOT)
    for type_ in ["peritumor#3#lung", "peritumor#5#lung", "peritumor#7#lung", "tumor#lung"]:
        task_settings["Normalization#01"]["pickle"][type_] = task_settings["Normalization#01"]["pickle"][type_].format(PKL_DIR=PKL_DIR)
        task_settings["SubRegionClustering"]["predict"]["pickle"][type_] = task_settings["SubRegionClustering"]["predict"]["pickle"][type_].format(PKL_DIR=PKL_DIR)
        task_settings["TumorHeterogeneity"]["predict"]["pickle_dir"][type_] = task_settings["TumorHeterogeneity"]["predict"]["pickle_dir"][type_].format(PKL_DIR=PKL_DIR)
    return task_settings, output_dir

def run(pre_image_path, pre_mask_path, post_image_path, post_mask_path):
    task_settings, output_dir = prepare_task(pre_image_path, pre_mask_path, post_image_path, post_mask_path)

    """1. HI"""
    load_state = dict(Step="LoadData", Datasets=copy.deepcopy(task_settings["Datasets"]))
    print("percentage@5", flush=True)


    prep_state = medi_ai.call.run_step("Preprocessing", task_settings, load_state, output_dir)
    prep_crop_state = medi_ai.call.run_step("Cropper#Tumor", task_settings, prep_state, output_dir)
    print("percentage@10", flush=True)

    # need to get the background intensity
    df = pd.read_csv(prep_state["Datasets"]["lung"]["path"])
    df.set_index("ID", inplace=True)
    data_dir = Path(prep_state["Datasets"]["lung"]["path"]).parent
    pre_bg_intensity = sitk.GetArrayFromImage(sitk.ReadImage(str(data_dir / df.loc["pre_X", "image_path"]))).min()
    post_bg_intensity = sitk.GetArrayFromImage(sitk.ReadImage(str(data_dir / df.loc["post_X", "image_path"]))).min()

    varm_state = medi_ai.call.run_step("VariousMask", task_settings, prep_crop_state, output_dir)
    print("percentage@15", flush=True)
    seg_state = medi_ai.call.run_step("PreSegmentation", task_settings, varm_state, output_dir)
    for type_ in seg_state["Datasets"].keys():
        df = pd.read_csv(seg_state["Datasets"][type_]["path"])
        # print(df.columns)
        df.set_index("ID", inplace=True, drop=False)
        data_dir = Path(seg_state["Datasets"][type_]["path"]).parent
        pre_idx = df.query("ID == 'pre_X'").index
        post_idx = df.query("ID == 'post_X'").index
        df.loc[pre_idx, "background_intensity"] = pre_bg_intensity
        df.loc[post_idx, "background_intensity"] = post_bg_intensity
        df.to_csv(seg_state["Datasets"][type_]["path"], index=False)

    # from IPython import embed; embed()
    print("percentage@20", flush=True)
    sv_fe_state = medi_ai.call.run_step("FeatureExtraction#SV", task_settings, seg_state, output_dir)
    print("percentage@25", flush=True)
    sv_dl_state = medi_ai.call.run_step("DeepLearningFeatureExtraction#SV", task_settings, seg_state, output_dir)
    print("percentage@30", flush=True)
    sv_ff_state = medi_ai.call.run_step(
        "FeatureFusion#SV-Rad+DL", 
        task_settings, 
        dict(Datasets=pd.DataFrame([sv_fe_state, sv_dl_state])["Datasets"].values.tolist()), 
        output_dir)
    print("percentage@35", flush=True)
    sv_ff_norm_state = medi_ai.call.run_step("Normalization#01", task_settings, sv_ff_state, output_dir)
    print("percentage@40", flush=True)
    sv_ff_norm_state["image_src_dir"] = seg_state["Step"]
    sr_state = medi_ai.call.run_step("SubRegionClustering", task_settings, sv_ff_norm_state, output_dir)
    print("percentage@45", flush=True)

    for type_ in sr_state["Datasets"].keys():
        df = pd.read_csv(sr_state["Datasets"][type_]["path"])
        df = df[["ID", "original_image_path", "original_mask_path"]].drop_duplicates()
        df.set_index("ID", inplace=True)

        type_dir = Path(sr_state["Datasets"][type_]["path"]).parent

        type_ = type_.replace("#lung", "")

        image_path = type_dir / df.loc["pre_X", "original_image_path"]
        mask_path = type_dir / df.loc["pre_X", "original_mask_path"]
        print(f"callback@pre_{type_}_image_path@{image_path}", flush=True)
        print(f"callback@pre_{type_}_mask_path@{mask_path}", flush=True)

        image_path = type_dir / df.loc["post_X", "original_image_path"]
        mask_path = type_dir / df.loc["post_X", "original_mask_path"]
        print(f"callback@post_{type_}_image_path@{image_path}", flush=True)
        print(f"callback@post_{type_}_mask_path@{mask_path}", flush=True)
    # import IPython; IPython.embed()

    sr_fe_state = medi_ai.call.run_step("FeatureExtraction#SR", task_settings, sr_state, output_dir)
    print("percentage@50", flush=True)
    sr_dl_state = medi_ai.call.run_step("DeepLearningFeatureExtraction#SR", task_settings, sr_state, output_dir)
    print("percentage@55", flush=True)
    sr_ff_state = medi_ai.call.run_step(
        "FeatureFusion#SR-Rad+DL", 
        task_settings, 
        dict(Datasets=pd.DataFrame([sr_fe_state, sr_dl_state])["Datasets"].values.tolist()), 
        output_dir)
    print("percentage@60", flush=True)
    th_state = medi_ai.call.run_step("TumorHeterogeneity", task_settings, sr_ff_state, output_dir)
    print("percentage@65", flush=True)
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

    """2. Rad"""
    tu_fe_state = medi_ai.call.run_step("FeatureExtraction#Tumor", task_settings, load_state, output_dir)
    print("percentage@70", flush=True)

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
    rad_df

    rad_df.to_csv(f"{output_dir}/FinalFeature/Rad.csv", index=False)

    """3. DL"""
    tu_crop_state = medi_ai.call.run_step("Cropper", task_settings, load_state, output_dir) # load_state is better, because I'm not sure that prep_crop_state is big enough
    print("percentage@75", flush=True)
    tu_dl_state = medi_ai.call.run_step("DeepLearningFeatureExtraction#Tumor", task_settings, tu_crop_state, output_dir)
    print("percentage@80", flush=True)
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
    dl_768_df
    print("percentage@85", flush=True)

    import pickle
    dl_pca_df = []
    for phase in ["pre", "post", "delta"]:
        df = dl_768_df.query(f"`phase` == '{phase}'").copy().reset_index(drop=True)
        cols = df.columns[df.columns.get_loc("#EC#")+1:]
        feat = df.loc[:, cols].values
        scaler = pickle.load(open(f"{PKL_DIR}/DeepLearningFeatureExtraction#Tumor/dl_{phase}_norm.pkl", "rb"))
        feat = scaler.transform(feat)
        pca = pickle.load(open(f"{PKL_DIR}/DeepLearningFeatureExtraction#Tumor/dl_{phase}_PCA.pkl", "rb"))
        feat_pca = pca.transform(feat)
        pca_df = pd.DataFrame(feat_pca)
        pca_df.rename(columns={i: f"DL_PCA_{i+1}" for i in range(pca_df.shape[-1])}, inplace=True)
        df = pd.concat([df, pca_df], axis=1)
        df.drop(columns=cols, inplace=True)
        dl_pca_df.append(df)
    dl_pca_df = pd.concat(dl_pca_df, axis=0).reset_index(drop=True)
    dl_pca_df.to_csv(f"{output_dir}/FinalFeature/DL_PCA.csv", index=False)
    dl_pca_df
    print("percentage@90", flush=True)
    fs_cols = pickle.load(open(f"{PKL_DIR}/Feature/feature_selection.pkl", "rb"))
    feat_cols = pickle.load(open(f"{PKL_DIR}/Feature/feature_merge.pkl", "rb"))


    """4. LC-NICER"""
    all_feat = {}

    # Rad
    df = pd.read_csv(f"{output_dir}/FinalFeature/Rad.csv")
    for phase in ["pre", "post", "delta"]:
        phase_df = df.query(f"`phase` == '{phase}'").copy().reset_index(drop=True)
        all_feat[(phase, "Rad")] = feature_selection(phase_df, feat_cols[(phase, "Rad")])

    # DL
    df = pd.read_csv(f"{output_dir}/FinalFeature/DL_PCA.csv")
    for phase in ["pre", "post", "delta"]:
        phase_df = df.query(f"`phase` == '{phase}'").copy().reset_index(drop=True)
        all_feat[(phase, "DL")] = feature_selection(phase_df, feat_cols[(phase, "DL")])

    # HI
    for phase in ["pre", "post", "delta"]:
        df = []
        first = True
        for type_ in ["HI_peritumor3", "HI_peritumor5", "HI_peritumor7", "HI_tumor"]:
            phase_df = pd.read_csv(f"{output_dir}/FinalFeature/{type_}.csv").query(f"`phase` == '{phase}'").copy().reset_index(drop=True)
            cols = phase_df.columns[phase_df.columns.get_loc("#EC#")+1:].tolist()

            if first:
                phase_df = phase_df[["ID", "PID", "#EC#"] + cols].copy()
                first = False
            else:
                phase_df = phase_df[cols].copy() 

            phase_df.rename(columns={col: f"{phase}#{type_}#{col}" for col in cols}, inplace=True)
            df.append(phase_df)
        df = pd.concat(df, axis=1).reset_index(drop=True)
        all_feat[(phase, "HI")] = feature_selection(df, feat_cols[(phase, "HI")])

    all_score_dfs = {}
    for phase in ["pre", "post", "delta"]:
        print(f"{phase}:")
        for type_ in ["Rad", "DL", "HI"]:
            df = all_feat[(phase, type_)].copy()
            feat = df.loc[:, df.columns[df.columns.get_loc("#EC#")+1:]]
            scaler = pickle.load(open(f"{PKL_DIR}/LC-NICER/{phase}_{type_}/norm.pkl", "rb"))
            model = pickle.load(open(f"{PKL_DIR}/LC-NICER/{phase}_{type_}/model.pkl", "rb"))
            score = model.predict_proba(scaler.transform(feat))[:, 1]
            # all_scores[(phase, type_)] = model.predict_proba(scaler.transform(feat))[:, 1]

            sdf = df.copy()
            sdf.drop(columns=feat.columns.tolist() + ["#EC#"], inplace=True)
            sdf["Probability"] = score
            all_score_dfs[(phase, type_)] = sdf
            print(f"\t{type_}:", feat.shape[1], "→", type(model))
            del df, feat, scaler, model, score

    print("percentage@95", flush=True)
    for phase in ["pre", "post", "delta"]:
        scaler = pickle.load(open(f"{PKL_DIR}/LC-NICER/{phase}/norm.pkl", "rb"))
        model = pickle.load(open(f"{PKL_DIR}/LC-NICER/{phase}/model.pkl", "rb"))

        feat = None
        for type_ in ["Rad", "DL", "HI"]:
            sdf = all_score_dfs[(phase, type_)].copy()
            sdf.set_index("PID", inplace=True)
            sdf.rename(columns={"Probability": type_}, inplace=True)
            if feat is None:
                feat = sdf.copy()
            else:
                feat.loc[sdf.index, type_] = sdf.loc[:, type_].copy()
            del sdf
        score = model.predict_proba(scaler.transform(feat[["Rad", "DL", "HI"]].values))[:, 1]
        sdf = feat.copy()
        sdf["Probability"] = score
        all_score_dfs[phase] = sdf
        print(f"{phase}:\n\t", type(model))


    scaler = pickle.load(open(f"{PKL_DIR}/LC-NICER/final/norm.pkl", "rb"))
    model = pickle.load(open(f"{PKL_DIR}/LC-NICER/final/model.pkl", "rb"))

    feat = None
    for phase in ["pre", "post", "delta"]:
        sdf = all_score_dfs[phase].copy()
        sdf.rename(columns={"Probability": phase}, inplace=True)
        if feat is None:
            feat = sdf.copy()
        else:
            feat.loc[sdf.index, phase] = sdf.loc[:, phase].copy()
        del sdf
    score = model.predict_proba(scaler.transform(feat[["pre", "post", "delta"]].values))[:, 1]
    sdf = feat.copy()
    sdf["final"] = score
    all_score_dfs["final"] = sdf

    print("percentage@100", flush=True)
    print(f"final:\n\t", type(model))
    print(all_score_dfs["final"].to_markdown())

    ret_df = pd.concat([all_score_dfs["pre"], all_score_dfs["post"], all_score_dfs["delta"]], axis=0, ignore_index=True)
    ret_df.insert(0, "Phase", ret_df["ID"].apply(lambda x: x.split("_")[0]))
    ret_df.drop(columns=["ID"], inplace=True)

    ret_df.rename(columns={"Phase": "Mode", "Probability": "Total"}, inplace=True)
    ret_df["Mode"] = ret_df["Mode"].apply(lambda x: x.capitalize())
    for type_ in ["Rad", "DL", "HI", "Total"]:
        ret_df[type_] = ret_df[type_].apply(lambda x: f"{x:.3f}")

    # import IPython; IPython.embed()
    return {
        "Probability": all_score_dfs["final"].loc["X", "final"],
        "DataFrame": ret_df,
    }

if __name__ == "__main__":
    ret = run(
        pre_image_path="/home/wzt/src/LC-NICER/examples/test/image/DT60508.nii.gz",
        pre_mask_path="/home/wzt/src/LC-NICER/examples/test/mask/DT60508.nii.gz",
        post_image_path="/home/wzt/src/LC-NICER/examples/test/image/T5179839.nii.gz",
        post_mask_path="/home/wzt/src/LC-NICER/examples/test/mask/T5179839.nii.gz",
    )
    print(ret)