from pathlib import Path
import torch
import pandas as pd
import os
from omegaconf import OmegaConf
import SimpleITK as sitk
from tqdm import tqdm
import monai
from monai.transforms.transform import Transform, MapTransform, RandomizableTransform
import traceback

from lib.models import vit_3d_base_patchsize8


class PathJoin(Transform):
    """
    Transform to join the image path with the root path.
    """
    def __init__(self, base_dir, name_key, path_key):
        super().__init__()
        self.name_key = name_key
        self.base_dir = base_dir
        self.path_key = path_key

    def __call__(self, data: dict):
        data[self.path_key] = os.path.join(self.base_dir, data[self.name_key])
        return data

def run(src_csv_path, base_dir, dst_csv_path):
    model = vit_3d_base_patchsize8(img_size=48, in_chans=1, num_classes=2)
    checkpoint = torch.load("./lung_CT_checkpoint.pth.tar", map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    linear_keyword = 'head'
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder.') and not k.startswith('module.base_encoder.%s' % linear_keyword):
            # remove prefix
            state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k] 
        if k.startswith('base_encoder.') and not k.startswith('base_encoder.%s' % linear_keyword):
            # remove prefix
            state_dict[k[len("base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k] 
        if k.startswith('encoder.') and not k.startswith('encoder.%s' % linear_keyword):
            # remove prefix
            state_dict[k[len("encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k] 
        if k == 'encoder_pos_embed':
            pe = torch.zeros([1, 1, state_dict[k].size(-1)])
            state_dict['pos_embed'] = torch.cat([pe, state_dict[k]], dim=1)
            del state_dict[k]
        if k == 'patch_embed.proj.weight' and \
            state_dict['patch_embed.proj.weight'].shape != model.patch_embed.proj.weight.shape:
            del state_dict['patch_embed.proj.weight']
            del state_dict['patch_embed.proj.bias']
        if k == 'pos_embed' and \
            state_dict['pos_embed'].shape != model.pos_embed.shape:
            del state_dict[k]
        if k in state_dict:
            del state_dict[k]
         

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    model.forward = model.forward_features

    # auto detect gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = model.to(device)
    model.eval()

    transform = monai.transforms.Compose([
        # 1. load the image file
        PathJoin(base_dir, "image_path", "#image#"),
        monai.transforms.LoadImaged(keys=["#image#"]),
        monai.transforms.EnsureChannelFirstd(keys=["#image#"], channel_dim="no_channel"),
        monai.transforms.ScaleIntensityRangeD(keys=["#image#"], a_min=-1024, a_max=3072, b_min=0., b_max=1., clip=True), 
        monai.transforms.ToTensord(keys=["#image#"], track_meta=False),
        # 2. scale the image data
        # monai.transforms.ResizeWithPadOrCropd(keys=["#image#"], spatial_size=[48, 48, 48], mode="minimum"),
        monai.transforms.Resized(keys=["#image#"], spatial_size=[48, 48, 48]), 
    ])
    ds = monai.data.CSVDataset(src=src_csv_path, transform=transform)
    feat_df = []
    err_df = []
    for i in tqdm(range(len(ds))):
    # for i, row in tqdm(enumerate(ds)):
        try:
            row = ds[i]
            image = row["#image#"][None] # add batch dim
            image = image.to(device)
            with torch.no_grad():
                feat = model(image)
                feat = feat[:, 0]
                feat_df.append(feat.cpu().numpy().flatten())
            err_df.append({"Error": False, "Msg": ""})
        except Exception as e:
            feat_df.append({})
            err_df.append({"Error": True, "Msg": str(e) + "\n" + traceback.format_exc()})
            print(traceback.format_exc())
            
    feat_df = pd.DataFrame(feat_df)
    feat_df.rename(columns={i: f"DL_{i + 1}" for i in range(len(feat_df.columns))}, inplace=True)
    feat_df.insert(0, "#EC#", None)

    err_df = pd.DataFrame(err_df)

    df = pd.read_csv(src_csv_path)
    if "#EC#" in df.columns:
        df = df.drop(columns=["#EC#"])
    
    new_df = pd.concat([df, err_df, feat_df], axis=1)
    Path(dst_csv_path).parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(dst_csv_path, index=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_csv_path", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--dst_csv_path", type=str, required=True)

    args = parser.parse_args()
    run(args.src_csv_path, args.base_dir, args.dst_csv_path)