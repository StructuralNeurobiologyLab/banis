import os

import numpy as np
import torch
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import zarr

from BANIS import BANIS, parse_args

from data import comp_affinities, load_data


def train_model_with_samples(model_path, dataloader):
    model = BANIS.load_from_checkpoint(model_path)

    trainer = Trainer(
        max_steps=1000,
        accelerator="gpu",
        devices=1,
    )
    trainer.fit(model, dataloader)


def compare_model_weights(path_a, path_b):
    model_a = BANIS.load_from_checkpoint(path_a)
    model_b = BANIS.load_from_checkpoint(path_b)

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    diffs = {}
    total_diff = 0.0
    total_norm = 0.0
    total_cos_sim = 0.0
    biggest_diff = 0
    nonfinite_a = 0
    nonfinite_b = 0
    total_params = 0

    n_layers = 0

    for key in tqdm(state_a):
        #print(f"getting {key}")
        param_a = state_a[key].flatten()
        param_b = state_b[key].flatten()

        diff = torch.abs(param_a - param_b).mean().item()
        if diff > biggest_diff:
            biggest_diff = diff
        norm = torch.norm(param_a - param_b).item()
        cos_sim = torch.nn.functional.cosine_similarity(param_a.unsqueeze(0), param_b.unsqueeze(0)).item()

        diffs[key] = {
            'mean_abs_diff': diff,
            'l2_norm': norm,
            'cosine_similarity': cos_sim,
        }

        total = param_a.numel()
        nonfinite_params_a = (~torch.isfinite(param_a)).sum().item()
        nonfinite_params_b = (~torch.isfinite(param_b)).sum().item()
        total_params += total
        nonfinite_a += nonfinite_params_a
        nonfinite_b += nonfinite_params_b

        total_diff += diff
        total_norm += norm
        total_cos_sim += cos_sim
        n_layers += 1

    avg_diff = total_diff / n_layers
    avg_norm = total_norm / n_layers
    avg_cos_sim = total_cos_sim / n_layers

    return {
        'avg_mean_abs_diff': avg_diff,
        'avg_l2_norm': avg_norm,
        'avg_cosine_similarity': avg_cos_sim,
        'biggest_diff': biggest_diff,
        'nonfinite_a': nonfinite_a,
        'nonfinite_b': nonfinite_b,
        'total_params': total_params,
        #'layerwise': diffs
    }

    #print(compare_model_weights(
    #    "/cajal/scratch/projects/misc/zuzur/ss3/debug1GPU-seed0-batch_size1-small_size128/default/checkpoints/epoch=0-step=110000.ckpt",
    #    "/cajal/scratch/projects/misc/zuzur/ss3/debug1GPU-seed0-batch_size1-small_size128/default/checkpoints/epoch=0-step=115000.ckpt"
    #        ))
    ## {'avg_mean_abs_diff': nan, 'avg_l2_norm': nan, 'avg_cosine_similarity': nan, 'biggest_diff': 0, 'nonfinite_a': 0, 'nonfinite_b': 62993031, 'total_params': 62993031}
    #print(compare_model_weights(
    #    "/cajal/scratch/projects/misc/zuzur/ss3/debug1GPU-seed0-batch_size1-small_size128/default/checkpoints/epoch=0-step=100000.ckpt",
    #    "/cajal/scratch/projects/misc/zuzur/ss3/debug1GPU-seed0-batch_size1-small_size128/default/checkpoints/epoch=0-step=105000.ckpt"
    #        ))
    ## {'avg_mean_abs_diff': 0.017999131043465913, 'avg_l2_norm': 2.9829090611515583, 'avg_cosine_similarity': 0.9655602666255757, 'biggest_diff': 0.06504751741886139}
    #print(compare_model_weights(
    #    "/cajal/scratch/projects/misc/zuzur/ss3/debug1GPU-seed0-batch_size1-small_size128/default/checkpoints/epoch=0-step=105000.ckpt",
    #    "/cajal/scratch/projects/misc/zuzur/ss3/debug1GPU-seed0-batch_size1-small_size128/default/checkpoints/epoch=0-step=110000.ckpt"
    #        ))
    ## {'avg_mean_abs_diff': 0.01891954702438203, 'avg_l2_norm': 3.084119017241339, 'avg_cosine_similarity': 0.9619980035223629, 'biggest_diff': 0.09889261424541473}
    #print(compare_model_weights(
    #    "/cajal/scratch/projects/misc/zuzur/ss3/debug1GPU-seed0-batch_size1-small_size128/default/checkpoints/epoch=0-step=90000.ckpt",
    #    "/cajal/scratch/projects/misc/zuzur/ss3/debug1GPU-seed0-batch_size1-small_size128/default/checkpoints/epoch=0-step=110000.ckpt"
    #        ))
    ## {'avg_mean_abs_diff': 0.03352864947696027, 'avg_l2_norm': 5.531613261509161, 'avg_cosine_similarity': 0.9207515456647084, 'biggest_diff': 0.1563815325498581}


def prepare_good_samples():
    args = parse_args()
    train_data, val_data, n_channels = load_data(args)
    return train_data


class SimpleDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "img": torch.from_numpy(sample["img"]),
            "aff": torch.from_numpy(sample["aff"]),
            "seg": torch.from_numpy(sample["seg"]),
        }

def prepare_bad_samples():
    samples = []
    runs_root = "/cajal/scratch/projects/misc/zuzur/ss3/"
    for run in os.listdir(runs_root):
        if run.startswith("debug1GPU-seed"):
            for candidate in os.listdir(os.path.join(runs_root, run)):
                if candidate.endswith("0_img.zarr"):
                    img = zarr.open(os.path.join(runs_root, run, candidate))
                    seg_name = candidate.replace("img", "seg")
                    seg = zarr.open(os.path.join(runs_root, run, seg_name))
                    aff, _ = comp_affinities(seg[:])
                    data = {
                        "img": img.astype(np.float16),
                        "seg": seg,
                        "aff": aff,
                    }
                    samples.append(data)
                    if len(samples) >= 1000:
                        return SimpleDataset(samples)
    return SimpleDataset(samples)


if __name__ == "__main__":
    good_model_path = "/cajal/scratch/projects/misc/zuzur/ss3/debug1GPU-seed0-batch_size1-small_size128/default/checkpoints/epoch=0-step=110000.ckpt"
    bad_model_path = "/cajal/scratch/projects/misc/zuzur/ss3/debug1GPU-seed0-batch_size1-small_size128/default/checkpoints/epoch=0-step=115000.ckpt"
    early_model_path = "/cajal/scratch/projects/misc/zuzur/ss3/debug1GPU-seed0-batch_size1-small_size128/default/checkpoints/epoch=0-step=50000.ckpt"

    bad_samples = DataLoader(prepare_bad_samples(), batch_size=1, num_workers=8, shuffle=True, drop_last=True)
    good_samples = DataLoader(prepare_good_samples(), batch_size=1, num_workers=8, shuffle=True, drop_last=True)

    train_model_with_samples(good_model_path, good_samples)
    train_model_with_samples(good_model_path, bad_samples)
