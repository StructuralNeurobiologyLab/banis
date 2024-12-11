import time
import os
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
import argparse
import torch
from datetime import datetime

from BANIS import BANIS


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)

    ckpt_path = os.path.join(args.save_path, args.exp_name, "default", "checkpoints", "last.ckpt")
    print(f"ckpt path: {ckpt_path}")
    if os.path.exists(ckpt_path):
        model = BANIS.load_from_checkpoint(ckpt_path)
        model = model.to("cuda")
        trainer = pl.Trainer(logger=tb_logger, accelerator="gpu", devices=-1)
        model.trainer = trainer  # for the logger

        # global step of model loaded from checkpoint is 0 by default, until trainer is started
        # see https://github.com/Lightning-AI/pytorch-lightning/issues/12819
        checkpoint = torch.load(ckpt_path)
        model.full_cube_inference("val", checkpoint["global_step"])



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--base_data_path", type=str, default="/cajal/nvmescratch/projects/NISB/", help="Base path for the dataset.")
    parser.add_argument("--data_setting", type=str, default="base", help="Data setting identifier.")
    parser.add_argument("--eval_ranges", type=float, nargs="+", default=torch.sigmoid(torch.tensor(list(range(-1, 12))).double() * 0.2).numpy().round(4).tolist(), help="List of evaluation thresholds.")
    parser.add_argument("--save_path", type=str, default="/cajal/scratch/projects/misc/riegerfr/aff_nis/", help="Path to save the model and logs.")
    parser.add_argument("--small_size", type=int, default=128, help="Size of the patches.")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name (if empty, will be filled automatically).")

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    main()
