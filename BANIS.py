import argparse
import gc
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import zarr
from nnunet_mednext import create_mednext_v1
from pytorch_lightning import LightningModule, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import load_data
from inference import scale_sigmoid, patched_inference, compute_connected_component_segmentation
from metrics import compute_metrics


class BANIS(LightningModule):
    """
    PyTorch Lightning module for BANIS: Baseline for Affinity-based Neuron Instance Segmentation
    """

    def __init__(self, **kwargs: Any):
        super().__init__()
        self.save_hyperparameters()
        print(f"hparams: \n{self.hparams}")

        self.model = create_mednext_v1(
            num_input_channels=self.hparams.num_input_channels,
            num_classes=6,  # 3 short + 3 long range affinities
            model_id=self.hparams.model_id,
            kernel_size=self.hparams.kernel_size,
        )
        self.model.outside_block_checkpointing = True  # Save GPU memory

        if self.hparams.compile:
            self.model = torch.compile(self.model)

        self.best_nerl_so_far = defaultdict(float)  # for train/val/test
        self.best_thr_so_far = defaultdict(float)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["best_thr_so_far"] = self.best_thr_so_far
        checkpoint["best_nerl_so_far"] = self.best_nerl_so_far

    def on_load_checkpoint(self, checkpoint):
        self.best_thr_so_far = checkpoint.get("best_thr_so_far", defaultdict(float))
        self.best_nerl_so_far = checkpoint.get("best_nerl_so_far", defaultdict(float))

    def on_fit_start(self):
        self.logger.experiment.add_text("hparams", str(self.hparams))

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        if self.hparams.scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.n_steps)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        return optimizer

    def on_train_epoch_start(self):
        self.plotted = False

    def on_validation_epoch_start(self):
        self.plotted = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, data: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(data, "train")

    def validation_step(self, data: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(data, "val")

    def _step(self, data: Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
        self.log_input(data["img"])
        self.log_weight_stats()
        pred = self(data["img"])
        target = data["aff"].half()
        loss_mask = data["aff"] >= 0
        loss = binary_cross_entropy_with_logits(pred[loss_mask], target[loss_mask])
        self.log(f"{mode}_loss", loss)
        if not self.plotted:
            self._log_images(data, pred, mode)
            self.plotted = True
        return loss

    def _log_images(self, data: Dict[str, torch.Tensor], pred: torch.Tensor, mode: str):
        middle = data["img"].shape[2] // 2
        self._add_image(f"{mode}_img", data["img"][:, :3, middle])
        self._add_image(f"{mode}_aff", data["aff"][:, :3, middle])
        self._add_image(f"{mode}_aff_pred", scale_sigmoid(pred[:, :3, middle]))
        self._add_image(f"{mode}_lr_aff", data["aff"][:, 3:, middle])
        self._add_image(f"{mode}_lr_aff_pred", scale_sigmoid(pred[:, 3:6, middle]))

        seg_middle = data["seg"][:, middle]
        colormap = torch.rand(seg_middle.max() + 1, 3)
        colormap[0] = 0
        colored_seg = colormap[seg_middle.cpu()].permute(0, 3, 1, 2)
        self._add_image(f"{mode}_seg", colored_seg)

    def _add_image(self, tag: str, img: torch.Tensor) -> None:
        self.logger.experiment.add_image(tag, torchvision.utils.make_grid(img, value_range=(0, 1)),
                                         global_step=self.global_step)

    def on_validation_epoch_end(self):
        if self.hparams.validate_extern:
            if self.trainer.is_global_zero:
                def format_value(value):
                    if isinstance(value, bool):
                        return str(value).lower()  # Convert booleans to lowercase strings (true/false)
                    elif isinstance(value, list):
                        return ' '.join(map(str, value))  # Convert list to a space-separated string
                    elif value is None:
                        return ''  # Skip None values
                    else:
                        return str(value)  # Convert other types to string

                args_list = [f"--{key} {format_value(value)}" for key, value in self.hparams.items()]
                args = ' '.join(args_list)

                command = f"sbatch --job-name {self.hparams.exp_name}_val --output {self.hparams.save_dir}/slurm-validation-log.txt validation_watcher.sh {args}"
                os.system(command)
                print(f"running validation: {command}")

        else:
            self.full_cube_inference("val")

    def on_train_end(self):
        assert self.best_nerl_so_far["val"] > 0, "No best NERL found in validation"
        self.eval()
        print(f"device {next(self.parameters()).device}")
        self.cuda()
        self.full_cube_inference("test")
        self.full_cube_inference("train")

    @torch.no_grad()
    def full_cube_inference(self, mode: str, global_step=None):
        """Perform full cube inference. Expensive!

        Args:
            mode: Either "train", "val", or "test".
        """
        assert mode in ["train", "val", "test"], f"Invalid mode: {mode}"
        print(f"Full cube inference for {mode}")

        base_path_mode = os.path.join(self.hparams.base_data_path, self.hparams.data_setting, mode)
        seeds_path_mode = sorted([f for f in os.listdir(base_path_mode) if f.startswith("seed")])
        assert len(seeds_path_mode) >= 1, f"No seeds found in {base_path_mode}"
        seed_path = os.path.join(base_path_mode, seeds_path_mode[0])

        img_data = zarr.open(os.path.join(seed_path, "data.zarr"), mode="r")["img"]

        aff_pred = patched_inference(img_data, model=self, do_overlap=True, prediction_channels=3, divide=255,
                                     small_size=self.hparams.small_size)

        aff_pred = zarr.array(aff_pred, dtype=np.float16, store=f"{self.hparams.save_dir}/pred_aff_{mode}.zarr",
                              chunks=(3, 512, 512, 512), overwrite=True)

        self._evaluate_thresholds(aff_pred, os.path.join(seed_path, "skeleton.pkl"), mode, global_step)

    def _evaluate_thresholds(self, aff_pred: zarr.Array, skel_path: str, mode: str, global_step=None):
        best_voi = best_voi_no_merge = 1e100
        best_nerl = best_nerl_no_merge = -1
        best_nerl_metrics = None
        thresholds = self.hparams.eval_ranges if mode != "test" else [self.best_thr_so_far["val"]]

        for thr in tqdm(thresholds):
            gc.collect()
            torch.cuda.empty_cache()
            print(f"threshold {thr}")

            pred_seg = compute_connected_component_segmentation(
                aff_pred[:3] > thr  # hard affinities
            )

            metrics = compute_metrics(pred_seg, skel_path)

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.safe_add_scalar(f"{mode}_{k}_thr_{thr}", v, global_step)

            if metrics["n_non0_mergers"] == 0:
                best_nerl_no_merge = max(best_nerl_no_merge, metrics["nerl"])
                best_voi_no_merge = min(best_voi_no_merge, metrics["voi_sum"])

            if metrics["nerl"] > best_nerl:
                best_nerl = metrics["nerl"]
                best_nerl_metrics = metrics
                if self.best_nerl_so_far[mode] < best_nerl:
                    self.best_nerl_so_far[mode] = best_nerl
                    self.best_thr_so_far[mode] = thr
                    with open(f"{self.hparams.save_dir}/best_thr_{mode}.txt", "w") as f:
                        f.write(str(self.best_thr_so_far[mode]))
                    seg_pred = zarr.array(pred_seg, dtype=np.uint32,
                                          store=f"{self.hparams.save_dir}/pred_seg_{mode}.zarr",
                                          chunks=(512, 512, 512), overwrite=True)
            best_voi = min(best_voi, metrics["voi_sum"])

        self.safe_add_scalar(f"{mode}_best_nerl", best_nerl, global_step)
        self.safe_add_scalar(f"{mode}_best_voi", best_voi, global_step)
        self.safe_add_scalar(f"{mode}_best_nerl_no_merge", best_nerl_no_merge, global_step)
        self.safe_add_scalar(f"{mode}_best_voi_no_merge", best_voi_no_merge, global_step)

        for k, v in best_nerl_metrics.items():
            if isinstance(v, (int, float)):
                self.safe_add_scalar(f"{mode}_best_nerl_{k}", v, global_step)

    def safe_add_scalar(self, name: str, value: float, global_step=None) -> None:
        try:  # s.t. full_cube_inference can be called outside of .fit() without error
            self.logger.experiment.add_scalar(name, value, self.global_step if global_step is None else global_step)
        except Exception as e:
            print(f"Error logging {name}: {e}")

    def log_input(self, input):
        self.log_dict({
                f"input/min": input.min(),
                f"input/max": input.max(),
                f"input/mean": input.mean(),
                f"input/std": input.std(),
        })

    def register_activation_hooks(self):
        for name, module in self.named_modules():
            def hook_fn(module, input, output, block_name=name):  # capture name in default arg
                if not self.training:  # don't log during validation
                    return
                self.log_dict({
                    f"activations/{block_name}_min": output.min(),
                    f"activations/{block_name}_max": output.max(),
                    f"activations/{block_name}_mean": output.mean(),
                    f"activations/{block_name}_std": output.std(),
                })
                if torch.isnan(output).any():
                    print(f"NaN in output of {block_name}")
            module.register_forward_hook(hook_fn)

    def setup(self, stage: str):
        if stage == 'fit':
            self.register_activation_hooks()

    def log_weight_stats(self):
        for name, param in self.named_parameters():
            self.log_dict({
                f"weights/{name}_min": param.data.min(),
                f"weights/{name}_max": param.data.max(),
                f"weights/{name}_mean": param.data.mean(),
                f"weights/{name}_std": param.data.std(),
            })

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        total_norm_before = torch.norm(torch.stack([p.grad.norm(2) for p in self.parameters() if p.grad is not None]))
        self.log("gradients/total_norm", total_norm_before.item())
        max_grad_before = max([p.grad.abs().max().item() for p in self.parameters() if p.grad is not None])
        self.log("gradients/max_grad", max_grad_before)

        self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm)

        total_norm_after = torch.norm(torch.stack([p.grad.norm(2) for p in self.parameters() if p.grad is not None]))
        self.log("gradients/total_norm_clipped", total_norm_after.item(), on_step=True)
        max_grad_after = max([p.grad.abs().max().item() for p in self.parameters() if p.grad is not None])
        self.log("gradients/max_grad_clipped", max_grad_after)


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)

    torch.set_float32_matmul_precision("medium")

    exp_name = (
            datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")
            + f"ds_{args.data_setting}"
              f"_lrng{args.long_range}_s{args.seed}_b{args.batch_size}_m{args.model_id}_k{args.kernel_size}_"
              f"lr{args.learning_rate}_wd{args.weight_decay}_sch{args.scheduler}_syn_{args.synthetic}"
              f"_drop{args.drop_slice_prob}_shift{args.shift_slice_prob}_int{args.intensity_aug}_noise{args.noise_scale}"
              f"_affine{args.affine}_ns{args.n_steps}_ss{args.small_size}"
    ) if not args.exp_name else args.exp_name

    save_dir = os.path.join(args.save_path, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"save dir: {save_dir}")
    tb_logger = TensorBoardLogger(
        save_dir=args.save_path,
        name=exp_name,
        version="default",
    )
    tb_logger.experiment.add_text("save dir", save_dir)

    model_checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_last=True,
        mode="min",
        save_top_k=100,
        verbose=True,
        save_on_train_epoch_end=False  # automatically runs at the end of the validation
    )
    trainer = pl.Trainer(
        callbacks=[
            DeviceStatsMonitor(),
            model_checkpoint_callback,
            LearningRateMonitor(
                logging_interval='step'
            ),
        ],
        logger=tb_logger,
        max_steps=args.n_steps,
        accelerator="gpu",
        devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=False) if args.distributed else "auto",
        num_nodes=int(os.environ["SLURM_NNODES"]) if args.distributed else 1,
        log_every_n_steps=args.log_every_n_steps,
        limit_val_batches=100,
        precision="16-mixed",
        profiler="simple",
        default_root_dir=save_dir,
        val_check_interval=args.val_check_interval,  # validation full cube inference expensive so less frequent
        check_val_every_n_epoch=None,
        num_sanity_val_steps=args.n_debug_steps,
        gradient_clip_val=1.0,
    )
    print(f"Checkpoints will be saved in: {trainer.default_root_dir}/checkpoints")

    train_data, val_data, n_channels = load_data(args)
    args.save_dir = save_dir
    args.num_input_channels = n_channels

    if os.path.exists(args.model_from_checkpoint):
        print(f"Loading model from checkpoint: {args.model_from_checkpoint}")
        model = BANIS(**vars(args))
        checkpoint = torch.load(args.model_from_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        model.hparams.update(vars(args))
    else:
        model = BANIS(**vars(args))

    trainer.fit(
        model=model,
        train_dataloaders=DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, drop_last=True),
        val_dataloaders=DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers),
        ckpt_path="last" if args.resume_from_last_checkpoint else None
    )

    print("Training complete")


def parse_args():
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name (if empty, will be filled automatically).")
    parser.add_argument("--save_path", type=str, default="/cajal/scratch/projects/misc/riegerfr/aff_nis/", help="Path to save the model and logs.")

    # Training arguments
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--n_steps", type=int, default=20_000, help="Number of training steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for the optimizer.")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers for data loading.")
    parser.add_argument("--scheduler", action=argparse.BooleanOptionalAction, default=True, help="Whether to use a learning rate scheduler.")
    parser.add_argument("--devices", type=int, default=-1, help="Number GPU devices to use (-1: all).")
    parser.add_argument("--n_debug_steps", type=int, default=0, help="Number of debug steps.")
    parser.add_argument("--log_every_n_steps", type=int, default=100, help="Log every n steps.")
    parser.add_argument("--val_check_interval", type=int, default=5000, help="Validation check interval.")
    parser.add_argument("--resume_from_last_checkpoint", action=argparse.BooleanOptionalAction, default=False, help="Resume training from the last checkpoint.")
    parser.add_argument("--model_from_checkpoint", type=str, default="", help="Load model from defined checkpoint.")
    parser.add_argument("--validate_extern", action=argparse.BooleanOptionalAction, default=False, help="Long training with a separate validation process.")
    parser.add_argument("--distributed", action=argparse.BooleanOptionalAction, default=False, help="Use distributed training.")

    # Data arguments
    parser.add_argument("--base_data_path", type=str, default="/cajal/nvmescratch/projects/NISB/", help="Base path for the dataset.")
    parser.add_argument("--data_setting", type=str, default="base", help="Data setting identifier.")
    parser.add_argument("--real_data_path", type=str, default="/cajal/scratch/projects/misc/mdraw/data/funke/zebrafinch/training/", help="Path to the real dataset. See https://colab.research.google.com/github/funkelab/lsd/blob/master/lsd/tutorial/notebooks/lsd_data_download.ipynb ")
    parser.add_argument("--synthetic", type=float, default=1.0, help="Ratio of synthetic data to real data.")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True, help="Use augmentations")
    parser.add_argument("--drop_slice_prob", type=float, default=0.05, help="Probability of dropping a slice during augmentation.")
    parser.add_argument("--shift_slice_prob", type=float, default=0.05, help="Probability of shifting a slice during augmentation.")
    parser.add_argument("--intensity_aug", action=argparse.BooleanOptionalAction, default=True, help="Whether to apply intensity augmentation.")
    parser.add_argument("--noise_scale", type=float, default=0.5, help="Scale of the noise to be added during augmentation.")
    parser.add_argument("--affine", type=float, default=0.5, help="Affine transformation probability.")
    parser.add_argument("--affine_scale", type=float, default=0.2, help="Scale for affine augmentation.")
    parser.add_argument("--affine_shear", type=float, default=0.5, help="Shear for affine augmentation.")
    parser.add_argument("--shift_magnitude", type=int, default=10, help="Shift augmentation magnitude (voxels).")
    parser.add_argument("--mul_int", type=float, default=0.1, help="Multiplicative augmentation intensity.")
    parser.add_argument("--add_int", type=float, default=0.1, help="Additive augmentation intensity.")

    # Model arguments
    parser.add_argument("--long_range", type=int, default=10, help="Long range affinities (voxels).")
    parser.add_argument("--model_id", type=str, default="S", help="Identifier for the mednext model architecture.")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for the convolutional layers.")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="Whether to compile the model using torch.compile.")
    parser.add_argument("--eval_ranges", type=float, nargs="+", default=torch.sigmoid(torch.tensor(list(range(-1, 12))).double() * 0.2).numpy().round(4).tolist(), help="List of evaluation thresholds.")
    parser.add_argument("--small_size", type=int, default=128, help="Size of the patches.")

    return parser.parse_args()


if __name__ == "__main__":
    main()
