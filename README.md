# BANIS: Baseline for Affinity-based Neuron Instance Segmentation

**An easily adaptable baseline for the [Neuron Instance Segmentation Benchmark (NISB)](https://structuralneurobiologylab.github.io/nisb/), [predicting affinities](https://arxiv.org/abs/1706.00120) with [modern architectures](https://arxiv.org/abs/2303.09975) and simple connected components for post-processing**

## Prerequisites

[Download NISB datasets](https://structuralneurobiologylab.github.io/nisb/) and set up a conda/mamba environment:

```bash
# With environment.yaml
mamba env create -f environment.yaml
mamba activate nisb

# Without yaml
mamba create -n nisb -c conda-forge python=3.11 -y
mamba activate nisb 
pip install torch torchvision torchaudio numpy connected-components-3d numba pytorch-lightning zarr monai scipy cython tensorboard
pip install git+https://github.com/MIC-DKFZ/MedNeXt.git#egg=mednextv1
pip install git+https://github.com/funkelab/funlib.evaluate.git 
```

Tested on a Slurm cluster with nodes equipped with 1 NVIDIA A40 GPU and 500 GB RAM (stay tuned for a less RAM-intensive version).

## Usage

Run a single training session (BANIS-S(mall)):

```bash
python BANIS.py --seed 0 --batch_size 8 --n_steps 50000 --data_setting base --base_data_path /local/dataset/dir/ --save_path /local/logging/dir/
```
Results are logged to TensorBoard. For GPUs with less than 48 GB memory, reduce `batch_size` (and adjust `n_steps` / `learning_rate`). For BANIS-L(arge) add `--model_id L --kernel_size 5`. Additional options are in `parse_args` of `BANIS.py`.

To run multiple jobs on Slurm, adjust `config.yaml` and `aff_train.sh`, then:

```bash
python slurm_job_scheduler.py
```

Adding an `auto_resubmit` argument to `config.yaml` allows Slurm to automatically resubmit jobs that reach the Slurm time limit (see `aff_train.sh`).

## Evaluation

To evaluate a predicted segmentation (`.zarr` or `.npy`):

```bash
python metrics.py --pred_seg /path/to/predictions.zarr --skel_path /path/to/skeleton.pkl [--load_to_memory]
```

## Visualization

To visualize the validation cube of each dataset, run:

```bash
 show_data.py --base_path /local/benchmark/dir/ 
```


