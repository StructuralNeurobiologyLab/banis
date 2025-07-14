import gc
import os
import shutil
import time
from collections import defaultdict
from datetime import timedelta
from typing import Union, List, Tuple

import numba
import numpy as np
import torch
import torch.utils
import zarr
import dask
from dask.distributed import (Client, LocalCluster)
import dask.array as da
from filelock import FileLock
from numba import jit
from scipy.ndimage import distance_transform_cdt
from torch import autocast
from torch.nn.functional import sigmoid
from tqdm import tqdm
from tqdm.dask import TqdmCallback


def scale_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Scale sigmoid to avoid numerical issues in high confidence fp16."""
    return sigmoid(0.2 * x)

def timing(func):
    def wrapper(*args, **kwargs):
        print(f"Starting '{func.__name__}'...")
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = timedelta(seconds=end - start)
        print(f"Finished '{func.__name__}' in {elapsed}")
        return result
    return wrapper


@jit(nopython=True)
def compute_connected_component_segmentation(hard_aff: np.ndarray) -> np.ndarray:
    """
    Compute connected components from affinities.

    Args:
        hard_aff: The (thresholded, boolean) short range affinities. Shape: (3, x, y, z).

    Returns:
        The segmentation. Shape: (x, y, z).
    """
    visited = np.zeros(tuple(hard_aff.shape[1:]), dtype=np.uint8)
    seg = np.zeros(tuple(hard_aff.shape[1:]), dtype=np.uint32)
    cur_id = 1
    for i in range(visited.shape[0]):
        for j in range(visited.shape[1]):
            for k in range(visited.shape[2]):
                if hard_aff[:, i, j, k].any() and not visited[i, j, k]:  # If foreground
                    cur_to_visit = [(i, j, k)]
                    visited[i, j, k] = True
                    while cur_to_visit:
                        x, y, z = cur_to_visit.pop()
                        seg[x, y, z] = cur_id

                        # Check all neighbors
                        if x + 1 < visited.shape[0] and hard_aff[0, x, y, z] and not visited[x + 1, y, z]:
                            cur_to_visit.append((x + 1, y, z))
                            visited[x + 1, y, z] = True
                        if y + 1 < visited.shape[1] and hard_aff[1, x, y, z] and not visited[x, y + 1, z]:
                            cur_to_visit.append((x, y + 1, z))
                            visited[x, y + 1, z] = True
                        if z + 1 < visited.shape[2] and hard_aff[2, x, y, z] and not visited[x, y, z + 1]:
                            cur_to_visit.append((x, y, z + 1))
                            visited[x, y, z + 1] = True
                        if x - 1 >= 0 and hard_aff[0, x - 1, y, z] and not visited[x - 1, y, z]:
                            cur_to_visit.append((x - 1, y, z))
                            visited[x - 1, y, z] = True
                        if y - 1 >= 0 and hard_aff[1, x, y - 1, z] and not visited[x, y - 1, z]:
                            cur_to_visit.append((x, y - 1, z))
                            visited[x, y - 1, z] = True
                        if z - 1 >= 0 and hard_aff[2, x, y, z - 1] and not visited[x, y, z - 1]:
                            cur_to_visit.append((x, y, z - 1))
                            visited[x, y, z - 1] = True
                    cur_id += 1
    return seg


@torch.no_grad()
@autocast(device_type="cuda")
@timing
def patched_inference(
        img: Union[np.ndarray, zarr.Array],
        model: torch.nn.Module,
        small_size: int = 128,
        do_overlap: bool = True,
        prediction_channels: int = 6,
        divide: int = 1,
) -> np.ndarray:
    """
    Perform patched inference with a model on an image.

    Args:
        img: The input image. Shape: (x, y, z, channel).
        model: The model to use for predictions.
        small_size: The size of the patches. Defaults to 128.
        do_overlap: Whether to perform overlapping predictions. Defaults to True:
            half of patch size for all 3 axes.
        prediction_channels: The number of channels in the output (additional model output
            dimensions are discarded). Defaults to 6 (3 short + 3 long range affinities).
        divide: The divisor for the image. Typically, 1 or 255 if img in [0, 255]

    Returns:
        The full prediction. Shape: (channel, x, y, z).
    """
    print(f"Performing patched inference with do_overlap={do_overlap} for img of shape {img.shape} and dtype {img.dtype}")
    img = img[:]  # load into memory (expensive!)

    patch_coordinates = get_coordinates(img.shape[:3], small_size, do_overlap)
    single_pred_weight = get_single_pred_weight(do_overlap, small_size)
    # to weight overlapping predictions lower close to the boundaries

    weight_sum = np.zeros((1, *img.shape[:3]), dtype=np.float32)
    weighted_pred = np.zeros((prediction_channels, *img.shape[:3]), dtype=np.float32)

    device = next(model.parameters()).device
    assert device.type != 'cpu'

    for x, y, z in tqdm(patch_coordinates):
        img_patch = torch.tensor(
            np.moveaxis(img[x: x + small_size, y: y + small_size, z: z + small_size], -1, 0)[None]).half().to(
            device) / divide
        pred = scale_sigmoid(model(img_patch))[0, :prediction_channels]

        weight_sum[:, x: x + small_size, y: y + small_size,
        z: z + small_size] += single_pred_weight if do_overlap else 1
        weighted_pred[:, x: x + small_size, y: y + small_size, z: z + small_size] += pred.cpu().numpy() * (
            single_pred_weight[None] if do_overlap else 1)
    del img  # to save memory before division
    # assert np.all(weight_sum > 0)
    np.divide(weighted_pred, weight_sum, out=weighted_pred)

    return weighted_pred


def get_coordinates(
        shape: Tuple[int, int, int], small_size: int, do_overlap: bool
) -> List[Tuple[int, int, int]]:
    """
    Get coordinates for cubes to be predicted.

    Args:
        shape: The shape of the input image (x, y, z).
        small_size: The size of the patches.
        do_overlap: Whether to perform overlapping predictions.

    Returns:
        List of (x, y, z) coordinates for prediction cubes.
    """
    offsets = [get_offsets(s, small_size) for s in shape]
    xyzs = [(x, y, z) for x in offsets[0] for y in offsets[1] for z in offsets[2]]
    if do_overlap:  # Add shifted cubes (half cube overlap)
        offset = small_size // 2

        xyzs_shifted = [
            set((x + offset, y, z) for x, y, z in xyzs),
            set((x, y + offset, z) for x, y, z in xyzs),
            set((x, y, z + offset) for x, y, z in xyzs),
            set((x + offset, y + offset, z) for x, y, z in xyzs),
            set((x + offset, y, z + offset) for x, y, z in xyzs),
            set((x, y + offset, z + offset) for x, y, z in xyzs),
            set((x + offset, y + offset, z + offset) for x, y, z in xyzs),
        ]
        xyzs_shifted = set(
            (x, y, z)
            for s in xyzs_shifted
            for x, y, z in s
            if x + small_size <= shape[0]
            and y + small_size <= shape[1]
            and z + small_size <= shape[2]
        )
        xyzs = list(set.union(set(xyzs), xyzs_shifted))
    return xyzs


def get_offsets(big_size: int, small_size: int) -> List[int]:
    """
    Calculate offsets for image patching.

    Args:
        big_size: The size of the whole image.
        small_size: The size of the patches.

    Returns:
        List of offsets.
    """
    offsets = list(range(0, big_size - small_size + 1, small_size))
    if offsets[-1] != big_size - small_size:
        offsets.append(big_size - small_size)
    return offsets


def get_single_pred_weight(do_overlap: bool, small_size: int) -> Union[np.ndarray, None]:
    """
    Get the weight for a single prediction.

    Args:
        do_overlap: Whether to perform overlapping predictions.
        small_size: The size of the patches.

    Returns:
        The weight array for a single prediction, or None if no overlap.
    """
    if do_overlap:
        # The weight (confidence/expected quality) of the predictions:
        # Low at the surface of the predicted cube, high in the center
        pred_weight_helper = np.pad(np.ones((small_size,) * 3), 1, mode='constant')
        return distance_transform_cdt(pred_weight_helper).astype(np.float32)[1:-1, 1:-1, 1:-1]
    else:
        return None


@torch.no_grad()
@autocast(device_type="cuda")
@timing
def predict_aff(
        img: Union[np.ndarray, zarr.Array],
        model: torch.nn.Module,
        zarr_path: str = "data.zarr",
        small_size: int = 128,
        do_overlap: bool = True,
        prediction_channels: int = 6,
        divide: int = 1,
        chunk_cube_size: int = 1024
):
    """
    Perform patched affinity prediction with a model on an image.

    Args:
        img: The input image. Shape: (x, y, z, channel).
        model: The model to use for predictions.
        small_size: The size of the patches. Defaults to 128.
        do_overlap: Whether to perform overlapping predictions. Defaults to True:
            half of patch size for all 3 axes.
        prediction_channels: The number of channels in the output (additional model output
            dimensions are discarded). Defaults to 6 (3 short + 3 long range affinities).
        divide: The divisor for the image. Typically, 1 or 255 if img in [0, 255]
        chunk_cube_size: The maximal side length of a cube held in memory.

    Returns:
        The full prediction. Shape: (channel, x, y, z).
    """
    print(f"Performing patched inference with do_overlap={do_overlap} for img of shape {img.shape} and dtype {img.dtype}")

    all_patch_coordinates = get_coordinates(img.shape[:3], small_size, do_overlap)
    chunked_patch_coordinates = chunk_xyzs(all_patch_coordinates, chunk_cube_size)

    z = zarr.open_group(zarr_path, mode='w')
    z.create_dataset('sum_pred', shape=(prediction_channels, *img.shape[:3]), chunks=(1, chunk_cube_size), dtype='f4')
    z.create_dataset('sum_weight', shape=(1, *img.shape[:3]), chunks=(1, chunk_cube_size), dtype='f4')

    # TODO: parallelize this!!!!!!!!!!!!!
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)
    print("Dask Client Dashboard:", client.dashboard_link)

    tasks = [dask.delayed(predict_aff_patches_chunked)(chunk, img, model, zarr_path, small_size, do_overlap, prediction_channels, divide)
                for chunk in chunked_patch_coordinates
            ]
    with TqdmCallback(desc="Overall Dask Progress (chunks)", unit="chunk", total=len(tasks)) as pbar:
        dask.compute(*tasks)
    #for chunk in tqdm(chunked_patch_coordinates):
    #    predict_aff_patches_chunked(chunk, img, model, zarr_path, small_size, do_overlap, prediction_channels, divide)

    tmp_sum_pred = da.from_zarr(f"{zarr_path}/sum_pred")
    tmp_sum_weight = da.from_zarr(f"{zarr_path}/sum_weight")
    aff = tmp_sum_pred / tmp_sum_weight
    aff.to_zarr(f"{zarr_path}/aff", overwrite=True)

    for key in ['sum_pred', 'sum_weight']:
        path = os.path.join(zarr_path, key)
        if os.path.exists(path):
            shutil.rmtree(path)

    return zarr.open(f"{zarr_path}/aff", mode="r")


def chunk_xyzs(xyzs, chunk_cube_size=1024):
    """
    Chunks the patch coordinates into chunks containing coordinates from the same part of the big cube.
    Args:
        xyzs: list of all coordinates
        chunk_cube_size: side length of each chunk
    Returns:
        chunked coordinates
    """
    chunks = defaultdict(list)
    for x, y, z in xyzs:
        chunks[(x // chunk_cube_size, y // chunk_cube_size, z // chunk_cube_size)].append((x, y, z))
    return list(chunks.values())


@torch.no_grad()
@autocast(device_type="cuda")
def predict_aff_patches_chunked(patch_coordinates, img, model, zarr_path, small_size, do_overlap, prediction_channels, divide):
    """
    Patch-wise predicts affinities in-memory, using coordinates of all patches inside a chunk.
    Args:
        patch_coordinates: List of patch coordinates. The extension of the coordinates must fit in memory (use adequate chunk size).
    Returns:
        Affinity prediction of the input chunk.
    """
    max_x = max(x for x, y, z in patch_coordinates)
    max_y = max(y for x, y, z in patch_coordinates)
    max_z = max(z for x, y, z in patch_coordinates)
    min_x = min(x for x, y, z in patch_coordinates)
    min_y = min(y for x, y, z in patch_coordinates)
    min_z = min(z for x, y, z in patch_coordinates)

    img_tmp = img[
             min_x: max_x + small_size,
             min_y: max_y + small_size,
             min_z: max_z + small_size,
             ]
    pred_tmp = np.zeros((prediction_channels, img_tmp.shape[0], img_tmp.shape[1], img_tmp.shape[2]), dtype=np.float32)
    weight_tmp = np.zeros((1, img_tmp.shape[0], img_tmp.shape[1], img_tmp.shape[2]), dtype=np.float32)
    single_pred_weight = get_single_pred_weight(do_overlap, small_size)

    for x_global, y_global, z_global in patch_coordinates:
        x = x_global - min_x
        y = y_global - min_y
        z = z_global - min_z
        img_patch = torch.tensor(np.moveaxis(
            img_tmp[x: x + small_size, y: y + small_size, z: z + small_size],
            -1, 0)[None]).to(model.device) / divide
        pred = scale_sigmoid(model(img_patch))[0, :prediction_channels]

        weight_tmp[:, x: x + small_size, y: y + small_size, z: z + small_size] += single_pred_weight if do_overlap else 1
        pred_tmp[:, x: x + small_size, y: y + small_size, z: z + small_size] += pred.detach().cpu().numpy() * (single_pred_weight[None] if do_overlap else 1)


    z = zarr.open_group(zarr_path, mode='a')
    weight_mask = z['sum_weight']
    full_pred = z['sum_pred']

    with FileLock(f"{zarr_path}/sum_weight.lock"):
        weight_mask[
            :,
            min_x: max_x + small_size,
            min_y: max_y + small_size,
            min_z: max_z + small_size,
        ] += weight_tmp

    with FileLock(f"{zarr_path}/sum_pred.lock"):
        full_pred[
            :,
            min_x: max_x + small_size,
            min_y: max_y + small_size,
            min_z: max_z + small_size,
        ] += pred_tmp



if __name__ == "__main__":
    input_path = "/cajal/nvmescratch/projects/NISB/base/val/seed100/data.zarr"
    img_data = zarr.open(input_path, mode="r")["img"]

    model_path = "/cajal/scratch/projects/misc/zuzur/ss3/debug1GPU-seed0-batch_size1-small_size128/default/checkpoints/epoch=0-step=70000.ckpt"
    from BANIS import BANIS
    model = BANIS.load_from_checkpoint(model_path)

    #aff_pred = patched_inference(img_data, model=model, do_overlap=True, prediction_channels=3, divide=255,small_size=model.hparams.small_size)
    #store = zarr.DirectoryStore('/cajal/scratch/projects/misc/zuzur/test0.zarr')
    #store.rmdir('')
    #z = zarr.array(aff_pred, store=store, override=True)

    #aff_pred2 = predict_aff(img_data, model, zarr_path="/cajal/scratch/projects/misc/zuzur/test.zarr", do_overlap=True, prediction_channels=3, divide=255,small_size=model.hparams.small_size)
    # ValueError('Codec does not support buffers of > 2147483647 bytes')

    aff_pred2 = predict_aff(img_data, model, chunk_cube_size=250, zarr_path="/cajal/scratch/projects/misc/zuzur/test2.zarr", do_overlap=True, prediction_channels=3, divide=255,small_size=model.hparams.small_size)

