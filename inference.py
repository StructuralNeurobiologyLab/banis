import gc
from typing import Union, List, Tuple

import numba
import numpy as np
import torch
import torch.utils
import zarr
from numba import jit
from scipy.ndimage import distance_transform_cdt
from torch import autocast
from torch.nn.functional import sigmoid
from tqdm import tqdm


def scale_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Scale sigmoid to avoid numerical issues in high confidence fp16."""
    return sigmoid(0.2 * x)


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
    print(
        f"Performing patched inference with do_overlap={do_overlap} for img of shape {img.shape} and dtype {img.dtype}")
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
