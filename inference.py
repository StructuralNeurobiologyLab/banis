import shutil
from collections import defaultdict
from copy import deepcopy
from typing import Union, List, Tuple

import argparse
import cc3d
import numba
import numpy as np
import torch
import torch.utils
import zarr
import dask
from dask import compute, persist, delayed
from dask.distributed import Client, LocalCluster
import dask.array as da
from distributed import progress
from filelock import FileLock
from numba import jit
from scipy.ndimage import distance_transform_cdt
from torch import autocast
from torch.nn.functional import sigmoid
from tqdm import tqdm
import mwatershed


class Utils:
    @staticmethod
    def get_coordinates(shape: Tuple[int, int, int], small_size: int, overlap: int = 0,
                        last_has_smaller_overlap: bool = True) -> List[Tuple[int, int, int]]:
        """
        Get coordinates for smaller patches to process a big cube in memory.
        Args:
            shape: The shape of the input (x, y, z).
            small_size: The size of the patches.
            overlap: The overlap between patches. The default 0 means no overlap (next patch starts on the next pixel from the previous patch). For half-cube overlap set overlap=small_size//2, for 1-pixel overlap set overlap=1.
            last_has_smaller_overlap: If the last patch with the specified size and overlap would exceed the big cube, move the patch so that it ends with the big cube, creating a bigger overlap in this patch.
        Returns:
            List of (x, y, z) coordinates (starting voxel of a patch) for processing of smaller patches.
        """
        if overlap < 0 or overlap >= small_size:
            raise ValueError(f"Overlap must be between 0 and {small_size}.")
        offsets = [Utils.get_offsets(s, small_size, small_size - overlap, last_has_smaller_overlap) for s in shape]
        xyzs = [(x, y, z) for x in offsets[0] for y in offsets[1] for z in offsets[2]]
        return xyzs

    @staticmethod
    def get_offsets(big_size, small_size, step, last_has_smaller_overlap):
        offsets = list(range(0, big_size - small_size + 1, step))
        if small_size > big_size:
            offsets.append(0)
        elif offsets[-1] != big_size - small_size and last_has_smaller_overlap:
            offsets.append(big_size - small_size)
        elif offsets[-1] != big_size - small_size and not last_has_smaller_overlap:
            offsets.append(len(offsets) * step)
        return offsets

    @staticmethod
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

    @staticmethod
    def scale_sigmoid(x: torch.Tensor) -> torch.Tensor:
        """Scale sigmoid to avoid numerical issues in high confidence fp16."""
        return sigmoid(0.2 * x)

    @staticmethod
    def get_xyz_end(chunk, chunk_cube_size, aff_shape):
        """
        Returns the end indices of a chunk, that correspond either to the chunk size, or align with the size of the affinities.
        """
        x, y, z = chunk
        x_end, y_end, z_end = (min(x + chunk_cube_size, aff_shape[1]),
                               min(y + chunk_cube_size, aff_shape[2]),
                               min(z + chunk_cube_size, aff_shape[3]))
        return (x_end, y_end, z_end)


class AffinityPredictor:
    def __init__(self,
                 chunk_cube_size: int = 1024,
                 compute_backend: str = "local",
                 model: torch.nn.Module = None,
                 model_path: str = None,
                 small_size: int = 128,
                 do_overlap: bool = True,
                 prediction_channels: int = 6,
                 divide: int = 1,
                 ):
        self.chunk_cube_size = chunk_cube_size
        self.compute_backend = compute_backend

        self.model = model  # only for local prediction
        self.model_path = model_path  # loads model in the worker in case of distributed inference (model not pickleable)
        self.small_size = small_size
        self.do_overlap = do_overlap
        self.prediction_channels = prediction_channels
        self.divide = divide

    def img_to_aff(self, img, zarr_path):
        """
        Complete prediction of affinities from the input image, with the model previously specified in AffinityPredictor.
        """
        print(f"Performing patched inference with do_overlap={self.do_overlap} for img of shape {img.shape} and dtype {img.dtype}")
        print(f"Parameters: cube size {self.chunk_cube_size}, compute backend {self.compute_backend}.")

        all_patch_coordinates = Utils.get_coordinates(img.shape[:3], self.small_size, overlap=self.small_size // 2 if self.do_overlap else 0, last_has_smaller_overlap=True)
        chunked_patch_coordinates = Utils.chunk_xyzs(all_patch_coordinates, self.chunk_cube_size)

        z = zarr.open_group(zarr_path + "_tmp", mode='w')
        zarr_chunk_size = min(self.chunk_cube_size, 512)
        z.create_dataset('sum_pred', shape=(self.prediction_channels, *img.shape[:3]), chunks=(1, zarr_chunk_size, zarr_chunk_size, zarr_chunk_size), dtype='f4')
        z.create_dataset('sum_weight', shape=(1, *img.shape[:3]), chunks=(1, zarr_chunk_size, zarr_chunk_size, zarr_chunk_size), dtype='f4')

        if self.compute_backend == "local":
            for chunk in tqdm(chunked_patch_coordinates, desc="chunks"):
                self.predict_aff_patches_chunked(chunk, img, zarr_path + "_tmp")
                torch.cuda.empty_cache()
        else:
            if self.compute_backend == "local_cluster":
                from dask_cuda import LocalCUDACluster
                cluster = LocalCUDACluster(threads_per_worker=1)  # 1 worker per GPU
            elif self.compute_backend == "slurm":
                from dask_jobqueue import SLURMCluster
                cluster = SLURMCluster(
                    cores=8,
                    memory="400GB",
                    processes=1,
                    worker_extra_args=["--resources processes=1", "--nthreads=1"],
                    job_extra_directives=["--gres=gpu:1"],
                    walltime="1-00:00:00"
                )
                cluster.adapt(minimum_jobs=1, maximum_jobs=32)

            else:
                raise NotImplementedError(f"Compute backend {self.compute_backend} not available.")

            client = Client(cluster)
            print(f"Waiting for workers...")
            client.wait_for_workers(n_workers=1)
            print("Dask Client Dashboard:", client.dashboard_link)
            tasks = [dask.delayed(self.predict_aff_patches_chunked)(chunk, img, zarr_path + "_tmp") for chunk in chunked_patch_coordinates]
            futures = persist(tasks)
            progress(futures)  # progress bar
            compute(futures)

        tmp_sum_pred = da.from_zarr(f"{zarr_path}_tmp/sum_pred")
        tmp_sum_weight = da.from_zarr(f"{zarr_path}_tmp/sum_weight")
        aff = tmp_sum_pred / tmp_sum_weight
        aff.to_zarr(zarr_path, overwrite=True)

        shutil.rmtree(zarr_path + "_tmp")

        return

    def predict_aff_patches_chunked(self, patch_coordinates, img, zarr_path):
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
                  min_x: max_x + self.small_size,
                  min_y: max_y + self.small_size,
                  min_z: max_z + self.small_size,
                  ]
        pred_tmp = np.zeros((self.prediction_channels, img_tmp.shape[0], img_tmp.shape[1], img_tmp.shape[2]), dtype=np.float32)
        weight_tmp = np.zeros((1, img_tmp.shape[0], img_tmp.shape[1], img_tmp.shape[2]), dtype=np.float32)
        single_pred_weight = self.get_single_pred_weight(self.do_overlap, self.small_size)

        if not self.model:
            from BANIS import BANIS
            print(self.model_path, flush=True)
            model = BANIS.load_from_checkpoint(self.model_path)
        else:
            model = self.model

        for x_global, y_global, z_global in tqdm(patch_coordinates, desc=f'cube ({min_x}, {max_x + self.small_size}), ({min_y}, {max_y + self.small_size}), ({min_z}, {max_z + self.small_size})'):
            x = x_global - min_x
            y = y_global - min_y
            z = z_global - min_z
            img_patch = torch.tensor(np.moveaxis(img_tmp[x: x + self.small_size, y: y + self.small_size, z: z + self.small_size], -1, 0)[None]).to(model.device) / self.divide
            pred = Utils.scale_sigmoid(model(img_patch))[0, :self.prediction_channels]

            weight_tmp[:, x: x + self.small_size, y: y + self.small_size, z: z + self.small_size] += single_pred_weight if self.do_overlap else 1
            pred_tmp[:, x: x + self.small_size, y: y + self.small_size, z: z + self.small_size] += pred.detach().cpu().numpy() * (single_pred_weight[None] if self.do_overlap else 1)

        z = zarr.open_group(zarr_path, mode='a')
        weight_mask = z['sum_weight']
        full_pred = z['sum_pred']

        with FileLock(f"{zarr_path}/sum_weight.lock"):
            weight_mask[
            :,
            min_x: max_x + self.small_size,
            min_y: max_y + self.small_size,
            min_z: max_z + self.small_size,
            ] += weight_tmp

        with FileLock(f"{zarr_path}/sum_pred.lock"):
            full_pred[
            :,
            min_x: max_x + self.small_size,
            min_y: max_y + self.small_size,
            min_z: max_z + self.small_size,
            ] += pred_tmp

    def get_single_pred_weight(self, do_overlap: bool, small_size: int) -> Union[np.ndarray, None]:
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


class Postprocessing:
    def __init__(self,
                 chunk_cube_size: int = 1024,
                 compute_backend: str = "local"
                 ):
        self.chunk_cube_size = chunk_cube_size
        self.compute_backend = compute_backend

    def aff_to_seg(self, aff, zarr_path):
        chunks = Utils.get_coordinates(aff.shape[1:], self.chunk_cube_size, overlap=1, last_has_smaller_overlap=False)
        reverse_chunks = {chunk: i for i, chunk in enumerate(chunks)}
        patched_zarr_path = zarr_path + "_tmp"

        zarr_chunk_size = min(self.chunk_cube_size, 512)
        z_root = zarr.create(shape=(len(chunks), self.chunk_cube_size, self.chunk_cube_size, self.chunk_cube_size),
                             store=patched_zarr_path, dtype='i4', overwrite=True,
                             chunks=(1, zarr_chunk_size, zarr_chunk_size, zarr_chunk_size))

        # SEGMENT AFFINITIES IN CHUNKS THAT FIT IN MEMORY
        self.patched_segment_affinities(aff, patched_zarr_path, chunks)

        # FIND GROUPS OF FRAGMENTS THAT SHOULD BE MERGED BETWEEN CHUNKS
        fragment_agglomeration, max_id = self.agglomerate_fragments(chunks, reverse_chunks, patched_zarr_path, aff.shape)

        # MERGE AND RELABEL INSTANCES GLOBALLY
        self.merge_and_relabel(fragment_agglomeration, max_id, patched_zarr_path, zarr_path, chunks, aff.shape)

        return

    def patched_segment_affinities(self, aff, patched_zarr_path, chunks):
        if self.compute_backend == "local":
            for i, chunk in enumerate(tqdm(chunks)):
                self.segment_chunk_wrapped(chunk, i, aff, patched_zarr_path)
        else:
            if self.compute_backend == "local_cluster":
                from dask_cuda import LocalCUDACluster
                cluster = LocalCUDACluster(threads_per_worker=1)  # 1 worker per GPU
            elif self.compute_backend == "slurm":
                from dask_jobqueue import SLURMCluster
                cluster = SLURMCluster(
                    cores=8,
                    memory="400GB",
                    processes=1,
                    worker_extra_args=["--resources processes=1", "--nthreads=1"],
                    job_extra_directives=["--gres=gpu:1"],
                    walltime="1-00:00:00"
                )
                cluster.adapt(minimum_jobs=1, maximum_jobs=32)
            else:
                raise NotImplementedError(f"Compute backend {self.compute_backend} not available.")

            client = Client(cluster)
            print(f"Waiting for workers...")
            client.wait_for_workers(n_workers=1)
            print("Dask Client Dashboard:", client.dashboard_link)
            tasks = [dask.delayed(self.segment_chunk_wrapped)(chunk, i, aff, patched_zarr_path) for (i, chunk) in enumerate(chunks)]
            futures = persist(tasks)
            progress(futures)  # progress bar
            compute(futures)

    def agglomerate_fragments(self, chunks, reverse_chunks, patched_zarr_path, aff_shape):
        if self.compute_backend == "local":
            fragment_agglomeration = {}
            for i, chunk in enumerate(tqdm(chunks)):
                chunk_agglomeration = self.agglomerate_chunk(chunk, reverse_chunks, patched_zarr_path, aff_shape)
                for node, nbrs in chunk_agglomeration.items():
                    for nbr in nbrs:
                        fragment_agglomeration.setdefault(node, set()).add(nbr)
                if len(fragment_agglomeration) > 10_000_000:
                    print("WARNING: fragment agglomeration too long, might cause problems!")
                    # TODO: solve this

            curr_id, fragment_agglomeration_flattened = self.flatten_agglomeration(fragment_agglomeration)
            #print("MERGING CHUNKS FLATTENED AGGLOMERATION LENGTH", len(fragment_agglomeration_flattened))
            #fragment_agglomeration_flattened = self.add_all_fragments_to_agglomeration(fragment_agglomeration_flattened, curr_id, chunks, patched_zarr_path)
            #print("ALL CHUNKS FLATTENED AGGLOMERATION LENGTH", len(fragment_agglomeration_flattened))

        else:
            # TODO: add slurm (and measure memory)
            raise NotImplementedError(f"Compute backend {self.compute_backend} not available.")

        return fragment_agglomeration_flattened, curr_id

    def agglomerate_chunk(self, chunk, reverse_chunks, patched_zarr_path, aff_shape):
        fragment_agglomeration = {}
        x, y, z = chunk
        x_end, y_end, z_end = Utils.get_xyz_end(chunk, self.chunk_cube_size, aff_shape)
        z_root = zarr.open(patched_zarr_path, mode='r')

        # for (x,y,z) get the last slice of the current cube (l, low) and the first slice of the next cube (h, high)
        # these slices overlap, so the voxels should have the same global id

        if x_end < aff_shape[1]:
            chunk_l = reverse_chunks[chunk]
            chunk_h = reverse_chunks[x + self.chunk_cube_size - 1, y, z]
            result_l = z_root[chunk_l, -1:, :, :]
            result_h = z_root[chunk_h, :1, :, :]
            combined = np.stack([result_l.flatten(), result_h.flatten()]).T
            uniques = np.unique(combined, axis=0)
            fragment_agglomeration = self.update_fragment_agglomeration(fragment_agglomeration, uniques, chunk_l, chunk_h)

        if y_end < aff_shape[2]:
            chunk_l = reverse_chunks[chunk]
            chunk_h = reverse_chunks[x, y + self.chunk_cube_size - 1, z]
            result_l = z_root[chunk_l, :, -1:, :]
            result_h = z_root[chunk_h, :, :1, :]
            combined = np.stack([result_l.flatten(), result_h.flatten()]).T
            uniques = np.unique(combined, axis=0)
            fragment_agglomeration = self.update_fragment_agglomeration(fragment_agglomeration, uniques, chunk_l, chunk_h)

        if z_end < aff_shape[3]:
            chunk_l = reverse_chunks[chunk]
            chunk_h = reverse_chunks[x, y, z + self.chunk_cube_size - 1]
            result_l = z_root[chunk_l, :, :, -1:]
            result_h = z_root[chunk_h, :, :, :1]
            combined = np.stack([result_l.flatten(), result_h.flatten()]).T
            uniques = np.unique(combined, axis=0)
            fragment_agglomeration = self.update_fragment_agglomeration(fragment_agglomeration, uniques, chunk_l, chunk_h)

        return fragment_agglomeration

    def update_fragment_agglomeration(self, fragment_agglomeration, uniques, chunk_l, chunk_h):
        for id_l, id_h in uniques:
            if id_l > 0 and id_h > 0:
                fragment_agglomeration.setdefault((chunk_h, id_h), set()).add(
                    (chunk_l, id_l)
                )
                fragment_agglomeration.setdefault((chunk_l, id_l), set()).add(
                    (chunk_h, id_h)
                )
        return fragment_agglomeration

    def flatten_agglomeration(self, fragment_agglomeration):
        """
        Computes connected components in the fragment agglomeration graph, and assigns the fragments new ids starting from 1.
        Args:
            fragment_agglomeration: dictionary with keys (chunk_id, fragment_id), and values a set of (chunk_id, fragment_id) in another chunk (cube) that should be connected
        Returns:
            fragment_agglomeration_flattened: dictionary with keys (chunk_id, fragment_id) and values the global component index
        """
        cur_id = 1
        fragment_agglomeration_flattened = dict()
        for position_id in tqdm(fragment_agglomeration):  # (chunk, idx) = position_id
            if position_id not in fragment_agglomeration_flattened:
                to_visit = {position_id}
                visited = set()
                while len(to_visit) > 0:
                    current = to_visit.pop()
                    if current not in visited:
                        visited.add(current)
                        for neighbor in fragment_agglomeration[current]:
                            to_visit.add(neighbor)
                for v in visited:
                    assert v not in fragment_agglomeration_flattened
                    fragment_agglomeration_flattened[v] = cur_id
                cur_id += 1

        return cur_id, fragment_agglomeration_flattened

    #def add_all_fragments_to_agglomeration(self, fragment_agglomeration_flattened, cur_id, chunks, patched_zarr_path):
    #    z_root = zarr.open(patched_zarr_path)
    #    for i, chunk in enumerate(tqdm(chunks)):
    #        data = z_root[i, :, :, :]
    #        for idx in range(1, int(data.max()) + 1):  # assuming each chunk has contiguous indices from 0 to max
    #            if (i, idx) not in fragment_agglomeration_flattened:
    #                fragment_agglomeration_flattened[(i, idx)] = cur_id
    #                cur_id += 1
    #    return fragment_agglomeration_flattened

    def merge_and_relabel(self, fragment_agglomeration, max_id, zarr_patched, zarr_final, chunks, aff_shape):
        zarr_chunk_size = min(self.chunk_cube_size, 512)
        z_root = zarr.open(zarr_patched)
        z_final = zarr.create(shape=aff_shape[1:],
                              store=zarr_final, dtype='i4', overwrite=True,
                              chunks=(zarr_chunk_size, zarr_chunk_size, zarr_chunk_size))

        if self.compute_backend == "local":
            for i, chunk in enumerate(tqdm(chunks)):
                x, y, z = chunk
                x_end, y_end, z_end = Utils.get_xyz_end(chunk, self.chunk_cube_size, aff_shape)
                data = z_root[i, : x_end - x, : y_end - y, : z_end - z]
                perm = [0]
                for idx in range(1, int(data.max()) + 1):  # assuming each chunk has contiguous indices from 0 to max
                    if not (i, idx) in fragment_agglomeration:
                        max_id += 1
                        perm.append(max_id)
                    else:
                        perm.append(fragment_agglomeration[(i, idx)])
                perm = np.array(perm, dtype=np.uint64)
                relabeled = perm[data]
                z_final[x: x_end, y: y_end, z: z_end] = relabeled

        else:
            raise NotImplementedError(f"Compute backend {self.compute_backend} not implemented.")

        shutil.rmtree(zarr_patched)

    def segment_chunk_wrapped(self, chunk, i, aff, zarr_path):
        x, y, z = chunk
        x_end, y_end, z_end = Utils.get_xyz_end(chunk, self.chunk_cube_size, aff.shape)
        curr_aff = aff[:, x : x_end, y : y_end, z : z_end]
        curr_seg = self.segment_chunk(curr_aff)
        z_root = zarr.open(zarr_path, mode="r+")
        z_root[i, : x_end - x, : y_end - y, : z_end - z] = curr_seg

    def segment_chunk(self, curr_aff):
        """
        In-memory segmentation of a chunk of affinities.
        Args:
            curr_aff: The affinities to segment (must fit in memory).
        Returns:
            Segmentation of the given affinities.
        """
        raise NotImplementedError(f"This method should be overridden in a subclass.")


class MutexWatershed(Postprocessing):
    def __init__(self, chunk_cube_size, compute_backend, mws_bias_short, mws_bias_long, long_range=10):
        super().__init__(chunk_cube_size, compute_backend)
        self.mws_bias_short = mws_bias_short
        self.mws_bias_long = mws_bias_long
        self.long_range = long_range

    def compute_mws_segmentation(self, cur_aff):
        cur_aff = deepcopy(cur_aff).astype(np.float64)
        cur_aff[:3] += self.mws_bias_short
        cur_aff[3:] += self.mws_bias_long

        cur_aff[:3] = np.clip(cur_aff[:3], 0, 1)  # short-range attractive edges
        cur_aff[3:] = np.clip(cur_aff[3:], -1, 0)  # long-range repulsive edges (see the Mutex Watershed paper)

        mws_pred = mwatershed.agglom(
            affinities=cur_aff,
            offsets=(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [self.long_range, 0, 0],
                    [0, self.long_range, 0],
                    [0, 0, self.long_range],
                ]
            ),
        )

        # mwatershed is wasteful with IDs (not contiguous) -> filter out single voxel objects and relabel again
        # size filter. single voxel objects are irrelevant for merging, take ~95% of IDs in an example cube, causing OOM when creating fragment_agglomeration
        dusted = cc3d.dust(  # does a cc first (reducing false mergers in add_to_agglomeration)
            mws_pred,
            threshold=2,
            connectivity=6,
            in_place=False,
        )
        # relabeling to save IDs
        pred_relabeled, N = cc3d.connected_components(
            dusted, return_N=True, connectivity=6
        )

        assert (pred_relabeled[mws_pred == 0] == 0).all()  # 0 stays 0
        assert N <= np.iinfo(np.uint32).max

        pred = pred_relabeled.astype(np.uint32)
        return pred

    def segment_chunk(self, curr_aff):
        return self.compute_mws_segmentation(curr_aff)



class Thresholding(Postprocessing):
    def __init__(self, chunk_cube_size, compute_backend, thr):
        super().__init__(chunk_cube_size, compute_backend)
        self.thr = thr

    @staticmethod
    @jit(nopython=True)
    def compute_connected_component_segmentation(hard_aff: np.ndarray) -> np.ndarray:
        """
        Compute connected components from affinities.

        Args:
            hard_aff: The (thresholded, boolean) short range affinities. Shape: (3, x, y, z).

        Returns:
            The segmentation. Shape: (x, y, z).
        """
        visited = np.zeros(tuple(hard_aff.shape[1:]), dtype=numba.boolean)
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

    def segment_chunk(self, curr_aff):
        return self.compute_connected_component_segmentation(curr_aff[:3] > self.thr)


def full_inference(
        # RESOURCES ARGUMENTS:
        chunk_cube_size: int = 3000,
        compute_backend: str = "local",
        # AFFINITY PREDICTION ARGUMENTS:
        img: Union[np.ndarray, zarr.Array] = None,
        model_path: str = None,
        aff_zarr_path: str = "aff_prediction.zarr",
        small_size: int = 128,
        do_overlap: bool = True,
        prediction_channels: int = 6,
        divide: int = 1,
        # POSTPROCESSING ARGUMENTS:
        postprocessing_type: str = "thresholding",
        seg_zarr_path: str = "seg_prediction.zarr",
        thr: float = 0.5,
        mws_bias_short: float = -0.5,
        mws_bias_long: float = -0.5,
):
    affinity_predictor = AffinityPredictor(
        chunk_cube_size=chunk_cube_size,
        compute_backend=compute_backend,
        model_path=model_path,
        small_size=small_size,
        do_overlap=do_overlap,
        prediction_channels=prediction_channels,
        divide=divide,
    )
    affinity_predictor.img_to_aff(img, zarr_path=aff_zarr_path)
    aff = zarr.open(aff_zarr_path, mode="r")

    if postprocessing_type == "thresholding":
        postprocessor = Thresholding(chunk_cube_size, compute_backend, thr)
    elif postprocessing_type == "mws":
        postprocessor = MutexWatershed(chunk_cube_size, compute_backend, mws_bias_short, mws_bias_long)
    else:
        raise NotImplementedError(f"Postprocessing type {postprocessing_type} is not implemented")
    postprocessor.aff_to_seg(aff, zarr_path=seg_zarr_path)
    seg = zarr.open(seg_zarr_path, mode="r")

    print(f"Segmentation saved at {seg_zarr_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_cube_size", type=int, default=3000, help="The maximal side length of a cube held in memory.")
    parser.add_argument("--compute_backend", type=str, default="local", help="Compute backend to use: local, slurm, or local_cluster.")
    parser.add_argument("--img_path", type=str, help="The image to segment (path to zarr).")
    parser.add_argument("--model_path", type=str, help="The path to the trained model.")
    parser.add_argument("--aff_zarr_path", type=str, default="aff_prediction.zarr", help="Where to save the predicted affinities.")
    parser.add_argument("--small_size", type=int, default=128, help="Size of the small patches for affinity prediction (model parameter).")
    parser.add_argument("--do_overlap", type=bool, default=True, help="Use overlapping patches for affinity prediction for better precision.")
    parser.add_argument("--prediction_channels", type=int, default=6, help="The number of prediction channels. Defaults to 6 (3 short + 3 long range affinities).")
    parser.add_argument("--divide", type=int, default=255, help="The divisor for the image. Typically, 1 or 255 if img in [0, 255].")
    parser.add_argument("--postprocessing_type", type=str, default="thresholding", help="Type of postprocessing to use: thresholding, or mws (mutex watershed).")
    parser.add_argument("--seg_zarr_path", type=str, default="seg_prediction.zarr", help="Where to save the final segmentation.")
    parser.add_argument("--thr", type=float, default=0.5, help="Threshold in case of thresholding.")
    parser.add_argument("--mws_bias_short", type=float, default=-0.5, help="Short-range bias for mutex watershed.")
    parser.add_argument("--mws_bias_long", type=float, default=-0.5, help="Long-range bias for mutex watershed.")

    args = parser.parse_args()

    img = zarr.open(args.img_path, mode="r")["img"]
    full_inference(
        chunk_cube_size=args.chunk_cube_size,
        compute_backend=args.compute_backend,
        img=img,
        model_path=args.model_path,
        aff_zarr_path=args.aff_zarr_path,
        small_size=args.small_size,
        do_overlap=args.do_overlap,
        prediction_channels=args.prediction_channels,
        divide=args.divide,
        postprocessing_type=args.postprocessing_type,
        seg_zarr_path=args.seg_zarr_path,
        thr=args.thr,
        mws_bias_short=args.mws_bias_short,
        mws_bias_long=args.mws_bias_long,
    )
