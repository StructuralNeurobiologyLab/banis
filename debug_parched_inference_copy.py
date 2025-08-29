import os
import pickle
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import timedelta
from functools import partial
import hydra
from omegaconf import DictConfig, open_dict
import joblib
import itertools
import gc

import cc3d
import configargparse
import fastremap
import mwatershed
import numpy as np
import pandas as pd
import zarr
from dask import config as dask_cfg
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, LocalCluster, as_completed
from tqdm import tqdm
from numba import jit
from datetime import datetime
import psutil, os


from metrics import compute_metrics

# this only changes the configuration in the local process, and not subprocesses (like remote workers)
dask_cfg.set(
    {
        "distributed.scheduler.worker-ttl": "1h",  # required because mwatershed blocks for a long time
        "distributed.comm.timeouts.connect": "1h",
        "distributed.comm.timeouts.tcp": "1h",
        "distributed.admin.tick.limit": "30s"  # increase time before triggering a warning (default limit of 3s remains in workers - https://github.com/dask/distributed/issues/3882)
    }
)


def chunk_list(list_to_chunk, chunk_size):  # todo:use zarr bag instead?
    return [
        list_to_chunk[i: i + chunk_size]
        for i in range(0, len(list_to_chunk), chunk_size)
    ]


def patched_thresholding(aff, conf):
    """
    Creates a segmentation from an affinity map.
    Segmentation is created patchwise using thresholding or mutex watershed,
    and subsequently merging segments that span multiple patches.
    """
    ijk_to_idx, patch_to_coords = get_mappings(aff, conf.patch_size)

    # Predict segments for all patches
    if conf.debug_patched_seg_path:
        print(f"Using patched segmentation from {conf.debug_patched_seg_path}")
        patched_seg = zarr.open(conf.debug_patched_seg_path, mode="r")
    else:
        patched_seg = segment_patches(
            aff,
            conf,
            patch_to_coords,
            ijk_to_idx
        )

    # Agglomerate segments at the edges of neighboring patches
    if conf.debug_fragment_agglomeration_path:
        print(f"Using fragment agglomeration from {conf.debug_fragment_agglomeration_path}")
        if not os.path.normpath(conf.debug_fragment_agglomeration_path) == os.path.normpath(f"{conf.path_root}/agglo_pkl_chunks"):
            print(f"CONF PATH {conf.debug_fragment_agglomeration_path} NOT EQUAL TO {conf.path_root}/agglo_pkl_chunks - MIGHT CAUSE PROBLEMS")
        else:
            fragment_agglomeration_flattened = None
            print(f"NOT LOADING PATH {conf.debug_fragment_agglomeration_path} - WILL DO IT IN THE WORKERS")
        fragment_agglomeration_flattened = None
    else:
        fragment_agglomeration_flattened = compute_fragment_agglomeration(
            patched_seg,
            aff,
            conf,
            ijk_to_idx,
            patch_to_coords,
        )

    # Unify indexing of all patches, including the merged agglomerations at the border
    if conf.debug_relabeled_seg_path:
        print(f"Using relabeled agglomerated segmentation from {conf.debug_relabeled_seg_path}")
        agglomerated_seg = zarr.open(conf.debug_relabeled_seg_path, mode="r")
    else:
        agglomerated_seg = relabel_globally(
            fragment_agglomeration_flattened,
            patched_seg,
            aff,
            conf,
            patch_to_coords,
            ijk_to_idx,
        )

    # Filter out segments that are too small
    filtered_seg = size_filter_relabel(agglomerated_seg, conf)

    # Delete intermediary files that are not needed anymore
    if conf.delete_files:
        try:
            zarr.DirectoryStore(f"{conf.path_root}/patched_seg.zarr").rmdir()
            os.rmdir(f"{conf.path_root}/agglo_pkl_chunks")
            zarr.DirectoryStore(f"{conf.path_root}/agglomerated_seg.zarr").rmdir()
            os.remove(f"{conf.path_root}/id_mapping.csv")
            shutil.rmtree(conf.path_root + "/dask-worker-space/", ignore_errors=True)
        except:
            print("Exception while deleting files.")

    return filtered_seg


def get_mappings(aff, patch_size):
    """
    Returns coordinates of patches, and their indices.
    """
    # x,y,z: coordinates
    # i,j,k: patch indices (i.e. i*n <= x < (i+1)*n)
    # idx: patch index (i.e. i * len(ys) * len(zs) + j * len(zs) + k)

    xs = list(range(0, aff.shape[1], patch_size))
    ys = list(range(0, aff.shape[2], patch_size))
    zs = list(range(0, aff.shape[3], patch_size))

    ijk_to_idx = {
        (i, j, k): i * len(ys) * len(zs) + j * len(zs) + k
        for i in range(len(xs))
        for j in range(len(ys))
        for k in range(len(zs))
    }

    patch_to_coords = {
        (i, j, k): (
            (xs[i], xs[i + 1] if i + 1 < len(xs) else None),
            (ys[j], ys[j + 1] if j + 1 < len(ys) else None),
            (zs[k], zs[k + 1] if k + 1 < len(zs) else None),
        )
        for i in range(len(xs))
        for j in range(len(ys))
        for k in range(len(zs))
    }

    return ijk_to_idx, patch_to_coords


def segment_patches(aff, conf, patch_to_coords, ijk_to_idx):
    """
    Predict segmentation for each patch independently.
    Returns segmentation of shape (len(patch_to_coords), patch_size + overlap, patch_size + overlap, patch_size + overlap)
    """

    print(f"Computing patch segmentation...")
    print(f"Store: {conf.path_root}/patched_seg.zarr")
    patched_seg = zarr.zeros(
        (
            len(patch_to_coords),
            conf.patch_size + conf.overlap,
            conf.patch_size + conf.overlap,
            conf.patch_size + conf.overlap,
        ),
        chunks=(1, conf.patch_size + conf.overlap, conf.patch_size + conf.overlap, conf.patch_size + conf.overlap),
        dtype=np.uint32,
        store=f"{conf.path_root}/patched_seg.zarr",
    )
    print(patched_seg.shape)

    ijks_chunked = chunk_list(
        list(patch_to_coords.keys()),
        chunk_size=1  # faster than bigger chunks
    )

    def chunked_thresholding(ijks):
        for ijk in ijks:
            threshold_ijk(ijk, aff, patched_seg, patch_to_coords, ijk_to_idx, conf)

    if not conf.use_parallelization:
        result = list(map(chunked_thresholding, tqdm(ijks_chunked)))
        # we don't need the result, the iteration writes the segmentation directly into the zarr array
    else:
        if conf.use_slurm:
            cluster = SLURMCluster(
                cores=8,
                memory="500GB",
                processes=1,
                worker_extra_args=["--resources processes=1"],
                log_directory=f"/cajal/scratch/projects/misc/zuzur/slurm_logs/segment/",
                walltime="3:00:00"  # default is 30mins and then worker gets killed, chunked ijks can take more time
            )
            cluster.adapt(minimum_jobs=1, maximum_jobs=32)

        else:
            cluster = LocalCluster(
                n_workers=min(os.cpu_count(), 16),
                threads_per_worker=1,
                local_directory=conf.path_root + "/dask-worker-space/"
                # to avoid independent runs deleting each other's directories
            )

        with Client(cluster) as client:
            print("Dask threshold Client Dashboard:", client.dashboard_link)

            start_time = time.time()
            futures = client.map(
                chunked_thresholding,
                ijks_chunked,
                batch_size=1,
                resources={"processes": 1},
            )
            for future in tqdm(as_completed(futures), total=len(ijks_chunked), smoothing=0):
                future.release()
                pass  # tqdm progress bar is nicer and shows remaining time
            print(f"Computing patch segmentations took {timedelta(seconds=int(time.time() - start_time))}")
        #cluster.close()

    return patched_seg


def threshold_ijk(ijk, aff, patched_seg, patch_to_coords, ijk_to_idx, conf):
    """
    Predicts segmentation from the affinities for one patch, and writes the result into patched_seg.
    """
    i, j, k = ijk
    #dask.distributed.print(f"processing: {ijk_to_idx[i, j, k]}")
    ((x_start, x_end), (y_start, y_end), (z_start, z_end)) = patch_to_coords[(i, j, k)]
    cur_aff = aff[
              :,
              max(0, x_start - conf.overlap - conf.surrounding): (
                  x_end + conf.surrounding if x_end is not None else None),
              max(0, y_start - conf.overlap - conf.surrounding): (
                  y_end + conf.surrounding if y_end is not None else None),
              max(0, z_start - conf.overlap - conf.surrounding): (
                  z_end + conf.surrounding if z_end is not None else None),
              ]

    cur_aff[np.isnan(cur_aff)] = 0.0
    cur_aff = np.clip(cur_aff, 0.0, 1.0)  # todo: enforce clip + not nan + not inf in aff inference

    # extend on all cut off sides
    cur_aff_tmp = cur_aff
    cur_aff = np.zeros(
        (
            aff.shape[0],
            conf.patch_size + (conf.overlap + 2 * conf.surrounding),
            conf.patch_size + (conf.overlap + 2 * conf.surrounding),
            conf.patch_size + (conf.overlap + 2 * conf.surrounding),
        )
    )

    x_start_tmp = (conf.overlap + conf.surrounding) if x_start == 0 else 0
    y_start_tmp = (conf.overlap + conf.surrounding) if y_start == 0 else 0
    z_start_tmp = (conf.overlap + conf.surrounding) if z_start == 0 else 0
    cur_aff[
        :,
        x_start_tmp: x_start_tmp + cur_aff_tmp.shape[1],
        y_start_tmp: y_start_tmp + cur_aff_tmp.shape[2],
        z_start_tmp: z_start_tmp + cur_aff_tmp.shape[3],
    ] = cur_aff_tmp

    if conf.mws:
        cur_aff = deepcopy(cur_aff).astype(np.float64)
        cur_aff[:3] += conf.mws_bias_short
        cur_aff[3:] += conf.mws_bias_long if conf.mws_bias_long is not None else 0.0

        cur_aff[:3] = np.clip(cur_aff[:3], 0, 1)
        cur_aff[3:] = np.clip(cur_aff[3:], -1, 0)

        mws_pred = mwatershed.agglom(
            affinities=cur_aff if conf.mws_bias_long is not None else cur_aff[:3],
            offsets=(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [conf.long_range, 0, 0],
                    [0, conf.long_range, 0],
                    [0, 0, conf.long_range],
                ]
                if conf.mws_bias_long is not None
                else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
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

    else:
        pred = conn_comps(cur_aff >= conf.thr)

    pred_no_surrounding = (
        pred[
            conf.surrounding:-conf.surrounding,
            conf.surrounding:-conf.surrounding,
            conf.surrounding:-conf.surrounding,
        ]
        if conf.surrounding > 0
        else pred
    )
    patched_seg[ijk_to_idx[i, j, k]] = pred_no_surrounding
    print(f"processed: {ijk_to_idx[i, j, k]}")
    return

@jit(nopython=True)
def conn_comps(hard_aff):
    visited = np.zeros(tuple(hard_aff.shape[1:]), dtype=np.bool_)
    seg = np.zeros(tuple(hard_aff.shape[1:]), dtype=np.uint32)
    cur_id = 1
    cur_id_used = False
    for i in range(visited.shape[0]):
        for j in range(visited.shape[1]):
            for k in range(visited.shape[2]):
                if hard_aff[
                   :, i, j, k
                   ].any() and not visited[i, j, k]:  # if foreground
                    cur_to_visit = [(i, j, k)]  # todo: use 3 array.array instead? or np.array and append?
                    visited[i, j, k] = True
                    while len(cur_to_visit) > 0:
                        x, y, z = cur_to_visit.pop()
                        # if not visited[x, y, z]:
                        # visited[x, y, z] = True
                        seg[x, y, z] = cur_id
                        cur_id_used = True
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
                    if cur_id_used:
                        cur_id += 1
                        cur_id_used = False
    return seg




def compute_fragment_agglomeration(
        patched_seg,
        aff,
        conf,
        ijk_to_idx,
        patch_to_coords,
):
    """
    From the patched segmentation, merges fragments at the border of adjacent patches.
    Computes flattened agglomeration, a dict with the keys (i, j, k, idx), where (i, j, k) is a patch and idx is an id in that patch, and values of a global id for this fragment.
    """

    print("Computing fragment agglomeration...")
    data_chunked = list(
        enumerate(chunk_list(list(patch_to_coords.items()), chunk_size=8))
    )

    if conf.use_slurm:
        cluster = SLURMCluster(
            cores=16,
            memory="500GB",
            processes=1,
            log_directory=f"/cajal/scratch/projects/misc/zuzur/slurm_logs/agglo/",
            walltime="3:00:00"
        )
        cluster.adapt(minimum_jobs=1, maximum_jobs=32)
    else:
        cluster = LocalCluster(
            n_workers=min(os.cpu_count(), 16),
            threads_per_worker=1,
            local_directory=conf.path_root + "/dask-worker-space/"
        )

    with Client(cluster) as client:
        print("Dask Client Dashboard:", client.dashboard_link)

        start_time = time.time()
        futures = client.map(
            partial(
                compute_agglomeration_part,
                patched_seg=patched_seg,
                ijk_to_idx=ijk_to_idx,
                conf=conf,
                aff=aff
            ),
            data_chunked
        )

        fragment_agglomeration = {}
        # agglomerate all fragments from all chunks as they complete
        for future, frag_aggl in tqdm(as_completed(futures, with_results=True), total=len(data_chunked), smoothing=0):
            for k, v in frag_aggl.items():
                fragment_agglomeration.setdefault(k, set()).update(v)

        print(f"Computing fragment agglomeration in patches took {timedelta(seconds=time.time() - start_time)}")
    #cluster.close()

    fragment_agglomeration_flattened = flatten_agglomeration(fragment_agglomeration, f"{conf.path_root}/agglo_pkl_chunks")

    return fragment_agglomeration_flattened


def compute_agglomeration_part(
        idx_samples,
        patched_seg,
        ijk_to_idx,
        conf,
        aff
):
    """
    Merges neighboring voxels from different cubes, creates a graph with connected fragments.
    Args:
        idx_samples: idx of the chunk, chunked patch indices

    Returns:
         fragment_agglomeration: Dictionary representing a graph between vertices (i, j, k, idx)
         where (i, j, k) is a patch index, and idx a fragment id in this patch.
         An edge means the fragments should be merged.
    """
    print("Entered compute_agglomeration_part", flush=True)
    idx, samples = idx_samples
    fragment_agglomeration = {}
    for sample in samples:
        print(f"{datetime.now()}: computing {sample}", flush=True)
        (i, j, k), ((x_start, x_end), (y_start, y_end), (z_start, z_end)) = sample

        # for x,y,z get the last slice of the current cube (l, low) and the first slice of the next cube (h, high)
        # they overlap, the voxels should have the same id

        if x_end is not None:
            if conf.overlap > 0:
                result_l = patched_seg[ijk_to_idx[i, j, k], -conf.overlap:]
                result_h = patched_seg[ijk_to_idx[i + 1, j, k], :conf.overlap]
                uniques = compute_uniques(conf.do_overlap_filter, result_h, result_l, min_overlap=conf.min_overlap)
            else:
                # merge according to short range affinities between each pair of IDs in neighboring cubes
                cur_aff = (
                        aff[0, x_end - 1: x_end, y_start:y_end, z_start:z_end] >= conf.merge_thr
                )
                # todo: this simple thresholding can re-introduce catastrophic mergers.
                #  tune thr? use mean instead of max? better use overlap strategy?

                result_l = patched_seg[ijk_to_idx[i, j, k], -1:][
                           : cur_aff.shape[0], : cur_aff.shape[1], : cur_aff.shape[2]
                           ][cur_aff]
                result_h = patched_seg[ijk_to_idx[i + 1, j, k], :1][
                           : cur_aff.shape[0], : cur_aff.shape[1], : cur_aff.shape[2]
                           ][cur_aff]
                combined = np.stack([result_l, result_h]).T
                uniques = np.unique(combined, axis=0)

            for id_l, id_h in uniques:
                if id_l > 0 and id_h > 0:
                    fragment_agglomeration.setdefault((i + 1, j, k, id_h), set()).add(
                        (i, j, k, id_l)
                    )
                    fragment_agglomeration.setdefault((i, j, k, id_l), set()).add(
                        (i + 1, j, k, id_h)
                    )

        if y_end is not None:
            if conf.overlap > 0:
                result_l = patched_seg[ijk_to_idx[i, j, k], :, -conf.overlap:]
                result_h = patched_seg[ijk_to_idx[i, j + 1, k], :, :conf.overlap]
                uniques = compute_uniques(conf.do_overlap_filter, result_h, result_l, min_overlap=conf.min_overlap)
            else:
                cur_aff = (
                        aff[1, x_start:x_end, y_end - 1: y_end, z_start:z_end] >= conf.merge_thr
                )
                result_l = patched_seg[ijk_to_idx[i, j, k], :, -1:][
                           : cur_aff.shape[0], : cur_aff.shape[1], : cur_aff.shape[2]
                           ][cur_aff]
                result_h = patched_seg[ijk_to_idx[i, j + 1, k], :, :1][
                           : cur_aff.shape[0], : cur_aff.shape[1], : cur_aff.shape[2]
                           ][cur_aff]
                combined = np.stack([result_l, result_h]).T
                uniques = np.unique(combined, axis=0)

            for id_l, id_h in uniques:
                if id_l > 0 and id_h > 0:
                    fragment_agglomeration.setdefault((i, j + 1, k, id_h), set()).add(
                        (i, j, k, id_l)
                    )
                    fragment_agglomeration.setdefault((i, j, k, id_l), set()).add(
                        (i, j + 1, k, id_h)
                    )

        if z_end is not None:
            if conf.overlap > 0:
                result_l = patched_seg[ijk_to_idx[i, j, k], :, :, -conf.overlap:]
                result_h = patched_seg[ijk_to_idx[i, j, k + 1], :, :, :conf.overlap]
                uniques = compute_uniques(conf.do_overlap_filter, result_h, result_l, min_overlap=conf.min_overlap)
            else:
                cur_aff = (
                        aff[2, x_start:x_end, y_start:y_end, z_end - 1: z_end] >= conf.merge_thr
                )
                result_l = patched_seg[ijk_to_idx[i, j, k], :, :, -1:][
                           : cur_aff.shape[0], : cur_aff.shape[1], : cur_aff.shape[2]
                           ][cur_aff]
                result_h = patched_seg[ijk_to_idx[i, j, k + 1], :, :, :1][
                           : cur_aff.shape[0], : cur_aff.shape[1], : cur_aff.shape[2]
                           ][cur_aff]
                combined = np.stack([result_l, result_h]).T
                uniques = np.unique(combined, axis=0)

            for id_l, id_h in uniques:
                if id_l > 0 and id_h > 0:
                    fragment_agglomeration.setdefault((i, j, k + 1, id_h), set()).add(
                        (i, j, k, id_l)
                    )
                    fragment_agglomeration.setdefault((i, j, k, id_l), set()).add(
                        (i, j, k + 1, id_h)
                    )
        print(f"{datetime.now()}: done", flush=True)
    return fragment_agglomeration


def compute_uniques(do_overlap_filter, result_h, result_l, min_overlap=0.9):
    if do_overlap_filter:
        result_l_ccs = cc3d.connected_components(result_l, connectivity=6)
        result_h_ccs = cc3d.connected_components(result_h, connectivity=6)

        l_ccs_to_l = np.unique(
            np.stack([result_l_ccs.flatten(), result_l.flatten()]), axis=1
        )
        l_ccs_to_l = {l_ccs: l for l_ccs, l in l_ccs_to_l.T}

        h_ccs_to_h = np.unique(
            np.stack([result_h_ccs.flatten(), result_h.flatten()]), axis=1
        )
        h_ccs_to_h = {h_ccs: h for h_ccs, h in h_ccs_to_h.T}

        combined_ccs = np.stack([result_l_ccs.flatten(), result_h_ccs.flatten()])
        uniques_ccs, counts_ccs = np.unique(combined_ccs, axis=1, return_counts=True)
        uniques_ccs = uniques_ccs.T
        # uniques_ccs = exact_overlap_filter(uniques_ccs)
        uniques_ccs = mutual_largest_overlap_filter(
            counts_ccs, uniques_ccs,
            min_overlap=min_overlap
        )

        uniques = [
            (l_ccs_to_l[l_ccs], h_ccs_to_h[h_ccs]) for l_ccs, h_ccs in uniques_ccs
        ]

    else:
        combined = np.stack([result_l.flatten(), result_h.flatten()]).T
        uniques, counts = np.unique(combined, axis=0, return_counts=True)
    return uniques


def exact_overlap_filter(uniques):
    #  keep only non-zero IDs with mutually exact corresponding count to reduce merge errors

    l_partners = {}
    h_partners = {}
    for id_l, id_h in uniques:
        l_partners.setdefault(id_l, []).append(id_h)
        h_partners.setdefault(id_h, []).append(id_l)

    uniques = [
        (id_l, id_h)
        for id_l, id_h in uniques
        if (
                (len(l_partners[id_l]) == len(h_partners[id_h]) == 1)
                and id_l != 0
                and id_h != 0
        )
    ]
    return uniques


def mutual_largest_overlap_filter(
        counts,
        uniques,
        min_overlap=0.5,
        # 1.0: exact overlap, 0.0: any overlap, 0.5: at least half of total count
):
    #  keep only non-zero IDs with mutually largest corresponding count to reduce merge errors

    # todo: Merge only perfect matches? i.e. except for 0 there are no other IDs in the uniques (ignore counts)

    highest_count_l = {}
    highest_count_h = {}

    total_count_l = {}
    total_count_h = {}

    for (id_l, id_h), count in zip(uniques, counts):
        total_count_l[id_l] = total_count_l.get(id_l, 0) + count
        total_count_h[id_h] = total_count_h.get(id_h, 0) + count

        # if id_l > 0 and id_h > 0: don't filter background here: if there is more overlap with background than with another ID, it should not be merged
        cur_highest_count_l, cur_highest_id_l = highest_count_l.setdefault(
            id_l, (-1, -1)
        )
        cur_highest_count_h, cur_highest_id_h = highest_count_h.setdefault(
            id_h, (-1, -1)
        )

        if count > cur_highest_count_l:
            highest_count_l[id_l] = (count, id_h)
        if count > cur_highest_count_h:
            highest_count_h[id_h] = (count, id_l)
    # uniques = [(id_l, id_h) for id_l, (count, id_h) in highest_count_l.items()] + [
    #    (id_l, id_h) for id_h, (count, id_l) in highest_count_h.items()
    # ] # for non ccs case: but snakes get split because only 1 assignment per ID but should be several

    uniques = [
        (id_l, id_h)
        for id_l, (count, id_h) in highest_count_l.items()
        if highest_count_h[id_h][1] == id_l
           and count >= min_overlap * total_count_l[id_l]
           and count >= min_overlap * total_count_h[id_h]
           and id_l != 0
           and id_h != 0
           and count >= 2  # single voxel branches get split
    ]

    return np.array(uniques)


def flatten_agglomeration(fragment_agglomeration, output_dir):
    """
    Computes connected components in the fragment agglomeration graph, and relabels the fragments with ids starting from 1.
    Args:
        fragment_agglomeration: dictionary with keys (i, j, k, id) indicating cube (i, j, k) and component id in that cube, and values a set of (i, j, k, id) in other cubes that should be connected
    Returns:
        fragment_agglomeration_flattened: dictionary with keys (i, j, k, id) and values the global component index
    """
    cur_id = 1
    fragment_agglomeration_flattened = dict()
    fragment_agglomeration_final = dict()
    flattened_ids = set()
    chunk_n = 0
    os.makedirs(output_dir, exist_ok=True)
    for position_id in tqdm(fragment_agglomeration):  # (i, j, k, idx) = position_id
        if position_id not in flattened_ids:
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
                flattened_ids.add(v)
                if len(fragment_agglomeration_flattened) >= 10_000_000:
                    file_path = os.path.join(output_dir, f"chunk_{chunk_n:02}.pkl")
                    with open(file_path, "wb") as f:
                        pickle.dump(fragment_agglomeration_flattened, f)
                    print(f"Saved {len(fragment_agglomeration_flattened)} items to {file_path}")
                    fragment_agglomeration_final.update(fragment_agglomeration_flattened)
                    fragment_agglomeration_flattened = dict()
                    chunk_n += 1
            cur_id += 1

    if fragment_agglomeration_flattened:
        file_path = os.path.join(output_dir, f"chunk_{chunk_n:02}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(fragment_agglomeration_flattened, f)
        print(f"Saved final {len(fragment_agglomeration_flattened)} items to {file_path}")
        fragment_agglomeration_final.update(fragment_agglomeration_flattened)

    return fragment_agglomeration_final


def relabel_cube_batched_wrapped(kwargs):
    return relabel_cube_batched(**kwargs)


def relabel_globally(
        fragment_agglomeration_flattened,
        patched_seg,
        aff,
        conf,
        patch_to_coords,
        ijk_to_idx,
):
    """
    Unite indexing within multiple cubes - if an object spans multiple cubes, it should have the same index everywhere
    """
    print(f"Global relabeling...")
    print(f"Agglomerated segmentation: {conf.path_root}/agglomerated_seg.zarr")
    agglomerated_seg = zarr.zeros(
        (aff.shape[1:]),
        chunks=(conf.patch_size, conf.patch_size, conf.patch_size),
        dtype=np.uint64,  # cheap because of zarr compression
        store=f"{conf.path_root}/agglomerated_seg.zarr",
    )

    cubes = list(patch_to_coords.items())
    cubes_batched = chunk_list(cubes, 100)

    if not conf.use_parallelization:
        result = list(tqdm(map(
            partial(
                relabel_cube,
                patched_seg=patched_seg,
                fragment_agglomeration_flattened=fragment_agglomeration_flattened,
                ijk_to_idx=ijk_to_idx,
                agglomerated_seg=agglomerated_seg,
                conf=conf
            ),
            cubes), total=len(cubes)))
    else:
        if not conf.use_slurm:
            with ThreadPoolExecutor(max_workers=32) as executor:
                result = list(tqdm(executor.map(
                    partial(
                        relabel_cube,
                        patched_seg=patched_seg,
                        fragment_agglomeration_flattened=fragment_agglomeration_flattened,
                        ijk_to_idx=ijk_to_idx,
                        agglomerated_seg=agglomerated_seg,
                        conf=conf
                    ),
                    cubes,
                    chunksize=8
                ), total=len(cubes), smoothing=0))
        else:
            cluster = SLURMCluster(
                cores=32,
                memory="500GB",
                processes=1,
                worker_extra_args=["--resources", "processes=1"],
                log_directory=f"/cajal/scratch/projects/misc/zuzur/slurm_logs/relabel/",
                walltime="24:00:00"
            )
            cluster.adapt(minimum_jobs=1, maximum_jobs=32)

            with Client(cluster) as client:
                print("Dask relabeling Client Dashboard:", client.dashboard_link)

                start_time = time.time()
                print(len(ijk_to_idx))
                with open(f"{conf.path_root}/ijk_to_idx.pkl", "wb") as f:
                    pickle.dump(ijk_to_idx, f)
                #ijk_to_idx_future = client.scatter(ijk_to_idx, broadcast=True)
                #print(f"Broadcasted ijk_to_idx in {timedelta(seconds=int(time.time() - start_time))}")
                #print(list(ijk_to_idx.items())[:10])
                configs = [
                    {
                        "cubes": cubes,
                        "patched_seg": patched_seg,
                        "fragment_agglomeration_chunks_path": f"{conf.path_root}/agglo_pkl_chunks",
                        #"ijk_to_idx": ijk_to_idx_future,
                        "agglomerated_seg": agglomerated_seg,
                        "conf": conf,
                    }
                    for cubes in cubes_batched
                ]
                futures = client.map(
                    relabel_cube_batched_wrapped,
                    configs,
                    resources={'processes': 1},
                    #batch_size=1
                )
                for _ in tqdm(as_completed(futures), total=len(cubes_batched), smoothing=0):
                    pass  # tqdm progress bar
                print(f"Relabeling fragments took {timedelta(seconds=int(time.time() - start_time))}")
            #cluster.close()

    return agglomerated_seg


def relabel_cube_batched(cubes, patched_seg, fragment_agglomeration_chunks_path, agglomerated_seg, conf):
    print(f"{datetime.now()}: start relabel_cube_batched", flush=True)
    with open(f"{conf.path_root}/ijk_to_idx.pkl", "rb") as f:
        ijk_to_idx = pickle.load(f)
    print(f"{datetime.now()}: ijk_to_idx loaded", flush=True)
    fragment_agglomeration_flattened = dict()
    chunk_files = os.listdir(fragment_agglomeration_chunks_path)
    for chunk_file in tqdm(chunk_files):
        with open(os.path.join(fragment_agglomeration_chunks_path, chunk_file), "rb") as f:
            chunk = pickle.load(f)
            fragment_agglomeration_flattened.update(chunk)
    print(f"{datetime.now()}: fragment_agglomeration_flattened loaded {len(fragment_agglomeration_flattened)}", flush=True)

    print(f"{datetime.now()}: Relabeling cube chunk", flush=True)
    for cube in cubes:
        relabel_cube(cube, patched_seg, fragment_agglomeration_flattened, ijk_to_idx, agglomerated_seg, conf)
    print(f"{datetime.now()}: End relabeling cube chunk", flush=True)


def relabel_cube(cube, patched_seg, fragment_agglomeration_flattened, ijk_to_idx, agglomerated_seg, conf):
    """
    If an object spans multiple cubes, relabel the indices to be the same
    """
    (i, j, k), ((x_start, x_end), (y_start, y_end), (z_start, z_end)) = cube
    # todo: for dask: fragment_agglomeration_flattened is big, load it in here from disk (once for several items?)

    cube = patched_seg[ijk_to_idx[i, j, k]]
    perm = [0]
    for idx in range(1, int(cube.max()) + 1):  # assuming cube has continuous indices from 0 to max
        if (i, j, k, idx) in fragment_agglomeration_flattened:  # object (idx) continued in neighboring cube -> already has a unique id
            perm.append(fragment_agglomeration_flattened[i, j, k, idx])
        else:  # object only in this cube
            # use upper 32 bits to indicate cube, lower 32 bits to indicate id
            perm.append((ijk_to_idx[i, j, k] + 1) * np.uint64(2 ** 32) + idx)
    perm = np.array(perm, dtype=np.uint64)

    relabeled = perm[cube[conf.overlap:, conf.overlap:, conf.overlap:]]
    if len(perm) > 1:
        print(cube.shape, np.max(cube), (i,j,k), ijk_to_idx[i,j,k])
        print(len(perm), perm[1] if len(perm) > 1 else "out of bounds")
        print(np.max(relabeled))
    # can't just use agglomerated_seg[x_start:x_end, y_start:y_end, z_start:z_end] = relabeled
    #  because chunks at the boundary can be smaller
    cur_shape = agglomerated_seg[x_start:x_end, y_start:y_end, z_start:z_end].shape
    # this is exactly 1 chunk (chunk-borders) -> no race conditions / overwriting
    agglomerated_seg[x_start:x_end, y_start:y_end, z_start:z_end] = relabeled[: cur_shape[0], : cur_shape[1],
                                                                    : cur_shape[2]]


def size_filter_relabel(seg, conf):
    """
    Filters out segments that are too small (less than minsize voxels)
    and relabels the remaining segments contiguously from 1.
    """
    start_time = time.time()

    block_indices = [(i, j, k) for i in range(seg.cdata_shape[0]) for j in range(seg.cdata_shape[1]) for k in
                     range(seg.cdata_shape[2])]
    block_indices_batched = chunk_list(block_indices, 16)
    combined_counter = Counter()

    print("Counting occurences of fragments...")
    if not conf.use_parallelization:
        result = list(tqdm(map(
            partial(batched_unique, seg=seg),
            block_indices_batched,
        ), total=len(block_indices_batched), smoothing=0))
        for counter in tqdm(result, total=len(block_indices_batched), smoothing=0):
            combined_counter.update(counter)
    else:
        if not conf.use_slurm:
            with ThreadPoolExecutor(max_workers=8) as executor:
                result = list(tqdm(executor.map(
                    partial(batched_unique, seg=seg),
                    block_indices_batched,
                ), total=len(block_indices_batched), smoothing=0))
                for counter in tqdm(result, total=len(block_indices_batched), smoothing=0):
                    combined_counter.update(counter)
        else:
            cluster = SLURMCluster(
                cores=32,
                memory="800GB",
                log_directory=f"/cajal/scratch/projects/misc/zuzur/slurm_logs/count/",
                processes=1,
                worker_extra_args=["--resources processes=1"],
                walltime="12:00:00"
            )
            cluster.adapt(minimum_jobs=1, maximum_jobs=32)
            with Client(cluster) as client:
                print("Dask counting Client Dashboard:", client.dashboard_link)
                futures = client.map(batched_unique, block_indices_batched, seg=seg, resources={'processes': 1})

                for future in (pbar := tqdm(as_completed(futures), total=len(block_indices_batched), smoothing=0)):
                    counter = future.result()
                    filtered = {k: v for k, v in counter.items() if v > 10}
                    combined_counter.update(filtered)
                    del counter
                    del future
                    mem = psutil.Process().memory_full_info()
                    rss = mem.rss / 1e9
                    vms = mem.vms / 1e9
                    pbar.set_postfix(rss=f"{rss:.1f} GB", vms=f"{vms:.1f} GB")
                    gc.collect()

            #cluster.close()

    remaining_ids = {id for id, count in combined_counter.items() if count > conf.minsize}
    id_mapping_remaining = {old_id: new_id for new_id, old_id in enumerate(sorted(remaining_ids))}
    assert id_mapping_remaining[0] == 0
    with open(f"{conf.path_root}/id_mapping.csv", 'w') as f:
        for k, v in id_mapping_remaining.items():
            f.write(f"{k} {v}\n")

    print(f"Store: {conf.path_root}/relabeled_seg.zarr")
    relabeled_seg = zarr.zeros(
        seg.shape,
        chunks=seg.chunks,
        dtype=np.uint32,
        store=f"{conf.path_root}/relabeled_seg.zarr",
    )

    print("Filtering out small fragments and relabeling contiguously...")
    if not conf.use_parallelization:
            result = list(tqdm(map(partial(batched_relabel, seg=seg, relabeled_seg=relabeled_seg, conf=conf),
                                     block_indices_batched)))
    else:
        if not conf.use_slurm:
            with ThreadPoolExecutor(max_workers=16) as executor:
                result = list(tqdm(executor.map(partial(batched_relabel, seg=seg, relabeled_seg=relabeled_seg, conf=conf),
                                     block_indices_batched)))
        else:
            cluster = SLURMCluster(
                cores=16,
                memory="500GB",
                log_directory=f"/cajal/scratch/projects/misc/zuzur/slurm_logs/filter/",
            )
            cluster.adapt(minimum_jobs=1, maximum_jobs=32)
            with Client(cluster) as client:
                print("Dask relabeling Client Dashboard:", client.dashboard_link)
                futures = client.map(partial(batched_relabel, seg=seg, relabeled_seg=relabeled_seg, conf=conf),
                                     block_indices_batched)
                for _ in tqdm(as_completed(futures), total=len(block_indices_batched), smoothing=0):
                    pass
            #cluster.close()
    print(f"Filtering small fragments and relabeling took {timedelta(seconds=int(time.time() - start_time))}")
    return relabeled_seg


def batched_unique(block_indices, seg):
    print(f"{datetime.now()}: start count", flush=True)
    c = Counter()
    for idx in block_indices:
        chunk_series = pd.Series(seg.blocks[idx].ravel())
        c.update(chunk_series.value_counts().to_dict())
    print(f"{datetime.now()}: end count", flush=True)
    return c


def batched_relabel(block_indices, seg, relabeled_seg, conf):
    mapping = {}
    with open(f"{conf.path_root}/id_mapping.csv", 'r') as f:
        for line in f:
            key, value = line.split()
            mapping[int(key)] = int(value)
    for block_index in block_indices:
        block = seg.blocks[block_index]
        masked_block = fastremap.mask_except(block, list(mapping.keys()))
        relabeled_block = fastremap.remap(masked_block, mapping)
        relabeled_seg.blocks[block_index] = relabeled_block
    return None


def main(conf):
    print(conf)
    aff = zarr.open(conf.aff_path, mode="r")

    if len(conf.thresholds) > 0:
        thr = conf.thresholds[0]

        path_root = f"{conf.path_base}/{f'thr_{thr}'}/"
        print(f"Root path: {path_root}")
        os.makedirs(path_root, exist_ok=True)

        with open_dict(conf):
            conf.path_root = path_root
            conf.thr = thr

        start_time = time.time()
        segmentation = patched_thresholding(
            aff,
            conf
        )
        print(f"Patched thresholding took {timedelta(seconds=int(time.time() - start_time))}")

    elif len(conf.mws_biases_short) > 0:
        biases = list(itertools.product(conf.mws_biases_short, conf.mws_biases_long))
        for (short, long) in biases:
            print(f"SEGMENTATION FOR {short}, {long}")
            path_root = f"{conf.path_base}/{f'mws_{short}_{long}'}/"
            print(f"Root path: {path_root}")
            os.makedirs(path_root, exist_ok=False)
            with open_dict(conf):
                conf.path_root = path_root
                conf.mws_bias_short = short
                conf.mws_bias_long = long

            start_time = time.time()
            segmentation = patched_thresholding(
                aff,
                conf
            )
            print(f"Patched thresholding took {timedelta(seconds=int(time.time() - start_time))}")
        pass
    return


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


@hydra.main(config_path=".")
def main_wrapper(conf: DictConfig):
    return main(conf)

if __name__ == "__main__":
    main_wrapper()