import argparse
import pickle
from typing import Union, Dict, Any

import numpy as np
import zarr
from funlib.evaluate import rand_voi, expected_run_length, get_skeleton_lengths
from funlib.evaluate.run_length import SkeletonScores
from networkx import get_node_attributes
from tqdm import tqdm


def compute_metrics(pred_seg: Union[np.ndarray, zarr.Array], skel_path: str) -> Dict[str, Any]:
    """
    Compute various metrics for evaluating segmentation quality.

    Args:
        pred_seg: The predicted segmentation. Shape: (x, y, z).
        skel_path: Path to the skeleton file (pickle format).

    Returns:
        dict: A dictionary containing various metrics.
    """
    with open(skel_path, "rb") as f:
        skel = pickle.load(f)
    for node in tqdm(skel.nodes):  # slow for zarr
        x, y, z = skel.nodes[node]["index_position"]
        skel.nodes[node]["pred_id"] = pred_seg[x, y, z]

    voi_report = rand_voi(
        np.array(list(get_node_attributes(skel, "id").values())).astype(np.uint64),
        np.array(list(get_node_attributes(skel, "pred_id").values())).astype(np.uint64),
        return_cluster_scores=False,
    )
    voi_split = voi_report["voi_split"]
    voi_merge = voi_report["voi_merge"]
    voi_sum = voi_report["voi_split"] + voi_report["voi_merge"]

    erl_report = expected_run_length(
        skel,
        "id",
        "edge_length",
        get_node_attributes(skel, "pred_id"),
        skeleton_position_attributes=["nm_position"],
        return_merge_split_stats=True,
    )
    merge_stats = erl_report[1]["merge_stats"]
    n_mergers = sum([len(v) for v in merge_stats.values()])

    merge_stats.pop(0, None)  # ignore "mergers" with background
    merge_stats.pop(0.0, None)
    n_non0_mergers = sum([len(v) for v in merge_stats.values()])

    split_stats = erl_report[1]["split_stats"]
    n_splits = sum([len(v) for v in split_stats.values()])
    max_erl_report = expected_run_length(
        skel,
        "id",
        "edge_length",
        get_node_attributes(skel, "id"),
        skeleton_position_attributes=["nm_position"],
        return_merge_split_stats=True,
    )
    erl = erl_report[0]
    max_erl = max_erl_report[0]
    nerl = erl / max_erl

    metrics = {
        "voi_sum": voi_sum,
        "voi_split": voi_split,
        "voi_merge": voi_merge,
        "erl": erl,
        "max_erl": max_erl,
        "nerl": nerl,
        "n_mergers": n_mergers,
        "n_splits": n_splits,
        # "erl_report": erl_report,
        # "max_erl_report": max_erl_report,
        "voi_report": voi_report,
        "n_non0_mergers": n_non0_mergers,
    }

    adapted_nerls = {}
    for ignored_merger_size in [5, 20, 100, np.inf]:
        adapted_erl_report = adapted_erl(
            skel,
            "id",
            "edge_length",
            get_node_attributes(skel, "pred_id"),
            skeleton_position_attributes=["nm_position"],
            return_merge_split_stats=True,
            ignored_merger_size=ignored_merger_size
        )
        adapted_nerls[ignored_merger_size] = adapted_erl_report[0] / max_erl
    for ignored_mergers, adapted_nerl in adapted_nerls.items():
        metrics[f"nerl_{ignored_mergers}"] = adapted_nerl

    print(f"metrics: {metrics}")
    return metrics


def adapted_erl(
            skeletons,
            skeleton_id_attribute,
            edge_length_attribute,
            node_segment_lut,
            skeleton_lengths=None,
            skeleton_position_attributes=None,
            return_merge_split_stats=False,
            ignored_merger_size=0
    ):
    """
    Adapted from `funlib.evaluate.expected_run_length`.
    Args:
        ignored_merger_size:
            Maximum number of nodes in a segment with the same predicted and ground truth id,
            so that the segment is never considered to be part of a merger.
            ignored_merger_size=0 is the same as the original ERL.
    """

    if skeleton_position_attributes is not None:

        if skeleton_lengths is not None:
            raise ValueError(
                "If skeleton_position_attributes is given, skeleton_lengths"
                "should not be given")

        skeleton_lengths = get_skeleton_lengths(
            skeletons,
            skeleton_position_attributes,
            skeleton_id_attribute,
            store_edge_length=edge_length_attribute)

    total_skeletons_length = np.sum([l for _, l in skeleton_lengths.items()])

    res = evaluate_skeletons(
        skeletons,
        skeleton_id_attribute,
        node_segment_lut,
        return_merge_split_stats=return_merge_split_stats,
        ignored_merger_size=ignored_merger_size
    )

    if return_merge_split_stats:
        skeleton_scores, merge_split_stats = res
    else:
        skeleton_scores = res

    skeletons_erl = 0

    for skeleton_id, scores in skeleton_scores.items():

        skeleton_length = skeleton_lengths[skeleton_id]
        skeleton_erl = 0

        for segment_id, correct_edges in scores.correct_edges.items():
            correct_edges_length = np.sum([
                skeletons.edges[e][edge_length_attribute]
                for e in correct_edges])

            skeleton_erl += (
                    correct_edges_length *
                    (correct_edges_length / skeleton_length)
            )

        skeletons_erl += (
                (skeleton_length / total_skeletons_length) *
                skeleton_erl
        )

    if return_merge_split_stats:
        return skeletons_erl, merge_split_stats
    else:
        return skeletons_erl


def evaluate_skeletons(
        skeletons,
        skeleton_id_attribute,
        node_segment_lut,
        return_merge_split_stats=False,
        ignored_merger_size=1
):
    """
    Adapted from `funlib.evaluate.expected_run_length`.
    Args:
        ignored_merger_size:
            Maximum number of nodes in a segment with the same predicted and ground truth id,
            so that the segment is never considered to be part of a merger.
            ignored_merger_size=0 is the same as the original ERL.
    """

    # find all merging segments (skeleton edges on merging segments will be
    # counted as wrong)

    # pairs of (skeleton, segment), one for each node
    skeleton_segment = np.array([
        [data[skeleton_id_attribute], node_segment_lut[n]]
        for n, data in skeletons.nodes(data=True)
    ])

    # unique pairs of (skeleton, segment)
    # THIS IS THE CHANGE - only consider segments of bigger size than ignored_merger_size
    skeleton_segment, counts = np.unique(skeleton_segment, axis=0, return_counts=True)
    skeleton_segment = skeleton_segment[counts > ignored_merger_size]

    # number of times that a segment was mapped to a skeleton
    segments, num_segment_skeletons = np.unique(
        skeleton_segment[:, 1],
        return_counts=True)

    # all segments that merge at least two skeletons
    merging_segments = segments[num_segment_skeletons > 1]

    merging_segments_mask = np.isin(skeleton_segment[:, 1], merging_segments)
    merged_skeletons = skeleton_segment[:, 0][merging_segments_mask]
    merging_segments = set(merging_segments)

    merges = {}
    splits = {}

    if return_merge_split_stats:

        merged_segments = skeleton_segment[:, 1][merging_segments_mask]

        for segment, skeleton in zip(merged_segments, merged_skeletons):
            if segment not in merges:
                merges[segment] = []
            merges[segment].append(skeleton)

    merged_skeletons = set(np.unique(merged_skeletons))

    skeleton_scores = {}

    for u, v in skeletons.edges():

        skeleton_id = skeletons.nodes[u][skeleton_id_attribute]
        segment_u = node_segment_lut[u]
        segment_v = node_segment_lut[v]

        if skeleton_id not in skeleton_scores:
            scores = SkeletonScores()
            skeleton_scores[skeleton_id] = scores
        else:
            scores = skeleton_scores[skeleton_id]

        if segment_u == 0 or segment_v == 0:
            scores.ommitted += 1
            continue

        if segment_u != segment_v:
            scores.split += 1

            if return_merge_split_stats:
                if skeleton_id not in splits:
                    splits[skeleton_id] = []
                splits[skeleton_id].append((segment_u, segment_v))
            continue

        # segment_u == segment_v != 0
        segment = segment_u

        # potentially merged edge?
        if skeleton_id in merged_skeletons:
            if segment in merging_segments:
                scores.merged += 1
                continue

        scores.correct += 1

        if segment not in scores.correct_edges:
            scores.correct_edges[segment] = []
        scores.correct_edges[segment].append((u, v))

    if return_merge_split_stats:

        merge_split_stats = {
            'merge_stats': merges,
            'split_stats': splits
        }

        return skeleton_scores, merge_split_stats

    else:

        return skeleton_scores



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute segmentation quality metrics.")
    parser.add_argument("--pred_seg", type=str, required=True, help="Path to predicted segmentation (Zarr format)")
    parser.add_argument("--skel_path", type=str, required=True, help="Path to skeleton file (pickle format)")
    parser.add_argument("--load_to_memory", action="store_true",
                        help="Load the entire segmentation to memory to speedup")
    args = parser.parse_args()

    if args.pred_seg.endswith('.zarr'):
        pred_seg = zarr.open(args.pred_seg, mode='r')
    elif args.pred_seg.endswith('.npy'):
        pred_seg = np.load(args.pred_seg)
    if args.load_to_memory:
        pred_seg = pred_seg[:]
    compute_metrics(pred_seg, args.skel_path)
