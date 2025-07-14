import argparse
import os
import pickle
from typing import Tuple
from collections import defaultdict, Counter, deque

import dask.array as da
import neuroglancer
import numpy as np
from dask.array import clip
from dask_image.ndfilters import gaussian
from neuroglancer import CoordinateSpace, LocalVolume, Viewer, SegmentationLayer
import zarr
from tqdm import tqdm
import networkx as nx
from networkx import connected_components, subgraph, convert_node_labels_to_integers

from data import comp_affinities

"""
Visualizes where the errors in the prediction are.
"""

class SkeletonSource(neuroglancer.skeleton.SkeletonSource):
    def __init__(self, dimensions, skel):
        super().__init__(dimensions)
        self.skel = skel

    def get_skeleton(self, i):
        print(f"Getting skeleton for {i}")
        cv_s = self.skel[i]
        try:
            s = neuroglancer.skeleton.Skeleton(vertex_positions=(cv_s.vertices / [9,9,20]), edges=cv_s.edges)
        except Exception as e:
            print(e)
        return s



# Coordinate spaces
COORDS = {
    "standard": CoordinateSpace(names=['x', 'y', 'z'], units=['nm', 'nm', 'nm'], scales=[9, 9, 20]),
    "standard_c": CoordinateSpace(names=["x", "y", "z", "c^"], units=["nm", "nm", "nm", ""], scales=[9, 9, 20, 1]),
    "liconn": CoordinateSpace(names=['x', 'y', 'z'], units=['nm', 'nm', 'nm'], scales=[9, 9, 12]),
    "liconn_c": CoordinateSpace(names=["x", "y", "z", "c^"], units=["nm", "nm", "nm", ""], scales=[9, 9, 12, 1]),
    "aff": CoordinateSpace(names=[ "c^", "x", "y", "z"], units=["", "nm", "nm", "nm"], scales=[1, 9, 9, 20]),
}


def load_data(data_path: str):
    """Load image, segmentation, and skeleton data."""
    seg = da.from_zarr(os.path.join(data_path, "data.zarr", "seg")).astype(np.uint32)[500:1000, 500:1000, 500:1000]
    img = da.from_zarr(os.path.join(data_path, "data.zarr", "img"))[500:1000, 500:1000, 500:1000]
    skel = da.from_zarr(os.path.join(data_path, "data.zarr", "skel")).astype(np.uint32)
    with open(os.path.join(data_path, "skeleton_dense.pkl"), 'rb') as f:
        skel_pkl = pickle.load(f)
    return img, seg, skel, skel_pkl


def add_image_layer(s, name: str, img: da.Array, c_res: CoordinateSpace):
    """Add an image layer to the viewer."""
    layer = LocalVolume(img, dimensions=c_res)
    s.layers.append(name=f'img_{name}', layer=layer)


def add_segmentation_layer(s, name: str, seg: da.Array, skel: dict, res: CoordinateSpace):
    """Add a segmentation layer to the viewer."""
    layer = SegmentationLayer(
        source=[LocalVolume(seg, dimensions=res, volume_type="segmentation"), SkeletonSource(res, skel)],
        skeleton_shader='void main() { emitRGB(vec3(.3, .8, .76)); }',
        mesh_silhouette_rendering=2.0
    )
    layer.skeleton_rendering.mode3d = "lines" #"lines_and_points"
    s.layers.append(name=f'seg_{name}', layer=layer)


def create_viewer(args) -> Viewer:
    """Create and configure the Neuroglancer viewer."""
    neuroglancer.set_server_bind_address('localhost', args.port)
    viewer = Viewer()

    with viewer.txn() as s:
        img, seg, skel, skel_pkl = load_data(args.data_path)

        coord_space = COORDS["standard_c"]
        add_image_layer(s, "gt", img, coord_space)

        seg_space = COORDS["standard"]
        add_segmentation_layer(s, "gt", seg, skel_pkl, seg_space)

        if True:
            aff, _ = comp_affinities(seg)
            aff = da.from_array(aff).astype(np.float32)
            s.layers["gt_aff"] = neuroglancer.ImageLayer(
                source=neuroglancer.LocalVolume(
                    aff[:3], dimensions=COORDS["aff"], voxel_offset=[0, 0, 0, 0]
                ),
                shader="""void main() {
                                    emitRGB(vec3(toNormalized(getDataValue(0)),
                                    toNormalized(getDataValue(1)),
                                    toNormalized(getDataValue(2))));
                                    }""",
            )

        pred_aff = da.from_zarr(args.pred_path).astype(np.float32)
        s.layers["pred_aff"] = neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(
                pred_aff[:3], dimensions=COORDS["aff"], voxel_offset=[0, 0, 0, 0]
            ),
            shader="""void main() {
                                emitRGB(vec3(toNormalized(getDataValue(0)),
                                toNormalized(getDataValue(1)),
                                toNormalized(getDataValue(2))));
                                }""",
        )


        print("If on a remote server, remember port forwarding. Meshes may take time to load.")

    return viewer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neuroglancer Viewer for NISB project")
    parser.add_argument("--data_path", type=str, default="/cajal/scratch/users/zuzur/NISB_corrected/base/val/seed100", help="Directory which contains data.zarr with segmentation + EM image + skeleton, and skeleton.pkl")
    parser.add_argument("--pred_path", type=str, default="/cajal/scratch/projects/misc/zuzur/test.zarr/aff")
    parser.add_argument("--port", type=int, default=8589, help="Port to run the viewer")
    args = parser.parse_args()

    viewer = create_viewer(args)
    print(viewer.get_viewer_url())
    input("Press Enter to quit")
