import argparse
import os
import pickle
from typing import Tuple

import dask.array as da
import networkx
import neuroglancer
import numpy as np
from dask.array import clip
from dask_image.ndfilters import gaussian
from networkx import connected_components, convert_node_labels_to_integers, subgraph
from neuroglancer import CoordinateSpace, LocalVolume, Viewer, SegmentationLayer


class SkeletonSource(neuroglancer.skeleton.SkeletonSource):
    def __init__(self, dimensions, skels):
        super().__init__(dimensions)
        self.skels = skels
        self.id_to_comp = {
            skels.nodes[next(iter(c))]["id"]: c for c in connected_components(skels)
        }

    def get_skeleton(self, i):
        comp = self.id_to_comp.get(i)
        if not comp:
            return None
        comp = convert_node_labels_to_integers(subgraph(self.skels, comp))
        vertices = [comp.nodes[v]["index_position"] for v in range(len(comp))]
        return neuroglancer.skeleton.Skeleton(vertex_positions=vertices, edges=list(comp.edges))


# Coordinate spaces
COORDS = {
    "standard": CoordinateSpace(names=['x', 'y', 'z'], units=['nm', 'nm', 'nm'], scales=[9, 9, 20]),
    "standard_c": CoordinateSpace(names=["x", "y", "z", "c^"], units=["nm", "nm", "nm", ""], scales=[9, 9, 20, 1]),
    "liconn": CoordinateSpace(names=['x', 'y', 'z'], units=['nm', 'nm', 'nm'], scales=[9, 9, 12]),
    "liconn_c": CoordinateSpace(names=["x", "y", "z", "c^"], units=["nm", "nm", "nm", ""], scales=[9, 9, 12, 1])
}


def load_data(base_path: str) -> Tuple[da.Array, da.Array, dict]:
    """Load image, segmentation, and skeleton data."""
    seg = da.from_zarr(os.path.join(base_path, "data.zarr/seg")).astype(np.uint32)
    img = da.from_zarr(os.path.join(base_path, "data.zarr/img"))
    with open(os.path.join(base_path, "skeleton.pkl"), 'rb') as f:
        skels = pickle.load(f)
    return img, seg, skels


def add_image_layer(s, name: str, img: da.Array, c_res: CoordinateSpace, multichannel=False):
    """Add an image layer to the viewer."""
    layer = LocalVolume(img[:, :, :, :3] if multichannel else img, dimensions=c_res)
    shader = """
        void main() {
            emitRGB(vec3(
                toNormalized(getDataValue(0)),
                toNormalized(getDataValue(1)),
                toNormalized(getDataValue(2))
            ));
        }
    """ if multichannel else None

    s.layers.append(name=f'img_{name}', layer=layer, shader=shader)

    if multichannel:
        blurred_img = (clip(gaussian(img[:, :, :, :3], sigma=(2, 2, 2, 0)), 112, 144) - 112) * 8
        s.layers.append(name=f'img_blurred_{name}', layer=LocalVolume(blurred_img, dimensions=c_res), shader=shader)


def add_segmentation_layer(s, name: str, seg: da.Array, skels: dict, res: CoordinateSpace):
    """Add a segmentation layer to the viewer."""
    s.layers.append(
        name=f'seg_{name}',
        layer=SegmentationLayer(
            source=[LocalVolume(seg, dimensions=res, volume_type="segmentation"), SkeletonSource(res, skels)],
            skeleton_shader='void main() { emitRGB(colormapJet(affinity)); }',
            mesh_silhouette_rendering=2.0
        )
    )


def create_viewer(base_path: str, port: int) -> Viewer:
    """Create and configure the Neuroglancer viewer."""
    neuroglancer.set_server_bind_address('localhost', port)
    viewer = Viewer()

    with viewer.txn() as s:
        for i in range(8):
            seg = da.from_zarr(args.input + f"_{i}_seg.zarr").astype(np.uint32)
            img = da.from_zarr(args.input + f"_{i}_img.zarr")
            skels = networkx.Graph()

            coord_space = COORDS["standard_c"]
            add_image_layer(s, f"img_{i}", img, coord_space)

            seg_space = COORDS["standard"]
            add_segmentation_layer(s, f"seg_{i}", seg, skels, seg_space)

        print("If on a remote server, remember port forwarding. Meshes may take time to load.")

    return viewer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neuroglancer Viewer for NISB project")
    parser.add_argument("--base_path", type=str, default="/cajal/nvmescratch/projects/NISB/",
                        help="Base path for NISB data")
    parser.add_argument("--input", type=str,
                        default="/cajal/scratch/projects/misc/zuzur/ss3/samplingPosColors-seed0/0_00087")
    parser.add_argument("--port", type=int, default=8085, help="Port to run the viewer")
    args = parser.parse_args()

    viewer = create_viewer(args.base_path, args.port)
    print(viewer.get_viewer_url())
    input("Press Enter to quit")
