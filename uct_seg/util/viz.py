"""Neuroglancer + neuroglancer-precomputed helpers.

Optional dep: `neuroglancer` (install via `pip install -e ".[viz]"`).
"""
from __future__ import annotations


def ng_layer(data, res, offset=(0, 0, 0), kind: str = "segmentation"):
    """Legacy LocalVolume layer (positional voxel_size API)."""
    import neuroglancer
    return neuroglancer.LocalVolume(
        data, volume_type=kind, voxel_size=list(res), offset=list(offset)
    )


def ng_layer3(data, res, offset=(0, 0, 0), kind: str = "segmentation"):
    """ZYX numpy -> XYZ neuroglancer LocalVolume with explicit CoordinateSpace."""
    import neuroglancer
    dim = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=list(res)
    )
    return neuroglancer.LocalVolume(
        data.transpose([2, 1, 0]),
        volume_type=kind,
        dimensions=dim,
        voxel_offset=list(offset),
    )


def launch_viewer(ip: str = "localhost", port: int = 9092):
    """Bind a neuroglancer server and return a fresh viewer."""
    import neuroglancer
    neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
    return neuroglancer.Viewer()


# Legacy aliases
ngLayer = ng_layer
ngLayer3 = ng_layer3
