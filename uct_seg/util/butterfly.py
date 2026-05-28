"""Butterfly JSON spec writer for legacy tile-pyramid viewers."""
from __future__ import annotations

import json


def write_bfly_v2(
    sz,
    numT,
    imN,
    tsz: int = 1024,
    tile_st=(0, 0),
    zPad=(0, 0),
    im_id=None,
    outName: str | None = None,
    st: int = 0,
    ndim: int = 1,
    rsz: float = 1,
    dt: str = "uint8",
):
    """Build a butterfly v2 manifest (per-section tile).

    sz = (depth, height, width); numT = (n_rows, n_columns); imN = lambda z: path.
    """
    if im_id is None:
        im_id = (
            list(range(zPad[0] + st, st, -1))
            + list(range(st, sz[0] + st))
            + list(range(sz[0] - 2 + st, sz[0] - zPad[1] - 2 + st, -1))
        )
    else:
        if zPad[0] > 0:
            im_id = [im_id[x] for x in range(zPad[0], 0, -1)] + im_id
        if zPad[1] > 0:
            im_id += [im_id[x] for x in range(sz[0] - 2, sz[0] - zPad[1] - 2, -1)]
    sec = [imN(x) for x in im_id]
    out = {
        "image": sec,
        "depth": sz[0] + sum(zPad),
        "height": sz[1],
        "width": sz[2],
        "tile_st": list(tile_st),
        "dtype": dt,
        "n_columns": numT[1],
        "n_rows": numT[0],
        "tile_size": tsz,
        "ndim": ndim,
        "tile_ratio": rsz,
    }
    if outName is None:
        return out
    with open(outName, "w") as f:
        json.dump(out, f)
