from __future__ import annotations
from collections import Counter
import numpy as np
from arc_utils import bbox, bbox_coords, crop_bbox, find_components, get_bg, paste, components_mask

def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill holes inside a large template frame using shape-matched outside sprites.

    Input structure:
    - bg = most common color. One large monochrome frame object (template).
    - Inside the frame: holes of various shapes (non-frame-color regions).
    - Outside the frame: small monochrome sprites scattered on bg.

    Rule:
    - Crop to the frame's bounding box; fill entirely with frame color.
    - For each hole, find an outside sprite whose shape matches (rotation/flip allowed).
    - Fill the hole with the matched sprite's color. If multiple sprites match, use the most frequent color.

    Key insight: Sprites act as shape-coded color keys — their shape identifies which hole they fill, their color determines the fill color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # STEP 1) Find the template frame: largest connected component of any non-bg color (8-connected).
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert not (not comps)
    frame = max(comps, key=lambda d: int(d["area"]))
    frame_col = int(frame["color"])
    fb = frame["bbox"]
    r0, c0, r1, c1 = map(int, fb)

    sub = crop_bbox(g, fb)
    th, tw = sub.shape

    # STEP 2) Extract hole components inside the bbox (anything not equal to frame color), 4-connected.
    hole_mask = (sub != int(frame_col))
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []  # (key, (r0,c0), mask)
    for coords in components_mask(hole_mask, connectivity=4):
        if coords.size == 0:
            continue
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True
        key = (m.shape, m.tobytes())
        holes.append((key, (int(hr0), int(hc0)), m))

    # STEP 3) Extract sprites outside the bbox.
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    # key -> color -> count
    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}

    def _all_mask_keys(mask: np.ndarray):
        """Generate keys for all rotations (and optional vertical flip) of a boolean mask."""
        m0 = mask.astype(bool)
        for flip in (False, True):
            m1 = np.flipud(m0) if flip else m0
            for k in range(4):
                m2 = np.rot90(m1, k)
                yield (m2.shape, m2.tobytes())

    for comp in spr_comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue
        bb = comp["bbox"]
        sr0, sc0, sr1, sc1 = map(int, bb)
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)

        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        if not vals:
            continue
        scol = int(Counter(vals).most_common(1)[0][0])

        rel = coords - np.array([sr0, sc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True

        for key in _all_mask_keys(m):
            bucket = sprites_by_key.setdefault(key, {})
            bucket[scol] = int(bucket.get(scol, 0)) + 1

    # STEP 4) Render output: start from solid frame color, then fill each hole by shape match.
    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        cmap = sprites_by_key.get(key)
        if not cmap:
            continue
        # choose the most frequent sprite color for this shape; break ties by preferring smaller color id
        scol = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        patch = np.full(hmask.shape, int(scol), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=hmask)

    return out


