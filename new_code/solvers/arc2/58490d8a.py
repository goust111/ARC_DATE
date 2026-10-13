from __future__ import annotations
import numpy as np
from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, label_mask

def main(grid: np.ndarray) -> np.ndarray:
    """
    Count colored objects and record tallies inside a legend box.

    Input structure:
    - A black rectangular legend box with key-color pixels in its leftmost non-black column.
    - Outside the box, distinct objects of each key color are scattered across the grid.

    Rule:
    - For each key color, count separate same-colored objects outside the legend box.
    - Output the legend box repainted: each key row gets one dot per object, spaced every other cell rightward from the key pixel.

    Key insight: The legend is both the color index and the output canvas — object counts become dot sequences on the matching row.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    _bg = int(get_bg(g))

    black = 0

    # STEP 1) Find the legend bbox: largest black CC whose bbox contains some non-black pixels.
    best = None  # (area, bbox)
    for coords in components_mask(g == int(black), connectivity=8):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if (sub != int(black)).any():
            area = int(coords.shape[0])
            cand = (area, tuple(map(int, b)))
            if best is None or cand[0] > best[0]:
                best = cand

    assert not (best is None)

    r0, c0, r1, c1 = map(int, best[1])
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)

    # STEP 2) Inside the legend bbox, find the key column:
    # leftmost column with any non-black pixels (the "key pixels").
    key_col = None
    for cc in range(bw):
        if (box[:, int(cc)] != int(black)).any():
            key_col = int(cc)
            break
    if key_col is None:
        return np.zeros_like(box)

    keys = [(int(rr), int(box[int(rr), int(key_col)])) for rr in range(bh) if int(box[int(rr), int(key_col)]) != int(black)]
    keys.sort(key=lambda t: int(t[0]))

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    # STEP 3/4) Count CCs of each key color outside the legend, then draw 1 dot per CC.
    for rr, col in keys:
        mask = (g == int(col)) & outside
        _labc, nc = label_mask(mask, connectivity=8)
        for i in range(int(nc)):
            cc = int(key_col + 2 * i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(col)

    return out

