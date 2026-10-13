from __future__ import annotations
from collections import Counter
import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


def mask_from_coords(coords, shape: tuple[int, int]) -> np.ndarray:
    """Local helper: build a boolean mask with True at the given coords (in-bounds only)."""
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            m[rr, cc] = True
    return m

def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort objects to top or bottom edge by whether their pixel-count is shared.

    Input structure:
    - bg = most common color.
    - Several multicolor connected components (8-connected) scattered across the grid.

    Rule:
    - Count each object's pixel area.
    - Objects whose area appears in two or more components → translate to TOP edge (row 0).
    - Objects whose area is unique → translate to BOTTOM edge (row H−h).
    - Horizontal position (left column) is preserved; only vertical translation changes.

    Key insight: The sorting criterion is area frequency, not area magnitude — duplicated sizes go up, singleton sizes go down.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert not (not comps)

    extracted = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (r0, c0, r1, c1))

        # Isolate this component exactly (bbox rectangle may include background).
        rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        m = mask_from_coords(rel_coords, patch.shape)
        pat = np.full_like(patch, bg)
        pat[m] = patch[m]

        extracted.append(
            {
                "c0": int(c0),
                "pat": pat,
                "area": int(comp["area"]),
            }
        )

    freq = Counter(int(d["area"]) for d in extracted)
    out = np.full((H, W), int(bg), dtype=int)

    for d in extracted:
        pat = d["pat"]
        h, w = map(int, pat.shape)
        dest_r0 = 0 if int(freq[int(d["area"])]) >= 2 else int(H - h)
        dest_c0 = int(d["c0"])
        paste(out, pat, dest_r0, dest_c0, transparent=int(bg))

    return out

