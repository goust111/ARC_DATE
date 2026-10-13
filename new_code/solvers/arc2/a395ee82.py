from __future__ import annotations
import math
from collections import Counter
import numpy as np
from arc_utils import bbox_h, bbox_w, find_components, get_bg

def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp translated copies of a monochrome template guided by a singleton pixel layout.

    Input structure:
    - bg = background. A large monochrome shape (template, color T).
    - Singleton pixels elsewhere: exactly one is color T (layout origin); the rest share color O (overlap color).
    - Singletons sit on a uniform-spacing lattice.

    Rule:
    - Convert each singleton's offset from the origin into coarse lattice units.
    - Place a template copy at each offset, scaled by template height/width.
    - The origin's copy is painted O instead of T.
    - Any pixel covered by two or more copies is also recolored to O.

    Key insight: The single T-colored singleton identifies both the origin position and the "self" copy, which is uniquely recolored.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert not (not comps)

    template = max(comps, key=lambda d: int(d["area"]))
    templ_coords = template["coords"]
    templ_col = int(g[int(templ_coords[0, 0]), int(templ_coords[0, 1])])
    r0, c0, r1, c1 = map(int, template["bbox"])
    h = int(bbox_h((r0, c0, r1, c1)))
    w = int(bbox_w((r0, c0, r1, c1)))

    templ_rel = [(int(r) - int(r0), int(c) - int(c0)) for r, c in templ_coords]

    # Singleton layout points (excluding the template component).
    singletons: list[tuple[int, int, int]] = []
    for comp in comps:
        if comp is template:
            continue
        if int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((int(rr), int(cc), int(g[int(rr), int(cc)])))

    origin = [(r, c) for (r, c, col) in singletons if int(col) == int(templ_col)]
    assert not (len(origin) != 1)
    orr, orc = origin[0]

    other_cols = [int(col) for (_r, _c, col) in singletons if int(col) != int(templ_col)]
    assert not (not other_cols)
    overlap_col = int(Counter(other_cols).most_common(1)[0][0])

    deltas = [(int(r - orr), int(c - orc), int(col)) for (r, c, col) in singletons]
    gcd_vals = [abs(int(x)) for dr, dc, _ in deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = int(math.gcd(*gcd_vals))
    gunit = max(1, int(gunit))
    deltas = [(int(dr // gunit), int(dc // gunit), int(col)) for dr, dc, col in deltas]

    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)  # coverage count from the origin singleton

    for dr, dc, col in deltas:
        off_r = int(dr * h)
        off_c = int(dc * w)
        is_origin = int(col) == int(templ_col)
        for rr, cc in templ_rel:
            R = int(r0 + off_r + rr)
            C = int(c0 + off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
                if is_origin:
                    center[R, C] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[total > 0] = int(templ_col)
    out[(total == 1) & (center > 0)] = int(overlap_col)
    out[total >= 2] = int(overlap_col)
    return out

