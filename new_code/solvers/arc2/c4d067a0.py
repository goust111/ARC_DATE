import numpy as np

from arc_utils import bbox_pts as _bbox, cc4 as _cc4, get_bg as _bg_color, most_common as _most_common
from arc_utils import gcd_list as _gcd_list

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete a grid of colored blocks to match a minimap legend in the top-left corner.

    Input structure:
    - bg = most common color.
    - Top-left corner: a small sprite of single colored pixels forming a minimap pattern on a regular lattice.
    - Main area: uniformly colored square blocks (NxN) arranged on a regular grid; some positions are empty (bg).

    Rule:
    - The legend pixels define a color pattern; each pixel's lattice position maps to a block position in the main grid via a fixed scale factor and offset.
    - Existing colored blocks confirm the scale/offset alignment.
    - For every legend pixel whose corresponding grid position is empty, place a new block of that color.

    Key insight: The top-left pixel pattern is a scaled-down minimap of the block layout; the mapping is inferred from blocks already present.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = _bg_color(g)

    comps = _cc4(g != bg)
    singles: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int, int]] = []  # (r0,c0,size,color)
    for pts in comps:
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(g[r, c])))
            continue
        r0, c0, r1, c1 = _bbox(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh != ww or hh < 2:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        col = int(sub[0, 0])
        if col == bg:
            continue
        if np.all(sub == col) and sub.size == len(pts):
            blocks.append((int(r0), int(c0), int(hh), col))

    if not singles or not blocks:
        return out

    # Legend pixels: singleton colored pixels in a small top-left window.
    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    if not legend:
        return out

    # Infer legend lattice step (usually 2).
    rs = sorted({r for r, _, _ in legend})
    cs = sorted({c for _, c, _ in legend})
    drs = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    dcs = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
    stepL = _gcd_list(drs + dcs)
    if stepL <= 0:
        stepL = 1
    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)

    legend_pts: list[tuple[int, int, int]] = []
    for r, c, v in legend:
        rr = (r - rmin) // stepL
        cc = (c - cmin) // stepL
        legend_pts.append((int(rr), int(cc), int(v)))

    # Determine typical block size and step between blocks.
    sizes = [s for _, _, s, _ in blocks]
    block_size = _most_common(np.array(sizes, dtype=int))
    blocks = [b for b in blocks if b[2] == block_size]
    if not blocks:
        return out

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r0, c0, s, col in blocks:
        by_color.setdefault(int(col), []).append((int(r0), int(c0)))
    for col in by_color:
        by_color[col].sort()

    # Compute step from block top-left positions (gcd of diffs).
    tops = sorted({r0 for r0, _, _, _ in blocks})
    lefts = sorted({c0 for _, c0, _, _ in blocks})
    step_candidates = []
    step_candidates += [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
    step_candidates += [lefts[i + 1] - lefts[i] for i in range(len(lefts) - 1)]
    step = _gcd_list(step_candidates)
    if step <= 0:
        step = block_size + 1

    # Infer base offset: choose base that best maps legend pixels with existing block colors.
    bases: dict[tuple[int, int], int] = {}
    for rr, cc, col in legend_pts:
        if col not in by_color:
            continue
        for br0, bc0 in by_color[col]:
            base_r = br0 - rr * step
            base_c = bc0 - cc * step
            bases[(base_r, base_c)] = bases.get((base_r, base_c), 0) + 1
    if not bases:
        return out
    base = max(bases.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
    base_r, base_c = base

    # Paint blocks for all legend pixels (usually adding missing ones).
    for rr, cc, col in legend_pts:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        if not (0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size):
            continue
        sub = out[tr0 : tr0 + block_size, tc0 : tc0 + block_size]
        if np.all(sub == bg) or np.all(sub == col):
            out[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = col
    return out
