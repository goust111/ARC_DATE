from __future__ import annotations
import numpy as np
from arc_utils import colors_present, get_bg

def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean each color layer by enforcing vertical mirror symmetry.

    Input structure:
    - bg = background. Several non-bg colors, each forming a nearly-symmetric pattern about some vertical axis.
    - Some pixels in each color are "noise" that breaks the symmetry.

    Rule:
    - For each non-bg color, find the vertical axis that best explains its pixel layout.
    - Remove any pixel that lacks a same-color mirror counterpart across that axis.

    Key insight: Each color has its own independent symmetry axis; the task is per-color denoising, not whole-grid symmetry.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    if not colors:
        return out

    # Search all possible vertical axes: k in [0, 2*(W-1)] corresponds to axis at c = k/2.
    for col in colors:
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue

        best_score = -1
        best_k = 0
        for k in range(0, 2 * (W - 1) + 1):
            score = 0
            for r, c in pts:
                cc = int(k) - int(c)
                if 0 <= cc < W and int(g[int(r), cc]) == int(col):
                    score += 1
            if score > best_score:
                best_score = int(score)
                best_k = int(k)

        # Delete unpaired pixels under the chosen vertical axis.
        k = int(best_k)
        for r, c in pts:
            rr = int(r)
            cc = int(k) - int(c)
            if not (0 <= cc < W and int(g[rr, cc]) == int(col)):
                out[rr, int(c)] = int(bg)

    return out

