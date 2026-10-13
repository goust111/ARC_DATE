from __future__ import annotations
import numpy as np
from arc_utils import components_mask

def main(grid: np.ndarray) -> np.ndarray:
    """
    Highlight black defects in blue foreground by surrounding them with orange.

    Input structure:
    - Blue (1) foreground on black (0) bg; small black holes or protrusions within blue regions; occasional thin blue spurs into black areas.

    Rule:
    - Thin blue pixels mostly surrounded by black are absorbed into the background.
    - Small black components (≤3 cells) and leaf pixels of larger black components are "defects."
    - Each defect gets an orange (7) 8-neighbor halo; the defect pixels themselves stay black.

    Key insight: Only structural anomalies—tiny isolated patches and dangling endpoints—are marked, not bulk black regions.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    hole_col = 0
    border_col = 7

    # STEP 0) Fill blue (1) pixels to black (0) when mostly surrounded by black.
    base = g.copy()
    for r in range(H):
        for c in range(W):
            if int(base[r, c]) != 1:
                continue
            nb = 0
            oob = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if not (0 <= rr < H and 0 <= cc < W):
                    oob += 1
                    continue
                if int(base[rr, cc]) == int(hole_col):
                    nb += 1
            # only count outside-as-black for corners (two out-of-bounds sides)
            if oob >= 2:
                nb += int(oob)
            if nb >= 3:
                base[int(r), int(c)] = int(hole_col)

    # STEP 1/2) Collect target black pixels (they must remain black).
    preserve: set[tuple[int, int]] = set()
    comps = components_mask(base == int(hole_col), connectivity=4)
    for coords in comps:
        pts_list = [(int(r), int(c)) for r, c in coords]
        if len(pts_list) <= 3:
            preserve.update(pts_list)
            continue
        for r, c in pts_list:
            nb = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(base[rr, cc]) == int(hole_col):
                    nb += 1
            if nb <= 1:
                preserve.add((int(r), int(c)))

    # STEP 3) Paint orange in the Chebyshev-1 neighborhood of each target set/pixel.
    out = base.copy()
    for coords in comps:
        pts_list = [(int(r), int(c)) for r, c in coords]
        targets = pts_list if len(pts_list) <= 3 else [p for p in pts_list if p in preserve]
        for r, c in targets:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in preserve:
                        out[rr, cc] = int(border_col)

    # Ensure target black pixels remain black.
    for r, c in preserve:
        out[int(r), int(c)] = int(hole_col)

    return out

