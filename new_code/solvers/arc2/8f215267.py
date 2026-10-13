from __future__ import annotations
import numpy as np
from arc_utils import bbox_h, bbox_w, find_components, get_bg

def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color patterns per frame and represent the tally as dots inside.

    Input structure:
    - bg = most common color.
    - Hollow rectangular frames in various colors.
    - Small monochrome pattern sprites scattered outside frames.

    Rule:
    - Remove all small patterns from the grid.
    - For each frame of color C, count how many patterns share color C.
    - Draw that many dots along the frame's center row, spaced by 2, starting from the right.

    Key insight: Frames act as tally displays — scattered same-color patterns are "absorbed" and their count is shown as interior dots.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        b = comp["bbox"]
        h = bbox_h(b)
        w = bbox_w(b)
        area = int(comp["area"])
        is_frame = (h >= 5) and (w >= 5) and (area < (h * w * 0.7))
        if is_frame:
            frames.append(comp)
        else:
            patterns.append(comp)

    # Remove patterns from output.
    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    # For each frame, count matching patterns and draw dots.
    for f in frames:
        col = int(f["color"])
        num_patterns = sum(1 for p in patterns if int(p["color"]) == col)
        if num_patterns <= 0:
            continue

        r0, c0, r1, c1 = map(int, f["bbox"])
        r_mid = (r0 + r1) // 2
        for i in range(int(num_patterns)):
            cc = int(c1 - 2 - 2 * i)
            if cc > int(c0):
                out[int(r_mid), int(cc)] = col

    return out

