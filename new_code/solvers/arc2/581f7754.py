from __future__ import annotations
from collections import Counter
import numpy as np
from arc_utils import find_components, get_bg, neighbors4

def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide multi-color sprites to align their marker pixels with matching isolated border anchors.

    Input structure:
    - bg = most common color.
    - Multi-color sprites in the interior; each has a minority-color "marker" pixel.
    - Isolated border pixels (non-bg with all 4-neighbors bg) act as anchors.

    Rule:
    - Match each sprite to the border anchor sharing its marker color.
    - Top/bottom-edge anchor (non-corner): slide sprite horizontally so marker column = anchor column.
    - Left/right-edge anchor (or corner): slide sprite vertically so marker row = anchor row.

    Key insight: The anchor's edge position determines the slide axis — top/bottom means horizontal, left/right (or corner) means vertical.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    # --- STEP 1) Detect anchors: isolated border pixels (non-bg) ---
    anchors_by_color: dict[int, tuple[int, int]] = {}
    for r in range(H):
        for c in range(W):
            if not is_border(r, c):
                continue
            v = int(g[r, c])
            if v == int(bg):
                continue
            ok = True
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) != int(bg):
                    ok = False
                    break
            if ok and v not in anchors_by_color:
                anchors_by_color[int(v)] = (int(r), int(c))

    assert not (not anchors_by_color)

    anchor_cells = {(int(r), int(c)) for (r, c) in anchors_by_color.values()}

    # --- STEP 2) Extract all foreground connected components as candidate sprites. ---
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    def component_marker_color(coords: np.ndarray) -> int | None:
        # For multi-color sprites, the marker color is the least frequent color inside the component.
        vals = [int(g[int(r), int(c)]) for r, c in coords]
        cnt = Counter(vals)
        if len(cnt) <= 1:
            return None
        return int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])

    def marker_pixel(coords: np.ndarray, marker: int) -> tuple[int, int] | None:
        # If marker appears multiple times, prefer one attached to the main body (4-adjacent to non-marker).
        pts = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(marker)]
        if not pts:
            return None
        if len(pts) == 1:
            return pts[0]
        S = {(int(r), int(c)) for r, c in coords}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in S and int(g[int(rr), int(cc)]) != int(marker):
                    return (int(r), int(c))
        return pts[0]

    moves: list[tuple[np.ndarray, int, int]] = []

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        # --- STEP 3) Decide marker color for this component. ---
        marker_col = component_marker_color(coords)
        if marker_col is None:
            # Single-color component: move only if it's a singleton of an anchor color, not the anchor itself.
            if int(coords.shape[0]) != 1:
                continue
            r, c = map(int, coords[0])
            v = int(g[r, c])
            if v == int(bg) or v not in anchors_by_color or (r, c) in anchor_cells:
                continue
            marker_col = int(v)

        if int(marker_col) not in anchors_by_color:
            continue

        mp = marker_pixel(coords, int(marker_col))
        if mp is None:
            continue

        ar, ac = anchors_by_color[int(marker_col)]

        # --- STEP 4) Compute translation (dr,dc). ---
        on_top_bottom = int(ar) in (0, int(H - 1))
        on_left_right = int(ac) in (0, int(W - 1))
        horizontal = bool(on_top_bottom and not on_left_right)  # top/bottom but not a corner

        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])

        moves.append((coords, int(dr), int(dc)))

    out = g.copy()

    # --- STEP 5) Apply all moves. ---
    # Clear moved components first (prevents collisions with their original locations).
    for coords, _dr, _dc in moves:
        out[coords[:, 0], coords[:, 1]] = int(bg)

    # Paste moved components:
    # - clip out-of-bounds pixels
    # - do not overwrite existing non-bg pixels
    for coords, dr, dc in moves:
        for r, c in coords:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(g[int(r), int(c)])

    return out



