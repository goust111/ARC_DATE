from __future__ import annotations
import numpy as np
from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste, random_free_location
from arc_utils import paste

def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack rectangles by proximity to two colored edge lines.

    Input structure:
    - bg = most common color.
    - Two edge lines (h + v) meet at a corner marked by a unique color.
    - "Blue" line is color 1 if present, else the longer line; other is "red."
    - Solid rectangles in the interior.

    Rule:
    - Sort rectangles by (distance to blue edge, distance to red edge).
    - If the blue edge is vertical, rotate each rectangle 90° CCW.
    - Stack top-to-bottom, centered horizontally.

    Key insight: Edge-line colors encode a coordinate system; proximity to each axis determines stacking order and rotation.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- locate the marked corner ---
    corners = [(0, 0), (0, int(W - 1)), (int(H - 1), 0), (int(H - 1), int(W - 1))]
    corner = None
    for r, c in corners:
        if int(g[r, c]) == int(bg):
            continue
        nbr = []
        if r == 0:
            nbr.append(int(g[r + 1, c]))
        if r == H - 1:
            nbr.append(int(g[r - 1, c]))
        if c == 0:
            nbr.append(int(g[r, c + 1]))
        if c == W - 1:
            nbr.append(int(g[r, c - 1]))
        if nbr and all(int(n) != int(g[r, c]) for n in nbr):
            corner = (int(r), int(c))
            break
    assert corner is not None

    cr, cc = map(int, corner)
    marker = int(g[cr, cc])

    # --- line colors adjacent to the corner ---
    hstep = 1 if cc == 0 else -1
    vstep = 1 if cr == 0 else -1
    hcol = int(g[cr, cc + hstep])
    vcol = int(g[cr + vstep, cc])

    # Choose "blue" line color.
    if hcol == 1 or vcol == 1:
        blue_col = 1
        red_col = int(vcol if hcol == 1 else hcol)
    else:
        # Fallback: longer line is blue.
        hlen = 0
        c = int(cc + hstep)
        while 0 <= c < W and int(g[cr, c]) == int(hcol):
            hlen += 1
            c += int(hstep)
        vlen = 0
        r = int(cr + vstep)
        while 0 <= r < H and int(g[r, cc]) == int(vcol):
            vlen += 1
            r += int(vstep)
        if vlen >= hlen:
            blue_col = int(vcol)
            red_col = int(hcol)
        else:
            blue_col = int(hcol)
            red_col = int(vcol)

    blue_orient = "h" if int(hcol) == int(blue_col) else "v"
    red_orient = "h" if int(hcol) == int(red_col) else "v"

    # --- mask out the two edge lines (entire corner row + col) ---
    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    # --- extract solid rectangles (no bg inside bbox) ---
    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if bool((sub == int(bg)).any()):
            continue
        rects.append({"box": tuple(map(int, b)), "arr": sub.copy()})

    assert not (not rects)

    def dist_to_edge(rect: dict, orient: str, edge_idx: int) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            col = int(edge_idx)
            if int(c0) > col:
                return int(c0 - col - 1)
            if int(c1) < col:
                return int(col - c1 - 1)
            return 0
        row = int(edge_idx)
        if int(r0) > row:
            return int(r0 - row - 1)
        if int(r1) < row:
            return int(row - r1 - 1)
        return 0

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    def along_from_corner(rect: dict, orient: str) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            return int(r0 if cr == 0 else -r1)
        return int(c0 if cc == 0 else -c1)

    for rct in rects:
        rct["db"] = int(dist_to_edge(rct, blue_orient, blue_edge_idx))
        rct["dr"] = int(dist_to_edge(rct, red_orient, red_edge_idx))

    rects.sort(
        key=lambda r: (
            int(r["db"]),
            int(r["dr"]),
            along_from_corner(r, blue_orient),
            along_from_corner(r, red_orient),
        )
    )

    # Rotate all rectangles if blue edge is vertical.
    seq = []
    for rct in rects:
        a = rct["arr"]
        if blue_orient == "v":
            a = np.rot90(a, k=1)  # CCW
        seq.append(a)

    outW = int(max(int(a.shape[1]) for a in seq))
    outH = int(sum(int(a.shape[0]) for a in seq))
    out = np.full((outH, outW), int(bg), dtype=int)

    out_r = 0
    for a in seq:
        h, w = map(int, a.shape)
        out_c = int((outW - w) // 2)
        paste(out, a, int(out_r), int(out_c))
        out_r += int(h)

    return out

