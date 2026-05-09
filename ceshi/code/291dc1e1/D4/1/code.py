# Variant of 291dc1e1
# Dimension: D4 — 源对象的band分布结构 (src)
# Change: 矩形分布在两条平行band中（距蓝线1和4的位置），每band 2-5个矩形... → 单band分布（所有矩形在同一行带）
#
# Design analysis for this variant
# - a) Concrete behavior and trace:
#   In the original task, the interior rectangles occupy two parallel bands in the
#   direction perpendicular to the blue edge. In canonical view that means rows
#   2-3 and 5-6 under a top blue line / left red line layout. The solver sorts by
#   (distance to blue, distance to red), so all near-band rectangles are emitted
#   before far-band rectangles.
#   In this variant every rectangle lives in one shared 2-cell-high band, fixed at
#   distance 1 from the blue edge in canonical view. Their only meaningful rank is
#   how far they are from the red edge, so stacking order is determined by red-edge
#   distance alone. Example: if three rectangles start at columns 2, 8, 14, then
#   original logic would compute identical blue distances and increasing red
#   distances; variant logic keeps exactly that red-driven order but the input no
#   longer exhibits a second blue-distance layer at all.
# - b) Complexity calibration:
#   A single band can become too trivial if there are only one or two identical
#   blocks. The generator therefore uses 3-6 rectangles, variable widths, and
#   occasional minority-color pixels inside a rectangle. This preserves enough
#   diversity for humans to see a nontrivial ordering and for the rotation branch to
#   remain visible when blue is vertical.
# - c) Implicit assumption audit against the original solver:
#   `rects.sort(key=(db, dr, ...))` relied on two blue-distance tiers existing.
#   That premise is false here, so the solver is changed to sort primarily by `dr`.
#   Corner detection, line-color selection, line masking, component extraction, and
#   the "rotate if blue is vertical" rule do not depend on the number of bands and
#   remain valid.
# - d) Placement legality:
#   The generator places all rectangles in one band with a 2-cell horizontal gap,
#   never touching each other, the corner marker, or the edge lines. The canonical
#   grid includes at least one background row/column beyond the objects, so later
#   flips/rotations still stay in-bounds and keep objects separated.
# - e) Visual observability:
#   Across examples, the distinctive cue is that every source rectangle lies on one
#   strip parallel to the red edge counterpart, rather than being split across two
#   strips. When blue becomes vertical, the output rotation remains easy to notice
#   because many rectangles are 2xw with w > 2 and some contain asymmetric pixels.
# - f) Visual cleanliness:
#   The scene still has only four element types: background, corner marker, two edge
#   lines, and separated rectangles. No overlaps or layered encodings are introduced.
# - g) Color-change inferability:
#   There is no rule-level color remapping. Rectangle colors are preserved exactly;
#   minority pixels only make object orientation visible and do not encode any extra
#   conditional mapping.
# - h) New solver assumptions after the edit:
#   The red-distance ordering assumes rectangles do not share the same `dr` in normal
#   cases; the generator guarantees this by placing them sequentially with positive
#   gaps. Component extraction assumes each rectangle's bbox is solid (no background
#   holes); the generator always fills every bbox completely before optional internal
#   recoloring.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)


def _locate_structure(grid: np.ndarray) -> tuple[int, int, int, str, str, list[dict]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

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

    hstep = 1 if cc == 0 else -1
    vstep = 1 if cr == 0 else -1
    hcol = int(g[cr, cc + hstep])
    vcol = int(g[cr + vstep, cc])

    if hcol == 1 or vcol == 1:
        blue_col = 1
        red_col = int(vcol if hcol == 1 else hcol)
    else:
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

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = bbox_coords(coords)
        sub = crop_bbox(g, box)
        if bool((sub == int(bg)).any()):
            continue
        rects.append({"box": tuple(map(int, box)), "arr": sub.copy()})

    assert rects
    return int(cr), int(cc), int(bg), str(blue_orient), str(red_orient), rects


def _dist_to_edge(box: tuple[int, int, int, int], orient: str, edge_idx: int) -> int:
    r0, c0, r1, c1 = map(int, box)
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


def _along_from_corner(
    box: tuple[int, int, int, int], orient: str, corner_r: int, corner_c: int
) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if corner_r == 0 else -r1)
    return int(c0 if corner_c == 0 else -c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Rectangles are all sourced from a single band relative to the blue edge.
    - Stack them by increasing distance to the red edge.
    - If the blue edge is vertical, rotate each rectangle 90 degrees CCW first.
    """
    g = np.array(grid, dtype=int)
    cr, cc, bg, blue_orient, red_orient, rects = _locate_structure(g)

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    for rect in rects:
        box = tuple(map(int, rect["box"]))
        rect["db"] = int(_dist_to_edge(box, blue_orient, blue_edge_idx))
        rect["dr"] = int(_dist_to_edge(box, red_orient, red_edge_idx))

    rects.sort(
        key=lambda rect: (
            int(rect["dr"]),
            _along_from_corner(tuple(map(int, rect["box"])), red_orient, cr, cc),
            int(rect["db"]),
            _along_from_corner(tuple(map(int, rect["box"])), blue_orient, cr, cc),
        )
    )

    seq = []
    for rect in rects:
        arr = rect["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)

    out_w = int(max(int(arr.shape[1]) for arr in seq))
    out_h = int(sum(int(arr.shape[0]) for arr in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = int((out_w - w) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(h)

    return out


def _transform(grid: np.ndarray, blue_orient: str, corner: str) -> np.ndarray:
    if blue_orient == "h":
        if corner == "tl":
            return grid
        if corner == "tr":
            return np.fliplr(grid)
        if corner == "bl":
            return np.flipud(grid)
        return np.rot90(grid, 2)

    base = np.rot90(grid, 1)
    if corner == "bl":
        return base
    if corner == "tl":
        return np.flipud(base)
    if corner == "tr":
        return np.rot90(grid, 3)
    return np.flipud(np.rot90(grid, 3))


def _sample_widths(count: int) -> list[int]:
    widths = [2 + 2 * int(np.random.randint(0, 4)) for _ in range(count)]
    if max(widths) == 2:
        widths[int(np.random.randint(count))] = 4
    return widths


def _sprinkle_minorities(patch: np.ndarray, dominant: int) -> np.ndarray:
    area = int(patch.size)
    if area < 8 or np.random.rand() < 0.45:
        return patch

    max_minor = min(3, area // 4)
    if max_minor < 2:
        return patch

    n_minor = int(np.random.randint(2, max_minor + 1))
    n_colors = int(np.random.randint(1, min(2, n_minor) + 1))
    minority_choices = [int(c) for c in MINORITY_COLORS if int(c) != dominant]
    colors = np.random.choice(minority_choices, size=n_colors, replace=False)
    cells = np.random.choice(area, size=n_minor, replace=False)
    flat = patch.reshape(-1)
    for idx, cell in enumerate(cells):
        flat[int(cell)] = int(colors[idx % n_colors])
    return patch


def _build_canonical() -> np.ndarray:
    n_rects = int(np.random.randint(3, 7))
    widths = _sample_widths(n_rects)
    used = int(sum(widths) + 2 * (n_rects - 1))
    trailing = int(np.random.randint(1, 4))
    long_dim = int(used + trailing + 2)

    grid = np.full((5, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    c = 2
    for width in widths:
        color = int(np.random.choice(DOMINANT_COLORS))
        patch = np.full((2, int(width)), color, dtype=int)
        patch = _sprinkle_minorities(patch, color)
        paste(grid, patch, 2, c)
        c += int(width) + 2

    return grid


def _single_band_metrics(grid: np.ndarray) -> tuple[int, int]:
    cr, cc, _bg, blue_orient, red_orient, rects = _locate_structure(grid)
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    db_values = {
        int(_dist_to_edge(tuple(map(int, rect["box"])), blue_orient, blue_edge_idx))
        for rect in rects
    }
    dr_values = [
        int(_dist_to_edge(tuple(map(int, rect["box"])), red_orient, red_edge_idx))
        for rect in rects
    ]
    return len(db_values), len(set(dr_values))


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    n_db, n_dr = _single_band_metrics(grid)
    if n_db != 1 or n_dr < 3:
        return None

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate single-band variant input for 291dc1e1")
