# Variant of 291dc1e1
# Dimension: D9 — 边缘线的角色与识别 (src)
# Change: 蓝线(色1)和红线(色2)分别沿水平和垂直方向贯穿整行/列，在角落相交处有标记色(0)像素... → 无标记像素——由线长度判断蓝/红

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


# Design analysis for the D9 variant
# ----------------------------------
# a) Concrete behavior and trace
# - Original input had one border row and one border column crossing at a corner,
#   but the crossing cell was a third marker color. The solver used that unique
#   marker to find the corner and then read the adjacent row/column colors.
# - This variant removes the marker. The two lines now "meet directly" at the
#   corner by letting the longer line own the corner pixel, while the shorter
#   line starts from the next cell on its edge. Example in canonical form:
#       old top row / left col: 0 1 1 1 ... and 0 / 2 / 2 / ...
#       new top row / left col: 1 1 1 1 ... and 1 / 2 / 2 / ...
#   The horizontal border row is still a line, and the vertical border column is
#   still a line, but there is no unique marker color to anchor detection.
# - The changed step is the entry point of the pipeline:
#     old: locate special corner marker -> read h/v line colors -> maybe prefer
#          literal color 1 as "blue"
#     new: locate the only all-non-background border row and column -> infer each
#          line's dominant color from that row/column -> declare the longer line
#          to be "blue"
#   Downstream sorting, optional rotation, and stacking stay unchanged.
# - Small trace:
#     if the top row is all color 2 and the left column is [2,1,1,1,1,1,1,1],
#     then the horizontal line length visible in its own row is 12 cells, the
#     vertical line length visible in its own column is 7 cells, so color-2
#     horizontal is "blue" and color-1 vertical is "red". Rectangles are sorted
#     by distance to top edge first, left edge second, with no rotation.
#
# b) Complexity calibration
# - The variant keeps only two line colors and one geometric decision: the longer
#   edge line is blue. This is enough to create genuine ambiguity because color 1
#   is no longer guaranteed to be blue, yet the hypothesis space remains small.
# - To make the length cue visible, the long axis is always at least 9 while the
#   short axis is 8, and the corner belongs to the longer line. Therefore the
#   dominant-color counts on the border row/column differ clearly.
#
# c) Hidden-assumption audit of the original solver
# - Original corner scan assumed a unique non-background pixel at a corner whose
#   color differed from both neighbors. This is invalid after marker removal.
# - Original "if color 1 exists, it is blue" is also invalid: in this variant
#   either color 1 or color 2 may be the longer line.
# - Original masking of the full corner row/column still remains valid because the
#   border-line geometry is unchanged.
# - Rectangle extraction still assumes objects stay away from the border lines and
#   are solid within their bounding boxes; the generator preserves that.
#
# d) Placement legality
# - The generator keeps the same two interior bands at rows 2 and 5 of the
#   canonical 8-row layout. Rectangles are pasted with 2-cell gaps and never touch
#   the border row/column, so they do not overlap lines or each other.
# - After flips/rotations, the unique all-non-background border row and column are
#   preserved, so the solver can still detect the corner deterministically.
#
# e) Visual observability
# - Across examples, the long line may be color 1 or color 2, and its orientation
#   may be horizontal or vertical. Because output rotation depends on blue-line
#   orientation, examples where color 2 is longer visibly contradict the original
#   "color 1 means blue" reading.
# - Rectangles get occasional asymmetric minority pixels so the rotation branch is
#   visually detectable.
#
# f) Visual cleanliness
# - Only the border row/column carry the structural cue. Interior rectangles remain
#   separated by background, and the line colors are not reused as dominant object
#   colors, preventing the edges from blending into the object field.
#
# g) Color inferability
# - No new color mapping is introduced. Colors 1 and 2 still denote the two edge
#   lines, but role assignment is now geometric: longer line = blue, shorter line
#   = red. That relation is repeated in every sample and is supported by visible
#   line spans rather than arbitrary remapping.
#
# New solver assumptions and how the generator guarantees them
# - Exactly one border row and one border column are fully non-background.
# - Each of those border lines uses one dominant color, with at most one foreign
#   pixel at the shared corner.
# - The longer line owns the corner cell, so dominant-color counts differ.
# - Interior components are solid rectangles not touching the border lines.


BG = 8
LINE_COLORS = np.array([1, 2], dtype=int)
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)


def _majority_color(vec: np.ndarray) -> int:
    vals, counts = np.unique(vec.astype(int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _locate_lines(g: np.ndarray, bg: int) -> tuple[int, int, int, int, str, str]:
    H, W = map(int, g.shape)
    edge_rows = [r for r in (0, H - 1) if bool(np.all(g[r, :] != int(bg)))]
    edge_cols = [c for c in (0, W - 1) if bool(np.all(g[:, c] != int(bg)))]
    assert len(edge_rows) == 1 and len(edge_cols) == 1

    cr = int(edge_rows[0])
    cc = int(edge_cols[0])
    row = g[cr, :]
    col = g[:, cc]

    hcol = int(_majority_color(row))
    vcol = int(_majority_color(col))
    assert hcol != vcol

    hlen = int(np.count_nonzero(row == int(hcol)))
    vlen = int(np.count_nonzero(col == int(vcol)))
    assert hlen != vlen

    blue_orient = "h" if hlen > vlen else "v"
    red_orient = "v" if blue_orient == "h" else "h"
    return cr, cc, hcol, vcol, blue_orient, red_orient


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack rectangles by proximity to two colored edge lines.

    D9 variant:
    - No corner marker exists.
    - The unique all-non-background border row and border column define the axes.
    - Their dominant colors define the two lines.
    - The longer line is "blue"; the shorter line is "red".
    - If the blue line is vertical, rotate each rectangle 90 degrees CCW.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    cr, cc, _hcol, _vcol, blue_orient, red_orient = _locate_lines(g, bg)

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask)
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if bool((sub == int(bg)).any()):
            continue
        rects.append({"box": tuple(map(int, b)), "arr": sub.copy()})

    assert rects

    def dist_to_edge(rect: dict, orient: str, edge_idx: int) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            if c0 > edge_idx:
                return int(c0 - edge_idx - 1)
            if c1 < edge_idx:
                return int(edge_idx - c1 - 1)
            return 0
        if r0 > edge_idx:
            return int(r0 - edge_idx - 1)
        if r1 < edge_idx:
            return int(edge_idx - r1 - 1)
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

    seq = []
    for rct in rects:
        arr = rct["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)

    outW = int(max(int(arr.shape[1]) for arr in seq))
    outH = int(sum(int(arr.shape[0]) for arr in seq))
    out = np.full((outH, outW), int(bg), dtype=int)

    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = int((outW - w) // 2)
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


def _distribute_band_widths(count: int, used_target: int) -> list[int]:
    extra_units = (used_target - (4 * count - 2)) // 2
    order = np.random.permutation(count)
    units = np.zeros(count, dtype=int)
    remaining = int(extra_units)
    for i, idx in enumerate(order):
        slots_left = count - i - 1
        lo = max(0, remaining - 3 * slots_left)
        hi = min(3, remaining)
        take = int(np.random.randint(lo, hi + 1))
        units[int(idx)] = take
        remaining -= take
    return [2 + 2 * int(u) for u in units]


def _sprinkle_minorities(patch: np.ndarray, dominant: int) -> np.ndarray:
    area = int(patch.size)
    if area < 8 or np.random.rand() < 0.6:
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


def _build_canonical(long_col: int, short_col: int) -> np.ndarray:
    long_dim = int(np.random.randint(9, 29))
    options = []
    for trailing in (1, 2):
        max_used = long_dim - 2 - trailing
        if max_used < 6 or max_used % 2:
            continue
        max_count = min(5, (max_used + 2) // 4)
        pairs = []
        for a in range(2, max_count + 1):
            for b in range(2, max_count + 1):
                if a + b > 9:
                    continue
                eligible = []
                if max_used <= 10 * a - 2:
                    eligible.append(0)
                if max_used <= 10 * b - 2:
                    eligible.append(1)
                if eligible:
                    pairs.append((a, b, eligible))
        if pairs:
            options.append((max_used, pairs))

    max_used, pairs = options[int(np.random.randint(len(options)))]
    n0, n1, eligible = pairs[int(np.random.randint(len(pairs)))]
    max_band = int(np.random.choice(eligible))
    max0 = min(max_used, 10 * n0 - 2)
    max1 = min(max_used, 10 * n1 - 2)
    used0 = int(max_used if max_band == 0 else np.random.choice(np.arange(4 * n0 - 2, max0 + 1, 2)))
    used1 = int(max_used if max_band == 1 else np.random.choice(np.arange(4 * n1 - 2, max1 + 1, 2)))
    band0 = _distribute_band_widths(n0, used0)
    band1 = _distribute_band_widths(n1, used1)

    grid = np.full((8, long_dim), BG, dtype=int)
    grid[0, :] = int(long_col)
    grid[:, 0] = int(short_col)
    grid[0, 0] = int(long_col)

    for r0, widths in ((2, band0), (5, band1)):
        c = 2
        for w in widths:
            color = int(np.random.choice(DOMINANT_COLORS))
            patch = np.full((2, w), color, dtype=int)
            patch = _sprinkle_minorities(patch, color)
            paste(grid, patch, r0, c)
            c += w + 2

    return grid


def _gen_attempt() -> np.ndarray:
    long_col = int(np.random.choice(LINE_COLORS))
    short_col = int(LINE_COLORS[0] + LINE_COLORS[1] - long_col)
    canonical = _build_canonical(long_col, short_col)
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    return _transform(canonical, blue_orient, corner)


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        try:
            _ = main(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D9 variant")
