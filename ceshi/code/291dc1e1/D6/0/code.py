# Variant of 291dc1e1
# Dimension: D6 — 堆叠方式与对齐 (dst)
# Change: 矩形从上到下紧密堆叠，水平方向居中对齐，无间隙... → 左对齐堆叠

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


# Design analysis for the D6 variant
# ----------------------------------
# a) Concrete behavior trace
# Input still contains one marked corner, one horizontal edge line, one vertical
# edge line, and several filled interior rectangles. The solver still:
# 1. finds the corner and line orientations,
# 2. sorts rectangles by distance to the "blue" and "red" axes,
# 3. rotates all rectangles 90 degrees CCW when the blue axis is vertical.
# The D6 change only affects the final packing step.
#
# Small trace:
# - Suppose the sorted output sequence has widths [6, 4, 2] after any required
#   rotation, with background color 8.
# - Original task: output width = 6, and the 4-wide / 2-wide rectangles are
#   pasted at columns 1 and 2 respectively, so the stack is centered.
# - This variant: output width is still 6, but every rectangle is pasted at
#   column 0, so the left edges are flush and the right edges create a staircase.
# - The difference appears only in the destination placement, not in the order
#   or in the rotation rule.
#
# b) Complexity calibration
# A naive generator could accidentally hide the variant. In the original
# generator, every rectangle had height 2. When blue is vertical, all rectangles
# are rotated to width 2, so left-aligned and centered outputs become identical.
# To avoid that degeneration, the new generator enforces:
# - at least two distinct widths among source rectangles,
# - at least two distinct heights among source rectangles,
# - at least one non-square rectangle.
# Therefore the final stacked widths vary both when blue is horizontal
# (final width = source width) and when blue is vertical (final width = source
# height after rotation).
#
# c) Implicit assumption audit on the solver
# - Corner detection assumes exactly one corner marker differs from both adjacent
#   line colors. The generator preserves that.
# - Blue/red axis inference assumes one edge line is color 1, or else uses the
#   longer line. The generator keeps blue as color 1 and red as color 2.
# - Rectangle extraction assumes each object is a solid filled bbox with no
#   background holes. The generator uses solid rectangles and only adds minority
#   colored cells, never background cells.
# - Distance sorting assumes rectangles do not touch the axis row/column. The
#   generator leaves at least one background cell between every rectangle and the
#   edge lines.
# - Final placement previously relied on center padding:
#       out_c = (outW - w) // 2
#   That line encoded the old D6 property. In this variant, that assumption is
#   invalid and must be replaced by:
#       out_c = 0
#   All other solver assumptions remain valid.
#
# d) Placement legality
# The generator builds a canonical top-left layout using a coarse cell grid. Each
# rectangle occupies the top-left of its own cell, cells are separated by at
# least two background cells, and the whole layout stays inside the frame after
# adding the border lines. The later orientation/corner transform is composed of
# flips and rotations, so non-overlap and in-bounds placement are preserved.
#
# e) Visual observability
# With 4-6 rectangles of varying widths/heights, the output regularly shows a
# clear left-flush staircase. Because widths vary after rotation as well, the
# left-alignment cue is visible in both blue-orientation branches.
#
# f) Visual cleanliness
# The input uses only three structural element types:
# - one corner marker,
# - two full border axes,
# - separated filled rectangles.
# Objects never overlap and gaps stay visible, so the scene remains easy to
# parse. Output is a single compact stack with one shared left edge.
#
# g) Color inference
# No new color mapping is introduced. Colors still belong to the rectangles
# themselves; line colors only define the ordering axes. This keeps the D6
# change focused on destination alignment rather than on any new color rule.
#
# 9) New assumptions introduced by the variant
# The updated solver assumes only that output width is the maximum width among
# the transformed rectangles and that all rectangles can be pasted starting from
# column 0. That is always true by construction. The generator additionally
# guarantees a positive centered-offset metric:
#     sum((max_final_width - rect_final_width) // 2) > 0
# so the produced sample set does not collapse back to the centered behavior.


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)


def _solve(grid: np.ndarray, *, left_align: bool) -> np.ndarray:
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

    if hcol == BLUE or vcol == BLUE:
        blue_col = BLUE
        red_col = int(vcol if hcol == BLUE else hcol)
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

    for rect in rects:
        rect["db"] = int(dist_to_edge(rect, blue_orient, blue_edge_idx))
        rect["dr"] = int(dist_to_edge(rect, red_orient, red_edge_idx))

    rects.sort(
        key=lambda rect: (
            int(rect["db"]),
            int(rect["dr"]),
            along_from_corner(rect, blue_orient),
            along_from_corner(rect, red_orient),
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
        out_c = 0 if left_align else int((out_w - w) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(h)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack rectangles by proximity to two colored edge lines.

    Variant rule change:
    - Sort rectangles exactly as in the source task.
    - If the blue edge is vertical, rotate each rectangle 90 degrees CCW.
    - Stack top-to-bottom with shared left edge alignment.
    """
    return _solve(grid, left_align=True)


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


def _sprinkle_minorities(patch: np.ndarray, dominant: int) -> np.ndarray:
    area = int(patch.size)
    if area < 6 or np.random.rand() < 0.35:
        return patch

    max_minor = min(4, max(1, area // 3))
    n_minor = int(np.random.randint(1, max_minor + 1))
    n_colors = int(np.random.randint(1, min(2, n_minor) + 1))
    minority_choices = [int(c) for c in MINORITY_COLORS if int(c) != dominant]
    colors = np.random.choice(minority_choices, size=n_colors, replace=False)
    cells = np.random.choice(area, size=n_minor, replace=False)
    flat = patch.reshape(-1)
    for idx, cell in enumerate(cells):
        flat[int(cell)] = int(colors[idx % n_colors])
    return patch


def _sample_rect_dims(count: int) -> list[tuple[int, int]]:
    for _ in range(200):
        heights = [int(np.random.randint(2, 6)) for _ in range(count)]
        widths = [int(np.random.randint(2, 7)) for _ in range(count)]
        dims = list(zip(heights, widths))
        if len(set(heights)) < 2:
            continue
        if len(set(widths)) < 2:
            continue
        if all(h == w for h, w in dims):
            continue
        return dims
    raise RuntimeError("failed to sample informative rectangle dimensions")


def _build_canonical() -> tuple[np.ndarray, list[tuple[int, int]]]:
    rows, cols = [(2, 2), (2, 3), (3, 2)][int(np.random.randint(3))]
    count = int(rows * cols)
    dims = _sample_rect_dims(count)

    row_heights = []
    for r in range(rows):
        row_heights.append(max(int(dims[r * cols + c][0]) for c in range(cols)))
    col_widths = []
    for c in range(cols):
        col_widths.append(max(int(dims[r * cols + c][1]) for r in range(rows)))

    vgap = 2
    hgap = 2
    trailing_bottom = int(np.random.randint(1, 3))
    trailing_right = int(np.random.randint(1, 3))
    H = 2 + int(sum(row_heights)) + int(vgap * (rows - 1)) + int(trailing_bottom)
    W = 2 + int(sum(col_widths)) + int(hgap * (cols - 1)) + int(trailing_right)

    grid = np.full((H, W), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    row_starts = []
    cur_r = 2
    for height in row_heights:
        row_starts.append(int(cur_r))
        cur_r += int(height) + int(vgap)

    col_starts = []
    cur_c = 2
    for width in col_widths:
        col_starts.append(int(cur_c))
        cur_c += int(width) + int(hgap)

    rect_dims = []
    used_colors: list[int] = []
    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            h, w = map(int, dims[idx])
            idx += 1
            color_options = [int(c) for c in DOMINANT_COLORS]
            if used_colors and np.random.rand() < 0.5:
                dominant = int(np.random.choice(used_colors))
            else:
                dominant = int(np.random.choice(color_options))
            used_colors.append(dominant)
            patch = np.full((h, w), dominant, dtype=int)
            patch = _sprinkle_minorities(patch, dominant)
            paste(grid, patch, int(row_starts[rr]), int(col_starts[cc]))
            rect_dims.append((h, w))

    return grid, rect_dims


def _alignment_signal(rect_dims: list[tuple[int, int]], blue_orient: str) -> int:
    final_widths = [int(w if blue_orient == "h" else h) for h, w in rect_dims]
    max_width = max(final_widths)
    return int(sum((max_width - width) // 2 for width in final_widths))


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    canonical, rect_dims = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    signal = int(_alignment_signal(rect_dims, blue_orient))
    if signal <= 0:
        return None
    grid = _transform(canonical, blue_orient, corner)
    meta = {"blue_orient": blue_orient, "corner": corner, "alignment_signal": signal}
    return grid, meta


def generate_input() -> np.ndarray:
    for _ in range(300):
        attempt = _gen_attempt()
        if attempt is None:
            continue
        grid, _meta = attempt
        left = _solve(grid, left_align=True)
        centered = _solve(grid, left_align=False)
        if np.array_equal(left, centered):
            continue
        return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D6 variant")
