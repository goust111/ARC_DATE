# Variant of 291dc1e1
# Dimension: D6 — 堆叠方式与对齐 (dst)
# Change: 矩形从上到下紧密堆叠，水平方向居中对齐，无间隙... → 堆叠间留1像素间隙

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


"""
Design analysis for the D6 variant
==================================

3a) Concrete behavior and trace
- Input stays the same as the source task: two colored edge lines meet at a marked
  corner; interior contains several solid 2-row rectangles separated by background.
- The original solver extracts those rectangles, sorts them by distance to the
  designated blue/red edges, optionally rotates them if the blue edge is vertical,
  then stacks them in a centered vertical column with no spacing.
- This variant changes only the final stacking layout: keep the same order,
  rotation, and centering, but insert exactly one full background row between every
  consecutive pair of rectangles.
- Small trace:
  Suppose sorted rectangle heights are [2, 2, 4] and output width is 6.
  Original output height = 2 + 2 + 4 = 8 and row starts are [0, 2, 4].
  Variant output height = 2 + 1 + 2 + 1 + 4 = 10 and row starts are [0, 3, 6].
  The only structural difference is the inserted full-background separator rows.

3b) Complexity calibration
- The gap rule is fixed at one pixel, not variable, so the concept is easy to infer.
- The generator still creates 4-9 rectangles across two interior bands, which gives
  repeated evidence that the spacing is systematic rather than accidental.
- The rectangles keep varied widths and sometimes asymmetric minority-color pixels,
  so ordering and optional rotation remain observable and the task does not collapse
  into identical blocks separated by empty rows.

3c) Implicit-assumption audit against the original main()
- Corner detection, line-color detection, rectangle extraction, distance sorting,
  and optional rotation do not depend on compact stacking. Those assumptions remain
  valid unchanged.
- The old output code assumed:
  1. output height is exactly the sum of rectangle heights;
  2. the next rectangle starts immediately after the previous one;
  3. no background-only rows occur in the output.
  Only these assumptions break under the variant, so only this segment must change.
- No other line in main() relies on adjacency between stacked rectangles.

3d) Placement legality
- The generator already spaces input rectangles by background columns and rows, so
  connected-component extraction stays unambiguous.
- In the output, the new formula `sum(h) + (n - 1)` allocates exactly one row for
  each separator, preventing overlap and out-of-bounds pastes.
- Because we paste rectangles only at computed centered positions inside `outW`,
  horizontal placement remains valid after rotation.

3e) Visual observability
- A human looking at 2-3 examples can see that rectangles are still ordered and
  centered as before, but are now separated by blank teal rows.
- This is especially visible because the generator always produces at least four
  rectangles, so there are multiple separator rows in every sample.

3f) Visual cleanliness
- The separator is a full background row, so it does not compete with any object
  pixels and makes the stacked output easier to segment visually.
- Input remains uncluttered: two edge lines, one marker, and disjoint rectangles.

3g) Color-rule inferability
- No new color mapping is introduced. The separator uses the existing background
  color, so there is no extra color relation for the solver to infer.

9) New assumptions introduced by the variant
- main() now assumes the desired spacing is always exactly one row. generate_input()
  guarantees this by defining the variant itself and validating produced samples.
- The generator also assumes there are at least two rectangles so the gap behavior is
  visible; it guarantees more strongly that there are at least four rectangles.
- Validation checks for at least one all-background row in the produced output,
  ensuring samples do not silently degenerate back to compact stacking.
"""


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack rectangles by proximity to two colored edge lines.

    Variant rule:
    - Sort rectangles by (distance to blue edge, distance to red edge).
    - If the blue edge is vertical, rotate each rectangle 90 degrees CCW.
    - Stack top-to-bottom, centered horizontally.
    - Insert exactly one background row between consecutive rectangles.
    """
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
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if bool((sub == int(bg)).any()):
            continue
        rects.append({"box": tuple(map(int, b)), "arr": sub.copy()})

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

    outW = int(max(int(a.shape[1]) for a in seq))
    total_rect_h = int(sum(int(a.shape[0]) for a in seq))
    total_gap_h = int(max(0, len(seq) - 1))
    outH = int(total_rect_h + total_gap_h)
    out = np.full((outH, outW), int(bg), dtype=int)

    out_r = 0
    for idx, arr in enumerate(seq):
        h, w = map(int, arr.shape)
        out_c = int((outW - w) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(h)
        if idx + 1 < len(seq):
            out_r += 1

    return out


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)


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


def _build_canonical() -> np.ndarray:
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
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    for r0, widths in ((2, band0), (5, band1)):
        c = 2
        for width in widths:
            color = int(np.random.choice(DOMINANT_COLORS))
            patch = np.full((2, width), color, dtype=int)
            patch = _sprinkle_minorities(patch, color)
            paste(grid, patch, r0, c)
            c += width + 2

    return grid


def _gen_attempt() -> np.ndarray:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    return _transform(canonical, blue_orient, corner)


def _output_has_separator_row(out: np.ndarray, bg: int) -> bool:
    return any(bool(np.all(out[r] == int(bg))) for r in range(int(out.shape[0])))


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        out = main(grid)
        if _output_has_separator_row(out, int(get_bg(grid))):
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D6 variant")
