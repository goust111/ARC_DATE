# Variant of 291dc1e1
# Dimension: D5 — 条件旋转规则 (transform)
# Change: 若蓝线为垂直方向，所有矩形做90度逆时针旋转；蓝线水平时不旋转... → 始终旋转90度（无论蓝线方向）

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


"""
Design analysis for the D5 variant
==================================

Source-task review:
- The variant metadata names no `source_task`, so there is no extra reference solver to inspect.

a) 具体化与行为追踪
- Input grids still contain one marked corner, one horizontal edge line, one vertical edge line,
  and several disconnected filled rectangles in the interior.
- In the canonical generator layout, rectangles are 2xh strips placed in two horizontal bands.
- Original behavior:
  if the blue line is horizontal, a 2x4 rectangle stays 2x4 in the output stack;
  if the blue line is vertical, that same rectangle is rotated to 4x2 before stacking.
- Variant behavior:
  the same 2x4 rectangle is always rotated to 4x2 before stacking, including when the blue line
  is horizontal.
- Concrete trace:
  suppose blue is horizontal and the sorted rectangle sequence is [green 2x4, yellow 2x2].
  Original output stack uses shapes [2x4, 2x2].
  Variant output stack uses shapes [4x2, 2x2].
  The sorting keys are unchanged; only the per-rectangle transform changes.

b) 复杂度校准
- The task keeps the original two-axis sorting logic and the same corner/line encoding, so the
  innovation is isolated to D5.
- To make the new rotation visually observable, the generator guarantees at least one rectangle
  has width > 2. Otherwise a grid of only 2x2 squares would make rotation invisible.
- Rectangles remain few (4-9 total) and widths come from a small even-valued set, so the puzzle
  stays legible and does not introduce extra free-form complexity.

c) 隐式假设审查
- Corner detection assumes exactly one corner cell differs from both adjacent edge-line colors.
  The variant does not change that; generator still constructs exactly one such marker.
- Blue/red identification assumes color 1 wins if present, else the longer line is blue.
  The variant does not change line semantics, so this remains valid.
- Component extraction assumes interior objects are disconnected rectangles with no background
  holes inside their bounding boxes. Generator still places rectangles with gaps and full bbox fill.
- The original rotation branch depended on `blue_orient == "v"`. That dependency is exactly what
  changes here: the transform no longer depends on orientation, so the solver must not reuse that
  condition.
- Stacking assumes each transformed rectangle can be centered into the output width independently.
  This still holds after unconditional rotation because every rectangle remains rectangular.

d) 放置合法性
- Canonical rectangles are placed in rows 2 and 5 with two-cell gaps, away from the edge lines.
- Global flips/rotations in `_transform` preserve non-overlap and keep everything in-bounds.
- Because the solver masks only the corner row/column, every rectangle must stay off those lines;
  the canonical placement guarantees this before and after the global transform.

e) 视觉可观察性
- The generator biases toward blue-horizontal cases, because that is where the variant differs from
  the original task.
- It also rejects any blue-horizontal sample whose output under the new solver matches the legacy
  conditional-rotation solver, preventing silent degeneration back to the original behavior.

f) 可视化整洁度
- The layout remains sparse: one corner marker, two border lines, two rectangle bands, and uniform
  background.
- No new overlay or marker system is added, so object boundaries stay easy to read.

g) 颜色变化可推断性
- There is no new color rule in this variant.
- Optional minority pixels are kept only as local asymmetry cues already present in the original
  generator style; they do not encode extra semantics.

h) 新 solver 的额外隐式假设
- Because rotation is now unconditional, the solver implicitly assumes each extracted object is a
  rectangle whose 90-degree rotation is still meaningful as a standalone object.
- The generator guarantees this by constructing every object as a full rectangular patch.
"""


def _locate_corner(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    H, W = g.shape
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
    cr, cc = corner
    return int(cr), int(cc), int(g[cr, cc])


def _line_orientations(g: np.ndarray, cr: int, cc: int) -> tuple[str, str]:
    H, W = g.shape
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
    return blue_orient, red_orient


def _solve(grid: np.ndarray, *, always_rotate: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    cr, cc, marker = _locate_corner(g, bg)
    blue_orient, red_orient = _line_orientations(g, cr, cc)

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
        if always_rotate or blue_orient == "v":
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack rectangles by proximity to the blue/red edge lines.

    D5 variant: every extracted rectangle is rotated 90 degrees CCW before stacking,
    regardless of whether the blue line is horizontal or vertical.
    """
    return _solve(grid, always_rotate=True)


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
    while True:
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
        used0 = int(
            max_used if max_band == 0 else np.random.choice(np.arange(4 * n0 - 2, max0 + 1, 2))
        )
        used1 = int(
            max_used if max_band == 1 else np.random.choice(np.arange(4 * n1 - 2, max1 + 1, 2))
        )
        band0 = _distribute_band_widths(n0, used0)
        band1 = _distribute_band_widths(n1, used1)
        if max(band0 + band1) <= 2:
            continue

        grid = np.full((8, long_dim), BG, dtype=int)
        grid[0, :] = BLUE
        grid[:, 0] = RED
        grid[0, 0] = MARKER

        for r0, widths in ((2, band0), (5, band1)):
            c = 2
            for w in widths:
                color = int(np.random.choice(DOMINANT_COLORS))
                patch = np.full((2, w), color, dtype=int)
                patch = _sprinkle_minorities(patch, color)
                paste(grid, patch, r0, c)
                c += w + 2

        return grid


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, always_rotate=False)


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = "h" if np.random.rand() < 0.8 else "v"
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    variant_out = main(grid)
    legacy_out = _legacy_main(grid)

    if blue_orient == "h":
        if np.array_equal(variant_out, legacy_out):
            return None
        return grid

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            _ = main(grid)
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D5 variant")
