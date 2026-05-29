# Variant of 291dc1e1
# Dimension: D14 — 干扰项与噪声 (src)
# Change: 网格中除矩形和边缘线外无额外对象... → 部分矩形被排除（需根据规则筛选）
#
# Design analysis for the D14 variant
# a) Concrete behavior and trace
#    The original task stacks every interior rectangle after sorting by distance to
#    the blue edge and then the red edge. This variant keeps the same geometric
#    sorting, but first filters rectangles by bbox area frequency: only rectangles
#    whose area appears at least twice are stacked. Rectangles with a unique area
#    stay visible in the input as distractors and disappear from the output.
#    Example in canonical orientation (blue = top edge, red = left edge):
#      top band:    widths [3, 5]
#      bottom band: widths [3, 2]
#    Original flow: stack all four rectangles in order
#      (top-3) -> (top-5) -> (bottom-3) -> (bottom-2).
#    Variant flow: areas are [6, 10, 6, 4], so only the two area-6 rectangles
#      remain. They are stacked in the same relative order:
#      (top-3) -> (bottom-3). The 5-wide and 2-wide rectangles are excluded.
# b) Complexity calibration
#    The generator uses 3-6 rectangles with 1-2 duplicate-area groups and at
#    least 1 unique-area rectangle. Widths come from a small pool, so repeated
#    sizes are easy to see, but there is enough variation that the filtering is
#    not trivial or visually ambiguous.
# c) Old assumptions that had to be checked
#    - Rectangle extraction assumes every object fully fills its bbox with non-bg
#      colors. That still holds.
#    - The old solver could stack every extracted rectangle. That is no longer
#      valid; area counts must be computed after extraction and before sorting.
#    - The old code could tolerate a single rectangle. The new rule can create an
#      empty selected set if every area is unique, so the generator must guarantee
#      at least one repeated area group. `main()` asserts that intended domain.
# d) Placement legality
#    The generator still uses two interior horizontal bands, with two columns of
#    padding from the edge lines and two columns of spacing between rectangles.
#    Grid width is derived from the widest band, so rectangles never overlap or
#    go out of bounds. Because the entire canonical grid is transformed as one
#    piece, those legality guarantees survive all corner/orientation variants.
# e) Visual observability
#    Repeated areas are represented by repeated rectangle widths, which is easier
#    to infer than matching arbitrary equal areas across different shapes. At
#    least one repeated width survives into the output, and at least one unique
#    width disappears, so the D14 change is visible across few-shot examples.
# f) Visual cleanliness
#    The scene still contains only the two edge lines, the corner marker, and
#    separated rectangles. Each rectangle gets a distinct dominant color, so the
#    surviving and excluded objects remain easy to track by eye.
# g) Color reasoning
#    No new color mapping is introduced. Minority pixels are only used to break
#    symmetry inside rectangles, so 90-degree rotation remains observable even
#    for near-square objects.

from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
WIDTH_POOL = np.array([2, 3, 4, 5, 6, 7], dtype=int)
AREA_CONFIGS = [
    ([2], 1),
    ([2], 2),
    ([3], 1),
    ([2, 2], 1),
    ([2, 2], 2),
    ([3, 2], 1),
]


def _locate_corner(g: np.ndarray, bg: int) -> tuple[int, int]:
    H, W = map(int, g.shape)
    corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
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
            return int(r), int(c)
    raise AssertionError("marked corner not found")


def _extract_rectangles(g: np.ndarray, bg: int, line_mask: np.ndarray, marker: int) -> list[dict]:
    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = bbox_coords(coords)
        sub = crop_bbox(g, box)
        if bool((sub == int(bg)).any()):
            continue
        rects.append(
            {
                "box": tuple(map(int, box)),
                "arr": sub.copy(),
                "area": int(sub.shape[0] * sub.shape[1]),
            }
        )
    return rects


def _solve(grid: np.ndarray, *, min_area_freq: int) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    cr, cc = _locate_corner(g, bg)
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

    rects = _extract_rectangles(g, bg, line_mask, marker)
    assert rects

    area_freq = Counter(int(rct["area"]) for rct in rects)
    rects = [rct for rct in rects if int(area_freq[int(rct["area"])]) >= int(min_area_freq)]
    assert rects

    def dist_to_edge(rect: dict, orient: str, edge_idx: int) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            col = int(edge_idx)
            if c0 > col:
                return int(c0 - col - 1)
            if c1 < col:
                return int(col - c1 - 1)
            return 0
        row = int(edge_idx)
        if r0 > row:
            return int(r0 - row - 1)
        if r1 < row:
            return int(row - r1 - 1)
        return 0

    def along_from_corner(rect: dict, orient: str) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            return int(r0 if cr == 0 else -r1)
        return int(c0 if cc == 0 else -c1)

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack only the rectangles whose bbox area is repeated.

    Input structure:
    - bg = most common color.
    - Two edge lines (horizontal + vertical) meet at a marked corner.
    - Interior objects are full rectangles; some rectangle areas repeat, others
      appear only once.

    Rule:
    - Extract all interior rectangles.
    - Keep only rectangles whose bbox area occurs at least twice.
    - Sort kept rectangles by (distance to blue edge, distance to red edge).
    - If the blue edge is vertical, rotate each kept rectangle 90 degrees CCW.
    - Stack the kept rectangles top-to-bottom, centered horizontally.
    """
    return _solve(grid, min_area_freq=2)


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
    if area < 8 or np.random.rand() < 0.45:
        return patch

    max_minor = min(3, area // 4)
    if max_minor < 2:
        return patch

    n_minor = int(np.random.randint(2, max_minor + 1))
    n_colors = int(np.random.randint(1, min(2, n_minor) + 1))
    minority_choices = [int(c) for c in MINORITY_COLORS if int(c) != int(dominant)]
    colors = np.random.choice(minority_choices, size=n_colors, replace=False)
    cells = np.random.choice(area, size=n_minor, replace=False)
    flat = patch.reshape(-1)
    for idx, cell in enumerate(cells):
        flat[int(cell)] = int(colors[idx % n_colors])
    return patch


def _sample_width_layout() -> tuple[list[int], list[int], Counter]:
    dup_groups, n_unique = AREA_CONFIGS[int(np.random.randint(len(AREA_CONFIGS)))]
    n_distinct = int(len(dup_groups) + n_unique)
    chosen = [int(x) for x in np.random.choice(WIDTH_POOL, size=n_distinct, replace=False)]

    dup_widths = chosen[: len(dup_groups)]
    unique_widths = chosen[len(dup_groups) :]

    widths = []
    for width, count in zip(dup_widths, dup_groups):
        widths.extend([int(width)] * int(count))
    widths.extend(int(width) for width in unique_widths)

    np.random.shuffle(widths)
    split = int(np.random.randint(1, len(widths)))
    top = widths[:split]
    bottom = widths[split:]
    if not top or not bottom:
        mid = len(widths) // 2
        top = widths[:mid]
        bottom = widths[mid:]

    freq = Counter(int(width * 2) for width in widths)
    return [int(w) for w in top], [int(w) for w in bottom], freq


def _build_canonical() -> tuple[np.ndarray, Counter]:
    top_widths, bottom_widths, area_freq = _sample_width_layout()
    top_used = int(sum(top_widths) + 2 * (len(top_widths) - 1))
    bottom_used = int(sum(bottom_widths) + 2 * (len(bottom_widths) - 1))
    trailing = int(np.random.randint(1, 3))
    width = int(2 + max(top_used, bottom_used) + trailing)

    grid = np.full((8, width), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    all_widths = top_widths + bottom_widths
    colors = np.random.permutation(DOMINANT_COLORS)[: len(all_widths)]
    color_idx = 0

    for row, widths in ((2, top_widths), (5, bottom_widths)):
        col = 2
        for rect_w in widths:
            dominant = int(colors[color_idx])
            color_idx += 1
            patch = np.full((2, int(rect_w)), dominant, dtype=int)
            patch = _sprinkle_minorities(patch, dominant)
            paste(grid, patch, int(row), int(col))
            col += int(rect_w) + 2

    return grid, area_freq


def _gen_attempt() -> np.ndarray:
    canonical, area_freq = _build_canonical()
    assert any(int(freq) >= 2 for freq in area_freq.values())
    assert any(int(freq) == 1 for freq in area_freq.values())

    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    out_variant = main(grid)
    out_original = _solve(grid, min_area_freq=1)
    assert out_variant.size < out_original.size or not np.array_equal(out_variant, out_original)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            return _gen_attempt()
        except (AssertionError, ValueError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 291dc1e1 D14 variant")
