# Combo variant of 291dc1e1 (index 12)
# combo_spec: D4/0+D11/3
# Name: 三band+band着色

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


# Design analysis for combo D4/0 + D11/3
# a) Behavior trace:
#    Baseline 291dc1e1 has two rectangle bands at db=1 and db=4 from the blue edge.
#    Rectangles are sorted by (db, dr, along_blue, along_red), optionally rotated when
#    blue is vertical, then stacked. This combo adds a third band at db=7 and recolors
#    the three db layers in output to fixed colors (1, 2, 3). Example in canonical form:
#    band rows at r=2,5,8 yield db values 1,4,7. A rectangle that used to keep its
#    original yellow/magenta/orange color now becomes blue if db=1, red if db=4, and
#    green if db=7. Sorting and rotation are unchanged; only the number of db layers and
#    output coloring change.
# b) Complexity calibration:
#    The combo keeps only three spatial layers and three output colors. That is enough to
#    make the fusion visually distinct from both the original task and the two-band color
#    transfer variant, without introducing too many categories. Each sample repeats every
#    band with 2-5 rectangles so the learner can infer the grouping.
# c) Hidden assumptions audited:
#    - Corner detection still assumes one special marker at a corner plus one full row
#      line and one full column line. The combo does not alter that.
#    - Rectangle extraction still assumes each rectangle fills its bbox without bg holes.
#      The generator preserves this by pasting solid 2xh patches and only changing
#      non-bg pixels for minor noise.
#    - Sorting logic assumes db/dr distances remain meaningful for every component.
#      Adding a third band preserves that because db now simply takes three values
#      {1,4,7} instead of two.
#    - Output coloring can no longer reuse source colors. We must map sorted db levels to
#      a fixed palette. The generator guarantees exactly three db levels so the mapping is
#      unambiguous.
# d) Placement legality:
#    The canonical generator uses short_dim=11 so three 2-cell-tall bands fit at
#    distances 1,4,7 from the blue line with 1-cell gaps between them and a final border.
#    Rectangles in each band are separated by 2 bg cells along the blue direction, so
#    connected-component extraction sees them as distinct non-overlapping objects.
# e) Visual observability:
#    Input samples show three clean parallel bands; output samples show the same stack
#    order but with three repeated colors. With 2-4 examples, a human can see both the
#    added far band and the fact that color now encodes band membership.
# f) Clean visualization:
#    Edge lines, marker, and rectangles remain in separate regions. Input rectangle colors
#    are drawn from {4,5,6,7,9}, reserving 1/2/3 for output-only semantics.
# g) Color-change inferability:
#    The output palette is fixed to three colors and directly tied to the three db layers.
#    Because each layer appears multiple times in every grid, the learner gets repeated
#    evidence that "same band => same output color".
# h) Fusion-specific visibility check:
#    To avoid degenerating into the original task or the two-band D11 variant, the
#    generator rejects samples unless there are exactly three db levels, the output uses
#    colors {1,2,3}, and the three-band output differs from the two-band recolor baseline.
# i) New solver assumptions and generator guarantees:
#    The solver's three-band recolor mode assumes the input contains exactly three db
#    layers. The generator guarantees this by constructing canonical bands at db=1,4,7
#    before rotating/flipping the whole grid.

BG = 8
BLUE = 1
RED = 2
THIRD = 3
MARKER = 0
BAND_OUTPUT_COLORS = (BLUE, RED, THIRD)
INPUT_COLORS = np.array([4, 5, 6, 7, 9], dtype=int)
BAND_ROWS = (2, 5, 8)
SHORT_DIM = 11
MAX_USED = 24


def _analyze_grid(grid: np.ndarray) -> tuple[np.ndarray, int, str, list[dict]]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    corners = [(0, 0), (0, int(w - 1)), (int(h - 1), 0), (int(h - 1), int(w - 1))]
    corner = None
    for r, c in corners:
        if int(g[r, c]) == int(bg):
            continue
        neighbors = []
        if r == 0:
            neighbors.append(int(g[r + 1, c]))
        if r == h - 1:
            neighbors.append(int(g[r - 1, c]))
        if c == 0:
            neighbors.append(int(g[r, c + 1]))
        if c == w - 1:
            neighbors.append(int(g[r, c - 1]))
        if neighbors and all(int(n) != int(g[r, c]) for n in neighbors):
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
        while 0 <= c < w and int(g[cr, c]) == int(hcol):
            hlen += 1
            c += int(hstep)
        vlen = 0
        r = int(cr + vstep)
        while 0 <= r < h and int(g[r, cc]) == int(vcol):
            vlen += 1
            r += int(vstep)
        if vlen >= hlen:
            blue_col = int(vcol)
            red_col = int(hcol)
        else:
            blue_col = int(hcol)
            red_col = int(vcol)
    del red_col

    blue_orient = "h" if int(hcol) == int(blue_col) else "v"
    red_orient = "v" if blue_orient == "h" else "h"

    line_mask = np.zeros((h, w), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = tuple(map(int, bbox_coords(coords)))
        sub = crop_bbox(g, box)
        if bool((sub == int(bg)).any()):
            continue
        rects.append({"box": box, "arr": sub.copy()})
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

    def along_from_corner(rect: dict, orient: str) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            return int(r0 if cr == 0 else -r1)
        return int(c0 if cc == 0 else -c1)

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

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
    return g, bg, blue_orient, rects


def _stack_sequence(seq: list[np.ndarray], bg: int) -> np.ndarray:
    out_w = int(max(int(arr.shape[1]) for arr in seq))
    out_h = int(sum(int(arr.shape[0]) for arr in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_r = 0
    for arr in seq:
        hh, ww = map(int, arr.shape)
        out_c = int((out_w - ww) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(hh)
    return out


def _solve(grid: np.ndarray, *, recolor_mode: str) -> np.ndarray:
    _, bg, blue_orient, rects = _analyze_grid(grid)
    db_levels = sorted({int(rect["db"]) for rect in rects})

    seq = []
    for rect in rects:
        arr = rect["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)

        if recolor_mode == "none":
            pass
        elif recolor_mode == "two_band":
            target = BLUE if int(rect["db"]) == int(db_levels[0]) else RED
            arr = np.where(arr != bg, int(target), int(bg))
        elif recolor_mode == "three_band":
            assert len(db_levels) == 3
            level_to_color = {
                int(db_levels[0]): BLUE,
                int(db_levels[1]): RED,
                int(db_levels[2]): THIRD,
            }
            arr = np.where(arr != bg, int(level_to_color[int(rect["db"])]), int(bg))
        else:
            raise ValueError(f"unknown recolor_mode: {recolor_mode}")
        seq.append(arr)
    return _stack_sequence(seq, bg)


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, recolor_mode="three_band")


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
    return [2 + 2 * int(unit) for unit in units]


def _sprinkle_minorities(patch: np.ndarray, dominant: int) -> np.ndarray:
    area = int(patch.size)
    if area < 8 or np.random.rand() < 0.45:
        return patch

    max_minor = min(3, area // 4)
    if max_minor < 2:
        return patch

    n_minor = int(np.random.randint(2, max_minor + 1))
    n_colors = int(np.random.randint(1, min(2, n_minor) + 1))
    minority_choices = [int(color) for color in INPUT_COLORS if int(color) != int(dominant)]
    colors = np.random.choice(minority_choices, size=n_colors, replace=False)
    cells = np.random.choice(area, size=n_minor, replace=False)
    flat = patch.reshape(-1)
    for idx, cell in enumerate(cells):
        flat[int(cell)] = int(colors[idx % n_colors])
    return patch


def _choose_band_counts() -> tuple[int, int, int]:
    choices = [
        (a, b, c)
        for a in range(2, 6)
        for b in range(2, 6)
        for c in range(2, 6)
        if 6 <= a + b + c <= 12
    ]
    return tuple(map(int, choices[int(np.random.randint(len(choices)))]))


def _choose_band_extent(count: int) -> int:
    min_used = 4 * count - 2
    max_used = min(MAX_USED, 10 * count - 2)
    options = np.arange(min_used, max_used + 1, 2)
    return int(np.random.choice(options))


def _build_canonical() -> np.ndarray:
    counts = _choose_band_counts()
    extents = [_choose_band_extent(int(count)) for count in counts]
    widths_by_band = [
        _distribute_band_widths(int(count), int(extent))
        for count, extent in zip(counts, extents)
    ]

    trailing = int(np.random.randint(1, 3))
    long_dim = int(2 + max(extents) + trailing)
    grid = np.full((SHORT_DIM, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    for r0, widths in zip(BAND_ROWS, widths_by_band):
        c = 2
        for width in widths:
            dominant = int(np.random.choice(INPUT_COLORS))
            patch = np.full((2, int(width)), dominant, dtype=int)
            patch = _sprinkle_minorities(patch, dominant)
            paste(grid, patch, int(r0), int(c))
            c += int(width) + 2
    return grid


def _gen_attempt() -> np.ndarray:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    return _transform(canonical, blue_orient, corner)


def _fusion_metrics(grid: np.ndarray) -> dict:
    rects = _analyze_grid(grid)[3]
    db_levels = tuple(sorted(int(rect["db"]) for rect in rects))
    unique_levels = tuple(sorted(set(db_levels)))
    band_counts = tuple(
        sum(1 for rect in rects if int(rect["db"]) == int(level)) for level in unique_levels
    )

    variant = _solve(grid, recolor_mode="three_band")
    two_band = _solve(grid, recolor_mode="two_band")
    baseline = _solve(grid, recolor_mode="none")

    non_bg_colors = tuple(sorted(set(map(int, variant.flatten())) - {BG}))
    return {
        "unique_levels": unique_levels,
        "band_counts": band_counts,
        "output_colors": non_bg_colors,
        "delta_to_two_band": int(np.count_nonzero(variant != two_band)),
        "delta_to_baseline": int(np.count_nonzero(variant != baseline)),
    }


def _variant_is_visible(grid: np.ndarray) -> bool:
    metrics = _fusion_metrics(grid)
    return (
        metrics["unique_levels"] == (1, 4, 7)
        and len(metrics["band_counts"]) == 3
        and min(metrics["band_counts"]) >= 2
        and metrics["output_colors"] == BAND_OUTPUT_COLORS
        and metrics["delta_to_two_band"] > 0
        and metrics["delta_to_baseline"] > 0
    )


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        try:
            _ = main(grid)
        except Exception:
            continue
        if _variant_is_visible(grid):
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 combo variant")
