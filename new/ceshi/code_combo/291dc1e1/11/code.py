# Combo variant of 291dc1e1 (index 11)
# combo_spec: D4/0+D10/0
# Name: 三band+三步堆叠

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

# Design analysis for the combo variant
# a) Concrete behavior:
#    The base task has rectangles arranged in parallel bands whose blue-edge distance `db`
#    is meaningful. Here the canonical layout is extended from two bands to three:
#    db = 1, 4, 7. Example in top-left canonical orientation:
#    - near band at rows 2:4
#    - mid band at rows 5:7
#    - far band at rows 8:10
#    If those bands contain A,B | C,D,E | F,G ordered by increasing red-edge distance `dr`,
#    the original task would output A,B,C,D,E,F,G as one uninterrupted stack.
#    This combo instead outputs A,B,[bg row],C,D,E,[bg row],F,G. If blue is vertical, each
#    rectangle is still rotated 90 degrees CCW before stacking, exactly as in the base task.
# b) Complexity calibration:
#    The third band must be visually real, not a corner case, so the generator enforces
#    exactly three distinct `db` levels and at least two rectangles per band. That creates
#    repeated evidence for three ordered phases while keeping the grammar simple: only one
#    extra band and one extra separator compared with the D10 single-variant reference.
# c) Hidden assumptions reviewed:
#    - Corner/axis detection assumes one marked corner and one full horizontal plus one full
#      vertical edge line. The combo keeps that unchanged.
#    - Rectangle extraction assumes each object fills its bbox with no background holes.
#      The generator keeps rectangles solid except for optional minority-color pixels.
#    - The original global sort assumed one uninterrupted total order over all rectangles.
#      That assumption is intentionally broken: the solver must now group by `db` first,
#      then sort by `dr` only inside each group.
#    - Output sizing can no longer be just sum(rect heights); it must also allocate one
#      separator row between consecutive band-steps.
# d) Placement legality:
#    The canonical generator uses short_dim = 11 so the three 2-cell-thick bands fit at
#    fixed rows 2, 5, 8 with 1-cell gaps between them and a 1-cell border after the far
#    band. Along the blue direction, rectangles start at dr = 1 and are separated by 2 bg
#    cells, so they never overlap and always stay in bounds.
# e) Visual observability:
#    Across a few examples, a human sees three parallel source bands in the input and three
#    corresponding stack phases in the output, because the output contains exactly two full
#    background separator rows. This makes the "three-step" behavior legible instead of
#    looking like a normal global sort.
# f) Visual cleanliness:
#    No extra markers or recoloring are introduced. Input still contains only the corner
#    marker, the two guide lines, and solid rectangles on uniform bg. Output uses plain bg
#    rows as separators, which is visually clean and easy to parse.
# g) Color-change inferability:
#    Colors do not encode the new rule. They remain object-local texture only. The combo is
#    conveyed entirely by geometry: three source bands and three stacked phases.
#
# New implicit assumptions introduced by the combo and guaranteed by the generator:
# - There are exactly three distinct blue-distance groups in generated inputs.
# - Every band is non-empty and, more strongly, has at least two rectangles so the banding
#   is supported by repetition rather than by a singleton anomaly.
# - Distinct bands remain distinguishable after global flips/rotations because their `db`
#   values stay separated by 3 cells.


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
BAND_ROWS = (2, 5, 8)


def _locate_structure(g: np.ndarray) -> tuple[int, int, int, int, str, str]:
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
    return bg, cr, cc, marker, blue_orient, red_orient


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


def _along_from_corner(box: tuple[int, int, int, int], orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _extract_rects(g: np.ndarray) -> tuple[int, str, str, int, int, list[dict]]:
    H, W = g.shape
    bg, cr, cc, marker, blue_orient, red_orient = _locate_structure(g)

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects: list[dict] = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = tuple(map(int, bbox_coords(coords)))
        sub = crop_bbox(g, box)
        if bool((sub == int(bg)).any()):
            continue
        rects.append({"box": box, "arr": sub.copy()})

    assert rects

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    for rect in rects:
        rect["db"] = int(_dist_to_edge(rect["box"], blue_orient, blue_edge_idx))
        rect["dr"] = int(_dist_to_edge(rect["box"], red_orient, red_edge_idx))

    return bg, blue_orient, red_orient, cr, cc, rects


def _global_sort_key(rect: dict, blue_orient: str, red_orient: str, cr: int, cc: int) -> tuple[int, int, int, int]:
    box = rect["box"]
    return (
        int(rect["db"]),
        int(rect["dr"]),
        _along_from_corner(box, blue_orient, cr, cc),
        _along_from_corner(box, red_orient, cr, cc),
    )


def _band_sort_key(rect: dict, red_orient: str, blue_orient: str, cr: int, cc: int) -> tuple[int, int, int]:
    box = rect["box"]
    return (
        int(rect["dr"]),
        _along_from_corner(box, red_orient, cr, cc),
        _along_from_corner(box, blue_orient, cr, cc),
    )


def _rotate_if_needed(arr: np.ndarray, blue_orient: str) -> np.ndarray:
    return np.rot90(arr, k=1) if blue_orient == "v" else arr


def _stack_sequence(seq: list[np.ndarray], bg: int, separator_rows: int = 0) -> np.ndarray:
    out_w = int(max(int(arr.shape[1]) for arr in seq))
    out_h = int(sum(int(arr.shape[0]) for arr in seq) + int(separator_rows))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = int((out_w - w) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(h)

    return out


def _stack_bands(bands: list[list[np.ndarray]], bg: int) -> np.ndarray:
    out_w = int(max(int(arr.shape[1]) for band in bands for arr in band))
    out_h = int(sum(int(arr.shape[0]) for band in bands for arr in band) + max(0, len(bands) - 1))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for band_idx, band in enumerate(bands):
        if band_idx > 0:
            out_r += 1
        for arr in band:
            h, w = map(int, arr.shape)
            out_c = int((out_w - w) // 2)
            paste(out, arr, int(out_r), int(out_c))
            out_r += int(h)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack rectangles in three blue-distance steps.

    Rule:
    - Detect rectangles and measure each one's distance to the blue edge (`db`) and red edge (`dr`).
    - Partition by `db` and process bands from near to far.
    - Within each band, sort by `dr` with the original corner-aware tie-breakers.
    - If blue is vertical, rotate each rectangle 90 degrees CCW.
    - Stack band 1, separator row, band 2, separator row, band 3.
    """
    g = np.array(grid, dtype=int)
    bg, blue_orient, red_orient, cr, cc, rects = _extract_rects(g)

    bands: list[list[np.ndarray]] = []
    for db in sorted({int(rect["db"]) for rect in rects}):
        group = [rect for rect in rects if int(rect["db"]) == int(db)]
        group.sort(key=lambda rect: _band_sort_key(rect, red_orient, blue_orient, cr, cc))
        bands.append([_rotate_if_needed(rect["arr"], blue_orient) for rect in group])

    return _stack_bands(bands, bg)


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


def _choose_band_plan(max_used: int) -> tuple[list[int], list[int]]:
    feasible_counts = [count for count in range(2, 5) if 4 * count - 2 <= max_used]
    assert feasible_counts

    while True:
        counts = [int(np.random.choice(feasible_counts)) for _ in range(3)]
        eligible = [idx for idx, count in enumerate(counts) if max_used <= 10 * count - 2]
        if eligible:
            break

    if max(feasible_counts) >= 3 and max(counts) == 2 and np.random.rand() < 0.7:
        upgradeable = [idx for idx, count in enumerate(counts) if count < max(feasible_counts)]
        if upgradeable:
            idx = int(np.random.choice(upgradeable))
            counts[idx] = 3
            eligible = [j for j, count in enumerate(counts) if max_used <= 10 * count - 2]

    full_band = int(np.random.choice(eligible))
    used_targets: list[int] = []
    for idx, count in enumerate(counts):
        upper = min(max_used, 10 * count - 2)
        choices = np.arange(4 * count - 2, upper + 1, 2)
        assert len(choices) > 0
        if idx == full_band:
            used_targets.append(int(max_used))
        else:
            used_targets.append(int(np.random.choice(choices)))
    return counts, used_targets


def _build_canonical() -> np.ndarray:
    max_used = int(np.random.choice(np.arange(10, 27, 2)))
    trailing = int(np.random.choice([1, 2]))
    long_dim = int(2 + max_used + trailing)

    counts, used_targets = _choose_band_plan(max_used)
    band_widths = [_distribute_band_widths(count, used) for count, used in zip(counts, used_targets)]

    grid = np.full((11, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    for r0, widths in zip(BAND_ROWS, band_widths):
        c = 2
        for w in widths:
            color = int(np.random.choice(DOMINANT_COLORS))
            patch = np.full((2, w), color, dtype=int)
            patch = _sprinkle_minorities(patch, color)
            paste(grid, patch, int(r0), int(c))
            c += int(w) + 2

    return grid


def _plain_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, blue_orient, red_orient, cr, cc, rects = _extract_rects(g)
    rects.sort(key=lambda rect: _global_sort_key(rect, blue_orient, red_orient, cr, cc))
    seq = [_rotate_if_needed(rect["arr"], blue_orient) for rect in rects]
    return _stack_sequence(seq, bg, separator_rows=0)


def _combo_metrics(grid: np.ndarray) -> dict[str, object]:
    bg, blue_orient, red_orient, cr, cc, rects = _extract_rects(np.array(grid, dtype=int))
    band_counts: dict[int, int] = {}
    for rect in rects:
        db = int(rect["db"])
        band_counts[db] = band_counts.get(db, 0) + 1

    combo_out = main(grid)
    plain_out = _stack_sequence(
        [_rotate_if_needed(rect["arr"], blue_orient) for rect in sorted(rects, key=lambda r: _global_sort_key(r, blue_orient, red_orient, cr, cc))],
        bg,
        separator_rows=0,
    )
    separator_rows = sum(
        1
        for r in range(1, int(combo_out.shape[0]) - 1)
        if bool((combo_out[r, :] == int(bg)).all())
    )

    return {
        "db_levels": sorted(band_counts),
        "band_sizes": [band_counts[k] for k in sorted(band_counts)],
        "separator_rows": int(separator_rows),
        "differs_from_plain": bool(combo_out.shape != plain_out.shape or not np.array_equal(combo_out, plain_out)),
    }


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    try:
        metrics = _combo_metrics(grid)
        if metrics["db_levels"] != [1, 4, 7]:
            return None
        if min(metrics["band_sizes"]) < 2:
            return None
        if metrics["separator_rows"] != 2:
            return None
        if not metrics["differs_from_plain"]:
            return None
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 combo variant")
