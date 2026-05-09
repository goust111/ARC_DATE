# Variant of 291dc1e1
# Dimension: D10 — 堆叠顺序与多步处理 (multi_step)
# Change: 一次性排序所有矩形后按序堆叠，无迭代... → 分band分步堆叠

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

# Design analysis for the D10 variant
# a) Concrete behavior:
#    The original task already has two visually clean "bands" measured by distance to the blue edge:
#    in the canonical layout, one row/column of rectangles is near the blue line and another is farther away.
#    Originally all rectangles were globally sorted by (db, dr) and then stacked with no pause.
#    In this variant, rectangles are first partitioned by db (distance to the blue line), then each band is
#    processed as one step: near band first, far band second. Within each band, rectangles are sorted by
#    distance to the red line. The output now contains a full background separator row between successive bands.
#    Small trace on a canonical top-left example:
#    near band at db=1 contains A,B,C with dr=1,5,9; far band at db=4 contains D,E with dr=2,8.
#    Original output order: A,B,C,D,E. Variant output: A,B,C,[bg row],D,E.
# b) Complexity calibration:
#    The reference mechanism from 2ba387bc is low-complexity grouped output, not a new geometric transform.
#    Keeping exactly two bands in generated examples makes the new stepwise behavior obvious while preserving
#    the original edge-distance logic. Each band contains 2-5 rectangles, so the separator is not the only clue:
#    there is repeated evidence that one whole band is exhausted before the next begins.
# c) Hidden assumptions reviewed:
#    - Corner finding still assumes a unique marker color at a corner and remains unchanged.
#    - Blue/red axis detection still assumes one horizontal and one vertical edge line meeting at that marker.
#    - Rectangle extraction still relies on full non-background bboxes; the generator keeps rectangles solid
#      (possibly with minority non-bg pixels) so connectivity and bbox-filling remain valid.
#    - The original single rects.sort(...) hard-coded a global total order. That assumption no longer matches
#      the variant, so sorting is moved inside db-groups. The rotation rule still applies per rectangle.
#    - Output height can no longer be just sum(rect heights); it must also reserve separator rows.
# d) Placement legality:
#    The generator keeps the original canonical construction: two non-overlapping bands with fixed vertical
#    spacing and at least one background column between rectangles. After flips/rotations, objects still stay
#    disjoint and remain at two distinct blue-edge distances, so the solver can recover the grouping reliably.
# e) Visual observability:
#    From 2-4 examples a human can see that the output is not merely sorted; it is built in phases because an
#    empty background row separates the "near blue" pile from the "far blue" pile. Both bands are guaranteed
#    non-empty, so the multi-step structure is always present.
# f) Visual cleanliness:
#    The input still contains only corner marker, two edge lines, and rectangles on a uniform background.
#    The output adds just one full background row between bands, avoiding noisy extra markers.
# g) Color-change inferability:
#    No new color semantics are introduced. Colors inside rectangles remain object-local decoration only, so the
#    new rule is conveyed by spatial grouping and the separator row rather than by arbitrary recoloring.


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)


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


def _band_sort_key(rect: dict, red_orient: str, blue_orient: str, cr: int, cc: int) -> tuple[int, int, int]:
    box = rect["box"]
    return (
        int(rect["dr"]),
        _along_from_corner(box, red_orient, cr, cc),
        _along_from_corner(box, blue_orient, cr, cc),
    )


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
    long_dim = int(np.random.randint(11, 29))
    options = []
    for trailing in (1, 2):
        max_used = long_dim - 2 - trailing
        if max_used < 6 or max_used % 2:
            continue
        max_count = min(5, (max_used + 2) // 4)
        pairs = []
        for near_count in range(2, max_count + 1):
            for far_count in range(2, max_count + 1):
                if near_count + far_count > 9:
                    continue
                eligible = []
                if max_used <= 10 * near_count - 2:
                    eligible.append(0)
                if max_used <= 10 * far_count - 2:
                    eligible.append(1)
                if eligible:
                    pairs.append((near_count, far_count, eligible))
        if pairs:
            options.append((max_used, pairs))

    max_used, pairs = options[int(np.random.randint(len(options)))]
    near_count, far_count, eligible = pairs[int(np.random.randint(len(pairs)))]
    max_band = int(np.random.choice(eligible))
    max_near = min(max_used, 10 * near_count - 2)
    max_far = min(max_used, 10 * far_count - 2)
    used_near = int(
        max_used if max_band == 0 else np.random.choice(np.arange(4 * near_count - 2, max_near + 1, 2))
    )
    used_far = int(
        max_used if max_band == 1 else np.random.choice(np.arange(4 * far_count - 2, max_far + 1, 2))
    )
    near_widths = _distribute_band_widths(near_count, used_near)
    far_widths = _distribute_band_widths(far_count, used_far)

    grid = np.full((8, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    for r0, widths in ((2, near_widths), (5, far_widths)):
        c = 2
        for w in widths:
            color = int(np.random.choice(DOMINANT_COLORS))
            patch = np.full((2, w), color, dtype=int)
            patch = _sprinkle_minorities(patch, color)
            paste(grid, patch, r0, c)
            c += w + 2

    return grid


def _band_profile(grid: np.ndarray) -> list[int]:
    _bg, _bo, _ro, _cr, _cc, rects = _extract_rects(grid)
    counts: dict[int, int] = {}
    for rect in rects:
        db = int(rect["db"])
        counts[db] = counts.get(db, 0) + 1
    return [counts[k] for k in sorted(counts)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack rectangles in multiple blue-distance steps.

    Input structure stays the same as 291dc1e1:
    - a marked corner where one horizontal and one vertical colored edge line meet
    - solid interior rectangles located at different distances from those two edges

    Variant rule:
    - Partition rectangles by distance to the blue edge (near band first, far band later).
    - Within each band, sort by distance to the red edge.
    - If the blue edge is vertical, rotate each rectangle 90° CCW.
    - Stack each band top-to-bottom, centered, and insert one full background separator row between bands.
    """
    g = np.array(grid, dtype=int)
    bg, blue_orient, red_orient, cr, cc, rects = _extract_rects(g)

    bands: list[list[np.ndarray]] = []
    for db in sorted({int(rect["db"]) for rect in rects}):
        group = [rect for rect in rects if int(rect["db"]) == int(db)]
        group.sort(key=lambda rect: _band_sort_key(rect, red_orient, blue_orient, cr, cc))
        band_arrays = []
        for rect in group:
            arr = rect["arr"]
            if blue_orient == "v":
                arr = np.rot90(arr, k=1)
            band_arrays.append(arr)
        bands.append(band_arrays)

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


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    try:
        band_sizes = _band_profile(grid)
        if len(band_sizes) != 2 or min(band_sizes) < 2:
            return None
        out = main(grid)
        if not any(bool((out[r, :] == BG).all()) for r in range(1, int(out.shape[0]) - 1)):
            return None
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D10 variant")
