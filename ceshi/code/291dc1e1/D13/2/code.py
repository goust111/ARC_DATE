# Variant of 291dc1e1
# Dimension: D13 — 源-目标基数关系 (meta)
# Change: 多对一——多个矩形合并到一个堆叠输出中（many:1）... → 多对多——按band分成两个独立堆叠
#
# Design analysis for this variant
# a) Concrete behavior and trace
#    - The original task already has two geometric "bands": rectangles sit in two parallel strips
#      at two different distances from the blue edge. In the canonical layout those are the upper
#      strip (distance 1 from the blue top line) and the lower strip (distance 4).
#    - Original solver behavior on a small canonical example:
#        blue = top edge, red = left edge
#        upper band rectangles widths = [4, 2], lower band widths = [6, 4]
#        distances to blue are [1, 1, 4, 4]; distances to red determine order inside each band
#        output = one single vertical stack: upper-1, upper-2, lower-1, lower-2
#    - Variant behavior:
#        keep the same grouping signal (distance to the blue edge), but do not merge the two bands.
#        Rectangles in the nearer band form the left pillar, rectangles in the farther band form
#        the right pillar. Each pillar is stacked independently using the same within-band ordering.
#    - The changed step is therefore the post-sorting aggregation stage: instead of one combined
#      sequence, we create two sequences keyed by blue-distance rank.
#
# b) Complexity calibration
#    - Keep exactly two blue-distance bands. More bands would complicate the rule beyond the D13
#      change; a single band would hide the change entirely.
#    - Keep 2-4 rectangles per band so the split is visible in every sample.
#    - Preserve occasional minority pixels inside rectangles to avoid uniformly bland outputs and
#      to keep rotation cases visually checkable without introducing extra symbolic machinery.
#
# c) Implicit-assumption audit against the original solver
#    - Corner detection still assumes exactly one marked corner cell touching two edge lines.
#      Generator keeps that invariant.
#    - Line masking still assumes the whole corner row/column are structural axes, not objects.
#      Generator keeps rectangles away from those axes.
#    - Component extraction assumes each object is a solid rectangle (its bbox contains no bg).
#      Generator still places rectangular patches with no holes.
#    - Original global sort relied on many:1 aggregation. That assumption is no longer valid.
#      We now rely on a new invariant: the rectangles occupy exactly two distinct blue-distance
#      levels. Generator enforces that by constructing exactly two bands.
#    - Rotation rule "if blue is vertical, rotate every rectangle 90° CCW" is unchanged.
#
# d) Placement legality
#    - Canonical generator places rectangles only in rows 2-3 and 5-6, with at least one bg row
#      between bands and two bg columns between neighboring rectangles.
#    - Width budgets are distributed so each band fits inside the canvas after accounting for gaps.
#    - Whole-grid transforms (flip/rotate) preserve non-overlap and edge-line legality.
#
# e) Visual observability
#    - A human can compare a few examples and see that rectangles still preserve the original
#      per-band ordering, but the result is now split into two side-by-side pillars instead of one
#      long column. The band split is visible because the two pillars often have different heights
#      and different internal width patterns.
#
# f) Visual cleanliness
#    - Input remains sparse: one corner marker, two full edge lines, two clean rectangle bands.
#    - Output uses a single bg spacer column between the two pillars, so the split is explicit
#      without adding new symbols.
#
# g) Color inferability
#    - Colors do not encode the new rule. They remain object-local texture only. The grouping cue
#      is geometric distance to the blue edge, which is stable across examples and independent of
#      palette variation.
#
# h) Variant metric used during validation
#    - Core metric: number of distinct blue-distance levels among extracted rectangles.
#      Valid samples must have exactly 2 levels and at least 2 rectangles in each level.
#    - Behavioral difference check: the variant output width must equal
#      near_pillar_width + 1 + far_pillar_width, which is structurally different from the legacy
#      many:1 single stack on the same input.
#
# i) New solver assumptions introduced by this variant
#    - The solver assumes exactly two blue-distance groups after rectangle extraction.
#      `generate_input()` guarantees this by construction.
#    - The solver assumes each group is non-empty and has a deterministic internal order given by
#      (distance to red edge, tie-break along blue, tie-break along red). Generator guarantees the
#      rectangles are separated along the red direction so this ordering is stable.

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


def _locate_corner_and_axes(g: np.ndarray, bg: int) -> tuple[int, int, int, str, str]:
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
    return int(cr), int(cc), int(marker), blue_orient, red_orient


def _dist_to_edge(rect: dict, orient: str, edge_idx: int) -> int:
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


def _along_from_corner(rect: dict, orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, rect["box"])
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _extract_rectangles(
    g: np.ndarray, bg: int, marker: int, cr: int, cc: int
) -> list[dict]:
    H, W = g.shape
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
    return rects


def _build_pillar(seq: list[np.ndarray], bg: int) -> np.ndarray:
    out_w = int(max(int(a.shape[1]) for a in seq))
    out_h = int(sum(int(a.shape[0]) for a in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_r = 0
    for a in seq:
        h, w = map(int, a.shape)
        out_c = int((out_w - w) // 2)
        paste(out, a, int(out_r), int(out_c))
        out_r += int(h)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Detect the same edge-line coordinate system as in the original task.
    - Group rectangles by their distance to the blue edge into two bands: near and far.
    - Sort rectangles independently inside each band by the original secondary keys.
    - If the blue edge is vertical, rotate every rectangle 90 degrees CCW.
    - Stack each band into its own pillar; place near on the left and far on the right.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cr, cc, marker, blue_orient, red_orient = _locate_corner_and_axes(g, bg)
    rects = _extract_rectangles(g, bg, marker, cr, cc)

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    for rect in rects:
        rect["db"] = int(_dist_to_edge(rect, blue_orient, blue_edge_idx))
        rect["dr"] = int(_dist_to_edge(rect, red_orient, red_edge_idx))
        rect["ab"] = int(_along_from_corner(rect, blue_orient, cr, cc))
        rect["ar"] = int(_along_from_corner(rect, red_orient, cr, cc))

    band_levels = sorted({int(rect["db"]) for rect in rects})
    assert len(band_levels) == 2

    pillars = []
    for band_db in band_levels:
        band = [rect for rect in rects if int(rect["db"]) == int(band_db)]
        assert band
        band.sort(key=lambda rect: (int(rect["dr"]), int(rect["ab"]), int(rect["ar"])))

        seq = []
        for rect in band:
            arr = rect["arr"]
            if blue_orient == "v":
                arr = np.rot90(arr, k=1)
            seq.append(arr)
        pillars.append(_build_pillar(seq, bg))

    near_pillar, far_pillar = pillars
    gap = 1
    out_h = int(max(int(near_pillar.shape[0]), int(far_pillar.shape[0])))
    out_w = int(near_pillar.shape[1] + gap + far_pillar.shape[1])
    out = np.full((out_h, out_w), int(bg), dtype=int)
    paste(out, near_pillar, 0, 0, transparent=bg)
    paste(out, far_pillar, 0, int(near_pillar.shape[1]) + gap, transparent=bg)
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
    long_dim = int(np.random.randint(14, 29))
    options = []
    for trailing in (1, 2):
        max_used = long_dim - 2 - trailing
        if max_used < 10 or max_used % 2:
            continue
        max_count = min(4, (max_used + 2) // 4)
        pairs = []
        for near_count in range(2, max_count + 1):
            for far_count in range(2, max_count + 1):
                if near_count + far_count > 8:
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
    near_cap = min(max_used, 10 * near_count - 2)
    far_cap = min(max_used, 10 * far_count - 2)
    near_used = int(
        max_used if max_band == 0 else np.random.choice(np.arange(4 * near_count - 2, near_cap + 1, 2))
    )
    far_used = int(
        max_used if max_band == 1 else np.random.choice(np.arange(4 * far_count - 2, far_cap + 1, 2))
    )
    near_widths = _distribute_band_widths(near_count, near_used)
    far_widths = _distribute_band_widths(far_count, far_used)

    # Avoid identical pillars too often; the D13 split should stay visually obvious.
    if near_count == far_count and near_widths == far_widths:
        far_widths = list(reversed(far_widths))

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
            c += int(w) + 2

    return grid


def _variant_structure_ok(grid: np.ndarray) -> bool:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cr, cc, marker, blue_orient, red_orient = _locate_corner_and_axes(g, bg)
    rects = _extract_rectangles(g, bg, marker, cr, cc)
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    counts: dict[int, int] = {}
    for rect in rects:
        db = int(_dist_to_edge(rect, blue_orient, blue_edge_idx))
        counts[db] = counts.get(db, 0) + 1
    levels = sorted(counts)
    return len(levels) == 2 and min(counts.values()) >= 2


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)
    if not _variant_structure_ok(grid):
        return None
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D13 variant")
