# Combo variant of 291dc1e1 (index 19)
# combo_spec: D10/0+D11/3
# Name: 分band分步+band着色

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    The source task uses the corner marker plus two edge lines as axes, then globally sorts all rectangles by
#    (db, dr) and stacks them. In the canonical layout there are already two clean blue-distance bands:
#    near band at db=1 and far band at db=4. This combo changes the same blue-distance signal in two coupled ways.
#    Example: near band has A,B,C with dr=1,5,9; far band has D,E with dr=1,7.
#    Original output: A,B,C,D,E with original colors preserved.
#    Combo output: recolor A,B,C to blue(1), stack them first; insert one full bg separator row; recolor D,E to
#    red(2), stack them after the separator. If blue edge is vertical, each rectangle is rotated 90 CCW before
#    recoloring/stacking, exactly as in the original task. The key difference is that band membership now controls
#    both process phase and output color, so the same grouping cue appears twice.
# b) Complexity calibration:
#    The design intentionally keeps only two visible blue-distance groups. That avoids ambiguous many-to-one color
#    mapping while making the fusion easy to observe: one block of blue rectangles, one separator, one block of red
#    rectangles. Each band still contains 2-5 rectangles, so the rule is supported by repeated evidence rather than
#    by a single separator row or a single recolored object.
# c) Hidden assumptions reviewed against the original main():
#    - Corner detection still assumes a unique marker at one outer corner with different-colored line neighbors.
#      The generator preserves that unchanged.
#    - Blue/red axis detection still assumes one horizontal and one vertical full edge line through the marker.
#    - Rectangle extraction still assumes each component's bbox contains no background. The generator keeps every
#      rectangle solid, allowing minority non-bg pixels but never holes.
#    - The original global rects.sort(...) assumed one total order was the whole rule. That no longer holds: now
#      db first defines the step/band, then dr orders within each band. The solver therefore groups by db and sorts
#      locally inside each group.
#    - The original seq.append(arr) reused rectangle colors. That assumption is invalid here because band identity
#      must be encoded in color; the solver now recolors non-bg pixels of each output rectangle to 1 or 2.
#    - The original out_h=sum(heights) assumed continuous stacking. The combo must reserve separator rows between
#      successive db-groups; the generator guarantees exactly two groups so one separator is always meaningful.
# d) Placement legality:
#    Generation keeps the original canonical geometry: two horizontal bands parallel to the blue axis, 2-cell gaps
#    between rectangles, first rectangle 1 cell from the red line, and a trailing margin of 1-2 cells. That ensures
#    no overlaps, no clipping, and exactly two recoverable db groups after any corner/orientation transform.
# e) Visual observability:
#    A human sees the same band structure twice in the output: first as stepwise stacking separated by a blank row,
#    second as a blue upper block vs red lower block. Across 2-4 examples, the consistent near/far split is easy to
#    infer because both bands repeat multiple times and the original rectangle colors disappear completely.
# f) Visual cleanliness:
#    Inputs remain sparse: one corner marker, two full edge lines, and non-touching rectangles on a uniform teal bg.
#    Outputs remain uncluttered because recoloring collapses rectangle interiors to a single color per band and the
#    only extra structure is one background separator row.
# g) Color-change inferability:
#    The output palette is intentionally limited to bg plus blue/red. Because input dominant colors exclude 1 and 2,
#    the recoloring is conspicuous. The spatial cue "near blue line vs farther from blue line" explains exactly why
#    one band becomes blue and the other red; the separator row reinforces the same partition instead of introducing
#    an unrelated color rule.
# h) New solver assumptions and generator guarantees:
#    The combo solver assumes at least two rectangles and expects band membership to be recoverable from db. For ARC
#    clarity it is best if there are exactly two non-empty db groups and each group has multiple rectangles. The
#    generator therefore constructs exactly two bands, each with 2-5 rectangles, and self-checks that combo output
#    differs from the base task and from each single-dimension variant.


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


def _along_from_corner(box: tuple[int, int, int, int], orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _extract_rects(grid: np.ndarray) -> tuple[int, str, str, int, int, list[dict]]:
    g = np.array(grid, dtype=int)
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
        arr = crop_bbox(g, box)
        if bool((arr == int(bg)).any()):
            continue
        rects.append({"box": box, "arr": arr.copy()})

    assert rects

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    for rect in rects:
        box = rect["box"]
        rect["db"] = int(_dist_to_edge(box, blue_orient, blue_edge_idx))
        rect["dr"] = int(_dist_to_edge(box, red_orient, red_edge_idx))

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
    if area < 8 or np.random.rand() < 0.55:
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


def _solve(grid: np.ndarray, *, separate_bands: bool, recolor_bands: bool) -> np.ndarray:
    bg, blue_orient, red_orient, cr, cc, rects = _extract_rects(grid)

    bands: list[tuple[int, list[np.ndarray]]] = []
    for band_idx, db in enumerate(sorted({int(rect["db"]) for rect in rects})):
        group = [rect for rect in rects if int(rect["db"]) == int(db)]
        group.sort(key=lambda rect: _band_sort_key(rect, red_orient, blue_orient, cr, cc))

        target_color = BLUE if band_idx == 0 else RED
        arrs: list[np.ndarray] = []
        for rect in group:
            arr = rect["arr"]
            if blue_orient == "v":
                arr = np.rot90(arr, k=1)
            if recolor_bands:
                arr = np.where(arr != int(bg), int(target_color), int(bg))
            arrs.append(arr)
        bands.append((int(db), arrs))

    out_w = int(max(int(arr.shape[1]) for _db, band in bands for arr in band))
    separator_rows = max(0, len(bands) - 1) if separate_bands else 0
    out_h = int(sum(int(arr.shape[0]) for _db, band in bands for arr in band) + separator_rows)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for band_idx, (_db, band) in enumerate(bands):
        if separate_bands and band_idx > 0:
            out_r += 1
        for arr in band:
            h, w = map(int, arr.shape)
            out_c = int((out_w - w) // 2)
            paste(out, arr, int(out_r), int(out_c))
            out_r += int(h)

    return out


def _band_profile(grid: np.ndarray) -> list[int]:
    _bg, _bo, _ro, _cr, _cc, rects = _extract_rects(grid)
    counts: dict[int, int] = {}
    for rect in rects:
        db = int(rect["db"])
        counts[db] = counts.get(db, 0) + 1
    return [counts[db] for db in sorted(counts)]


def _has_internal_separator(out: np.ndarray, bg: int) -> bool:
    return any(bool((out[r, :] == int(bg)).all()) for r in range(1, int(out.shape[0]) - 1))


def _combo_is_visible(grid: np.ndarray) -> bool:
    profile = _band_profile(grid)
    if len(profile) != 2 or min(profile) < 2:
        return False

    combo = _solve(grid, separate_bands=True, recolor_bands=True)
    d10_only = _solve(grid, separate_bands=True, recolor_bands=False)
    d11_only = _solve(grid, separate_bands=False, recolor_bands=True)
    base = _solve(grid, separate_bands=False, recolor_bands=False)

    if np.array_equal(combo, base) or np.array_equal(combo, d10_only) or np.array_equal(combo, d11_only):
        return False
    if not _has_internal_separator(combo, BG):
        return False

    colors = set(map(int, combo.flatten()))
    return BLUE in colors and RED in colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack rectangles band-by-band and recolor each band by its line color.

    - Group rectangles by distance to the blue edge.
    - Process near band first, far band later.
    - Sort inside each band by distance to the red edge, with the same corner-based tie breaks.
    - Rotate each rectangle 90 CCW when the blue edge is vertical.
    - Recolor the first band blue and later band(s) red, then stack with one bg separator row between bands.
    """
    return _solve(grid, separate_bands=True, recolor_bands=True)


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    try:
        _ = main(grid)
    except Exception:
        return None

    if _combo_is_visible(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 combo variant")
