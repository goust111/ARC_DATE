# Variant of 291dc1e1
# Dimension: D6 — 堆叠方式与对齐 (dst)
# Change: 矩形从上到下紧密堆叠，水平方向居中对齐，无间隙... → 重力堆叠到底部

from __future__ import annotations

import numpy as np
from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

# Design analysis for the D6 variant
# a) Concrete behavior:
#    The input is unchanged in spirit: a marked corner, one horizontal edge line,
#    one vertical edge line, and several filled interior rectangles. Rectangles are
#    still sorted by (distance to blue edge, distance to red edge), with a 90° CCW
#    rotation when the blue edge is vertical. The only change is the destination
#    stack direction. In the original solver, sorted rectangles are pasted from row 0
#    downward, so the first sorted rectangle ends up visually at the top. In this
#    variant, we mimic the gravity anchoring style seen in 62593bfd: start from the
#    bottom edge of the output canvas, place the first sorted rectangle there, then
#    keep stacking upward. Example with sorted sequence [A(2x4), B(2x2), C(2x6)]:
#    original top-stack gives rows top→bottom = A, B, C; variant bottom-stack gives
#    top→bottom = C, B, A. This makes the D6 change visible without adding new
#    object types or changing the ordering rule itself.
# b) Complexity calibration:
#    We keep the original 4-7 rectangles and keep the width variability / minority
#    pixels inside rectangles, so reversing stack direction is visually obvious.
#    No new color rule or multi-branch destination rule is introduced.
# c) Implicit assumption audit for main():
#    - Corner detection assumes exactly one corner marker differs from both adjacent
#      line colors; generator preserves that.
#    - Blue/red orientation inference assumes two full edge lines meeting at the
#      marker; generator preserves that.
#    - Rectangle extraction assumes every interior component has a bbox with no
#      background holes; generator still creates filled 2-row bands, optionally with
#      recolored cells but never with holes.
#    - Sorting keys depend only on rectangle bbox positions relative to the two axes;
#      unchanged by this variant.
#    - The old placement loop implicitly assumed row cursor starts at 0. That is the
#      exact D6-specific assumption we replace: now the cursor starts at outH and is
#      decremented before each paste.
# d) Placement legality:
#    Output rectangles never overlap because the cursor moves by each rectangle's
#    full height and total canvas height is exactly the sum of all heights. Centering
#    uses the same max-width canvas as before, so widths remain in bounds.
# e) Visual observability:
#    Human solvers can now detect that the nearest rectangle goes to the bottom,
#    not the top. Because generators produce multiple rectangles with differing widths
#    or internal minority patterns, top-stack and bottom-stack are usually distinct.
# f) Visual cleanliness:
#    The task still contains only edge axes plus separated filled rectangles. The
#    output is a single clean centered stack; no new overlays or marks are added.
# g) Color inference:
#    No new color mapping exists. Colors keep their original role as object identity
#    and interior asymmetry markers.
# 9) New solver assumptions introduced by the variant:
#    The bottom-stack implementation assumes the generator keeps at least two
#    rectangles often enough for the reversal to matter, and that the first and last
#    sorted rectangles are not always visually identical. The generator therefore
#    explicitly rejects inputs where bottom-stack output equals the legacy top-stack
#    output, preventing silent regression back to the original behavior.


def _solve(grid: np.ndarray, *, stack_from_bottom: bool) -> np.ndarray:
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

    outW = int(max(int(arr.shape[1]) for arr in seq))
    outH = int(sum(int(arr.shape[0]) for arr in seq))
    out = np.full((outH, outW), int(bg), dtype=int)

    if stack_from_bottom:
        out_r = int(outH)
        for arr in seq:
            h, w = map(int, arr.shape)
            out_r -= int(h)
            out_c = int((outW - w) // 2)
            paste(out, arr, int(out_r), int(out_c))
    else:
        out_r = 0
        for arr in seq:
            h, w = map(int, arr.shape)
            out_c = int((outW - w) // 2)
            paste(out, arr, int(out_r), int(out_c))
            out_r += int(h)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder rectangles by proximity to the blue/red edge axes, rotate them when the
    blue axis is vertical, then stack them from the bottom upward.
    """
    return _solve(grid, stack_from_bottom=True)


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
        for n0 in range(2, max_count + 1):
            for n1 in range(2, max_count + 1):
                if n0 + n1 > 9:
                    continue
                eligible = []
                if max_used <= 10 * n0 - 2:
                    eligible.append(0)
                if max_used <= 10 * n1 - 2:
                    eligible.append(1)
                if eligible:
                    pairs.append((n0, n1, eligible))
        if pairs:
            options.append((max_used, pairs))

    assert options
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
        for w in widths:
            color = int(np.random.choice(DOMINANT_COLORS))
            patch = np.full((2, w), color, dtype=int)
            patch = _sprinkle_minorities(patch, color)
            paste(grid, patch, r0, c)
            c += w + 2

    return grid


def _count_rectangles(grid: np.ndarray) -> int:
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
    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True
    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    count = 0
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        sub = crop_bbox(g, bbox_coords(coords))
        if bool((sub == int(bg)).any()):
            continue
        count += 1
    return int(count)


def _gen_attempt() -> np.ndarray:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    bottom_out = main(grid)
    top_out = _solve(grid, stack_from_bottom=False)
    assert _count_rectangles(grid) >= 4
    if np.array_equal(bottom_out, top_out):
        raise ValueError("degenerate sample: bottom-stack equals top-stack")
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (AssertionError, IndexError, ValueError):
            continue
    raise RuntimeError("failed to generate input for 291dc1e1 D6 variant")
