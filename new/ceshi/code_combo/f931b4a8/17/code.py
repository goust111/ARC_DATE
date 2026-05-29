# Combo variant of f931b4a8 (index 17)
# combo_spec: D4/2+D11/2
# Name: 遮罩双色填充+嵌套框架模板

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - The original task uses the top half only as two counters, then combines a coarse
#      lower-left color source with a lower-right overlay stencil.
#    - In this combo, the counting controls stay unchanged, but the bottom half changes role:
#      the lower-left quadrant must be a binary color source, and the lower-right quadrant must
#      be a nested rectangular mask rather than a literal paint stencil.
#    - Small Q=8 trace:
#        fill colors = {2, 7}; coarse cell (0,1) = 2, so that macro-block uses primary=2,
#        secondary=7.
#        stencil has an outer rectangle frame at rows/cols 1..6 and a centered 2x2 core at
#        rows/cols 3..4.
#        Old rule:
#          outer frame pixels would keep stencil colors, zeros would use only one coarse color.
#        New rule:
#          pixels outside the outer frame use 7,
#          pixels inside the outer frame but outside the center core use 2,
#          pixels in the center core flip back to 7.
#      The result is a repeated three-zone bullseye, and if another macro-block picks primary=7
#      then the same geometry is color-inverted there.
#
# b) Complexity calibration:
#    - Q can be at most 8 in the original family, so the nested template cannot support many
#      visually distinct loops. The generator therefore locks onto the cleanest realizable
#      structure: one inset frame plus one centered inner rectangle, which still yields three
#      alternating zones.
#    - The output dimensions are forced to use at least two coarse macro-blocks so the solver
#      must notice both the nested mask geometry and the lower-left coarse selector.
#
# c) Old-code assumptions that break:
#    - `stencil[r % q, c % q]` no longer contributes output colors directly; stencil colors are
#      structural only.
#    - Treating all stencil background cells identically is no longer valid: different black
#      regions can sit at different nesting depths and therefore map to different source colors.
#    - Random sparse foreground in the lower-right quadrant is no longer legal because the solver
#      now assumes nested rectangular components that can be ordered by containment.
#
# d) Placement legality:
#    - The generator uses only N=16 so Q=8, which is the smallest original size that can show
#      an exterior band, a middle band, and a central nested region at the same time.
#    - The outer frame is inset from the tile border, and the inner core is separated from it by
#      at least one black ring; all quadrants stay disjoint because the full task layout is still
#      the original four-quadrant split.
#
# e) Visual observability:
#    - The lower-right tile visibly looks like a nested frame template rather than a random mask.
#    - The output uses only the two lower-left colors, so a human can see that the right-lower
#      colors are not copied, only the layered geometry is.
#    - The central core flips the color parity again, which distinguishes the combo from the
#      single-boundary D11 behavior after only a few examples.
#
# f) Visual cleanliness:
#    - The output has only two colors and three broad concentric zones per repeated tile.
#      This keeps the pattern legible instead of noisy.
#    - The lower-left tile is also binary and uses simple repeated motifs, so the two source
#      colors are easy to read off visually.
#
# g) Change inferability:
#    - No hidden ordering by numeric color id is used. For each macro-block, the lower-left
#      coarse cell chooses one of the two visible source colors as the odd-depth color, and the
#      other source color automatically becomes the even-depth color.
#    - Because the output spans multiple macro-blocks and the nested template is fixed across
#      them, the observer can infer that geometry comes from the template while color polarity
#      comes from the source tile.
#
# h) Generator guarantees:
#    - lower-left contains exactly two non-black colors
#    - the used coarse coordinates contain both colors
#    - lower-right contains exactly two nested rectangular layers: one outer frame and one inner
#      core, with at least one black gap between them
#    - the combo output differs from both the original overlay rule and a reduced single-frame
#      D11 interpretation
#
# i) New implicit assumptions audit:
#    - `_extract_rect_layers` assumes each non-black component is either a one-cell-thick
#      rectangle frame or a solid rectangle. The generator constructs exactly those shapes.
#    - `_compose_nested_mask_output` assumes the lower-left tile is binary and fully non-black.
#      The generator enforces both.
#    - The coarse lookup still indexes `fill_tile[r // q, c // q]`, so output sizes are kept in
#      the original 3..16 family and the generator ensures the accessed coarse indices stay in
#      range automatically.


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (16,)
STENCIL_COLOR_COUNT = (1, 2)
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 8
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 300


def _palette(k: int, pool=COLORS) -> list[int]:
    """Return k distinct colors sampled from pool."""
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    """Fill an array by drawing each cell independently from palette."""
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


def _ordered_coords(q: int, mode: str) -> list[tuple[int, int]]:
    """Return all (r,c) in a qxq grid in a layout-dependent order."""
    coords = [(r, c) for r in range(q) for c in range(q)]
    if mode == "rect":
        r0, c0 = random.randint(0, q - 1), random.randint(0, q - 1)
        return [((r + r0) % q, (c + c0) % q) for r, c in coords]
    if mode == "rows":
        rows, cols = list(range(q)), list(range(q))
        random.shuffle(rows)
        random.shuffle(cols)
        return [(r, c) for r in rows for c in cols]
    if mode == "compact":
        ar, ac = random.randrange(q), random.randrange(q)
        random.shuffle(coords)
        return sorted(coords, key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()))
    random.shuffle(coords)
    return coords


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control quadrant with exactly `count` non-black pixels."""
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(2, min(max_colors, count, len(pool)))
    pal = _palette(k, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    factors = [(hh, ww) for hh in range(1, q + 1) for ww in range(1, q + 1) if hh * ww == count]
    if factors and random.random() < CONTROL_RECT_PROB:
        hh, ww = random.choice(factors)
        r0 = random.randint(0, q - hh)
        c0 = random.randint(0, q - ww)
        tile[r0 : r0 + hh, c0 : c0 + ww] = _random_color_array((hh, ww), pal)
    else:
        mode = random.choice(["rect", "compact", "rows"])
        coords = _ordered_coords(q, mode)[:count]
        for r, c in coords:
            tile[r, c] = random.choice(pal)
    return tile


def _make_lower_left(q: int, palette: list[int], required_coords: list[tuple[int, int]]) -> np.ndarray:
    """Build the two-color source tile."""
    c0, c1 = palette
    tile = np.empty((q, q), dtype=int)
    mode = random.choice(["rows", "cols", "checker", "diagonal", "quadrants"])

    if mode == "rows":
        for r in range(q):
            tile[r, :] = c0 if r % 2 == 0 else c1
    elif mode == "cols":
        for c in range(q):
            tile[:, c] = c0 if c % 2 == 0 else c1
    elif mode == "checker":
        rr, cc = np.mgrid[:q, :q]
        tile[:] = np.where((rr + cc) % 2 == 0, c0, c1)
    elif mode == "diagonal":
        rr, cc = np.mgrid[:q, :q]
        tile[:] = np.where((rr + 2 * cc) % 3 == 0, c0, c1)
    else:
        tile[: q // 2, : q // 2] = c0
        tile[: q // 2, q // 2 :] = c1
        tile[q // 2 :, : q // 2] = c1
        tile[q // 2 :, q // 2 :] = c0

    used_colors = {int(tile[r, c]) for r, c in required_coords}
    missing = [color for color in palette if color not in used_colors]
    if missing:
        fix_coords = list(required_coords)
        random.shuffle(fix_coords)
        for color, (r, c) in zip(missing, fix_coords):
            tile[r, c] = color
    return tile


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components of a boolean mask."""
    q = mask.shape[0]
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(q):
        for c in range(q):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < q and 0 <= nc < q and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _rect_perimeter(top: int, bottom: int, left: int, right: int) -> set[tuple[int, int]]:
    """Return the one-cell-thick perimeter of an axis-aligned rectangle."""
    coords: set[tuple[int, int]] = set()
    for c in range(left, right + 1):
        coords.add((top, c))
        coords.add((bottom, c))
    for r in range(top + 1, bottom):
        coords.add((r, left))
        coords.add((r, right))
    return coords


def _extract_rect_layers(stencil: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    """
    Parse the nested rectangular mask into an outer-to-inner sequence of rectangles.

    Each foreground component must be either a one-cell-thick rectangular frame or a solid
    rectangular core. The returned tuples are (top, bottom, left, right).
    """
    fg = stencil != bg
    if not np.any(fg):
        return []

    q = stencil.shape[0]
    layers: list[tuple[int, int, int, int]] = []
    for comp in _connected_components(fg):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        top, bottom = min(rows), max(rows)
        left, right = min(cols), max(cols)
        comp_set = set(comp)
        full_rect = {(r, c) for r in range(top, bottom + 1) for c in range(left, right + 1)}
        if comp_set == full_rect:
            layers.append((top, bottom, left, right))
            continue
        assert bottom - top >= 1 and right - left >= 1
        assert comp_set == _rect_perimeter(top, bottom, left, right)
        layers.append((top, bottom, left, right))

    layers.sort(key=lambda rect: ((rect[1] - rect[0] + 1) * (rect[3] - rect[2] + 1), -rect[0], -rect[2]), reverse=True)
    outer = layers[0]
    assert outer[0] > 0 and outer[2] > 0 and outer[1] < q - 1 and outer[3] < q - 1
    for prev, cur in zip(layers, layers[1:]):
        assert cur[0] >= prev[0] + 2
        assert cur[2] >= prev[2] + 2
        assert cur[1] <= prev[1] - 2
        assert cur[3] <= prev[3] - 2
    return layers


def _mask_depths(stencil: np.ndarray, bg: int) -> np.ndarray | None:
    """Return per-cell nesting depth: how many rectangular layers contain each cell."""
    layers = _extract_rect_layers(stencil, bg)
    if not layers:
        return None
    q = stencil.shape[0]
    rr, cc = np.mgrid[:q, :q]
    depths = np.zeros((q, q), dtype=int)
    for top, bottom, left, right in layers:
        inside = (top <= rr) & (rr <= bottom) & (left <= cc) & (cc <= right)
        depths[inside] += 1
    return depths


def _fill_colors(fill_tile: np.ndarray, bg: int) -> tuple[int, int]:
    """Extract the two lower-left source colors."""
    colors = [int(v) for v in np.unique(fill_tile)]
    assert bg not in colors
    assert len(colors) == 2
    return colors[0], colors[1]


def _compose_nested_mask_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Compose the output under the nested-frame parity mask rule."""
    bg = int(Color.BLACK)
    q = fill_tile.shape[0]
    depths = _mask_depths(stencil, bg)
    if depths is None:
        rr, cc = np.mgrid[:out_h, :out_w]
        return fill_tile[rr % q, cc % q]

    color0, color1 = _fill_colors(fill_tile, bg)
    rr, cc = np.mgrid[:out_h, :out_w]
    coarse = fill_tile[rr // q, cc // q]
    secondary = np.where(coarse == color0, color1, color0)
    odd_depth = depths[rr % q, cc % q] % 2 == 1
    return np.where(odd_depth, coarse, secondary).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the upper control quadrants to size an output whose colors come from a two-color source and whose repeated geometry comes from a nested frame mask.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants are control regions.
    - The lower-left quadrant is a two-color source tile, while the lower-right quadrant is a nested rectangular template made of non-black frame layers on black.

    Rule:
    - Output height is the number of non-background pixels in the upper-left quadrant.
    - Output width is the number of non-background pixels in the upper-right quadrant.
    - If the lower-right tile is empty, repeat the lower-left tile directly over the output.
    - Otherwise, treat each non-black rectangular layer in the lower-right tile as a mask boundary and count, for every repeated fine-grid position, how many nested layers contain it; odd-depth cells take the lower-left coarse color for that macro-block, and even-depth cells take the other lower-left color.

    Key insight:
    - The top half still determines only the output size, but the bottom half now splits cleanly into geometry and color polarity: the nested right-lower template defines alternating concentric zones, and the left-lower coarse cell decides which of the two source colors occupies the odd-depth zones in each repeated block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    bg = int(Color.BLACK)
    qh, qw = h // 2, w // 2
    assert qh == qw
    upper_left = g[:qh, :qw]
    upper_right = g[:qh, qw:]
    fill_tile = g[qh:, :qw]
    stencil = g[qh:, qw:]

    out_h = int(np.count_nonzero(upper_left != bg))
    out_w = int(np.count_nonzero(upper_right != bg))
    assert out_h > 0 and out_w > 0

    return _compose_nested_mask_output(fill_tile, stencil, out_h, out_w)


def _paint_coords(tile: np.ndarray, coords: list[tuple[int, int]], palette: list[int]) -> None:
    """Paint coordinates with one or two visible stencil colors."""
    if len(palette) == 1:
        for r, c in coords:
            tile[r, c] = palette[0]
        return
    for i, (r, c) in enumerate(coords):
        tile[r, c] = palette[i % len(palette)]


def _make_nested_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build the fixed clean nested template: one inset frame plus one centered core."""
    assert q == 8
    tile = np.zeros((q, q), dtype=int)
    outer = sorted(_rect_perimeter(1, q - 2, 1, q - 2))
    inner = [(q // 2 - 1, q // 2 - 1), (q // 2 - 1, q // 2), (q // 2, q // 2 - 1), (q // 2, q // 2)]
    _paint_coords(tile, outer, palette)
    _paint_coords(tile, inner, palette[::-1] if len(palette) == 2 and random.random() < 0.5 else palette)
    return tile


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Reproduce the solver's combo tiling logic for oracle validation."""
    return _compose_nested_mask_output(fill_tile, stencil, out_h, out_w)


def _legacy_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Original overlay rule, used to reject degenerate samples."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _single_frame_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Reduced D11-style interpretation that ignores the inner nested layer."""
    q = stencil.shape[0]
    outer_only = np.zeros_like(stencil)
    outer_only[list(range(1, q - 1)), 1] = 1
    outer_only[list(range(1, q - 1)), q - 2] = 1
    outer_only[1, 1 : q - 1] = 1
    outer_only[q - 2, 1 : q - 1] = 1
    return _compose_nested_mask_output(fill_tile, outer_only, out_h, out_w)


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid combo input; return None on failure."""
    n = random.choice(GRID_SIZES)
    q = n // 2

    ll_palette = _palette(2)
    stencil_pool = [c for c in COLORS if c not in ll_palette]
    st_k = random.randint(*STENCIL_COLOR_COUNT)
    stencil_palette = _palette(st_k, stencil_pool)

    control_pool = [c for c in COLORS if c not in set(ll_palette) | set(stencil_palette)]
    if not control_pool:
        return None

    valid_dims = [
        (out_h, out_w)
        for out_h in range(OUTPUT_DIM_MIN, OUTPUT_DIM_MAX + 1)
        for out_w in range(OUTPUT_DIM_MIN, OUTPUT_DIM_MAX + 1)
        if out_h * out_w <= OUTPUT_AREA_MAX and ceil(out_h / q) * ceil(out_w / q) >= 2
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    used_fill = [(r, c) for r in range(ceil(out_h / q)) for c in range(ceil(out_w / q))]
    fill_tile = _make_lower_left(q, ll_palette, used_fill)
    if len({int(fill_tile[r, c]) for r, c in used_fill}) < 2:
        return None
    stencil = _make_nested_stencil(q, stencil_palette)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    expected = _expected_output(fill_tile, stencil, out_h, out_w)
    legacy = _legacy_output(fill_tile, stencil, out_h, out_w)
    single_frame = _single_frame_output(fill_tile, stencil, out_h, out_w)
    depths = _mask_depths(stencil, BLACK)
    if depths is None or int(depths.max()) < 2:
        return None
    if np.array_equal(expected, legacy) or np.array_equal(expected, single_frame):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
