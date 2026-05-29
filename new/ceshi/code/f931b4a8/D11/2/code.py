# Variant of f931b4a8
# Dimension: D11 — 模板与颜色源的组合逻辑 (transform)
# Change: 简单覆盖：模板非零像素保持不变，模板零像素被颜色源替换... → 模板作为遮罩，颜色源的两种颜色分别填充遮罩的两侧

from __future__ import annotations

from collections import deque
from math import ceil
import random

import numpy as np

from arc_utils import Color


# Design analysis for the D11 variant
# ----------------------------------
# a) Concrete behavior and trace:
#    - The 4-quadrant layout and the top-half counting controls stay unchanged.
#    - The lower-left quadrant is no longer an arbitrary all-nonblack texture. It is a
#      two-color source tile. For each output macro-block, the coarse coordinate
#      (r // Q, c // Q) chooses a primary color from this tile; the other lower-left color
#      becomes the secondary color for that same macro-block.
#    - The lower-right quadrant is no longer a color overlay stencil. Its non-black cells
#      are treated as a closed boundary. Background cells enclosed by that boundary are the
#      "inside"; background cells that can reach the tile border are the "outside".
#    - Small trace with Q=8:
#      lower-left colors = {2, 7}, coarse cell (0,1) = 2, so block primary=2 and
#      secondary=7. Suppose the lower-right tile is a hollow rectangle.
#      Old rule on one tiled 8x8 block:
#        boundary cells keep stencil color (say 4), interior holes use coarse color 2,
#        outer holes also use coarse color 2.
#      New rule on the same block:
#        boundary + enclosed interior -> 2, border-connected exterior -> 7.
#      So the lower-right colors disappear from the output; what matters is the region split.
#
# b) Complexity calibration:
#    - The variant must be visibly different from "replace zeros with a fill color".
#      Therefore the generator always uses a non-empty closed mask, never an empty stencil,
#      and lower-left always has exactly two colors.
#    - The two lower-left colors repeat across multiple macro-blocks so the solver must notice
#      both the inside/outside split and the coarse block lookup.
#
# c) Old-code assumptions that break:
#    - `value = stencil[r % q, c % q]` used to mean "template foreground survives".
#      That is false now: stencil colors are structural only.
#    - `if value == bg: value = fill_tile[r // q, c // q]` used to treat every zero cell the
#      same. That is false now: zero cells split into inside vs outside.
#    - The old generator allowed random sparse stencils; those do not define an inside/outside
#      partition, so the stencil builder must change to guaranteed closed boundaries.
#
# d) Placement legality:
#    - The closed boundary is generated strictly inside the QxQ stencil tile so each tiled copy
#      has its own local outside region and a non-empty enclosed inside region.
#    - Output dimensions are kept in the original numeric range and also forced to use at least
#      two coarse macro-blocks, so both lower-left colors can appear in the output.
#
# e) Visual observability:
#    - The right-lower quadrant now looks like an outline or loop, and the output becomes a
#      repeated two-color region fill rather than a pasted multi-color stencil.
#    - Stencil colors are chosen distinct from lower-left colors, which makes it clear that the
#      stencil is not copied literally into the output.
#
# f) Visual cleanliness:
#    - Only two output colors are used, and the mask is a single clean closed outline.
#      This avoids the noisy look that random scatter stencils would create.
#
# g) Color inference:
#    - The lower-left tile always contains exactly two visible colors. For each macro-block,
#      one of them fills the boundary+inside side, and the other fills the outside side.
#      No hidden ordering by color id is used.
#
# h) Variant-specific generator guarantees for the new solver:
#    - lower-left has exactly two non-black colors
#    - lower-right has a non-empty boundary, no boundary pixel touches the tile edge,
#      and the enclosed inside region is non-empty
#    - the coarse coordinates actually used by the sampled output dimensions contain both
#      lower-left colors
#    - the new output is rejected if it accidentally matches the legacy overlay rule
#
# i) New implicit assumptions audit:
#    - `_classify_stencil_regions` assumes a border-connected outside region exists within the
#      base stencil tile; the generator guarantees that by keeping the boundary inset.
#    - `_compose_mask_output` assumes each used coarse coordinate maps to one of the two
#      lower-left colors; the generator guarantees lower-left is fully non-black and binary.


def _border_connected_background(mask: np.ndarray) -> np.ndarray:
    """Return background cells reachable from the tile border via 4-connectivity."""
    q = mask.shape[0]
    seen = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for r in range(q):
        for c in (0, q - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(q):
        for r in (0, q - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < q and 0 <= nc < q and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return seen


def _classify_stencil_regions(stencil: np.ndarray, bg: int) -> np.ndarray | None:
    """
    Return a boolean QxQ mask where True means "boundary or enclosed interior".
    Return None when the stencil is empty.
    """
    boundary = stencil != bg
    if not np.any(boundary):
        return None

    q = stencil.shape[0]
    assert not np.any(boundary[0, :])
    assert not np.any(boundary[q - 1, :])
    assert not np.any(boundary[:, 0])
    assert not np.any(boundary[:, q - 1])

    background = ~boundary
    outside = _border_connected_background(background)
    inside = background & ~outside

    assert np.any(inside), "stencil must enclose an interior"
    assert np.any(outside), "stencil must leave an exterior"
    return boundary | inside


def _fill_colors(fill_tile: np.ndarray, bg: int) -> tuple[int, int]:
    """Extract the two lower-left source colors."""
    colors = [int(v) for v in np.unique(fill_tile)]
    assert bg not in colors
    assert len(colors) == 2
    return colors[0], colors[1]


def _compose_mask_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Compose the output under the D11 inside/outside mask rule."""
    bg = int(Color.BLACK)
    q = fill_tile.shape[0]
    side_mask = _classify_stencil_regions(stencil, bg)
    if side_mask is None:
        rr, cc = np.mgrid[:out_h, :out_w]
        return fill_tile[rr % q, cc % q]

    color0, color1 = _fill_colors(fill_tile, bg)
    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        coarse_r = r // q
        fine_r = r % q
        assert coarse_r < q
        for c in range(out_w):
            coarse_c = c // q
            fine_c = c % q
            assert coarse_c < q
            primary = int(fill_tile[coarse_r, coarse_c])
            secondary = color1 if primary == color0 else color0
            out[r, c] = primary if side_mask[fine_r, fine_c] else secondary
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the upper control quadrants to size an output whose colors come from a two-color source and whose geometry comes from a boundary mask.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants are control regions.
    - The lower-left quadrant is a two-color source tile, while the lower-right quadrant is a boundary mask tile.

    Rule:
    - Output height is the number of non-background pixels in the upper-left quadrant.
    - Output width is the number of non-background pixels in the upper-right quadrant.
    - If the lower-right tile is empty, repeat the lower-left tile directly over the output.
    - Otherwise, classify each repeated lower-right tile into border-connected exterior versus boundary/enclosed interior; for each output macro-block, use the corresponding lower-left coarse cell as the interior color and the other lower-left color as the exterior color.

    Key insight:
    - The top half still controls only the output size, but the bottom half now separates roles: the lower-left tile supplies a two-color pair plus coarse block selection, and the lower-right tile supplies only the inside/outside partition.
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

    return _compose_mask_output(fill_tile, stencil, out_h, out_w)


# --------------- generator (D11 variant) ---------------

BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (16,)                    # Q=8 is the smallest original size that can show a clean closed mask with inside+outside
STENCIL_COLOR_COUNT = (1, 2)          # mostly monochrome boundary; sometimes two colors
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
    return np.array([random.choice(palette) for _ in range(int(np.prod(shape)))],
                    dtype=int).reshape(shape)


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
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else \
        random.randint(2, min(max_colors, count, len(pool)))
    pal = _palette(k, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    factors = [(hh, ww) for hh in range(1, q + 1) for ww in range(1, q + 1) if hh * ww == count]
    if factors and random.random() < CONTROL_RECT_PROB:
        hh, ww = random.choice(factors)
        r0 = random.randint(0, q - hh)
        c0 = random.randint(0, q - ww)
        tile[r0:r0 + hh, c0:c0 + ww] = _random_color_array((hh, ww), pal)
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
        tile[:] = np.where((rr + cc) % 3 == 0, c0, c1)
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


def _paint_boundary(tile: np.ndarray, coords: list[tuple[int, int]], palette: list[int]) -> None:
    """Paint the boundary coordinates using one or two visible stencil colors."""
    if len(palette) == 1:
        for r, c in coords:
            tile[r, c] = palette[0]
        return
    for i, (r, c) in enumerate(coords):
        tile[r, c] = palette[i % len(palette)]


def _rect_boundary(q: int) -> list[tuple[int, int]]:
    top = random.randint(1, q // 3)
    left = random.randint(1, q // 3)
    bottom = random.randint(q - q // 3 - 1, q - 2)
    right = random.randint(q - q // 3 - 1, q - 2)
    if bottom - top < 3:
        bottom = min(q - 2, top + 3)
    if right - left < 3:
        right = min(q - 2, left + 3)
    coords: list[tuple[int, int]] = []
    for c in range(left, right + 1):
        coords.append((top, c))
        coords.append((bottom, c))
    for r in range(top + 1, bottom):
        coords.append((r, left))
        coords.append((r, right))
    return coords


def _diamond_boundary(q: int) -> list[tuple[int, int]]:
    center = q // 2 - 1
    radius = random.choice([2, 3])
    if center - radius < 1:
        radius = 2
    coords = []
    for r in range(q):
        for c in range(q):
            if abs(r - center) + abs(c - center) == radius:
                coords.append((r, c))
    return coords


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build a closed boundary mask that leaves both exterior and enclosed interior."""
    tile = np.zeros((q, q), dtype=int)
    mode = random.choice(["rect", "rect", "diamond"])
    coords = _rect_boundary(q) if mode == "rect" else _diamond_boundary(q)
    _paint_boundary(tile, coords, palette)
    mask = _classify_stencil_regions(tile, BLACK)
    assert mask is not None
    return tile


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Reproduce the solver's D11 tiling logic for oracle validation."""
    return _compose_mask_output(fill_tile, stencil, out_h, out_w)


def _legacy_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Original overlay rule, used only to reject degenerate samples."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid D11 variant input; return None on failure."""
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
    stencil = _make_stencil(q, stencil_palette)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    expected = _expected_output(fill_tile, stencil, out_h, out_w)
    legacy = _legacy_output(fill_tile, stencil, out_h, out_w)
    if np.array_equal(expected, legacy):
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
