# Variant of f931b4a8
# Dimension: D9 — 前景/背景的判定 (match_cond)
# Change: 零值（颜色0）为背景，非零值为前景；控制象限中非零像素被计数，模板中零值位置被颜色源替换...
#   → 模板中两种颜色分别映射到模板结构色和颜色源色

from __future__ import annotations

from math import ceil
import random

import numpy as np

from arc_utils import Color


# Design analysis for the D9 variant
# ==================================
# a) Concrete behavior and trace
#    - The whole input is still an even square split into four equal quadrants.
#    - Upper-left and upper-right still only encode output height/width by counting non-black cells.
#    - Lower-left is still an all-non-black coarse color source tile.
#    - Lower-right is no longer a black-vs-foreground stencil. When it is used, it is filled with
#      exactly two non-black colors:
#        * the rarer color is the fixed template structure
#        * the more frequent color is the replaceable template "background"
#    - Example with q=4:
#        stencil =
#          7 7 7 7
#          7 2 2 7
#          7 2 2 7
#          7 7 7 7
#      Here 7 is more frequent than 2, so 7 is the replaceable marker and 2 is the kept structure.
#      Old logic on the same stencil would keep both 7 and 2 everywhere because neither is black.
#      New logic instead keeps only the 2-cells and fills the 7-cells from lower-left using
#      coarse coordinates (r // q, c // q).
# b) Complexity calibration
#    - The semantic template is constrained to exactly two colors so the new role split is visible
#      without introducing many ambiguous cases.
#    - The replaceable color is forced to be a strict majority, so the solver can distinguish roles
#      from a visible property of the template instead of a hidden color-id convention.
#    - Lower-left uses 1-3 colors with simple stripe/diagonal layouts so filled areas are legible.
# c) Old assumptions that must change
#    - `if value == bg:` used to mean "template hole". That is no longer valid because the template
#      holes are now non-black.
#    - `if not np.any(stencil != bg):` is still safe as an optional fallback branch for an all-black
#      lower-right quadrant, but the generator keeps the semantic two-color case dominant.
#    - The old code never had to identify two different foreground roles inside the stencil; the new
#      solver must inspect stencil color frequencies and treat the strict majority color as replaceable.
# d) Placement legality
#    - Quadrant layout is unchanged, so there are no new overlap risks.
#    - Output dimensions remain bounded by q*q, so coarse indices `r // q` and `c // q` stay within
#      the lower-left quadrant as before.
#    - The generator only emits non-empty semantic stencils with both colors present and unequal counts.
# e) Visual observability
#    - In each generated example, one stencil color survives into the output while the other disappears
#      and is replaced by lower-left colors. This is directly visible across 2-4 examples.
#    - Control colors are sampled disjointly from lower-half colors so the top quadrants cannot be
#      mistaken for output content.
# f) Visual cleanliness
#    - The lower-right template uses simple masks (inner box, single row/col, sparse scatter), so the
#      two semantic colors are readable even at q=2 or q=4.
#    - Lower-left stays fully non-black and patterned simply, avoiding noisy multi-layer overlaps.
# g) Color-change inferability
#    - Only one stencil color is ever replaced, and it is always the majority color in that quadrant.
#      This creates repeated evidence that the same visible role, not a numeric color id, controls the change.
# h) Variant-specific non-degeneracy target
#    - On generated inputs, applying the old solver to the same non-empty stencil would just tile both
#      stencil colors. The new solver must differ on every sample because the majority stencil color is
#      always present and always replaced by disjoint lower-left colors.
# i) New solver assumptions and generator guarantees
#    - Assumption: a non-empty semantic stencil has exactly two non-black colors and one strict majority.
#      Guarantee: `_make_semantic_stencil` enforces exactly that.
#    - Assumption: coarse lower-left lookup never hits black.
#      Guarantee: `_make_lower_left` fills every cell with non-black colors.
#    - Assumption: if lower-left uses multiple colors, enough coarse cells are used to expose them.
#      Guarantee: valid output dimensions require at least as many used coarse cells as lower-left colors.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the two upper control quadrants to size an output built from the lower quadrants.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is still the only true background for the control regions.
    - The upper-left and upper-right quadrants encode output height and width by foreground counts.
    - The lower-left quadrant is a coarse color source tile, while the lower-right quadrant is either empty or a two-color semantic template.

    Rule:
    - Output height is the number of non-black pixels in the upper-left quadrant.
    - Output width is the number of non-black pixels in the upper-right quadrant.
    - If the lower-right quadrant is all black, tile the lower-left quadrant directly over the output.
    - Otherwise, the lower-right quadrant contains exactly two non-black colors: the more frequent color marks replaceable positions, and the less frequent color is preserved as template structure. Tile that template over the output, replacing only the majority template color with lower-left values at coarse coordinates.

    Key insight:
    - The top half still supplies only dimensions, but the lower-right quadrant now distinguishes two different foreground roles instead of using black holes versus non-black structure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    bg = int(Color.BLACK)
    qh, qw = h // 2, w // 2
    upper_left = g[:qh, :qw]
    upper_right = g[:qh, qw:]
    fill_tile = g[qh:, :qw]
    stencil = g[qh:, qw:]

    out_h = int(np.count_nonzero(upper_left != bg))
    out_w = int(np.count_nonzero(upper_right != bg))
    assert out_h > 0 and out_w > 0
    assert np.all(fill_tile != bg)

    out = np.empty((out_h, out_w), dtype=int)
    nz = stencil[stencil != bg]
    if nz.size == 0:
        for r in range(out_h):
            for c in range(out_w):
                out[r, c] = int(fill_tile[r % qh, c % qw])
        return out

    colors, counts = np.unique(nz, return_counts=True)
    assert len(colors) == 2
    assert counts[0] != counts[1]
    replace_color = int(colors[int(np.argmax(counts))])

    for r in range(out_h):
        for c in range(out_w):
            value = int(stencil[r % qh, c % qw])
            assert value != bg
            if value == replace_color:
                fill_r = r // qh
                fill_c = c // qw
                assert fill_r < qh and fill_c < qw
                value = int(fill_tile[fill_r, fill_c])
                assert value != bg
            out[r, c] = value

    return out


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
STENCIL_EMPTY_PROB = 0.0
LL_SOLID_PROB = 0.45
LL_MULTI_COLOR_RANGE = (2, 3)
STRUCTURE_FRAC = (0.12, 0.40)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 400


def _palette(k: int, pool=COLORS) -> list[int]:
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


def _ordered_coords(q: int, mode: str) -> list[tuple[int, int]]:
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
        return sorted(
            coords,
            key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()),
        )
    random.shuffle(coords)
    return coords


def _make_lower_left(
    q: int,
    palette: list[int],
    required_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    k = len(palette)
    tile = np.empty((q, q), dtype=int)
    mode = "solid" if k == 1 else random.choice(["rows", "cols", "rows", "cols", "diagonal"])

    if mode == "solid":
        tile[:] = palette[0]
    elif mode == "rows":
        offset = random.randrange(k)
        for r in range(q):
            tile[r, :] = palette[(r + offset) % k]
    elif mode == "cols":
        offset = random.randrange(k)
        for c in range(q):
            tile[:, c] = palette[(c + offset) % k]
    else:
        offset = random.randrange(k)
        rr, cc = np.mgrid[:q, :q]
        idx = (rr + cc + offset) % k
        for i, color in enumerate(palette):
            tile[idx == i] = color

    existing = set(map(int, np.unique(tile)))
    missing = [c for c in palette if c not in existing]
    if missing:
        fix_coords = (
            list(required_coords)
            if required_coords is not None
            else [(r, c) for r in range(q) for c in range(q)]
        )
        random.shuffle(fix_coords)
        for color, (r, c) in zip(missing, fix_coords):
            tile[r, c] = color
    return tile


def _structure_mask(q: int) -> np.ndarray:
    area = q * q
    modes = ["dot", "random", "random", "inner_box"]
    if q >= 3:
        modes.extend(["single_row", "single_col"])
    mode = random.choice(modes)
    mask = np.zeros((q, q), dtype=bool)

    if q == 2:
        r, c = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
        mask[r, c] = True
        return mask

    if mode == "dot":
        count = random.randint(1, max(1, min(q, int(area * STRUCTURE_FRAC[1]))))
        coords = [(r, c) for r in range(q) for c in range(q)]
        random.shuffle(coords)
        for r, c in coords[:count]:
            mask[r, c] = True
    elif mode == "single_row":
        r = random.randrange(q)
        mask[r, :] = True
    elif mode == "single_col":
        c = random.randrange(q)
        mask[:, c] = True
    elif mode == "inner_box":
        h = random.randint(1, max(1, q // 2))
        w = random.randint(1, max(1, q // 2))
        r0 = random.randint(0, q - h)
        c0 = random.randint(0, q - w)
        mask[r0:r0 + h, c0:c0 + w] = True
    else:
        lo = max(1, int(np.floor(STRUCTURE_FRAC[0] * area)))
        hi = max(lo, min(area - 1, int(np.ceil(STRUCTURE_FRAC[1] * area))))
        count = random.randint(lo, hi)
        coords = [(r, c) for r in range(q) for c in range(q)]
        random.shuffle(coords)
        for r, c in coords[:count]:
            mask[r, c] = True

    count = int(mask.sum())
    if count == 0:
        mask[random.randrange(q), random.randrange(q)] = True
        count = 1
    if count * 2 >= area:
        coords = list(map(tuple, np.argwhere(mask)))
        random.shuffle(coords)
        keep = max(1, area // 2 - 1)
        new_mask = np.zeros_like(mask)
        for r, c in coords[:keep]:
            new_mask[r, c] = True
        mask = new_mask
    return mask


def _make_semantic_stencil(q: int, structure_color: int, replace_color: int) -> np.ndarray:
    tile = np.full((q, q), replace_color, dtype=int)
    mask = _structure_mask(q)
    tile[mask] = structure_color

    colors, counts = np.unique(tile, return_counts=True)
    assert len(colors) == 2
    assert counts.min() > 0
    assert counts[0] != counts[1]
    assert int(colors[int(np.argmax(counts))]) == replace_color
    return tile


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(
        2, min(max_colors, count, len(pool))
    )
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


def _expected_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    nz = stencil[stencil != BLACK]
    if nz.size == 0:
        return fill_tile[rr % q, cc % q]

    colors, counts = np.unique(nz, return_counts=True)
    assert len(colors) == 2 and counts[0] != counts[1]
    replace_color = int(colors[int(np.argmax(counts))])
    tiled = stencil[rr % q, cc % q].copy()
    replace_mask = tiled == replace_color
    tiled[replace_mask] = fill_tile[rr[replace_mask] // q, cc[replace_mask] // q]
    return tiled


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k)

    if stencil_empty:
        structure_color = replace_color = None
        stencil_colors: list[int] = []
    else:
        stencil_colors = _palette(2, [c for c in COLORS if c not in ll_palette])
        if len(stencil_colors) < 2:
            return None
        structure_color, replace_color = stencil_colors

    occupied = set(ll_palette) | set(stencil_colors)
    control_pool = [c for c in COLORS if c not in occupied]
    if not control_pool:
        return None

    lo = max(OUTPUT_DIM_MIN, q)
    hi = min(OUTPUT_DIM_MAX, q * q)
    valid_dims = [
        (out_h, out_w)
        for out_h in range(lo, hi + 1)
        for out_w in range(lo, hi + 1)
        if out_h * out_w <= OUTPUT_AREA_MAX
        and ceil(out_h / q) * ceil(out_w / q) >= ll_k
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    used_fill = [(r, c) for r in range(ceil(out_h / q)) for c in range(ceil(out_w / q))]
    fill_tile = _make_lower_left(q, ll_palette, used_fill)
    if stencil_empty:
        stencil = np.zeros((q, q), dtype=int)
    else:
        assert structure_color is not None and replace_color is not None
        stencil = _make_semantic_stencil(q, structure_color, replace_color)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    expected = _expected_output(fill_tile, stencil, out_h, out_w)
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
