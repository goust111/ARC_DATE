# Combo variant of f931b4a8 (index 16)
# combo_spec: D9/0+D11/1
# Name: 最频繁色为背景+颜色源覆盖前景

from __future__ import annotations

from collections import Counter
from math import ceil
import random

import numpy as np


# Design analysis for combo variant D9/0 + D11/1
# ------------------------------------------------
# spec adjustments relative to the base task:
# - The grid is still an even square split into four quadrants, but color 0 is no
#   longer globally special. The upper-left and upper-right quadrants each have
#   their own modal background color, and the lower-right stencil also has its own
#   modal background color.
# - The lower-left quadrant remains the coarse color source tile. Its colors are
#   copied literally and may include 0, because "background" is now determined by
#   frequency rather than by a fixed color id.
# - The combination rule is inverted relative to the source task: stencil
#   foreground cells (cells different from the stencil mode) are the ones replaced
#   by lower-left coarse colors; stencil background cells stay as the stencil
#   background.
# - A uniform stencil is no longer an instruction to tile the lower-left quadrant.
#   Under the inversion it means there is no foreground to cover, so the entire
#   output stays at the stencil background color.
#
# a) Concrete behavior and trace:
#   Take Q=4. Suppose UL is mostly color 7 with five accent cells of colors 2/3,
#   so output height is 5, not "count of non-zero cells". Suppose UR is mostly
#   color 4 with six accent cells, so output width is 6. Suppose the stencil is
#   mostly color 8, with a checker of colors 1 and 0 marking its foreground.
#   The original task would tile the stencil and only replace zeros; the D9-only
#   variant would replace stencil cells equal to 8; the fused rule instead keeps
#   every 8 in place and replaces every non-8 cell by the lower-left coarse color
#   for that output block. This changes both the sizing step and the source/target
#   roles inside the stencil.
#
# b) Complexity calibration:
#   The variant should be visibly different without becoming noisy. To keep the
#   "most frequent color = background" cue readable, control quadrants and stencil
#   use one dominant color plus a limited number of accent colors. To keep the
#   inversion readable, stencil foreground density is moderate rather than tiny, so
#   the lower-left colors clearly occupy the stencil motif instead of its holes.
#   Lower-left tiles are often multi-color so the coverage effect is observable.
#
# c) Implicit-assumption audit of the original main():
#   - `bg = int(Color.BLACK)` assumed one universal background. Broken by D9: the
#     solver must infer a local modal background for each relevant quadrant.
#   - `count_nonzero(upper_left != bg)` and the UR analogue assumed "foreground"
#     means non-zero. Broken by D9: it now means "different from that quadrant's
#     mode".
#   - `if not np.any(stencil != bg)` assumed an empty stencil is all zeros. Broken
#     by D9 and D11: emptiness now means "uniform in its own mode", and a uniform
#     stencil yields a uniform output because there is no foreground to cover.
#   - `if value == bg:` assumed lower-left colors fill stencil background holes.
#     Broken by D11/1: replacement now happens on stencil foreground cells
#     (`value != stencil_bg`), while stencil background survives.
#   - `assert value != bg` for lower-left cells is invalid now because the color
#     source may legitimately contain 0.
#
# d) Placement legality:
#   Quadrants still occupy the fixed 2x2 partition, so overlap is structurally
#   impossible. Generator legality reduces to:
#   - each control quadrant and stencil must have a unique modal color;
#   - output dimensions must be feasible under those modal-background counts;
#   - every coarse coordinate referenced by `r // Q, c // Q` must lie within the
#     QxQ lower-left tile;
#   - stencil foreground must exist often enough to make the inversion visible,
#     while uniform stencils remain legal as a special case.
#
# e) Visual observability:
#   The generator forces control backgrounds and stencil background to be non-zero,
#   so a human can see that the large flat non-zero color is being ignored for
#   sizing and preserved for stencil background. Stencil foreground colors never
#   survive into the output, while lower-left colors do; this makes the inversion
#   detectable from 2-4 examples. Some lower-left or stencil-foreground cells may
#   still be color 0, reinforcing that 0 is not the privileged background.
#
# f) Visual cleanliness:
#   Each quadrant keeps a single role and there is no overlapping clutter. The top
#   pair are dense control patches, the lower-left is a simple stripe/solid source
#   tile, and the lower-right is a sparse-to-moderate motif over a flat background.
#   The output therefore still reads as "template-shaped patches cut from a coarse
#   color source", just with the source copied onto foreground instead of holes.
#
# g) Change inferability:
#   There is no hidden color remapping. All copied colors come directly from the
#   lower-left quadrant, and the only semantic changes are frequency-based
#   background detection plus the foreground/background role reversal in the
#   stencil. The generator rejects samples where the fused solver would collapse to
#   the D9-only or D11-only behaviors, so both ingredients remain active.


ALL_COLORS = tuple(range(10))
NONZERO_COLORS = tuple(range(1, 10))

GRID_SIZES = (8, 16)
STENCIL_EMPTY_PROB = 0.12
LL_SOLID_PROB = 0.45
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.78
STENCIL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_FG_FRAC = (0.30, 0.48)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.70
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 4
UR_MAX_COLORS = 4
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 500


def _unique_mode(arr: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(arr).flatten()))
    ranked = counts.most_common()
    assert ranked
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1]
    return int(ranked[0][0])


def _palette(k: int, pool=ALL_COLORS) -> list[int]:
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


def _checker_coords(q: int) -> list[tuple[int, int]]:
    parity = random.randrange(2)
    coords = [(r, c) for r in range(q) for c in range(q) if (r + c) % 2 == parity]
    random.shuffle(coords)
    return coords


def _frame_coords(q: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(q):
        coords.append((0, c))
    for r in range(1, q - 1):
        coords.append((r, q - 1))
    if q > 1:
        for c in range(q - 1, -1, -1):
            coords.append((q - 1, c))
    for r in range(q - 2, 0, -1):
        coords.append((r, 0))
    return coords


def _split_count(total: int, k: int, max_part: int) -> list[int]:
    assert 1 <= k <= total
    assert max_part >= 1
    assert total <= k * max_part
    parts = [1] * k
    remaining = total - k
    while remaining > 0:
        choices = [i for i, part in enumerate(parts) if part < max_part]
        i = random.choice(choices)
        parts[i] += 1
        remaining -= 1
    random.shuffle(parts)
    return parts


def _control_feasible(q: int, count: int, max_colors: int, pool_size: int) -> bool:
    area = q * q
    bg_count = area - count
    if count <= 0 or bg_count <= 1 or pool_size <= 1:
        return False
    max_k = min(max_colors, count, pool_size - 1)
    return any(count <= k * (bg_count - 1) for k in range(1, max_k + 1))


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


def _make_stencil(q: int, bg_color: int, palette: list[int]) -> np.ndarray:
    area = q * q
    tile = np.full((q, q), bg_color, dtype=int)
    if not palette:
        return tile

    lo = max(1, int(ceil(STENCIL_FG_FRAC[0] * area)))
    hi = min(int(STENCIL_FG_FRAC[1] * area), (area - 1) // 2)
    hi = max(lo, hi)
    fg_count = random.randint(lo, hi)

    modes = ["compact", "rows", "rect", "checker", "random"]
    if q >= STENCIL_FRAME_MIN_Q:
        modes.append("frame")
    mode = random.choice(modes)

    if mode == "checker":
        coords = _checker_coords(q)[:fg_count]
    elif mode == "frame":
        frame = _frame_coords(q)
        random.shuffle(frame)
        coords = frame[:fg_count]
    else:
        coord_mode = mode if mode != "random" else random.choice(["rect", "rows", "compact"])
        coords = _ordered_coords(q, coord_mode)[:fg_count]

    max_k = min(len(palette), fg_count, STENCIL_MULTI_COLOR_RANGE[1])
    if max_k <= 1 or random.random() < STENCIL_SINGLE_COLOR_PROB:
        k = 1
    else:
        k = random.randint(STENCIL_MULTI_COLOR_RANGE[0], max_k)
    fg_palette = _palette(k, palette)
    bg_count = area - fg_count
    counts = _split_count(fg_count, k, bg_count - 1)

    paint: list[int] = []
    for color, n in zip(fg_palette, counts):
        paint.extend([color] * n)
    random.shuffle(paint)
    for (r, c), color in zip(coords, paint):
        tile[r, c] = color

    assert _unique_mode(tile) == bg_color
    assert np.any(tile != bg_color)
    return tile


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    area = q * q
    bg_count = area - count
    assert bg_count > 1

    bg = random.choice(pool)
    fg_pool = [c for c in pool if c != bg]
    feasible_k = [
        k
        for k in range(1, min(max_colors, count, len(fg_pool)) + 1)
        if count <= k * (bg_count - 1)
    ]
    assert feasible_k

    if 1 in feasible_k and random.random() < CONTROL_SINGLE_COLOR_PROB:
        k = 1
    else:
        k = random.choice(feasible_k)
    pal = _palette(k, fg_pool)
    counts = _split_count(count, k, bg_count - 1)

    tile = np.full((q, q), bg, dtype=int)
    factors = [(h, w) for h in range(1, q + 1) for w in range(1, q + 1) if h * w == count]
    if factors and random.random() < CONTROL_RECT_PROB:
        h, w = random.choice(factors)
        r0 = random.randint(0, q - h)
        c0 = random.randint(0, q - w)
        coords = [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]
    else:
        mode = random.choice(["rect", "compact", "rows"])
        coords = _ordered_coords(q, mode)[:count]

    paint: list[int] = []
    for color, n in zip(pal, counts):
        paint.extend([color] * n)
    random.shuffle(paint)
    for (r, c), color in zip(coords, paint):
        tile[r, c] = color

    assert _unique_mode(tile) == bg
    assert int(np.count_nonzero(tile != bg)) == count
    return tile


def _solve_variant(grid: np.ndarray, *, use_modal_bg: bool, fill_foreground: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    qh, qw = h // 2, w // 2
    upper_left = g[:qh, :qw]
    upper_right = g[:qh, qw:]
    fill_tile = g[qh:, :qw]
    stencil = g[qh:, qw:]

    if use_modal_bg:
        ul_bg = _unique_mode(upper_left)
        ur_bg = _unique_mode(upper_right)
        stencil_bg = _unique_mode(stencil)
    else:
        ul_bg = 0
        ur_bg = 0
        stencil_bg = 0

    out_h = int(np.count_nonzero(upper_left != ul_bg))
    out_w = int(np.count_nonzero(upper_right != ur_bg))
    assert out_h > 0 and out_w > 0

    rr, cc = np.mgrid[:out_h, :out_w]
    tiled = stencil[rr % qh, cc % qw].copy()

    if fill_foreground:
        if np.all(stencil == stencil_bg):
            return np.full((out_h, out_w), stencil_bg, dtype=int)
        fg_mask = tiled != stencil_bg
        tiled[fg_mask] = fill_tile[rr[fg_mask] // qh, cc[fg_mask] // qw]
        return tiled

    if np.all(stencil == stencil_bg):
        return fill_tile[rr % qh, cc % qw]
    bg_mask = tiled == stencil_bg
    tiled[bg_mask] = fill_tile[rr[bg_mask] // qh, cc[bg_mask] // qw]
    return tiled


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    stencil_bg = _unique_mode(stencil)
    if np.all(stencil == stencil_bg):
        return np.full((out_h, out_w), stencil_bg, dtype=int)
    tiled = stencil[rr % q, cc % q].copy()
    fg_mask = tiled != stencil_bg
    tiled[fg_mask] = fill_tile[rr[fg_mask] // q, cc[fg_mask] // q]
    return tiled


def _behavior_signature(grid: np.ndarray) -> dict[str, int]:
    combo = main(grid)
    base = _solve_variant(grid, use_modal_bg=False, fill_foreground=False)
    d9_only = _solve_variant(grid, use_modal_bg=True, fill_foreground=False)
    d11_only = _solve_variant(grid, use_modal_bg=False, fill_foreground=True)
    return {
        "same_as_base": int(np.array_equal(combo, base)),
        "same_as_d9_only": int(np.array_equal(combo, d9_only)),
        "same_as_d11_only": int(np.array_equal(combo, d11_only)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each quadrant's modal background, then stamp coarse source colors onto stencil foreground.

    Input structure:
    - The input is an even square divided into four equal quadrants.
    - The upper-left and upper-right quadrants are control panels whose own most frequent colors serve as local backgrounds.
    - The lower-left quadrant is a coarse color source tile.
    - The lower-right quadrant is a stencil whose own most frequent color is its background; other cells mark foreground positions to be covered.

    Rule:
    - Output height is the number of upper-left cells that differ from the upper-left modal color.
    - Output width is the number of upper-right cells that differ from the upper-right modal color.
    - Tile the lower-right stencil over that output size.
    - Keep stencil background cells unchanged, but replace every tiled stencil foreground cell with the lower-left color indexed by the coarse output block `(r // Q, c // Q)`.
    - If the stencil is uniform in its modal color, the whole output stays at that background color.

    Key insight:
    - Two things changed together: background must be discovered by frequency instead of assuming color 0, and the lower-left tile now paints the stencil's foreground motif rather than its holes.
    """
    return _solve_variant(grid, use_modal_bg=True, fill_foreground=True)


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    fill_palette = _palette(ll_k, ALL_COLORS)

    stencil_bg_pool = [c for c in NONZERO_COLORS if c not in fill_palette]
    if not stencil_bg_pool:
        return None
    stencil_bg = random.choice(stencil_bg_pool)

    stencil_fg_pool = [c for c in ALL_COLORS if c not in fill_palette and c != stencil_bg]
    if not stencil_empty and not stencil_fg_pool:
        return None
    if stencil_empty:
        stencil_palette: list[int] = []
    else:
        max_st_k = min(STENCIL_MULTI_COLOR_RANGE[1], len(stencil_fg_pool))
        if max_st_k <= 1 or random.random() < STENCIL_SINGLE_COLOR_PROB:
            st_k = 1
        else:
            st_k = random.randint(STENCIL_MULTI_COLOR_RANGE[0], max_st_k)
        stencil_palette = _palette(st_k, stencil_fg_pool)

    output_colors = set(fill_palette) | {stencil_bg}
    control_pool = [c for c in NONZERO_COLORS if c not in output_colors]
    if len(control_pool) < 2:
        return None

    max_count = min(OUTPUT_DIM_MAX, q * q - 2)
    valid_h = [
        h
        for h in range(OUTPUT_DIM_MIN, max_count + 1)
        if _control_feasible(q, h, UL_MAX_COLORS, len(control_pool))
    ]
    valid_w = [
        w
        for w in range(OUTPUT_DIM_MIN, max_count + 1)
        if _control_feasible(q, w, UR_MAX_COLORS, len(control_pool))
    ]
    valid_dims = [
        (h, w)
        for h in valid_h
        for w in valid_w
        if h * w <= OUTPUT_AREA_MAX and (stencil_empty or ceil(h / q) * ceil(w / q) >= ll_k)
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    used_fill = [(r, c) for r in range(ceil(out_h / q)) for c in range(ceil(out_w / q))]
    fill_tile = _make_lower_left(q, fill_palette, None if stencil_empty else used_fill)
    stencil = (
        np.full((q, q), stencil_bg, dtype=int)
        if stencil_empty
        else _make_stencil(q, stencil_bg, stencil_palette)
    )
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    grid = np.empty((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    try:
        actual = main(grid)
        expected = _expected_output(fill_tile, stencil, out_h, out_w)
        if not np.array_equal(actual, expected):
            return None
        signature = _behavior_signature(grid)
        if any(signature.values()):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for combo f931b4a8 D9/0+D11/1")
