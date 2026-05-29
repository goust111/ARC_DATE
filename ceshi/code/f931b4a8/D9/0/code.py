# Variant of f931b4a8
# Dimension: D9 — 前景/背景的判定 (match_cond)
# Change: 零值（颜色0）为背景，非零值为前景；控制象限中非零像素被计数，模板中零值位置被颜色源替换... → 最频繁出现的颜色为背景

from __future__ import annotations

from collections import Counter
from math import ceil
import random

import numpy as np


# Design analysis for the D9 variant
# ----------------------------------
# spec:
#   grid:
#     even square NxN, but N=4 is removed because a 2x2 control quadrant cannot
#     simultaneously have a unique modal background and still encode output size 3.
#     We therefore use N in {8, 16}, so each quadrant has Q in {4, 8}.
#   upper_left / upper_right:
#     each control quadrant is first filled with one background color, and the
#     background is defined as the unique most frequent color inside that quadrant.
#     Output height/width are the counts of cells that differ from that local mode.
#   lower_left:
#     still acts as the coarse color source tile. Unlike the original code, it is
#     allowed to contain color 0 because 0 is no longer privileged.
#   lower_right:
#     the stencil also has its own local modal background color. Cells equal to the
#     stencil mode are holes; all other cells are copied directly to the output.
#     An "empty" stencil is now a uniform tile of its modal background color.
#   output:
#     if the stencil is uniform background, tile the lower-left quadrant directly.
#     otherwise tile the stencil and replace stencil-background cells with the
#     lower-left color at coarse coordinate (r // Q, c // Q), exactly as before.
#
# a) Concrete behavior and trace:
#   Example with Q=4:
#   - UL background is color 7, with five cells recolored to 3/8/9, so out_h=5.
#   - UR background is color 4, with six non-4 cells, so out_w=6.
#   - Stencil background is color 2. When a tiled stencil cell equals 2, we pull a
#     coarse fill color from LL; when it is 5 or 0, that exact color survives.
#   Original logic would have counted "!= 0" in UL/UR and treated only 0 in the
#   stencil as holes. The variant instead counts "!= local mode" and treats only
#   the stencil's own dominant color as a hole. The structural change is therefore
#   visible in both sizing and fill decisions.
#
# b) Complexity calibration:
#   The rule stays single-mechanism: one modal-background test per relevant
#   quadrant. To avoid over-simplifying back into the original task, control and
#   stencil backgrounds are usually non-zero. To avoid excessive noise, the number
#   of control foreground colors is capped, and stencil foreground density is kept
#   below half so the background remains visually dominant.
#
# c) Implicit-assumption audit of the original main():
#   - `bg = int(Color.BLACK)` assumed one fixed background for all quadrants.
#     This is false now and must become per-quadrant mode detection.
#   - `count_nonzero(upper_left != bg)` / same for UR assumed foreground means
#     "non-zero". This now becomes "different from the quadrant mode".
#   - `if not np.any(stencil != bg)` assumed the empty stencil is all zeros.
#     This now becomes "stencil is uniform in its own background color".
#   - `if value == bg:` assumed stencil holes are exactly zero-valued.
#     This now becomes comparison against the stencil's mode color.
#   - `assert value != bg` on lower-left values was harmless before because LL had
#     no black. It would be wrong now because LL may legitimately contain 0, so the
#     assertion must be removed.
#
# d) Placement legality:
#   All quadrants still occupy the same fixed 2x2 partition, so there is no object
#   overlap issue. Generator-side legality reduces to guaranteeing that each control
#   quadrant has a unique modal color, and that every coarse LL coordinate used by
#   the solver lies inside the QxQ lower-left tile. The output-size bounds and the
#   `ceil(out_h/Q) * ceil(out_w/Q) >= palette_size` check preserve that.
#
# e) Visual observability:
#   The generator biases control/stencil backgrounds away from 0, so the new rule
#   is visible as "the large flat color is ignored even when it is not black".
#   Some LL or stencil-foreground cells may still use 0, which further shows that
#   zero is no longer a universal background marker.
#
# f) Visual cleanliness:
#   Each quadrant still has one clear role and no extra overlays. Control regions
#   remain dense monochrome backgrounds with a small number of accent cells; the
#   stencil remains a sparse repeated motif. This keeps the four-way partition easy
#   to read.
#
# g) Color-change inferability:
#   There is no hidden color remapping. Colors are copied literally; only the
#   classification of "background" changes, and it is always supported by the local
#   frequency structure of the quadrant itself.


ALL_COLORS = tuple(range(10))
NONZERO_COLORS = tuple(range(1, 10))

GRID_SIZES = (8, 16)
STENCIL_EMPTY_PROB = 0.16
LL_SOLID_PROB = 0.70
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.75
STENCIL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_FG_FRAC = (0.20, 0.45)
STENCIL_FRAME_MIN_Q = 8
CONTROL_SINGLE_COLOR_PROB = 0.65
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 4
UR_MAX_COLORS = 4
OUTPUT_DIM_MIN = 4
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 400


def _unique_mode(arr: np.ndarray) -> int:
    counts = Counter(map(int, arr.flatten()))
    ranked = counts.most_common()
    assert ranked
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1]
    return int(ranked[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each quadrant's dominant color as its background and use the lower pair to build the output.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - The two upper quadrants are control regions whose own most frequent colors act as local backgrounds.
    - The lower-left quadrant is a coarse color tile.
    - The lower-right quadrant is a stencil whose own most frequent color marks holes, or it may be uniformly background.

    Rule:
    - Output height is the number of cells in the upper-left quadrant that differ from the upper-left modal color.
    - Output width is the number of cells in the upper-right quadrant that differ from the upper-right modal color.
    - If the lower-right quadrant is uniform in its modal color, tile the lower-left quadrant directly over the output.
    - Otherwise tile the stencil; whenever a tiled stencil cell equals the stencil modal color, replace it with the lower-left cell indexed by the coarse tile position.

    Key insight:
    - The task no longer treats color 0 as globally special: each relevant quadrant defines its own background by frequency, and the rest of the construction is the same tiling scheme as the original puzzle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    qh, qw = h // 2, w // 2
    upper_left = g[:qh, :qw]
    upper_right = g[:qh, qw:]
    fill_tile = g[qh:, :qw]
    stencil = g[qh:, qw:]

    ul_bg = _unique_mode(upper_left)
    ur_bg = _unique_mode(upper_right)
    stencil_bg = _unique_mode(stencil)

    out_h = int(np.count_nonzero(upper_left != ul_bg))
    out_w = int(np.count_nonzero(upper_right != ur_bg))
    assert out_h > 0 and out_w > 0

    out = np.empty((out_h, out_w), dtype=int)
    if np.all(stencil == stencil_bg):
        for r in range(out_h):
            for c in range(out_w):
                out[r, c] = int(fill_tile[r % qh, c % qw])
        return out

    for r in range(out_h):
        for c in range(out_w):
            value = int(stencil[r % qh, c % qw])
            if value == stencil_bg:
                fill_r = r // qh
                fill_c = c // qw
                assert fill_r < qh and fill_c < qw
                value = int(fill_tile[fill_r, fill_c])
            out[r, c] = value

    return out


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
    hi = max(lo, int(STENCIL_FG_FRAC[1] * area))
    hi = min(hi, area - 2)
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
        coords = _ordered_coords(q, mode if mode != "random" else random.choice(["rect", "rows", "compact"]))[:fg_count]

    for r, c in coords:
        tile[r, c] = random.choice(palette)
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


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    stencil_bg = _unique_mode(stencil)
    if np.all(stencil == stencil_bg):
        return fill_tile[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == stencil_bg
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k, ALL_COLORS)

    if stencil_empty:
        stencil_palette: list[int] = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(*STENCIL_MULTI_COLOR_RANGE)
        stencil_palette = _palette(st_k, ALL_COLORS)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in NONZERO_COLORS if c not in output_colors]
    if len(control_pool) < 2:
        return None

    valid_h = [
        h
        for h in range(max(OUTPUT_DIM_MIN, q), min(OUTPUT_DIM_MAX, q * q - 2) + 1)
        if _control_feasible(q, h, UL_MAX_COLORS, len(control_pool))
    ]
    valid_w = [
        w
        for w in range(max(OUTPUT_DIM_MIN, q), min(OUTPUT_DIM_MAX, q * q - 2) + 1)
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
    fill_tile = _make_lower_left(q, ll_palette, used_fill)
    stencil_bg = random.choice(control_pool)
    stencil = np.full((q, q), stencil_bg, dtype=int) if stencil_empty else _make_stencil(q, stencil_bg, stencil_palette)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    grid = np.empty((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    try:
        actual = main(grid)
    except Exception:
        return None
    expected = _expected_output(fill_tile, stencil, out_h, out_w)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
