# Combo variant of f931b4a8 (index 15)
# combo_spec: D8/0+D9/0
# Name: 分隔线划分+最频繁色为背景

from __future__ import annotations

"""
Design analysis for the D8/0 + D9/0 combo variant.

a) concretization_and_trace:
  - The input is no longer split by the midpoint. A single solid divider row and a
    single solid divider column, both using the same reserved color, cut the grid
    into four rectangular regions. The cuts are allowed to be off-center.
  - The upper-left region still encodes output height and the upper-right region
    still encodes output width, but "foreground" is now defined relative to each
    region's own most frequent color instead of color 0.
  - The lower-left region is still the coarse fill tile. The lower-right region is
    still the stencil, but the stencil's holes are now the cells equal to that
    stencil region's own unique modal color.
  - Concrete trace:
    * Original: if the grid were 8x8, the code would use q=4, count `!= 0` in the
      two upper quadrants, and treat stencil value 0 as the only hole color.
    * Combo: suppose the divider row is r=4 and the divider column is c=6. Then the
      four regions have shapes 4x6, 4x5, 5x6, and 5x5. If the upper-left mode is 7
      and exactly five cells differ from 7, then out_h=5. If the stencil mode is 4,
      then every tiled stencil cell equal to 4 is replaced by `fill_tile[r // 5,
      c // 5]`, while all other stencil colors survive unchanged.
    * Difference: the solver must first locate the divider cross, then compute local
      modes, then count / substitute relative to those modes. Midpoint slicing and
      zero-background assumptions both fail.

b) complexity_calibration:
  - The partition cue is intentionally simple: exactly one full row and one full
    column of the same reserved color. This makes the D8 change visually explicit.
  - The background cue is also intentionally simple: each control region and the
    stencil region have one dominant flat color, usually non-zero, with a smaller
    set of accent cells. This makes the D9 change visible without adding extra
    object types.
  - Region side lengths are kept in a small range, and the lower patterns still use
    familiar stripe/checker/frame motifs, so the task remains inferable from a few
    examples rather than becoming a noisy search problem.

c) implicit_assumption_audit:
  - `h == w` and even midpoint splitting are no longer valid assumptions. The solver
    must detect the separator row/column directly.
  - Slices such as `g[:qh, :qw]` are invalid because the true boundaries are now
    data-dependent.
  - `bg = 0` is no longer valid for control counting or stencil-hole detection. The
    solver must compute a unique mode separately for upper-left, upper-right, and
    stencil.
  - `if not np.any(stencil != 0)` is no longer a correct emptiness test. The correct
    test is whether the stencil is uniform in its own modal color.
  - `assert value != 0` on lower-left colors is wrong in the combo variant because 0
    may legitimately appear inside the fill tile.
  - Coarse replacement still depends on `fill_r = r // stencil_h` and
    `fill_c = c // stencil_w`, so the generator must guarantee
    `ceil(out_h / stencil_h) <= fill_h` and `ceil(out_w / stencil_w) <= fill_w`.

d) placement_legality:
  - The generator samples top/bottom/left/right region sizes first, then inserts one
    divider row and one divider column between them. This guarantees four non-empty
    rectangular regions with no overlap.
  - Divider color is reserved and never reused in region content, so divider
    detection is unambiguous.
  - Control counts are filtered through a feasibility test that guarantees a unique
    modal background can coexist with the requested number of non-background cells.
  - When the stencil is non-uniform, output dimensions are filtered so every coarse
    fill index used by the solver stays inside the lower-left region.

e) visual_observability:
  - Every sample forces at least one off-center cut, so the frame-based partition is
    visible rather than collapsing back to equal quadrants.
  - Control and stencil backgrounds are sampled from non-zero colors, which makes it
    visually obvious that "largest color patch" matters more than "black means
    background".
  - Lower-left and stencil foregrounds may still contain color 0, which reinforces
    that zero is not globally special in this task family.

f) visual_cleanliness:
  - The divider is the only full-row/full-column overlay and uses one reserved color.
  - Region interiors use simple fills, stripes, diagonals, and sparse stencil motifs,
    so the four roles remain visually separable.
  - Control colors are excluded from the output palette, preserving the clue that the
    top half contributes counts rather than output colors.

g) color_inference_and_degeneration_plan:
  - There is no hidden color remapping: colors are copied literally. Only the
    background classification changes, and it is supported by visible frequency
    structure inside each relevant region.
  - Degeneration is checked by measuring both ingredients separately:
    * divider offsets from the geometric midpoint must be non-zero;
    * fused outputs must differ from a same-split solver that still treats 0 as the
      background.
    If either ingredient stopped affecting the generated samples, the combo would
    have collapsed toward a simpler task.

h) new_solver_assumptions_and_generator_guarantees:
  solver assumptions:
    - Exactly one solid divider row and one solid divider column share the same color.
    - That divider color is not reused elsewhere.
    - The upper-left, upper-right, and stencil regions each have a unique modal color.
    - If the stencil is non-uniform, coarse fill indices stay within the lower-left
      tile bounds.
  generator guarantees:
    - Off-center partitioning is enforced.
    - Divider color is reserved.
    - Unique modes are explicitly constructed, not left to chance.
    - Output sizes are filtered against both control feasibility and coarse fill
      capacity.
"""

from collections import Counter
from math import ceil
import random

import numpy as np


BLACK = 0
DIVIDER = 5
ALL_COLORS = tuple(range(10))
CONTENT_COLORS = tuple(c for c in ALL_COLORS if c != DIVIDER)
NONZERO_CONTENT_COLORS = tuple(c for c in CONTENT_COLORS if c != BLACK)

STENCIL_EMPTY_PROB = 0.16
LL_SOLID_PROB = 0.70
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.75
STENCIL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_FG_FRAC = (0.18, 0.40)
CONTROL_SINGLE_COLOR_PROB = 0.65
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 4
UR_MAX_COLORS = 4
REGION_SIDE_RANGE = (4, 6)
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 500


def _unique_mode(arr: np.ndarray) -> int:
    counts = Counter(map(int, arr.flatten()))
    ranked = counts.most_common()
    assert ranked
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1]
    return int(ranked[0][0])


def _find_dividers(g: np.ndarray) -> tuple[int, int, int]:
    """Return (divider_row, divider_col, divider_color)."""
    h, w = g.shape
    row_candidates = [
        (r, int(g[r, 0]))
        for r in range(h)
        if int(g[r, 0]) != BLACK and np.all(g[r, :] == g[r, 0])
    ]
    col_candidates = [
        (c, int(g[0, c]))
        for c in range(w)
        if int(g[0, c]) != BLACK and np.all(g[:, c] == g[0, c])
    ]
    colors = {
        row_color
        for _, row_color in row_candidates
        for _, col_color in col_candidates
        if row_color == col_color
    }
    assert len(colors) == 1
    divider_color = next(iter(colors))
    rows = [r for r, color in row_candidates if color == divider_color]
    cols = [c for c, color in col_candidates if color == divider_color]
    assert len(rows) == 1 and len(cols) == 1
    divider_r = rows[0]
    divider_c = cols[0]
    assert 0 < divider_r < h - 1 and 0 < divider_c < w - 1
    return divider_r, divider_c, divider_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the divider cross, infer local background colors by frequency, and then build the output from the two lower regions.

    Input structure:
    - The grid contains one full separator row and one full separator column using the same dedicated color.
    - Those dividers partition the grid into four rectangular regions, which do not need to be equal in size.
    - The upper-left and upper-right regions are controls, the lower-left region is a fill tile, and the lower-right region is a stencil.
    - Inside each relevant region, background is defined by the most frequent color rather than by a fixed global color id.

    Rule:
    - Output height is the number of cells in the upper-left region that differ from the upper-left modal color.
    - Output width is the number of cells in the upper-right region that differ from the upper-right modal color.
    - If the lower-right stencil is uniform in its own modal color, tile the lower-left region directly over the output.
    - Otherwise tile the stencil over the output, and whenever a tiled stencil cell equals the stencil modal color, replace it with the lower-left cell indexed by the coarse stencil-repeat position.

    Key insight:
    - The original "count above, pattern below" structure is preserved, but both preparatory assumptions changed: the four roles must be found from divider lines, and the relevant backgrounds must be inferred locally from color frequency rather than from color 0.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    divider_r, divider_c, _divider_color = _find_dividers(g)

    upper_left = g[:divider_r, :divider_c]
    upper_right = g[:divider_r, divider_c + 1 :]
    fill_tile = g[divider_r + 1 :, :divider_c]
    stencil = g[divider_r + 1 :, divider_c + 1 :]

    assert upper_left.size > 0 and upper_right.size > 0
    assert fill_tile.size > 0 and stencil.size > 0

    ul_bg = _unique_mode(upper_left)
    ur_bg = _unique_mode(upper_right)
    stencil_bg = _unique_mode(stencil)

    out_h = int(np.count_nonzero(upper_left != ul_bg))
    out_w = int(np.count_nonzero(upper_right != ur_bg))
    assert out_h > 0 and out_w > 0

    fill_h, fill_w = fill_tile.shape
    stencil_h, stencil_w = stencil.shape
    out = np.empty((out_h, out_w), dtype=int)

    if np.all(stencil == stencil_bg):
        for r in range(out_h):
            for c in range(out_w):
                out[r, c] = int(fill_tile[r % fill_h, c % fill_w])
        return out

    for r in range(out_h):
        for c in range(out_w):
            value = int(stencil[r % stencil_h, c % stencil_w])
            if value == stencil_bg:
                fill_r = r // stencil_h
                fill_c = c // stencil_w
                assert fill_r < fill_h and fill_c < fill_w
                value = int(fill_tile[fill_r, fill_c])
            out[r, c] = value
    return out


def _palette(k: int, pool: tuple[int, ...] = CONTENT_COLORS) -> list[int]:
    pool_list = list(pool)
    random.shuffle(pool_list)
    return pool_list[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


def _ordered_coords(height: int, width: int, mode: str) -> list[tuple[int, int]]:
    coords = [(r, c) for r in range(height) for c in range(width)]
    if mode == "rect":
        r0 = random.randint(0, height - 1)
        c0 = random.randint(0, width - 1)
        return [((r + r0) % height, (c + c0) % width) for r, c in coords]
    if mode == "rows":
        rows = list(range(height))
        cols = list(range(width))
        random.shuffle(rows)
        random.shuffle(cols)
        return [(r, c) for r in rows for c in cols]
    if mode == "compact":
        ar = random.randrange(height)
        ac = random.randrange(width)
        random.shuffle(coords)
        return sorted(
            coords,
            key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()),
        )
    random.shuffle(coords)
    return coords


def _checker_coords(height: int, width: int) -> list[tuple[int, int]]:
    parity = random.randrange(2)
    coords = [(r, c) for r in range(height) for c in range(width) if (r + c) % 2 == parity]
    random.shuffle(coords)
    return coords


def _frame_coords(height: int, width: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(width):
        coords.append((0, c))
    for r in range(1, height - 1):
        coords.append((r, width - 1))
    if height > 1:
        for c in range(width - 1, -1, -1):
            coords.append((height - 1, c))
    for r in range(height - 2, 0, -1):
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


def _control_feasible(height: int, width: int, count: int, max_colors: int, pool_size: int) -> bool:
    area = height * width
    bg_count = area - count
    if count <= 0 or bg_count <= 1 or pool_size <= 1:
        return False
    max_k = min(max_colors, count, pool_size - 1)
    return any(count <= k * (bg_count - 1) for k in range(1, max_k + 1))


def _make_lower_left(
    shape: tuple[int, int],
    palette: list[int],
    required_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Build the fill tile using simple repeated color patterns."""
    height, width = shape
    k = len(palette)
    tile = np.empty((height, width), dtype=int)
    mode = "solid" if k == 1 else random.choice(["rows", "cols", "rows", "cols", "diagonal"])

    if mode == "solid":
        tile[:] = palette[0]
    elif mode == "rows":
        offset = random.randrange(k)
        for r in range(height):
            tile[r, :] = palette[(r + offset) % k]
    elif mode == "cols":
        offset = random.randrange(k)
        for c in range(width):
            tile[:, c] = palette[(c + offset) % k]
    else:
        offset = random.randrange(k)
        rr, cc = np.mgrid[:height, :width]
        idx = (rr + cc + offset) % k
        for i, color in enumerate(palette):
            tile[idx == i] = color

    existing = set(map(int, np.unique(tile)))
    missing = [color for color in palette if color not in existing]
    if missing:
        fix_coords = (
            list(required_coords)
            if required_coords is not None
            else [(r, c) for r in range(height) for c in range(width)]
        )
        random.shuffle(fix_coords)
        for color, (r, c) in zip(missing, fix_coords):
            tile[r, c] = color
    return tile


def _make_stencil(shape: tuple[int, int], bg_color: int, palette: list[int]) -> np.ndarray:
    """Build the stencil tile with a unique modal background color."""
    height, width = shape
    area = height * width
    tile = np.full((height, width), bg_color, dtype=int)
    if not palette:
        return tile

    lo = max(1, int(ceil(STENCIL_FG_FRAC[0] * area)))
    hi = max(lo, int(STENCIL_FG_FRAC[1] * area))
    hi = min(hi, (area - 1) // 2)
    fg_count = random.randint(lo, hi)

    modes = ["compact", "rows", "rect", "checker", "random"]
    if min(height, width) >= 4:
        modes.append("frame")
    mode = random.choice(modes)

    if mode == "checker":
        coords = _checker_coords(height, width)[:fg_count]
    elif mode == "frame":
        frame = _frame_coords(height, width)
        random.shuffle(frame)
        coords = frame[:fg_count]
    else:
        base_mode = mode if mode != "random" else random.choice(["rect", "rows", "compact"])
        coords = _ordered_coords(height, width, base_mode)[:fg_count]

    for r, c in coords:
        tile[r, c] = random.choice(palette)

    assert _unique_mode(tile) == bg_color
    assert np.any(tile != bg_color)
    return tile


def _make_control(height: int, width: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control region whose unique modal color is its local background."""
    area = height * width
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
    pal = _palette(k, tuple(fg_pool))
    counts = _split_count(count, k, bg_count - 1)

    tile = np.full((height, width), bg, dtype=int)
    factors = [
        (rect_h, rect_w)
        for rect_h in range(1, height + 1)
        for rect_w in range(1, width + 1)
        if rect_h * rect_w == count
    ]
    if factors and random.random() < CONTROL_RECT_PROB:
        rect_h, rect_w = random.choice(factors)
        r0 = random.randint(0, height - rect_h)
        c0 = random.randint(0, width - rect_w)
        coords = [(r, c) for r in range(r0, r0 + rect_h) for c in range(c0, c0 + rect_w)]
    else:
        mode = random.choice(["rect", "compact", "rows"])
        coords = _ordered_coords(height, width, mode)[:count]

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
    fill_h, fill_w = fill_tile.shape
    stencil_h, stencil_w = stencil.shape
    stencil_bg = _unique_mode(stencil)
    rr, cc = np.mgrid[:out_h, :out_w]
    if np.all(stencil == stencil_bg):
        return fill_tile[rr % fill_h, cc % fill_w]
    tiled = stencil[rr % stencil_h, cc % stencil_w].copy()
    bg_mask = tiled == stencil_bg
    tiled[bg_mask] = fill_tile[rr[bg_mask] // stencil_h, cc[bg_mask] // stencil_w]
    return tiled


def _zero_bg_same_split_output(grid: np.ndarray) -> np.ndarray | None:
    """A degeneration probe: correct split, but still assuming color 0 is background."""
    try:
        g = np.asarray(grid, dtype=int)
        divider_r, divider_c, _ = _find_dividers(g)
        upper_left = g[:divider_r, :divider_c]
        upper_right = g[:divider_r, divider_c + 1 :]
        fill_tile = g[divider_r + 1 :, :divider_c]
        stencil = g[divider_r + 1 :, divider_c + 1 :]
        out_h = int(np.count_nonzero(upper_left != BLACK))
        out_w = int(np.count_nonzero(upper_right != BLACK))
        if out_h <= 0 or out_w <= 0:
            return None
        fill_h, fill_w = fill_tile.shape
        stencil_h, stencil_w = stencil.shape
        out = np.empty((out_h, out_w), dtype=int)
        if not np.any(stencil != BLACK):
            for r in range(out_h):
                for c in range(out_w):
                    out[r, c] = int(fill_tile[r % fill_h, c % fill_w])
            return out
        for r in range(out_h):
            for c in range(out_w):
                value = int(stencil[r % stencil_h, c % stencil_w])
                if value == BLACK:
                    fill_r = r // stencil_h
                    fill_c = c // stencil_w
                    if fill_r >= fill_h or fill_c >= fill_w:
                        return None
                    value = int(fill_tile[fill_r, fill_c])
                out[r, c] = value
        return out
    except Exception:
        return None


def _sample_region_sizes() -> tuple[int, int, int, int]:
    for _ in range(100):
        top = random.randint(*REGION_SIDE_RANGE)
        bottom = random.randint(*REGION_SIDE_RANGE)
        left = random.randint(*REGION_SIDE_RANGE)
        right = random.randint(*REGION_SIDE_RANGE)
        if top != bottom or left != right:
            return top, bottom, left, right
    raise RuntimeError("failed to sample an off-center partition")


def _gen_attempt() -> np.ndarray | None:
    top, bottom, left, right = _sample_region_sizes()
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k, CONTENT_COLORS)

    if stencil_empty:
        stencil_palette: list[int] = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(
            *STENCIL_MULTI_COLOR_RANGE
        )
        stencil_palette = _palette(st_k, CONTENT_COLORS)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in NONZERO_CONTENT_COLORS if c not in output_colors]
    if len(control_pool) < 2:
        return None

    valid_h = [
        out_h
        for out_h in range(OUTPUT_DIM_MIN, min(OUTPUT_DIM_MAX, top * left - 1) + 1)
        if _control_feasible(top, left, out_h, UL_MAX_COLORS, len(control_pool))
    ]
    valid_w = [
        out_w
        for out_w in range(OUTPUT_DIM_MIN, min(OUTPUT_DIM_MAX, top * right - 1) + 1)
        if _control_feasible(top, right, out_w, UR_MAX_COLORS, len(control_pool))
    ]
    valid_dims: list[tuple[int, int]] = []
    for out_h in valid_h:
        for out_w in valid_w:
            if out_h * out_w > OUTPUT_AREA_MAX:
                continue
            if not stencil_empty:
                if ceil(out_h / bottom) > bottom or ceil(out_w / right) > left:
                    continue
                if ceil(out_h / bottom) * ceil(out_w / right) < ll_k:
                    continue
            valid_dims.append((out_h, out_w))
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    used_fill = None
    if not stencil_empty:
        used_fill = [
            (r, c)
            for r in range(ceil(out_h / bottom))
            for c in range(ceil(out_w / right))
        ]
    fill_tile = _make_lower_left((bottom, left), ll_palette, used_fill)
    stencil_bg = random.choice(control_pool)
    stencil = (
        np.full((bottom, right), stencil_bg, dtype=int)
        if stencil_empty
        else _make_stencil((bottom, right), stencil_bg, stencil_palette)
    )
    upper_left = _make_control(top, left, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(top, right, out_w, UR_MAX_COLORS, control_pool)

    grid = np.empty((top + 1 + bottom, left + 1 + right), dtype=int)
    grid[top, :] = DIVIDER
    grid[:, left] = DIVIDER
    grid[:top, :left] = upper_left
    grid[:top, left + 1 :] = upper_right
    grid[top + 1 :, :left] = fill_tile
    grid[top + 1 :, left + 1 :] = stencil

    try:
        actual = main(grid)
    except Exception:
        return None
    expected = _expected_output(fill_tile, stencil, out_h, out_w)
    if not np.array_equal(actual, expected):
        return None

    probe = _zero_bg_same_split_output(grid)
    if probe is None or np.array_equal(actual, probe):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
