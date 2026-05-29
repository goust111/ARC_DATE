# Variant of f931b4a8
# Dimension: D8 — 象限的划分方式 (meta)
# Change: 输入网格沿中点均分为四个等面积象限（H/2 x W/2），以正方形或偶数尺寸矩形网格为前提... → 用分隔线（特定颜色的行/列）划分象限

from __future__ import annotations

"""
Design analysis for the D8 variant.

a) concretization_and_trace:
  variant_focus:
    - Keep the original four-region semantics: upper-left controls output height,
      upper-right controls output width, lower-left is the coarse fill tile, and
      lower-right is the stencil tile.
    - Replace midpoint splitting with one full horizontal divider row and one full
      vertical divider column, both using a dedicated separator color.
    - The four regions no longer need equal sizes. They only need to be non-empty.
  pixel_level_shape:
    - The input contains a single solid grey row and a single solid grey column.
    - The row/column intersect once and partition the grid into four rectangular regions.
    - Black remains the only background color inside regions; grey is reserved for dividers.
  behavior_trace:
    - Original: on an 8x8 grid, `qh = qw = 4`, so all four slices are fixed by the midpoint.
      A pixel at output (7, 5) looks up stencil[3, 1]; if that stencil cell is black,
      it uses fill_tile[1, 1] because `7 // 4 = 1` and `5 // 4 = 1`.
    - Variant: suppose the divider row is at r=3 and divider column at c=4, so the
      four regions are 3x4, 3x5, 5x4, and 5x5. The same output logic now uses the
      detected boundaries: stencil[7 % 5, 5 % 5] = stencil[2, 0], and a hole is filled
      from fill_tile[7 // 5, 5 // 5] = fill_tile[1, 1].
    - Difference: the split is no longer inferable from `h // 2` and `w // 2`; the
      solver must first detect the frame lines, then use the true region sizes.

b) complexity_calibration:
  - The separator mechanism is intentionally simple and visually strong: exactly one
    full row and one full column, sharing the same grey color.
  - Region sizes vary, but only through one horizontal cut and one vertical cut; this
    keeps the rule discoverable from a few examples.
  - The divider color is excluded from all non-divider content so the frame structure
    is unambiguous and does not compete with the pattern colors.

c) implicit_assumption_audit:
  - `assert h == w and h % 2 == 0 and w % 2 == 0` relied on midpoint splitting.
    This is removed because the variant allows arbitrary rectangular sizes.
  - `qh, qw = h // 2, w // 2` assumed all four regions were equal rectangles.
    This is replaced by explicit divider detection.
  - Slices such as `g[:qh, :qw]` depended on the midpoint being the true boundary.
    They are replaced by slices around the detected divider row/column.
  - `stencil[r % qh, c % qw]` and `fill_tile[r % qh, c % qw]` assumed both lower
    regions had the same shape. The variant keeps the shared bottom height but allows
    different widths, so width indexing is now handled separately.
  - `fill_r = r // qh` and `fill_c = c // qw` were only safe because the generator
    guaranteed output dimensions compatible with the equal quadrant size. The new
    generator must enforce `out_h <= fill_h * stencil_h` and `out_w <= fill_w * stencil_w`
    whenever stencil holes may trigger coarse fill lookup.

d) placement_legality:
  - The generator creates four region sizes first, then inserts the divider row/column.
  - All regions are at least 3 cells wide/high, so control counts can reach the minimum
    output size and the lower tiles have enough structure to be visible.
  - Lower-left contains no black pixels. Lower-right may be all black or a non-empty
    pattern with holes. When holes exist, the generator only chooses output sizes whose
    coarse tile indices stay within lower-left.
  - The divider row/column are written after region construction and occupy their own
    dedicated cells, so there is no overlap ambiguity.

e) visual_observability:
  - Every generated sample forces at least one off-center cut (`top != bottom` or
    `left != right`), so the frame-based partition is visible rather than silently
    collapsing back to a centered split.
  - The grey cross is a direct cue that the task now uses separators instead of equal
    quadrants. A human can infer the four roles by reading the regions around the cross.

f) visual_cleanliness:
  - Only one separator color is used and it appears only on the full divider row/column.
  - Region contents keep the original task's simple rect/stripe/checker patterns, so the
    input still reads as four clean areas rather than a heavily overlaid collage.

g) color_inference:
  - No hidden ordering or remapping depends on color ids.
  - Upper control colors remain irrelevant to the output and are sampled from colors not
    used by the lower regions, which preserves the original "count, don't recolor" clue.

h) degeneration_check_plan:
  - The critical D8 signal is the off-center divider-based split. The validation script
    will measure how often samples are asymmetric and how often midpoint-based assumptions
    are impossible on the raw grid.

i) new_solver_assumptions_and_generator_guarantees:
  solver_assumptions:
    - Exactly one solid non-black divider row and exactly one solid non-black divider
      column share the same color.
    - The divider row/column are not on the outer border, so four non-empty regions exist.
    - Lower-left has no black cells.
    - If the stencil has holes, output dimensions stay within the coarse fill capacity.
  generator_guarantees:
    - Grey is reserved for the divider only.
    - Region sizes are sampled until at least one cut is off-center.
    - Output dimensions are filtered against both control capacities and coarse fill bounds.
"""

from math import ceil
import random

import numpy as np

from arc_utils import Color


BLACK = int(Color.BLACK)
DIVIDER = int(Color.GREY)
COLORS = tuple(c for c in range(1, 10) if c != DIVIDER)

STENCIL_EMPTY_PROB = 0.16
LL_SOLID_PROB = 0.70
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.75
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.75)
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
REGION_SIDE_RANGE = (3, 5)
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 400


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
        color
        for _, color in row_candidates
        for _, col_color in col_candidates
        if color == col_color
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
    Detect divider lines that split the input into four regions, then build the output from the two lower regions.

    Input structure:
    - The grid contains one full non-black separator row and one full non-black separator column.
    - Those two divider lines partition the grid into four rectangular regions, which may have different sizes.
    - The upper-left and upper-right regions are controls, the lower-left region is a fill tile, and the lower-right region is a stencil tile.
    - Black is the region background; the divider color is only a boundary marker.

    Rule:
    - Output height is the number of non-black pixels in the upper-left region.
    - Output width is the number of non-black pixels in the upper-right region.
    - If the lower-right stencil is empty, tile the lower-left region directly over the output.
    - Otherwise tile the stencil over the output, and whenever a stencil cell is black, replace it with the lower-left cell indexed by the coarse stencil-repeat position.

    Key insight:
    - The task logic is still "count above, pattern below", but the first step is no longer midpoint splitting; the solver must locate the divider cross before interpreting the four roles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    divider_r, divider_c, _divider_color = _find_dividers(g)

    upper_left = g[:divider_r, :divider_c]
    upper_right = g[:divider_r, divider_c + 1:]
    fill_tile = g[divider_r + 1:, :divider_c]
    stencil = g[divider_r + 1:, divider_c + 1:]

    assert upper_left.size > 0 and upper_right.size > 0
    assert fill_tile.size > 0 and stencil.size > 0
    fill_h, fill_w = fill_tile.shape
    stencil_h, stencil_w = stencil.shape
    assert fill_h == stencil_h

    out_h = int(np.count_nonzero(upper_left != BLACK))
    out_w = int(np.count_nonzero(upper_right != BLACK))
    assert out_h > 0 and out_w > 0

    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != BLACK):
        for r in range(out_h):
            for c in range(out_w):
                value = int(fill_tile[r % fill_h, c % fill_w])
                assert value != BLACK
                out[r, c] = value
        return out

    for r in range(out_h):
        for c in range(out_w):
            value = int(stencil[r % stencil_h, c % stencil_w])
            if value == BLACK:
                fill_r = r // stencil_h
                fill_c = c // stencil_w
                assert fill_r < fill_h and fill_c < fill_w
                value = int(fill_tile[fill_r, fill_c])
                assert value != BLACK
            out[r, c] = value

    return out


def _palette(k: int, pool: tuple[int, ...] = COLORS) -> list[int]:
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


def _make_lower_left(
    shape: tuple[int, int],
    palette: list[int],
    required_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Build the fill tile (all non-black) using stripe/solid/diagonal patterns."""
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


def _make_stencil(shape: tuple[int, int], palette: list[int]) -> np.ndarray:
    """Build the stencil tile (black = holes filled from fill tile)."""
    height, width = shape
    tile = np.zeros((height, width), dtype=int)
    modes = ["checker", "row_stripe", "col_stripe", "random", "random"]
    if min(height, width) >= 4:
        modes.append("frame")
    mode = random.choice(modes)

    if mode == "checker":
        parity = random.randrange(2)
        rr, cc = np.mgrid[:height, :width]
        mask = (rr + cc) % 2 == parity
        tile[mask] = _random_color_array((int(mask.sum()),), palette)
    elif mode == "frame":
        t = random.randint(1, max(1, min(height, width) // 4))
        mask = np.ones((height, width), dtype=bool)
        mask[t:height - t, t:width - t] = False
        tile[mask] = _random_color_array((int(mask.sum()),), palette)
    elif mode == "row_stripe":
        parity = random.randrange(2)
        for r in range(height):
            if r % 2 == parity:
                tile[r, :] = _random_color_array((width,), palette)
    elif mode == "col_stripe":
        parity = random.randrange(2)
        for c in range(width):
            if c % 2 == parity:
                tile[:, c] = _random_color_array((height,), palette)
    else:
        area = height * width
        lo = max(1, int(ceil(STENCIL_FG_FRAC[0] * area)))
        hi = max(lo, int(STENCIL_FG_FRAC[1] * area))
        fg_count = random.randint(lo, hi)
        coords = [(r, c) for r in range(height) for c in range(width)]
        random.shuffle(coords)
        for r, c in coords[:fg_count]:
            tile[r, c] = random.choice(palette)
    return tile


def _make_control(height: int, width: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control region with exactly `count` non-black pixels."""
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(
        2, min(max_colors, count, len(pool))
    )
    pal = _palette(k, tuple(pool))
    tile = np.full((height, width), BLACK, dtype=int)

    factors = [
        (rh, rw)
        for rh in range(1, height + 1)
        for rw in range(1, width + 1)
        if rh * rw == count
    ]
    if factors and random.random() < CONTROL_RECT_PROB:
        rect_h, rect_w = random.choice(factors)
        r0 = random.randint(0, height - rect_h)
        c0 = random.randint(0, width - rect_w)
        tile[r0:r0 + rect_h, c0:c0 + rect_w] = _random_color_array((rect_h, rect_w), pal)
    else:
        mode = random.choice(["rect", "compact", "rows"])
        coords = _ordered_coords(height, width, mode)[:count]
        for r, c in coords:
            tile[r, c] = random.choice(pal)
    return tile


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Reproduce the solver's tiling logic for oracle validation."""
    fill_h, fill_w = fill_tile.shape
    stencil_h, stencil_w = stencil.shape
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % fill_h, cc % fill_w]
    tiled = stencil[rr % stencil_h, cc % stencil_w].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // stencil_h, cc[bg_mask] // stencil_w]
    return tiled


def _sample_region_sizes() -> tuple[int, int, int, int]:
    for _ in range(50):
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
    ll_palette = _palette(ll_k)
    if stencil_empty:
        stencil_palette: list[int] = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(
            *STENCIL_MULTI_COLOR_RANGE
        )
        stencil_palette = _palette(st_k)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    max_h = min(OUTPUT_DIM_MAX, top * left)
    max_w = min(OUTPUT_DIM_MAX, top * right)
    valid_dims = []
    for out_h in range(OUTPUT_DIM_MIN, max_h + 1):
        for out_w in range(OUTPUT_DIM_MIN, max_w + 1):
            if out_h * out_w > OUTPUT_AREA_MAX:
                continue
            if not stencil_empty:
                if out_h > bottom * bottom or out_w > left * right:
                    continue
                used_rows = ceil(out_h / bottom)
                used_cols = ceil(out_w / right)
                if used_rows > bottom or used_cols > left:
                    continue
                if used_rows * used_cols < ll_k:
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
    stencil = (
        np.zeros((bottom, right), dtype=int)
        if stencil_empty
        else _make_stencil((bottom, right), stencil_palette)
    )
    upper_left = _make_control(top, left, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(top, right, out_w, UR_MAX_COLORS, control_pool)

    grid = np.zeros((top + 1 + bottom, left + 1 + right), dtype=int)
    grid[top, :] = DIVIDER
    grid[:, left] = DIVIDER
    grid[:top, :left] = upper_left
    grid[:top, left + 1:] = upper_right
    grid[top + 1:, :left] = fill_tile
    grid[top + 1:, left + 1:] = stencil

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
