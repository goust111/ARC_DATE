# Combo variant of f931b4a8 (index 29)
# combo_spec: D2/2+D8/0+D9/0
# Name: 分隔线划分+最频繁背景+边缘编码

from __future__ import annotations

# Design analysis for the combo variant
# -------------------------------------
# a) concretization_and_trace:
#   - The grid is no longer midpoint-split. Instead, one solid divider row and one
#     solid divider column partition the canvas into four rectangular regions.
#   - The upper-left region still controls height and the upper-right region still
#     controls width, but each control is compressed onto a 1D boundary:
#       * height = number of cells on the leftmost column of the upper-left region
#         whose color differs from that region's modal background
#       * width = number of cells on the topmost row of the upper-right region
#         whose color differs from that region's modal background
#   - The lower-left region is the coarse fill tile; the lower-right region is the
#     stencil tile. The stencil's own modal background marks holes. If the stencil
#     is uniform in that modal color, it is treated as empty and the fill tile is
#     repeated directly.
#   - Example trace: suppose the divider row is at r=5 and the divider column at
#     c=4. The upper-left region is 5x4 and has modal color 7. If exactly four
#     cells on its left edge differ from 7, output height is 4 even if the region
#     also contains interior distractor cells. The upper-right region may have modal
#     color 3, with six non-3 cells on its top edge, so output width is 6. The
#     lower-right stencil may have modal color 8; tiled 8-cells are holes filled
#     from lower-left[rr // stencil_h, cc // stencil_w]. Compared to the source
#     task, midpoint slicing, fixed black background, and 2D control counts all fail.
#
# b) complexity_calibration:
#   - The divider mechanism is visually strong but minimal: exactly one solid row
#     plus one solid column in a reserved color.
#   - Control marks are concentrated on one edge per control region, so the edge
#     encoding is observable. Small interior distractors are sometimes added to the
#     control regions so whole-region counting is measurably wrong.
#   - Modal backgrounds are usually non-black, and the fill/stencil palettes may
#     still include 0, which makes the D9 change visible without adding extra rules.
#
# c) implicit_assumption_audit:
#   - The source `h == w and h % 2 == 0` assumption is invalid: divider-based
#     regions may be rectangular and off-center.
#   - Midpoint slices such as `g[:qh, :qw]` are invalid. Divider detection must
#     happen first.
#   - `count_nonzero(control != 0)` is invalid because control backgrounds are local
#     modes and are often non-zero.
#   - The control count is no longer a 2D area count. Only the upper-left left edge
#     and upper-right top edge are counted.
#   - `value == 0` is no longer the stencil-hole test. The hole color is the
#     lower-right region's modal background.
#   - Assertions that lower-left values cannot equal background must be removed,
#     because the fill tile may legitimately contain 0 or its own modal color.
#
# d) placement_legality:
#   - The generator samples region sizes first, then inserts the divider cross into
#     its own reserved row and column, so there is no overlap ambiguity.
#   - Control marks are placed only inside the valid control edge line, and optional
#     distractors avoid that line so they cannot corrupt the intended count.
#   - Lower-left and lower-right share the same height by construction. Output sizes
#     are capped by control-line lengths, while the lower regions are large enough
#     that coarse fill lookups stay in bounds.
#
# e) visual_observability:
#   - A human can first see the divider cross, then notice that the large flat
#     control-region color is ignored even when it is not black, and then match
#     output dimensions to the highlighted border marks rather than region area.
#   - Off-center cuts are forced, so the task does not silently collapse back into a
#     midpoint-split puzzle.
#
# f) visual_cleanliness:
#   - Divider color is reserved for the cross only.
#   - The four regions keep simple stripe/segment/scatter structure. There is no
#     multi-layer overdraw inside one cell.
#
# g) change_inferability:
#   - No hidden recoloring occurs. Colors are copied literally.
#   - The only semantic color change is the definition of background as the local
#     modal color, and that is always supported by strong frequency dominance.
#
# h) degeneration_check_plan:
#   - Measure how often the divider is off-center.
#   - Measure how often a black-background interpretation gives the wrong control
#     counts.
#   - Measure how often whole-region counting gives the wrong dimensions because of
#     interior distractors.
#
# i) new_solver_assumptions_and_generator_guarantees:
#   solver_assumptions:
#     - Exactly one full divider row and one full divider column share a reserved
#       color and lie strictly inside the grid.
#     - Upper-left and upper-right each have a unique modal background.
#     - The left edge of upper-left and the top edge of upper-right contain at least
#       one non-background control mark.
#     - Lower-right has a unique modal background; if non-uniform, that modal color
#       remains the strict mode so it can safely denote holes.
#   generator_guarantees:
#     - Divider color never appears outside the divider cross.
#     - Relevant modal backgrounds are unique and usually non-zero.
#     - Optional distractors in control regions never land on the counted control
#       line, so the intended dimension count remains exact.

from collections import Counter
from math import ceil
import random

import numpy as np

from arc_utils import Color


DIVIDER = int(Color.GREY)
BLACK = int(Color.BLACK)
ALL_REGION_COLORS = tuple(c for c in range(10) if c != DIVIDER)
NONZERO_REGION_COLORS = tuple(c for c in range(1, 10) if c != DIVIDER)

STENCIL_EMPTY_PROB = 0.18
FILL_SOLID_PROB = 0.65
FILL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.70
STENCIL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_FG_FRAC = (0.20, 0.40)
CONTROL_SINGLE_COLOR_PROB = 0.80
CONTROL_EDGE_SEGMENT_PROB = 0.80
CONTROL_NOISE_PROB = 0.75
CONTROL_MAX_NOISE = 2
CONTROL_MAX_COLORS = 3
REGION_SIDE_RANGE = (4, 7)
OUTPUT_DIM_MIN = 3
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
    h, w = g.shape
    row_candidates = [
        (r, int(g[r, 0]))
        for r in range(h)
        if 0 < r < h - 1 and np.all(g[r, :] == g[r, 0])
    ]
    col_candidates = [
        (c, int(g[0, c]))
        for c in range(w)
        if 0 < c < w - 1 and np.all(g[:, c] == g[0, c])
    ]
    shared_colors = {
        color
        for _, color in row_candidates
        for _, other in col_candidates
        if color == other
    }
    assert len(shared_colors) == 1
    divider_color = next(iter(shared_colors))
    rows = [r for r, color in row_candidates if color == divider_color]
    cols = [c for c, color in col_candidates if color == divider_color]
    assert len(rows) == 1 and len(cols) == 1
    return rows[0], cols[0], divider_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect a divider cross, infer local modal backgrounds, then read edge controls to size a tiled output.

    Input structure:
    - The grid contains one full separator row and one full separator column in a reserved divider color.
    - Those lines split the canvas into four rectangular regions that are not required to be equal or centered.
    - The upper-left and upper-right regions are control regions, but their counts are encoded only on the grid-facing outer edges: the upper-left left column and the upper-right top row.
    - The lower-left region is a fill tile, and the lower-right region is a stencil tile whose own modal color denotes holes.

    Rule:
    - Find the divider row and divider column first, then slice the four regions around them.
    - In the upper-left region, compute the modal background color and count how many cells on its leftmost column differ from that mode; this is the output height.
    - In the upper-right region, compute the modal background color and count how many cells on its top row differ from that mode; this is the output width.
    - If the lower-right stencil is uniform in its own modal color, tile the lower-left region directly. Otherwise tile the stencil, and every tiled stencil cell equal to the stencil mode is replaced by the lower-left value selected by the coarse repeated-tile block position.

    Key insight:
    - The puzzle must now be parsed in stages: locate the divider cross, infer each region's background by frequency rather than by color id, and only then read the 1D edge encodings that replace the old 2D control quadrants.
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

    ul_bg = _unique_mode(upper_left)
    ur_bg = _unique_mode(upper_right)
    stencil_bg = _unique_mode(stencil)

    out_h = int(np.count_nonzero(upper_left[:, 0] != ul_bg))
    out_w = int(np.count_nonzero(upper_right[0, :] != ur_bg))
    assert out_h > 0 and out_w > 0

    fill_h, fill_w = fill_tile.shape
    stencil_h, stencil_w = stencil.shape
    assert fill_h == stencil_h

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


def _palette(k: int, pool: tuple[int, ...] | list[int]) -> list[int]:
    values = list(pool)
    random.shuffle(values)
    return values[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


def _ordered_positions(length: int, mode: str) -> list[int]:
    positions = list(range(length))
    if mode == "compact":
        anchor = random.randrange(length)
        random.shuffle(positions)
        return sorted(positions, key=lambda idx: (abs(idx - anchor), random.random()))
    if mode == "ends":
        return positions[::2] + positions[1::2]
    random.shuffle(positions)
    return positions


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
        return sorted(coords, key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()))
    random.shuffle(coords)
    return coords


def _make_fill_tile(shape: tuple[int, int], palette: list[int]) -> np.ndarray:
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
        coords = [(r, c) for r in range(height) for c in range(width)]
        random.shuffle(coords)
        for color, (r, c) in zip(missing, coords):
            tile[r, c] = color
    return tile


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


def _make_stencil(shape: tuple[int, int], bg_color: int, palette: list[int]) -> np.ndarray:
    height, width = shape
    tile = np.full((height, width), bg_color, dtype=int)
    if not palette:
        return tile

    area = height * width
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
        coords = _frame_coords(height, width)
        random.shuffle(coords)
        coords = coords[:fg_count]
    elif mode == "rows":
        coords = _ordered_coords(height, width, "rows")[:fg_count]
    elif mode == "rect":
        coords = _ordered_coords(height, width, "rect")[:fg_count]
    elif mode == "compact":
        coords = _ordered_coords(height, width, "compact")[:fg_count]
    else:
        coords = _ordered_coords(height, width, random.choice(["rect", "rows", "compact"]))[:fg_count]

    for r, c in coords:
        tile[r, c] = random.choice(palette)
    assert _unique_mode(tile) == bg_color
    assert np.any(tile != bg_color)
    return tile


def _control_edge_positions(length: int, count: int) -> list[int]:
    assert 0 < count <= length
    if random.random() < CONTROL_EDGE_SEGMENT_PROB:
        start = random.randint(0, length - count)
        return list(range(start, start + count))
    mode = random.choice(["compact", "ends", "scatter"])
    return _ordered_positions(length, "compact" if mode == "compact" else mode if mode == "ends" else "scatter")[:count]


def _make_control_region(
    height: int,
    width: int,
    count: int,
    edge: str,
    pool: list[int],
) -> np.ndarray:
    assert edge in {"left", "top"}
    assert count > 0
    line_length = height if edge == "left" else width
    assert count <= line_length

    bg = random.choice(pool)
    fg_pool = [c for c in pool if c != bg]
    assert fg_pool
    tile = np.full((height, width), bg, dtype=int)

    color_count = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(
        1,
        min(CONTROL_MAX_COLORS, len(fg_pool)),
    )
    mark_palette = _palette(color_count, fg_pool)
    edge_positions = _control_edge_positions(line_length, count)
    for pos in edge_positions:
        color = random.choice(mark_palette)
        if edge == "left":
            tile[pos, 0] = color
        else:
            tile[0, pos] = color

    total_marks = count
    max_total_marks = (height * width - 1) // 2
    noise_budget = min(CONTROL_MAX_NOISE, max_total_marks - total_marks)
    if noise_budget > 0 and random.random() < CONTROL_NOISE_PROB:
        noise_count = random.randint(1, noise_budget)
        if edge == "left":
            base_coords = _ordered_coords(
                height,
                width - 1,
                random.choice(["compact", "rows", "rect"]),
            )
            mapped = [(r, c + 1) for r, c in base_coords]
        else:
            base_coords = _ordered_coords(
                height - 1,
                width,
                random.choice(["compact", "rows", "rect"]),
            )
            mapped = [(r + 1, c) for r, c in base_coords]
        for r, c in mapped[:noise_count]:
            tile[r, c] = random.choice(mark_palette)

    assert _unique_mode(tile) == bg
    if edge == "left":
        assert int(np.count_nonzero(tile[:, 0] != bg)) == count
    else:
        assert int(np.count_nonzero(tile[0, :] != bg)) == count
    return tile


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    fill_h, fill_w = fill_tile.shape
    stencil_h, stencil_w = stencil.shape
    stencil_bg = _unique_mode(stencil)
    rr, cc = np.mgrid[:out_h, :out_w]
    if np.all(stencil == stencil_bg):
        return fill_tile[rr % fill_h, cc % fill_w]
    tiled = stencil[rr % stencil_h, cc % stencil_w].copy()
    mask = tiled == stencil_bg
    tiled[mask] = fill_tile[rr[mask] // stencil_h, cc[mask] // stencil_w]
    return tiled


def _sample_region_sizes() -> tuple[int, int, int, int]:
    for _ in range(60):
        top = random.randint(*REGION_SIDE_RANGE)
        bottom = random.randint(*REGION_SIDE_RANGE)
        left = random.randint(*REGION_SIDE_RANGE)
        right = random.randint(*REGION_SIDE_RANGE)
        if top != bottom or left != right:
            return top, bottom, left, right
    raise RuntimeError("failed to sample an off-center divider layout")


def _gen_attempt() -> np.ndarray | None:
    top, bottom, left, right = _sample_region_sizes()
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    fill_k = 1 if random.random() < FILL_SOLID_PROB else random.randint(*FILL_MULTI_COLOR_RANGE)
    fill_palette = _palette(fill_k, ALL_REGION_COLORS)

    if stencil_empty:
        stencil_palette: list[int] = []
    else:
        stencil_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(*STENCIL_MULTI_COLOR_RANGE)
        stencil_palette = _palette(stencil_k, ALL_REGION_COLORS)

    output_colors = set(fill_palette) | set(stencil_palette)
    control_pool = [c for c in NONZERO_REGION_COLORS if c not in output_colors]
    if len(control_pool) < 2:
        return None

    valid_dims = [
        (out_h, out_w)
        for out_h in range(OUTPUT_DIM_MIN, top + 1)
        for out_w in range(OUTPUT_DIM_MIN, right + 1)
        if out_h * out_w <= OUTPUT_AREA_MAX
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    fill_tile = _make_fill_tile((bottom, left), fill_palette)
    stencil_bg = random.choice(control_pool)
    stencil = (
        np.full((bottom, right), stencil_bg, dtype=int)
        if stencil_empty
        else _make_stencil((bottom, right), stencil_bg, stencil_palette)
    )
    upper_left = _make_control_region(top, left, out_h, "left", control_pool)
    upper_right = _make_control_region(top, right, out_w, "top", control_pool)

    grid = np.empty((top + 1 + bottom, left + 1 + right), dtype=int)
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
