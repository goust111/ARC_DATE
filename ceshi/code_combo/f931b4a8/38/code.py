# Combo variant of f931b4a8 (index 38)
# combo_spec: D3/0+D4/0
# Name: 单色颜色源+空模板

from __future__ import annotations

import random

import numpy as np

from arc_utils import Color

# Design analysis for the fused variant:
# - Concrete behavior trace:
#   In the original task, UL and UR still count output height/width, but the
#   lower half supports two distinct behaviors:
#   1) non-empty stencil -> fine stencil pixels survive and black holes are
#      replaced from LL by coarse tile block coordinates;
#   2) empty stencil -> LL is tiled directly, which can still produce stripes or
#      multiple colors.
#   Under D3/0 + D4/0, both branches collapse together. LL is forced to be a
#   single non-black color everywhere, and LR is forced to be fully black. The
#   solver must still parse the four quadrants and read the two control counts,
#   but the produced output is now a solid rectangle filled with the unique LL
#   color. Example with Q=4: if UL has 6 colored cells, UR has 9, LL is all 7s,
#   and LR is all 0s, the original solver would tile LL directly and already
#   return a 6x9 all-7 rectangle. The fused variant makes that degenerate case
#   mandatory rather than occasional.
# - Complexity calibration:
#   This is intentionally a strong simplification, but it keeps two visible
#   degrees of freedom that are necessary for ARC inference: output height and
#   output width still vary independently through the top quadrants, and output
#   color varies through the lower-left quadrant. The bottom-right quadrant being
#   always empty is visually obvious, so the degeneration is still observable in
#   a few examples instead of looking like arbitrary pure-color outputs.
# - Implicit-assumption audit against the original main():
#   * Counting non-black cells in UL/UR still holds unchanged.
#   * The original empty-stencil branch indexed LL by (r % Q, c % Q). That
#     branch remains valid because every LL cell has the same non-black color, so
#     periodic indexing no longer matters.
#   * The original non-empty-stencil branch depended on coarse block coordinates
#     fill_r=r//Q and fill_c=c//Q staying inside LL. That entire dependency is
#     removed because LR is guaranteed to be empty.
#   * The original assertions that LL lookup values are non-black become a
#     stronger global invariant: every LL cell is the same non-black color.
# - Placement legality:
#   The generator keeps the original even-square four-quadrant layout. Control
#   pixels are placed only inside their own quadrants, LL is filled uniformly
#   inside its quadrant, and LR is all black, so there is no overlap or
#   out-of-bounds placement risk.
# - Visual observability / neatness:
#   Across examples, the user can still see that the top quadrants determine the
#   rectangle size while the lower-left quadrant determines the rectangle color.
#   The lower-right quadrant is uniformly empty and the output is visually clean:
#   one solid block, no mixed overlays, no hidden numeric ordering tricks.
# - New solver assumptions introduced here:
#   The solver now assumes LR is fully black and LL contains exactly one unique
#   non-black color with no black pixels. The generator enforces both conditions
#   directly, so these are safe contractual assumptions for this fused variant.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the two control quadrants, then emit a solid-color rectangle from the lower-left source.

    Input structure:
    - The input is an even square divided into four equal quadrants.
    - Black is background.
    - The upper-left quadrant encodes output height and the upper-right quadrant encodes output width by foreground counts.
    - The lower-left quadrant is a uniform non-black color patch, and the lower-right quadrant is entirely black.

    Rule:
    - Count non-black cells in the upper-left quadrant to get the output height.
    - Count non-black cells in the upper-right quadrant to get the output width.
    - Extract the unique non-black color used by the lower-left quadrant.
    - Return an output of that size filled entirely with that color.

    Key insight:
    - The original stencil/fill interaction has degenerated completely: once the template quadrant is empty and the color source is monochrome, only the two sizes and the single source color still matter.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0

    bg = int(Color.BLACK)
    q = h // 2
    upper_left = g[:q, :q]
    upper_right = g[:q, q:]
    fill_tile = g[q:, :q]
    stencil = g[q:, q:]

    out_h = int(np.count_nonzero(upper_left != bg))
    out_w = int(np.count_nonzero(upper_right != bg))
    assert out_h > 0 and out_w > 0

    assert not np.any(stencil != bg)
    assert np.all(fill_tile != bg)
    fill_colors = np.unique(fill_tile)
    assert fill_colors.size == 1

    fill_color = int(fill_colors[0])
    return np.full((out_h, out_w), fill_color, dtype=int)


BLACK = 0
COLORS = tuple(range(1, 10))
GRID_SIZES = (4, 8, 16)
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 300


def _palette(k: int, pool: tuple[int, ...] | list[int] = COLORS) -> list[int]:
    colors = list(pool)
    random.shuffle(colors)
    return colors[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    values = [random.choice(palette) for _ in range(int(np.prod(shape)))]
    return np.array(values, dtype=int).reshape(shape)


def _ordered_coords(q: int, mode: str) -> list[tuple[int, int]]:
    coords = [(r, c) for r in range(q) for c in range(q)]
    if mode == "rect":
        r0, c0 = random.randint(0, q - 1), random.randint(0, q - 1)
        return [((r + r0) % q, (c + c0) % q) for r, c in coords]
    if mode == "rows":
        rows = list(range(q))
        cols = list(range(q))
        random.shuffle(rows)
        random.shuffle(cols)
        return [(r, c) for r in rows for c in cols]
    if mode == "compact":
        anchor_r, anchor_c = random.randrange(q), random.randrange(q)
        random.shuffle(coords)
        return sorted(
            coords,
            key=lambda rc: (abs(rc[0] - anchor_r) + abs(rc[1] - anchor_c), random.random()),
        )
    random.shuffle(coords)
    return coords


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control quadrant with exactly `count` non-black cells."""
    assert 0 < count <= q * q
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(
        2, min(max_colors, count, len(pool))
    )
    palette = _palette(k, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    factors = [(hh, ww) for hh in range(1, q + 1) for ww in range(1, q + 1) if hh * ww == count]
    if factors and random.random() < CONTROL_RECT_PROB:
        hh, ww = random.choice(factors)
        r0 = random.randint(0, q - hh)
        c0 = random.randint(0, q - ww)
        tile[r0 : r0 + hh, c0 : c0 + ww] = _random_color_array((hh, ww), palette)
        return tile

    coords = _ordered_coords(q, random.choice(["rect", "compact", "rows"]))[:count]
    for r, c in coords:
        tile[r, c] = random.choice(palette)
    return tile


def _expected_output(fill_color: int, out_h: int, out_w: int) -> np.ndarray:
    return np.full((out_h, out_w), fill_color, dtype=int)


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2

    fill_color = random.choice(COLORS)
    control_pool = [color for color in COLORS if color != fill_color]
    if not control_pool:
        return None

    lo = max(OUTPUT_DIM_MIN, 1)
    hi = min(OUTPUT_DIM_MAX, q * q)
    valid_dims = [
        (out_h, out_w)
        for out_h in range(lo, hi + 1)
        for out_w in range(lo, hi + 1)
        if out_h * out_w <= OUTPUT_AREA_MAX
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)
    fill_tile = np.full((q, q), fill_color, dtype=int)
    stencil = np.zeros((q, q), dtype=int)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    expected = _expected_output(fill_color, out_h, out_w)
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
