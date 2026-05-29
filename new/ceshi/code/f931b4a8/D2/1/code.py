# Variant of f931b4a8
# Dimension: D2 — 控制象限的布局位置 (src)
# Change: 左上象限控制输出高度，右上象限控制输出宽度... → 四个象限中两个非对角象限为控制区

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color

# Design analysis for the D2 variant
# ----------------------------------
# variant_spec:
#   preserved_core:
#     - The input is still an even square split into four equal quadrants.
#     - One quadrant provides a full non-black color tile.
#     - One quadrant provides a stencil tile whose black cells are holes.
#     - Output size is still determined by two control quadrants via non-black counts.
#     - Output rendering is unchanged: tile the stencil if present, and fill stencil holes
#       from coarse positions in the color tile; if the stencil is empty, tile the color tile directly.
#   changed_layout:
#     - upper_left: color source tile (formerly lower_left)
#     - upper_right: width control
#     - lower_left: height control (formerly upper_left)
#     - lower_right: stencil tile
#   concrete_behavior_trace:
#     - Example with quadrant size Q=4:
#       upper_left is a full non-black striped tile, upper_right has 5 non-black cells,
#       lower_left has 7 non-black cells, lower_right is a checker stencil.
#       Original task would have read height from upper_left and treated lower_left as colors,
#       which would be impossible here because upper_left is fully non-black and lower_left
#       intentionally contains black background gaps.
#       Variant logic instead reads height=7 from lower_left and width=5 from upper_right,
#       then uses upper_left as the coarse fill tile under the lower_right stencil.
#     - Visible difference:
#       the bottom-left quadrant now looks sparse/background-bearing like a control panel,
#       while the top-left quadrant looks dense/full like a material tile. This is the main clue.
#   complexity_calibration:
#     - Only one semantic swap is introduced, so the rule stays learnable.
#     - The control quadrants still use compact or rectangular foreground layouts; this keeps
#       counts readable instead of visually noisy.
#     - The fill tile remains fully non-black so the stencil-hole substitution remains valid.
#   hidden_assumption_review:
#     - Any code that counts upper_left as the height control is invalid now.
#     - Any code that assumes lower_left has no black pixels is invalid now, because lower_left
#       is now a control quadrant and must contain black background around its markers.
#     - Any code that reads fill colors from lower_left during coarse substitution is invalid;
#       the fill source must be upper_left.
#     - The stencil code remains safe because lower_right keeps the same role.
#   placement_legality:
#     - Control quadrants are generated independently inside their own QxQ areas, so they never
#       overlap anything else.
#     - The fill tile is generated fully non-black, guaranteeing that every coarse lookup used by
#       the solver is legal.
#     - Output sizes are restricted so ceil(out_h / Q) and ceil(out_w / Q) never exceed Q,
#       matching the coarse lookup bounds in the solver.
#   visual_observability:
#     - Training examples will show a consistent relocation of the sparse height-control pattern
#       from upper-left to lower-left.
#     - The top-left region becomes visually dense and color-bearing, so a solver that keeps the
#       original quadrant roles will fail on many examples.
#   cleanliness:
#     - Each quadrant still has exactly one role, with no overlays or internal collisions.
#   color_inference:
#     - Control colors are sampled outside the output palette so they remain clearly irrelevant
#       to the produced image, just like in the original task.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count a right-and-lower pair of control quadrants to size an output made from the opposite-diagonal pattern quadrants.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is the background.
    - The lower-left and upper-right quadrants are the control regions.
    - The upper-left quadrant is a color source tile; the lower-right quadrant is either a stencil tile or empty.

    Rule:
    - Output height is the number of non-background pixels in the lower-left quadrant.
    - Output width is the number of non-background pixels in the upper-right quadrant.
    - If the lower-right stencil has foreground pixels, tile it over the output; each stencil background pixel is filled from the upper-left quadrant cell at the corresponding coarse tile position.
    - If the lower-right stencil is empty, tile the upper-left quadrant directly over the output.

    Key insight:
    - The two control areas are no longer both on the top row: the width cue stays in the upper-right, but the height cue moves to the lower-left, while the opposite diagonal supplies the repeated pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    bg = int(Color.BLACK)
    qh, qw = h // 2, w // 2
    fill_tile = g[:qh, :qw]
    upper_right = g[:qh, qw:]
    lower_left = g[qh:, :qw]
    stencil = g[qh:, qw:]

    out_h = int(np.count_nonzero(lower_left != bg))
    out_w = int(np.count_nonzero(upper_right != bg))
    assert out_h > 0 and out_w > 0

    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != bg):
        for r in range(out_h):
            for c in range(out_w):
                value = int(fill_tile[r % qh, c % qw])
                assert value != bg
                out[r, c] = value
        return out

    for r in range(out_h):
        for c in range(out_w):
            value = int(stencil[r % qh, c % qw])
            if value == bg:
                fill_r = r // qh
                fill_c = c // qw
                assert fill_r < qh and fill_c < qw
                value = int(fill_tile[fill_r, fill_c])
                assert value != bg
            out[r, c] = value

    return out


# --------------- generator (o5x, adapted for D2) ---------------

BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
STENCIL_EMPTY_PROB = 0.16
FILL_SOLID_PROB = 0.70
FILL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.75
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
HEIGHT_CONTROL_MAX_COLORS = 3
WIDTH_CONTROL_MAX_COLORS = 2
OUTPUT_DIM_MIN = 3
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
        return sorted(
            coords,
            key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()),
        )
    random.shuffle(coords)
    return coords


def _make_fill_tile(
    q: int,
    palette: list[int],
    required_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Build the color tile (all non-black) using stripe/solid/diagonal patterns."""
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
    elif mode == "diagonal":
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


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build the stencil tile (black = holes filled from the color tile)."""
    tile = np.zeros((q, q), dtype=int)
    modes = ["checker", "row_stripe", "col_stripe", "random", "random"]
    if q >= STENCIL_FRAME_MIN_Q:
        modes.append("frame")
    mode = random.choice(modes)

    if mode == "checker":
        parity = random.randrange(2)
        mask = (np.arange(q)[:, None] + np.arange(q)[None, :]) % 2 == parity
        tile[mask] = _random_color_array((int(mask.sum()),), palette)
    elif mode == "frame":
        t = random.randint(1, max(1, q // 4))
        mask = np.ones((q, q), dtype=bool)
        mask[t : q - t, t : q - t] = False
        tile[mask] = _random_color_array((int(mask.sum()),), palette)
    elif mode == "row_stripe":
        parity = random.randrange(2)
        for r in range(q):
            if r % 2 == parity:
                tile[r, :] = _random_color_array((q,), palette)
    elif mode == "col_stripe":
        parity = random.randrange(2)
        for c in range(q):
            if c % 2 == parity:
                tile[:, c] = _random_color_array((q,), palette)
    else:
        area = q * q
        lo = max(1, int(ceil(STENCIL_FG_FRAC[0] * area)))
        hi = max(lo, int(STENCIL_FG_FRAC[1] * area))
        fg_count = random.randint(lo, hi)
        coords = [(r, c) for r in range(q) for c in range(q)]
        random.shuffle(coords)
        for r, c in coords[:fg_count]:
            tile[r, c] = random.choice(palette)
    return tile


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control quadrant with exactly `count` non-black pixels."""
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(
        2, min(max_colors, count, len(pool))
    )
    pal = _palette(k, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    factors = [(h, w) for h in range(1, q + 1) for w in range(1, q + 1) if h * w == count]
    if factors and random.random() < CONTROL_RECT_PROB:
        h, w = random.choice(factors)
        r0 = random.randint(0, q - h)
        c0 = random.randint(0, q - w)
        tile[r0 : r0 + h, c0 : c0 + w] = _random_color_array((h, w), pal)
    else:
        mode = random.choice(["rect", "compact", "rows"])
        coords = _ordered_coords(q, mode)[:count]
        for r, c in coords:
            tile[r, c] = random.choice(pal)
    return tile


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Reproduce the solver's tiling logic for oracle validation."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q]
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid grid; return None on failure."""
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    fill_k = 1 if random.random() < FILL_SOLID_PROB else random.randint(*FILL_MULTI_COLOR_RANGE)
    fill_palette = _palette(fill_k)
    if stencil_empty:
        stencil_palette = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(
            *STENCIL_MULTI_COLOR_RANGE
        )
        stencil_palette = _palette(st_k)

    output_colors = set(fill_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    lo = max(OUTPUT_DIM_MIN, q)
    hi = min(OUTPUT_DIM_MAX, q * q)
    valid_dims = [
        (h, w)
        for h in range(lo, hi + 1)
        for w in range(lo, hi + 1)
        if h * w <= OUTPUT_AREA_MAX and (stencil_empty or ceil(h / q) * ceil(w / q) >= fill_k)
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    used_fill = [(r, c) for r in range(ceil(out_h / q)) for c in range(ceil(out_w / q))]
    upper_left = _make_fill_tile(q, fill_palette, used_fill)
    upper_right = _make_control(q, out_w, WIDTH_CONTROL_MAX_COLORS, control_pool)
    lower_left = _make_control(q, out_h, HEIGHT_CONTROL_MAX_COLORS, control_pool)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = lower_left
    grid[q:, q:] = stencil

    expected = _expected_output(upper_left, stencil, out_h, out_w)
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
