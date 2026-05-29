# Combo variant of f931b4a8 (index 22)
# combo_spec: D7/1+D15/0
# Name: 模板倍数输出+整数倍约束
#
# Design analysis
# a) Concrete behavior and trace:
#    - The input is still a square split into four quadrants of size q x q.
#    - The original task reads the upper counts as literal output height/width in pixels.
#      Here those same counts become repeat counts of whole template blocks.
#    - Small trace with q = 4, UL count = 3, UR count = 2:
#      original -> output shape = 3 x 2, so the lower-right stencil is only cropped at its top-left corner,
#      and any black holes can only read fill_tile coarse coords among the tiny set touched by that crop.
#      combo    -> output shape = (3 * 4) x (2 * 4) = 12 x 8, so the stencil repeats as a 3-by-2 array of
#      complete q x q blocks, and each black hole inside block (br, bc) is filled from fill_tile[br, bc].
#      difference -> the top counts now say "how many full blocks" rather than "how many output pixels".
#    - D15/0 is not implemented as a later rounding pass: once counts are interpreted as repeat counts,
#      the output is automatically an integer multiple of the stencil size, so only full blocks appear.
# b) Complexity calibration:
#    - Repeat counts stay in 1..q so fill_tile[block_r, block_c] always exists and the output stays <= 256 cells.
#    - The generator rejects the trivial (1, 1) repeat pair so examples usually show multiple full template units.
#    - The generator also rejects one-color outputs because they hide the coarse/fine two-level structure.
# c) Implicit assumption review:
#    - `count_nonzero(...)` used to mean direct output dimensions; it now means repeat counts.
#    - `fill_r = r // q` and `fill_c = c // q` now intentionally index template-block coordinates,
#      so the solver requires repeat_h <= q and repeat_w <= q; the generator enforces that.
#    - The empty-stencil branch still tiles lower-left directly, but now over a size that is guaranteed
#      to be a whole-number multiple of q in both dimensions.
# d) Placement validity:
#    - The four-quadrant layout is unchanged, so there is no new overlap risk.
#    - Control quadrants only need exact foreground counts, still using colors excluded from the output palette.
#    - Lower-left remains fully non-black, ensuring every coarse fill lookup is valid.
# e) Visual observability:
#    - Because outputs are full q x q blocks, the block lattice is directly visible instead of being truncated.
#    - When the stencil has black holes, neighboring template blocks can show different hole colors from lower-left,
#      which makes the block-count interpretation visible across only a few training examples.
# f) Visual cleanliness:
#    - Each quadrant keeps one role: counters on top, coarse color source bottom-left, fine stencil bottom-right.
#    - No extra markers or mixed encodings are added.
# g) Change inferability:
#    - There is no hidden color-id mapping. Colors only serve as visible texture and fill sources.
#    - The main inferable change is geometric: each extra top control pixel adds one more whole template block.

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat the top quadrants as counters for how many complete template blocks to place.

    Input structure:
    - The grid is an even square split into four equal quadrants of side length `q`.
    - Black is the background.
    - The upper-left and upper-right quadrants are control regions; only their foreground counts matter.
    - The lower-left quadrant is a non-black coarse fill tile, and the lower-right quadrant is either a stencil tile or an all-black empty tile.

    Rule:
    - Let `repeat_h` be the foreground count in the upper-left quadrant and `repeat_w` the count in the upper-right quadrant.
    - The output size is `(repeat_h * q, repeat_w * q)`, so the output always contains full qxq template copies.
    - If the lower-right quadrant is empty, tile the lower-left quadrant directly over the output.
    - Otherwise, tile the lower-right stencil; whenever a stencil cell is black, replace it with the lower-left value at coarse block coordinate `(r // q, c // q)`.

    Key insight:
    - The top counts no longer encode literal pixel dimensions. They encode how many whole stencil blocks to repeat, which simultaneously explains why the output dimensions are always integer multiples of the template size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    bg = int(Color.BLACK)
    q = h // 2
    upper_left = g[:q, :q]
    upper_right = g[:q, q:]
    fill_tile = g[q:, :q]
    stencil = g[q:, q:]

    repeat_h = int(np.count_nonzero(upper_left != bg))
    repeat_w = int(np.count_nonzero(upper_right != bg))
    assert 0 < repeat_h <= q and 0 < repeat_w <= q

    out_h = repeat_h * q
    out_w = repeat_w * q
    out = np.empty((out_h, out_w), dtype=int)

    if not np.any(stencil != bg):
        for r in range(out_h):
            for c in range(out_w):
                value = int(fill_tile[r % q, c % q])
                assert value != bg
                out[r, c] = value
        return out

    for r in range(out_h):
        for c in range(out_w):
            value = int(stencil[r % q, c % q])
            if value == bg:
                fill_r = r // q
                fill_c = c // q
                assert fill_r < q and fill_c < q
                value = int(fill_tile[fill_r, fill_c])
                assert value != bg
            out[r, c] = value

    return out


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
STENCIL_EMPTY_PROB = 0.16
LL_SOLID_PROB = 0.70
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.75
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
REPEAT_MIN = 1
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 400


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
        anchor_r, anchor_c = random.randrange(q), random.randrange(q)
        random.shuffle(coords)
        return sorted(
            coords,
            key=lambda rc: (abs(rc[0] - anchor_r) + abs(rc[1] - anchor_c), random.random()),
        )
    random.shuffle(coords)
    return coords


def _make_lower_left(
    q: int,
    palette: list[int],
    required_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Build the fill tile (all non-black) using simple repeated patterns."""
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
    missing = [color for color in palette if color not in existing]
    if missing:
        fix_coords = list(required_coords) if required_coords is not None else [
            (r, c) for r in range(q) for c in range(q)
        ]
        random.shuffle(fix_coords)
        for color, (r, c) in zip(missing, fix_coords):
            tile[r, c] = color
    return tile


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build the stencil tile (black cells are holes later filled from lower-left)."""
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
        mask[t:q - t, t:q - t] = False
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

    assert np.any(tile == BLACK)
    assert np.any(tile != BLACK)
    return tile


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control quadrant with exactly `count` non-black pixels."""
    max_k = min(max_colors, count, len(pool))
    if max_k == 1 or random.random() < CONTROL_SINGLE_COLOR_PROB:
        k = 1
    else:
        k = random.randint(2, max_k)
    palette = _palette(k, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    factors = [(h, w) for h in range(1, q + 1) for w in range(1, q + 1) if h * w == count]
    if factors and random.random() < CONTROL_RECT_PROB:
        h, w = random.choice(factors)
        r0 = random.randint(0, q - h)
        c0 = random.randint(0, q - w)
        tile[r0 : r0 + h, c0 : c0 + w] = _random_color_array((h, w), palette)
    else:
        mode = random.choice(["rect", "compact", "rows"])
        for r, c in _ordered_coords(q, mode)[:count]:
            tile[r, c] = random.choice(palette)
    return tile


def _expected_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    repeat_h: int,
    repeat_w: int,
) -> np.ndarray:
    """Mirror `main` for generator-side validation."""
    q = fill_tile.shape[0]
    out_h, out_w = repeat_h * q, repeat_w * q
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _valid_repeats(q: int, ll_k: int) -> list[tuple[int, int]]:
    """Enumerate repeat counts that fit the area cap and expose enough coarse cells."""
    pairs = []
    for repeat_h in range(REPEAT_MIN, q + 1):
        for repeat_w in range(REPEAT_MIN, q + 1):
            out_h, out_w = repeat_h * q, repeat_w * q
            if out_h * out_w > OUTPUT_AREA_MAX:
                continue
            if repeat_h * repeat_w < ll_k:
                continue
            pairs.append((repeat_h, repeat_w))
    return pairs


def _is_observable(
    out: np.ndarray,
    q: int,
    repeat_h: int,
    repeat_w: int,
    stencil_empty: bool,
    fill_tile: np.ndarray,
) -> bool:
    """Reject samples that visually collapse the fused rule."""
    if (repeat_h, repeat_w) == (1, 1):
        return False
    if out.shape != (repeat_h * q, repeat_w * q):
        return False
    if out.shape[0] % q != 0 or out.shape[1] % q != 0:
        return False
    if len(set(map(int, np.unique(out)))) < 2:
        return False
    if stencil_empty and np.array_equal(out, np.full(out.shape, out[0, 0], dtype=int)):
        return False
    if not stencil_empty and repeat_h * repeat_w > 1:
        block00 = out[:q, :q]
        all_same_blocks = True
        for br in range(repeat_h):
            for bc in range(repeat_w):
                block = out[br * q : (br + 1) * q, bc * q : (bc + 1) * q]
                if not np.array_equal(block, block00):
                    all_same_blocks = False
                    break
            if not all_same_blocks:
                break
        if all_same_blocks and len(set(map(int, np.unique(fill_tile[:repeat_h, :repeat_w])))) > 1:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid grid; return None on failure."""
    n = random.choice(GRID_SIZES)
    q = n // 2
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
    control_pool = [color for color in COLORS if color not in output_colors]
    if not control_pool:
        return None

    valid_repeats = _valid_repeats(q, ll_k)
    if not valid_repeats:
        return None
    preferred_repeats = [pair for pair in valid_repeats if pair != (1, 1) and max(pair) >= 2]
    repeat_h, repeat_w = random.choice(preferred_repeats or valid_repeats)

    used_fill = [(r, c) for r in range(repeat_h) for c in range(repeat_w)]
    fill_tile = _make_lower_left(q, ll_palette, used_fill)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)
    upper_left = _make_control(q, repeat_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, repeat_w, UR_MAX_COLORS, control_pool)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    expected = _expected_output(fill_tile, stencil, repeat_h, repeat_w)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _is_observable(actual, q, repeat_h, repeat_w, stencil_empty, fill_tile):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
