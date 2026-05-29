# Variant of f931b4a8
# Dimension: D8 — 象限的划分方式 (meta)
# Change: 输入网格沿中点均分为四个等面积象限（H/2 x W/2），以正方形或偶数尺寸矩形网格为前提... → 三区域布局（上方控制区 + 下方左右两源区）

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color


# Design analysis
# ----------------
# variant_spec:
#   layout:
#     - Keep the overall grid as an even square N x N with N in {4, 8, 16}.
#     - Let q = N / 2.
#     - Rows 0..q-1 are one unified control band of shape q x (2q).
#     - Rows q..N-1 keep the original two source tiles:
#       left q x q = coarse fill tile, right q x q = stencil tile.
#   control_encoding:
#     - The control band is no longer split into upper-left / upper-right quadrants.
#     - Instead it uses two colors mixed across the whole top band.
#     - The color anchored at the top-left corner encodes output height.
#     - The color anchored at the top-right corner encodes output width.
#     - To make the "mixed" layout visually obvious and capacity-stable, height cells
#       occupy only even-parity coordinates of the control band and width cells occupy
#       only odd-parity coordinates. Each parity class has exactly q*q slots, matching
#       the original per-dimension maximum.
#   traced_example:
#     - Original 8x8 logic: upper-left fg count = 6, upper-right fg count = 10, so
#       output is 6 x 10.
#     - Variant 8x8 logic: the top 4x8 control band might contain 6 red cells on even
#       slots and 10 blue cells on odd slots. Red touches (0,0); blue touches (0,7).
#       The solver counts all red cells in the whole top band to get height 6 and all
#       blue cells in the whole top band to get width 10.
#     - Difference: the old solver relied on geometric partition first and counting
#       second; the variant relies on role-marked colors inside one shared region.
#   complexity_check:
#     - Using two role colors avoids invisible rules such as ordering color ids.
#     - Checkerboard slot partition keeps the control area legible: the two roles are
#       interleaved, but never overwrite each other.
#     - Counts still vary over a useful range [3, q*q], so outputs remain diverse and
#       non-square often enough for humans to infer which control color maps to height
#       versus width.
#   implicit_assumption_audit:
#     - `assert h == w and h % 2 == 0`: still required because the lower half remains a
#       pair of q x q source tiles.
#     - `height_color = control[0, 0]`: assumes a non-black anchor at top-left.
#       Generator guarantees this.
#     - `width_color = control[0, -1]`: assumes a distinct non-black anchor at
#       top-right. Generator guarantees this.
#     - `control_colors == {height_color, width_color}`: assumes exactly two control
#       colors. Generator enforces that, so counting is unambiguous.
#     - `fill_r = r // q`, `fill_c = c // q`: unchanged from the original task, but now
#       depends on output dimensions still satisfying out_h <= q*q and out_w <= q*q.
#       Generator keeps those bounds.
#   placement_legality:
#     - Fill tile is all non-black, so every coarse lookup used by stencil holes is
#       valid.
#     - Control cells are assigned to disjoint parity slots, so the two encoded counts
#       never overlap.
#     - Generator forces each control color to appear on both left and right halves of
#       the top band, which makes the 1+2 layout visually different from the original
#       four-quadrant split and strongly reduces accidental agreement with the old
#       solver.
#   visual_observability:
#     - The top band clearly reads as one region because it spans the full width.
#     - The corner anchors make the two roles recoverable without relying on color id.
#     - The checkerboard interleaving is easy to see in 2-3 examples and prevents the
#       top band from collapsing into two hidden sub-rectangles.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count two mixed control colors in the top band to size an output assembled from the lower source tiles.

    Input structure:
    - The grid is an even square.
    - The upper half is one shared control band rather than two separate quadrants.
    - The lower-left half is a non-black fill tile and the lower-right half is a stencil tile or an all-black empty tile.
    - The control color at the top-left corner marks height cells; the control color at the top-right corner marks width cells.

    Rule:
    - Output height is the total number of top-band cells using the top-left anchor color.
    - Output width is the total number of top-band cells using the top-right anchor color.
    - If the stencil is empty, tile the lower-left fill tile directly across the output.
    - Otherwise tile the stencil; whenever the stencil contributes black, replace it with the lower-left fill tile value selected by coarse tile coordinates.

    Key insight:
    - The lower half keeps the original two-level pattern mechanism, but the sizing information is no longer separated spatially into quadrants; it is mixed into one control band and disambiguated by anchored control colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0

    bg = int(Color.BLACK)
    q = h // 2
    control = g[:q, :]
    fill_tile = g[q:, :q]
    stencil = g[q:, q:]

    height_color = int(control[0, 0])
    width_color = int(control[0, -1])
    assert height_color != bg and width_color != bg and height_color != width_color

    control_colors = set(map(int, np.unique(control))) - {bg}
    assert control_colors == {height_color, width_color}

    out_h = int(np.count_nonzero(control == height_color))
    out_w = int(np.count_nonzero(control == width_color))
    assert out_h > 0 and out_w > 0

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


# --------------- generator (o5x) ---------------

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
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 300
CONTROL_MIX_RETRIES = 60


def _palette(k: int, pool=COLORS) -> list[int]:
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


def _ordered_coords(height: int, width: int, mode: str) -> list[tuple[int, int]]:
    coords = [(r, c) for r in range(height) for c in range(width)]
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
        fix_coords = list(required_coords) if required_coords is not None else [
            (r, c) for r in range(q) for c in range(q)
        ]
        random.shuffle(fix_coords)
        for color, (r, c) in zip(missing, fix_coords):
            tile[r, c] = color
    return tile


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
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
    return tile


def _parity_slots(q: int, parity: int) -> list[tuple[int, int]]:
    width = 2 * q
    return [(r, c) for r in range(q) for c in range(width) if (r + c) % 2 == parity]


def _make_control_band(q: int, out_h: int, out_w: int, pool: list[int]) -> np.ndarray | None:
    width = 2 * q
    if len(pool) < 2:
        return None
    if out_h < 3 or out_w < 3 or out_h > q * q or out_w > q * q:
        return None

    height_color, width_color = _palette(2, pool)
    even_slots = _parity_slots(q, 0)
    odd_slots = _parity_slots(q, 1)
    h_anchor = (0, 0)
    w_anchor = (0, width - 1)
    assert h_anchor in even_slots and w_anchor in odd_slots

    even_right = [rc for rc in even_slots if rc != h_anchor and rc[1] >= q]
    odd_left = [rc for rc in odd_slots if rc != w_anchor and rc[1] < q]
    if not even_right or not odd_left:
        return None

    for _ in range(CONTROL_MIX_RETRIES):
        control = np.zeros((q, width), dtype=int)
        control[h_anchor] = height_color
        control[w_anchor] = width_color

        h_cross = random.choice(even_right)
        w_cross = random.choice(odd_left)
        control[h_cross] = height_color
        control[w_cross] = width_color

        h_remaining = out_h - 2
        w_remaining = out_w - 2

        h_pool = [rc for rc in even_slots if rc not in {h_anchor, h_cross}]
        w_pool = [rc for rc in odd_slots if rc not in {w_anchor, w_cross}]
        h_pool_set = set(h_pool)
        w_pool_set = set(w_pool)

        h_mode = random.choice(["compact", "rows", "random"])
        w_mode = random.choice(["compact", "rows", "random"])
        h_order = [rc for rc in _ordered_coords(q, width, h_mode) if rc in h_pool_set]
        w_order = [rc for rc in _ordered_coords(q, width, w_mode) if rc in w_pool_set]

        for r, c in h_order[:h_remaining]:
            control[r, c] = height_color
        for r, c in w_order[:w_remaining]:
            control[r, c] = width_color

        if np.count_nonzero(control == height_color) != out_h:
            continue
        if np.count_nonzero(control == width_color) != out_w:
            continue

        left = int(np.count_nonzero(control[:, :q] != BLACK))
        right = int(np.count_nonzero(control[:, q:] != BLACK))
        if left == out_h and right == out_w:
            continue

        if not np.any(control[:, :q] == height_color):
            continue
        if not np.any(control[:, q:] == height_color):
            continue
        if not np.any(control[:, :q] == width_color):
            continue
        if not np.any(control[:, q:] == width_color):
            continue

        return control

    return None


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q]
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k)
    if stencil_empty:
        stencil_palette = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(*STENCIL_MULTI_COLOR_RANGE)
        stencil_palette = _palette(st_k)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if len(control_pool) < 2:
        return None

    lo = max(OUTPUT_DIM_MIN, q)
    hi = min(OUTPUT_DIM_MAX, q * q)
    valid_dims = [
        (out_h, out_w)
        for out_h in range(lo, hi + 1)
        for out_w in range(lo, hi + 1)
        if out_h + out_w <= 2 * q * q
        if out_h * out_w <= OUTPUT_AREA_MAX
        if stencil_empty or ceil(out_h / q) * ceil(out_w / q) >= ll_k
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    used_fill = [(r, c) for r in range(ceil(out_h / q)) for c in range(ceil(out_w / q))]
    fill_tile = _make_lower_left(q, ll_palette, used_fill)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)
    control = _make_control_band(q, out_h, out_w, control_pool)
    if control is None:
        return None

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :] = control
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
