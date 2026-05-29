# Variant of f931b4a8
# Dimension: D10 — 控制象限前景的空间排列 (meta)
# Change: 控制象限的前景像素可以是任意排列（均匀填充、稀疏散布等），仅计数有意义，空间位置无额外语义
#   -> 前景颜色多样化，不同颜色分别控制不同参数

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color


# Design analysis for the D10 variant
# ----------------------------------
# a) Concrete behavior:
#    The four-quadrant layout stays the same. The lower half keeps the original
#    "fill tile + stencil tile" mechanism. The change is only in the two upper
#    control quadrants: each one now contains exactly two non-black colors.
#    Inside one control quadrant, the color with the larger count controls the
#    output size on that axis, while the color with the smaller count controls a
#    phase offset for the repeated lower pattern on that axis.
#    Small example with q=4:
#      UL has 6 red cells and 2 yellow cells  -> output height = 6, row phase = 2
#      UR has 5 green cells and 1 blue cell   -> output width  = 5, col phase = 1
#    Original logic would have read height=8 and width=6 by counting all
#    non-black cells. The new logic separates the two colors and uses them for
#    different parameters.
#
# b) Complexity calibration:
#    The rule uses only two roles per control quadrant (size and phase), so the
#    new freedom is visible but still learnable. Requiring exactly two colors and
#    unequal counts removes ambiguity. The phase is kept in 1..q-1 so it is
#    small, repeated, and easy to notice across samples.
#
# c) Old assumptions that must change:
#    - np.count_nonzero(control != bg) no longer gives the dimension, because the
#      smaller color must not be added into the size count.
#    - The old solver ignored control colors entirely. That is no longer valid:
#      main() must count per color and distinguish larger-count vs smaller-count.
#    - The lower tiling used zero phase. That must change so the secondary color
#      actually influences the output, not just the interpretation of the top.
#
# d) Placement legality:
#    The generator ensures each control quadrant has exactly two colors, distinct
#    counts, and enough cells for both counts: dim_count + phase_count <= q*q.
#    Lower-left remains all non-black. Lower-right may still be empty or non-empty.
#    All placements stay inside their quadrants; no cross-quadrant overlap exists.
#
# e) Visual observability:
#    The output size now visibly matches only the more frequent control color, not
#    the total foreground count. Additionally, the minority color shifts the lower
#    repeated pattern. The generator rejects samples whose output would be
#    identical to the zero-phase version, guaranteeing that the phase color has a
#    visible effect in the final grid.
#
# f) Visual cleanliness:
#    The variant does not add extra objects or overlays. It only recolors the
#    control pixels, so the grid remains easy to parse: top = controls, bottom =
#    pattern source/stencil, output = repeated pattern.
#
# g) Color inference:
#    No role depends on numeric color ids. The solver uses only visible structure:
#    in each control quadrant, the more frequent color is the size-controller and
#    the less frequent color is the phase-controller.


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
CONTROL_RECT_PROB = 0.85
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 400


def _palette(k: int, pool=COLORS) -> list[int]:
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    return np.array([random.choice(palette) for _ in range(int(np.prod(shape)))], dtype=int).reshape(shape)


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
        return sorted(coords, key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()))
    random.shuffle(coords)
    return coords


def _choose_subset(coords: list[tuple[int, int]], q: int, count: int, mode: str) -> list[tuple[int, int]]:
    coords = list(coords)
    if mode == "compact":
        ar, ac = random.choice(coords)
        random.shuffle(coords)
        coords.sort(key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()))
    elif mode == "rows":
        rows, cols = list(range(q)), list(range(q))
        random.shuffle(rows)
        random.shuffle(cols)
        rank = {(r, c): i for i, (r, c) in enumerate((rc for r in rows for c in cols for rc in [(r, c)]))}
        coords.sort(key=lambda rc: rank[rc])
    else:
        random.shuffle(coords)
    return coords[:count]


def _make_lower_left(q: int, palette: list[int], required_coords: list[tuple[int, int]] | None = None) -> np.ndarray:
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
        fix_coords = list(required_coords) if required_coords is not None else [(r, c) for r in range(q) for c in range(q)]
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


def _make_control(q: int, dim_count: int, phase_count: int, pool: list[int]) -> np.ndarray:
    assert 1 <= phase_count < dim_count
    assert dim_count + phase_count <= q * q

    dim_color, phase_color = random.sample(pool, 2)
    tile = np.full((q, q), BLACK, dtype=int)
    total = dim_count + phase_count

    factors = [(h, w) for h in range(1, q + 1) for w in range(1, q + 1) if h * w == total]
    if factors and random.random() < CONTROL_RECT_PROB:
        hh, ww = random.choice(factors)
        r0 = random.randint(0, q - hh)
        c0 = random.randint(0, q - ww)
        active_coords = [(r, c) for r in range(r0, r0 + hh) for c in range(c0, c0 + ww)]
    else:
        active_coords = _ordered_coords(q, random.choice(["rect", "compact", "rows", "random"]))[:total]

    for r, c in active_coords:
        tile[r, c] = dim_color

    accent_coords = _choose_subset(active_coords, q, phase_count, random.choice(["compact", "rows", "random"]))
    for r, c in accent_coords:
        tile[r, c] = phase_color
    return tile


def _decode_control(control: np.ndarray, bg: int) -> tuple[int, int]:
    fg = control[control != bg]
    colors, counts = np.unique(fg, return_counts=True)
    assert len(colors) == 2
    count_a, count_b = int(counts[0]), int(counts[1])
    assert count_a != count_b
    dim_count = max(count_a, count_b)
    phase_count = min(count_a, count_b)
    return dim_count, phase_count


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int, row_phase: int, col_phase: int) -> np.ndarray:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    sr = (rr + row_phase) % q
    sc = (cc + col_phase) % q
    if not np.any(stencil != BLACK):
        return fill_tile[sr, sc]
    tiled = stencil[sr, sc].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _zero_phase_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    return _expected_output(fill_tile, stencil, out_h, out_w, 0, 0)


def _legacy_output(fill_tile: np.ndarray, stencil: np.ndarray, upper_left: np.ndarray, upper_right: np.ndarray) -> np.ndarray:
    out_h = int(np.count_nonzero(upper_left != BLACK))
    out_w = int(np.count_nonzero(upper_right != BLACK))
    return _expected_output(fill_tile, stencil, out_h, out_w, 0, 0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode two-color control quadrants, then phase-shift the lower pattern to build the output.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is the background.
    - The two upper quadrants are controls, and each one contains exactly two non-black colors.
    - The lower-left quadrant is a non-black fill tile; the lower-right quadrant is a stencil tile or an empty black tile.

    Rule:
    - In each upper control quadrant, the more frequent non-black color sets the size on that axis.
    - In each upper control quadrant, the less frequent non-black color sets the phase offset on that axis.
    - The lower pattern is then tiled as in the original task, except the repeated stencil/fill pattern is shifted by the decoded row and column phases.
    - If the lower-right stencil is empty, the shifted lower-left tile is repeated directly.

    Key insight:
    - The top half no longer encodes only "how many foreground pixels"; it encodes two parameters per axis by splitting the foreground into a dominant color for size and a minority color for phase.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    bg = int(Color.BLACK)
    qh, qw = h // 2, w // 2
    assert qh == qw
    upper_left = g[:qh, :qw]
    upper_right = g[:qh, qw:]
    fill_tile = g[qh:, :qw]
    stencil = g[qh:, qw:]

    out_h, row_phase = _decode_control(upper_left, bg)
    out_w, col_phase = _decode_control(upper_right, bg)
    assert out_h > 0 and out_w > 0
    assert np.all(fill_tile != bg)

    return _expected_output(fill_tile, stencil, out_h, out_w, row_phase, col_phase)


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    row_phase = random.randint(1, q - 1)
    col_phase = random.randint(1, q - 1)

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    if stencil_empty:
        ll_k = max(2, ll_k)
    ll_palette = _palette(ll_k)

    if stencil_empty:
        stencil_palette: list[int] = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(*STENCIL_MULTI_COLOR_RANGE)
        stencil_palette = _palette(st_k)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if len(control_pool) < 2:
        return None

    lo = max(OUTPUT_DIM_MIN, q)
    max_h = min(OUTPUT_DIM_MAX, q * q - row_phase)
    max_w = min(OUTPUT_DIM_MAX, q * q - col_phase)
    valid_dims = [
        (hh, ww)
        for hh in range(lo, max_h + 1)
        for ww in range(lo, max_w + 1)
        if hh * ww <= OUTPUT_AREA_MAX and (stencil_empty or ceil(hh / q) * ceil(ww / q) >= ll_k)
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    used_fill = [(r, c) for r in range(ceil(out_h / q)) for c in range(ceil(out_w / q))]
    fill_tile = _make_lower_left(q, ll_palette, used_fill)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)
    upper_left = _make_control(q, out_h, row_phase, control_pool)
    upper_right = _make_control(q, out_w, col_phase, control_pool)

    expected = _expected_output(fill_tile, stencil, out_h, out_w, row_phase, col_phase)
    if np.array_equal(expected, _zero_phase_output(fill_tile, stencil, out_h, out_w)):
        return None

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

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
