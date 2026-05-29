# Combo variant of f931b4a8 (index 23)
# combo_spec: D1/1+D10/1
# Name: 多色控制通道+颜色值编码尺寸

from __future__ import annotations

import random
import sys
from math import ceil
from pathlib import Path

import numpy as np

try:
    from arc_utils import Color
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
    from arc_utils import Color


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    The lower half keeps the original two-level pattern system:
#    lower-left is the coarse color tile and lower-right is the repeated stencil.
#    The change is entirely in the two upper control quadrants. They now use the
#    same two non-black colors in both quadrants.
#      - In the upper-left quadrant, one color is the local majority.
#      - In the upper-right quadrant, the other color is the local majority.
#    The dominant color value itself gives the output size on that axis:
#      Example with q=4:
#        UL has seven cells of color 6 and two cells of color 4.
#        UR has one cell of color 6 and eight cells of color 4.
#        New rule: output height = 6, output width = 4.
#        Old rule: output height = 9, output width = 9 by counting foreground.
#    So the solver must separate the two visible color channels and read the
#    winning color id as a number instead of summing pixels.
#
# b) Complexity calibration:
#    The control logic uses exactly two colors across the whole top half. That is
#    enough to make "different colors control different parameters" visually
#    explicit without introducing many arbitrary mappings. Reusing the same pair
#    on both sides creates repeated evidence: one color is emphasized on the left
#    and the other on the right. Counts still matter only enough to identify the
#    dominant channel, not as the final numeric size.
#
# c) Hidden assumptions that change:
#    - `np.count_nonzero(upper_left != bg)` and the analogous width line are no
#      longer valid; total foreground count is intentionally decoupled from size.
#    - The original solver ignored control colors entirely. The combo solver must
#      now inspect per-color counts and also require that both control quadrants
#      share the same two-color alphabet with flipped dominance.
#    - The original vectorized coarse-fill logic implicitly relied on
#      `out_h <= q*q` and `out_w <= q*q` whenever stencil holes are present.
#      D1 breaks that for q=2 if we allow colors 5..9, so the generator must
#      avoid hole-using cases there. The rendering helper mirrors the original
#      loop structure so it only touches coarse indices when a stencil hole is
#      actually encountered.
#
# d) Placement legality:
#    Each upper quadrant is filled independently inside its own qxq box. The
#    generator builds a set of active cells, paints them with the dominant color,
#    then recolors a strict minority subset with the secondary color, so there is
#    no overlap ambiguity. Lower-left stays fully non-black. When stencil holes
#    may occur, dimensions are restricted so every coarse lookup stays inside the
#    qxq fill tile; for q=2 we force an empty stencil instead.
#
# e) Visual observability:
#    The generator rejects the original count-based reading by construction:
#    each control quadrant uses two colors, and its total foreground count is
#    chosen to differ from both size-encoding color values. Across examples, the
#    same visual motif repeats: left-majority color id = output height and
#    right-majority color id = output width. A human can see that the counts vary
#    while the output dimensions track color labels instead.
#
# f) Visual cleanliness:
#    No extra overlays are added. The picture still reads as "top = controls,
#    bottom = pattern source". The top remains easy to parse because only two
#    non-black colors appear there, and the same pair is reused on both sides.
#
# g) Change inferability:
#    The only color-to-number mapping used is the ARC palette id itself, which is
#    recoverable from the examples because output sizes repeat the dominant color
#    labels directly. There are only two control colors per task instance, both
#    repeated in both control quadrants, so the mapping is heavily constrained
#    rather than an arbitrary many-color relabeling.


BLACK = 0
COLORS = tuple(range(1, 10))
CONTROL_VALUE_COLORS = tuple(range(3, 10))

GRID_SIZES = (4, 8, 16)
STENCIL_EMPTY_PROB = 0.16
LL_SOLID_PROB = 0.70
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.75
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
CONTROL_RECT_PROB = 0.85
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


def _pick_control_total(q: int, banned: set[int]) -> int | None:
    candidates = [count for count in range(3, q * q + 1) if count not in banned]
    if not candidates:
        return None
    return random.choice(candidates)


def _make_control(q: int, dominant_color: int, secondary_color: int, banned_totals: set[int]) -> np.ndarray | None:
    total = _pick_control_total(q, banned_totals)
    if total is None:
        return None

    minority_max = (total - 1) // 2
    assert minority_max >= 1
    minority_min = 1 if minority_max == 1 else 2
    minority_count = random.randint(minority_min, minority_max)

    tile = np.full((q, q), BLACK, dtype=int)
    factors = [(hh, ww) for hh in range(1, q + 1) for ww in range(1, q + 1) if hh * ww == total]
    if factors and random.random() < CONTROL_RECT_PROB:
        hh, ww = random.choice(factors)
        r0 = random.randint(0, q - hh)
        c0 = random.randint(0, q - ww)
        active_coords = [(r, c) for r in range(r0, r0 + hh) for c in range(c0, c0 + ww)]
    else:
        active_coords = _ordered_coords(q, random.choice(["rect", "compact", "rows", "random"]))[:total]

    for r, c in active_coords:
        tile[r, c] = dominant_color

    accent_coords = _choose_subset(active_coords, q, minority_count, random.choice(["compact", "rows", "random"]))
    for r, c in accent_coords:
        tile[r, c] = secondary_color
    return tile


def _control_summary(control: np.ndarray, bg: int) -> tuple[int, int, int]:
    fg = control[control != bg]
    colors, counts = np.unique(fg, return_counts=True)
    assert len(colors) == 2
    order = np.argsort(counts)
    low = int(order[0])
    high = int(order[-1])
    assert int(counts[high]) > int(counts[low])
    return int(colors[high]), int(colors[low]), int(counts.sum())


def _decode_dimensions(upper_left: np.ndarray, upper_right: np.ndarray, bg: int) -> tuple[int, int]:
    left_dominant, left_secondary, _ = _control_summary(upper_left, bg)
    right_dominant, right_secondary, _ = _control_summary(upper_right, bg)
    assert {left_dominant, left_secondary} == {right_dominant, right_secondary}
    assert left_dominant == right_secondary
    assert right_dominant == left_secondary
    return left_dominant, right_dominant


def _render_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int, bg: int) -> np.ndarray:
    q = fill_tile.shape[0]
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


def _legacy_output(fill_tile: np.ndarray, stencil: np.ndarray, upper_left: np.ndarray, upper_right: np.ndarray) -> np.ndarray:
    out_h = int(np.count_nonzero(upper_left != BLACK))
    out_w = int(np.count_nonzero(upper_right != BLACK))
    return _render_output(fill_tile, stencil, out_h, out_w, BLACK)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the dominant color channels in the top half, then build the output with the original lower-half tiling rule.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is the background.
    - The two upper quadrants are control regions and both use the same two non-black colors.
    - In each upper quadrant one of those two colors is the majority, while the lower-left quadrant is an all-non-black fill tile and the lower-right quadrant is a stencil tile or an empty black tile.

    Rule:
    - The majority color value in the upper-left quadrant gives the output height.
    - The majority color value in the upper-right quadrant gives the output width.
    - The minority color in each control quadrant is the other axis's controller, so the same two colors cross-reference each other across the whole top half.
    - The output content is produced exactly like the original task: tile the lower-right stencil, and whenever a tiled stencil cell is black, replace it with the lower-left color at the corresponding coarse block coordinate; if the stencil is empty, tile the lower-left quadrant directly.

    Key insight:
    - The upper half no longer says "count these foreground pixels"; it says "separate the two control colors, identify which color wins on each side, and read that color id itself as the row or column count."
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

    out_h, out_w = _decode_dimensions(upper_left, upper_right, bg)
    assert out_h > 0 and out_w > 0
    assert np.all(fill_tile != bg)

    return _render_output(fill_tile, stencil, out_h, out_w, bg)


def _eligible_control_values(output_colors: set[int], q: int, stencil_empty: bool) -> list[int]:
    values = [color for color in CONTROL_VALUE_COLORS if color not in output_colors]
    if not stencil_empty:
        values = [color for color in values if color <= q * q]
    if q == 2 and stencil_empty:
        values = [color for color in values if color >= 5]
    return values


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = q == 2 or random.random() < STENCIL_EMPTY_PROB

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k)

    if stencil_empty:
        stencil_palette: list[int] = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(*STENCIL_MULTI_COLOR_RANGE)
        stencil_palette = _palette(st_k)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_values = _eligible_control_values(output_colors, q, stencil_empty)
    if len(control_values) < 2:
        return None

    out_h, out_w = random.sample(control_values, 2)
    if out_h * out_w > OUTPUT_AREA_MAX:
        return None

    if not stencil_empty and (out_h > q * q or out_w > q * q):
        return None

    required_fill = None
    if not stencil_empty:
        required_fill = [(r, c) for r in range(ceil(out_h / q)) for c in range(ceil(out_w / q))]
        if required_fill and (max(r for r, _ in required_fill) >= q or max(c for _, c in required_fill) >= q):
            return None

    fill_tile = _make_lower_left(q, ll_palette, required_fill)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)

    banned_totals = {out_h, out_w}
    upper_left = _make_control(q, out_h, out_w, banned_totals)
    upper_right = _make_control(q, out_w, out_h, banned_totals)
    if upper_left is None or upper_right is None:
        return None

    expected = _render_output(fill_tile, stencil, out_h, out_w, BLACK)
    legacy = _legacy_output(fill_tile, stencil, upper_left, upper_right)
    if np.array_equal(expected, legacy):
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
