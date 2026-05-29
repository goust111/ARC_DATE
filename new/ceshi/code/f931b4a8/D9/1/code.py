# Variant of f931b4a8
# Dimension: D9 — 前景/背景的判定 (match_cond)
# Change: 零值（颜色0）为背景，非零值为前景；控制象限中非零像素被计数，模板中零值位置被颜色源替换... → 控制象限中特定颜色为前景标记

from __future__ import annotations

from math import ceil
import random

import numpy as np

from arc_utils import Color


# Design analysis for the D9 variant
# a) Concrete behavior and trace
#    - The 4-quadrant geometry stays intact: UL controls output height, UR controls output width,
#      LL is the coarse fill tile, LR is the repeated stencil.
#    - The change is only in "what counts" inside the two control quadrants:
#      UL counts only the color that also appears in LL.
#      UR counts only the color that also appears in LR's foreground palette; when LR is empty,
#      UR falls back to counting the color that also appears in LL.
#    - Each control quadrant also contains extra non-black distractor pixels whose colors do not
#      appear in the relevant lower quadrant, so the visible non-black total is larger than the
#      effective count.
#    - Small trace:
#      LL colors = {6, 1}, LR colors = {4, 7}. If UL has five 6s plus two 9s, old logic gives
#      out_h = 7 while the variant gives out_h = 5. If UR has four 4s plus three 3s, old logic
#      gives out_w = 7 while the variant gives out_w = 4. The bottom-half tiling step is unchanged.
# b) Complexity calibration
#    - The marker rule is intentionally single-step: one marker color per control quadrant.
#    - Distractors are mandatory, so the variant is visually observable and does not collapse back
#      to the original "count every non-black" behavior.
#    - When LR is non-empty, LL and LR palettes are forced disjoint so the two marker sources stay
#      easy to separate by eye.
# c) Original assumptions that must change
#    - `np.count_nonzero(upper_left != bg)` and `np.count_nonzero(upper_right != bg)` relied on
#      "all non-black control pixels are meaningful". That is false in this variant.
#    - The bottom-half logic (`stencil == bg` means hole, LL has no black, coarse lookup by
#      `r // q`, `c // q`) does not rely on the old control rule and remains valid.
# d) Placement legality
#    - Output dimensions are capped at `q*q - 1`, guaranteeing at least one spare control cell for
#      a distractor color in each top quadrant.
#    - Control markers and distractors occupy disjoint cells inside the quadrant, and LL still has
#      no black pixels, so all coarse substitutions are legal.
# e) Visual observability
#    - Every sample contains distractor pixels in both control quadrants.
#    - The countable color is echoed below: UL shares one color with LL; UR shares one color with
#      LR (or LL when LR is empty). Neutral distractor colors are absent from the relevant source.
# f) Visual cleanliness
#    - Control regions remain simple compact/rectangular clusters rather than noisy scatter.
#    - Distractors use a small neutral palette and do not overlap lower-quadrant palettes.
# g) Color inference
#    - No hidden ordering or color-id arithmetic is used. The only cue is visible palette overlap
#      between an upper control quadrant and its relevant lower quadrant.
# h) Generator anti-degeneracy metric
#    - For every generated sample, `nonzero_count(control) - marker_count(control) >= 1` for both
#      controls, so the variant's marker rule changes the inferred dimensions.
# i) New solver assumptions to enforce
#    - UL must share exactly one non-black color with LL.
#    - UR must share exactly one non-black color with LR's foreground palette, or with LL if LR is
#      empty.
#    - The generator explicitly constructs those unique overlaps.


BLACK = int(Color.BLACK)
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
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_DISTRACTOR_MAX_COLORS = 2
CONTROL_DISTRACTOR_MAX_CELLS = 4
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
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
        ar, ac = random.randrange(q), random.randrange(q)
        random.shuffle(coords)
        return sorted(
            coords,
            key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()),
        )
    random.shuffle(coords)
    return coords


def _make_lower_left(
    q: int,
    palette: list[int],
    required_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Build the fill tile (all non-black) using stripe/solid/diagonal patterns."""
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
    """Build the stencil tile (black = holes filled from fill tile)."""
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


def _pick_marker_color(control: np.ndarray, source_colors: set[int]) -> int:
    """Find the unique control color that is echoed in the relevant lower quadrant."""
    control_colors = {int(v) for v in np.unique(control) if int(v) != BLACK}
    shared = sorted(control_colors & source_colors)
    assert len(shared) == 1
    return int(shared[0])


def _place_marker_coords(q: int, count: int) -> list[tuple[int, int]]:
    """Choose exactly `count` cells for the marker, usually as a rectangle."""
    factors = [(h, w) for h in range(1, q + 1) for w in range(1, q + 1) if h * w == count]
    if factors and random.random() < CONTROL_RECT_PROB:
        h, w = random.choice(factors)
        r0 = random.randint(0, q - h)
        c0 = random.randint(0, q - w)
        return [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]
    mode = random.choice(["rect", "compact", "rows"])
    return _ordered_coords(q, mode)[:count]


def _make_control(q: int, marker_color: int, count: int, distractor_pool: list[int]) -> np.ndarray | None:
    """Build a control quadrant where only `marker_color` should be counted."""
    area = q * q
    if not (1 <= count < area):
        return None
    if not distractor_pool:
        return None

    tile = np.full((q, q), BLACK, dtype=int)
    marker_coords = _place_marker_coords(q, count)
    for r, c in marker_coords:
        tile[r, c] = marker_color

    remaining = [(r, c) for r in range(q) for c in range(q) if tile[r, c] == BLACK]
    if not remaining:
        return None

    distractor_count = random.randint(1, min(len(remaining), CONTROL_DISTRACTOR_MAX_CELLS))
    if len(distractor_pool) == 1 or distractor_count == 1 or random.random() < CONTROL_SINGLE_COLOR_PROB:
        distractor_palette = [random.choice(distractor_pool)]
    else:
        k = min(CONTROL_DISTRACTOR_MAX_COLORS, distractor_count, len(distractor_pool))
        distractor_palette = [random.choice(distractor_pool)] if k == 1 else _palette(
            random.randint(2, k), distractor_pool
        )

    coords = _ordered_coords(q, random.choice(["compact", "rows", "rect"]))
    chosen = [(r, c) for r, c in coords if tile[r, c] == BLACK][:distractor_count]
    for r, c in chosen:
        tile[r, c] = random.choice(distractor_palette)
    return tile


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Reproduce the solver's tiling logic for oracle validation."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the two control colors from the lower quadrants, then size an output from the lower pattern.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - The upper-left and upper-right quadrants are control regions that may contain several non-black colors.
    - The lower-left quadrant is a non-black fill tile.
    - The lower-right quadrant is either an all-black empty stencil or a foreground stencil with black holes.

    Rule:
    - Output height is the number of pixels in the upper-left quadrant whose color also appears in the lower-left quadrant.
    - Output width is the number of pixels in the upper-right quadrant whose color appears in the lower-right foreground; if the stencil is empty, use the lower-left palette instead.
    - If the lower-right stencil is empty, tile the lower-left quadrant directly over the output.
    - Otherwise tile the stencil over the output, and replace each stencil black cell with the lower-left color taken from the corresponding coarse tile position.

    Key insight:
    - Extra colors in the top half are distractors: only the upper color echoed by the relevant lower quadrant is countable, while the bottom-half tiling rule itself is unchanged.
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

    fill_colors = {int(v) for v in np.unique(fill_tile) if int(v) != BLACK}
    assert fill_colors
    stencil_colors = {int(v) for v in np.unique(stencil) if int(v) != BLACK}

    height_marker = _pick_marker_color(upper_left, fill_colors)
    width_source = stencil_colors if stencil_colors else fill_colors
    width_marker = _pick_marker_color(upper_right, width_source)

    out_h = int(np.count_nonzero(upper_left == height_marker))
    out_w = int(np.count_nonzero(upper_right == width_marker))
    assert out_h > 0 and out_w > 0

    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != BLACK):
        for r in range(out_h):
            for c in range(out_w):
                value = int(fill_tile[r % qh, c % qw])
                assert value != BLACK
                out[r, c] = value
        return out

    for r in range(out_h):
        for c in range(out_w):
            value = int(stencil[r % qh, c % qw])
            if value == BLACK:
                fill_r = r // qh
                fill_c = c // qw
                assert fill_r < qh and fill_c < qw
                value = int(fill_tile[fill_r, fill_c])
                assert value != BLACK
            out[r, c] = value

    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid grid; return None on failure."""
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k)

    remaining_colors = [c for c in COLORS if c not in ll_palette]
    if stencil_empty:
        stencil_palette: list[int] = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(*STENCIL_MULTI_COLOR_RANGE)
        if len(remaining_colors) < st_k:
            return None
        stencil_palette = _palette(st_k, remaining_colors)

    output_colors = set(ll_palette) | set(stencil_palette)
    distractor_pool = [c for c in COLORS if c not in output_colors]
    if not distractor_pool:
        return None

    lo = max(OUTPUT_DIM_MIN, q)
    hi = min(OUTPUT_DIM_MAX, q * q - 1)
    valid_dims = [
        (h, w)
        for h in range(lo, hi + 1)
        for w in range(lo, hi + 1)
        if h * w <= OUTPUT_AREA_MAX and (stencil_empty or ceil(h / q) * ceil(w / q) >= ll_k)
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    used_fill = [(r, c) for r in range(ceil(out_h / q)) for c in range(ceil(out_w / q))]
    fill_tile = _make_lower_left(q, ll_palette, used_fill)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)

    height_marker = random.choice(ll_palette)
    width_marker = random.choice(ll_palette if stencil_empty else stencil_palette)
    upper_left = _make_control(q, height_marker, out_h, distractor_pool)
    upper_right = _make_control(q, width_marker, out_w, distractor_pool)
    if upper_left is None or upper_right is None:
        return None

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
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
