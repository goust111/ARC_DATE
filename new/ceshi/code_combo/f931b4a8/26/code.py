# Combo variant of f931b4a8 (index 26)
# combo_spec: D3/3+D5/0+D6/1
# Name: 渐变色源+镜像铺贴+行循环替换

from __future__ import annotations

from math import ceil
import random

import numpy as np

from arc_utils import Color


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - The 2x2 quadrant layout stays fixed. UL and UR still encode output height/width
#      by non-black pixel counts.
#    - LL is no longer treated as a coarse QxQ lookup table. It is constrained to a
#      row-gradient: each row is solid, and the colors form a visible cyclic sequence.
#    - LR is a stencil, but it is repeated with mirror tiling instead of plain modulo
#      tiling. Block row parity flips the local row index; block column parity flips
#      the local column index.
#    - Black holes in the mirrored stencil are no longer filled from coarse block
#      coordinates LL[r // Q, c // Q]. They are filled from the LL row sequence using
#      the global output row: sequence[r % Q].
#    - Small example with Q=2:
#        LL = [[2,2],
#              [5,5]]
#        stencil = [[7,0],
#                   [0,8]]
#      Original task on a 4x4 output uses periodic stencil and coarse fill, so the
#      holes on rows 0..1 use color 2 and rows 2..3 use color 5.
#      This combo uses mirrored stencil blocks:
#        block (0,0): [[7,0],[0,8]]
#        block (0,1): [[0,7],[8,0]]
#        block (1,0): [[0,8],[7,0]]
#        block (1,1): [[8,0],[0,7]]
#      and every hole on global rows 0,1,2,3 receives colors 2,5,2,5 respectively.
#      The difference from the original therefore has two simultaneous sources:
#      neighboring stencil blocks change orientation, and background color changes per
#      output row rather than per coarse Q-row block.
# b) Complexity calibration:
#    - LL is restricted to row gradients, not arbitrary row/column ambiguity. D6/1 is
#      explicitly row-driven, so allowing column gradients would add noise without
#      adding a human-inferable rule.
#    - The gradient uses 2-4 colors and repeats cyclically, which keeps the palette
#      small while making the horizontal stripes obvious.
#    - LR is forced to be non-empty, to contain holes, to span at least 2x2 repeated
#      blocks in the output, and to break active mirror symmetries. That prevents the
#      D5 effect from collapsing back to ordinary periodic tiling.
# c) Hidden-assumption audit for main():
#    - The original non-empty branch assumed stencil[r % Q, c % Q]. That fails here
#      and must be replaced by mirrored local indexing based on repeated-block parity.
#    - The original background fill assumed LL stores coarse block colors. That also
#      fails: LL now stores a visible row sequence, so the fill rule depends only on
#      the absolute output row.
#    - The empty-stencil branch still tiles LL directly. This remains safe because the
#      generator makes LL row-uniform, so direct tiling still yields coherent stripes.
#    - The solver now assumes LL is a valid non-black row gradient with at least two
#      distinct row colors. The generator enforces exactly that.
# d) Placement legality:
#    - The input remains an even square split into four equal quadrants, so placement
#      legality is inherited from the original task.
#    - Output dimensions are capped so out_h, out_w <= Q*Q. That keeps all repeated
#      block indices within the LL quadrant bounds for the original coarse-fill oracle
#      comparisons used in the generator.
#    - The generator only uses direct forward construction, so no objects overlap and
#      no quadrant exceeds its fixed boundary.
# e) Visual observability:
#    - A human can read the LL row sequence directly from the lower-left quadrant.
#    - The output then shows the same sequence as horizontal stripe colors only inside
#      stencil holes, while the stencil foreground itself exhibits neighbor-to-neighbor
#      reflections. These two cues are spatially distinct and can be inferred from a
#      few examples without using hidden numeric properties of colors.
# f) Visual cleanliness:
#    - Region roles remain clean: top = counting controls, bottom-left = stripe source,
#      bottom-right = reflected texture source.
#    - Control colors are kept disjoint from output colors so the upper quadrants do
#      not introduce distractor colors into the answer.
#    - The combined output has only two layers: stencil foreground and striped hole
#      fill. There is no third semantic overlay.
# g) Change inferability:
#    - Color change is structured entirely by row position. The rule never depends on
#      sorting or arithmetic over color ids.
#    - The generator rejects samples where the combo output matches plain periodic
#      tiling, mirrored coarse-fill tiling, or periodic row-fill tiling, so each of
#      the fused mechanisms has a visible effect.


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
STENCIL_EMPTY_PROB = 0.0
GRADIENT_DISTINCT_RANGE = (2, 4)
STENCIL_SINGLE_COLOR_PROB = 0.45
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 400
STENCIL_ATTEMPTS = 50
MIN_VARIANT_DIFF_PIXELS = 2


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


def _mirrored_local_index(idx: int, tile_size: int, block_idx: int) -> int:
    """Return the local coordinate inside one repeated block under mirror tiling."""
    local = int(idx % tile_size)
    if int(block_idx) % 2 == 1:
        return int(tile_size - 1 - local)
    return local


def _mirrored_stencil_lookup(stencil: np.ndarray, r: int, c: int) -> int:
    """Read the stencil value at output position (r,c) under mirrored repetition."""
    q = int(stencil.shape[0])
    block_r = int(r // q)
    block_c = int(c // q)
    local_r = _mirrored_local_index(int(r), q, block_r)
    local_c = _mirrored_local_index(int(c), q, block_c)
    return int(stencil[local_r, local_c])


def _extract_row_gradient(fill_tile: np.ndarray) -> np.ndarray:
    """Return the visible row sequence encoded by the lower-left gradient source."""
    qh, qw = fill_tile.shape
    assert qh == qw and qh >= 2
    seq = np.asarray(fill_tile[:, 0], dtype=int)
    assert np.all(seq != BLACK)
    assert np.all(fill_tile == seq[:, None])
    assert len(set(map(int, seq))) >= 2
    return seq


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the two upper control quadrants to size an output whose stencil repeats by reflection over row-driven stripes.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants are control regions counted by non-black pixels.
    - The lower-left quadrant is a non-black row-gradient source, and the lower-right quadrant is a stencil tile or empty.

    Rule:
    - Output height is the number of non-background pixels in the upper-left quadrant.
    - Output width is the number of non-background pixels in the upper-right quadrant.
    - Read the lower-left quadrant as a cyclic row-color sequence.
    - If the lower-right stencil is empty, repeat the lower-left gradient tile directly over the output.
    - Otherwise, repeat the stencil with mirror tiling: odd repeated block rows are vertically flipped and odd repeated block columns are horizontally flipped; any black stencil cell on output row r is replaced by the r-th cyclic color from the lower-left row sequence.

    Key insight:
    - The top half still specifies only dimensions, but the bottom half now splits into two coordinated roles: the right tile controls reflected stencil geometry, while the left tile controls a row-by-row background color cycle visible through the stencil holes.
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

    out_h = int(np.count_nonzero(upper_left != bg))
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

    row_sequence = _extract_row_gradient(fill_tile)
    for r in range(out_h):
        stripe_color = int(row_sequence[r % qh])
        for c in range(out_w):
            value = _mirrored_stencil_lookup(stencil, r, c)
            if value == bg:
                value = stripe_color
            out[r, c] = value
    return out


def _make_row_gradient_source(q: int, palette: list[int]) -> np.ndarray:
    """Build LL as visible horizontal color bands."""
    assert len(palette) >= 2
    tile = np.empty((q, q), dtype=int)
    offset = random.randrange(len(palette))
    for r in range(q):
        tile[r, :] = palette[(r + offset) % len(palette)]
    if len(set(map(int, tile[:, 0]))) < 2:
        tile[0, :] = palette[0]
        tile[1 % q, :] = palette[min(1, len(palette) - 1)]
    return tile


def _make_stencil_base(q: int, palette: list[int]) -> np.ndarray:
    """Build a candidate stencil tile (black cells are holes)."""
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


def _stencil_is_informative(stencil: np.ndarray, need_v: bool, need_h: bool) -> bool:
    """Require the stencil to show holes and to change under active mirror flips."""
    if not np.any(stencil != BLACK):
        return False
    if not np.any(stencil == BLACK):
        return False
    if need_v and np.array_equal(stencil, stencil[::-1, :]):
        return False
    if need_h and np.array_equal(stencil, stencil[:, ::-1]):
        return False
    hole_rows = np.unique(np.argwhere(stencil == BLACK)[:, 0])
    return len(hole_rows) >= min(2, stencil.shape[0])


def _make_stencil(q: int, palette: list[int], *, need_v: bool, need_h: bool) -> np.ndarray:
    """Build a stencil whose mirrored repeats differ visibly along active axes."""
    for _ in range(STENCIL_ATTEMPTS):
        tile = _make_stencil_base(q, palette)
        if _stencil_is_informative(tile, need_v, need_h):
            return tile

    tile = np.zeros((q, q), dtype=int)
    tile[0, 0] = palette[0]
    if q > 1:
        tile[0, q - 1] = palette[min(1, len(palette) - 1)]
        tile[q - 1, 0] = palette[min(2, len(palette) - 1)]
    if q > 2:
        tile[1, 0] = palette[0]
    assert _stencil_is_informative(tile, need_v, need_h)
    return tile


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control quadrant with exactly `count` non-black pixels."""
    k_hi = min(max_colors, count, len(pool))
    if k_hi <= 1 or random.random() < CONTROL_SINGLE_COLOR_PROB:
        k = 1
    else:
        k = random.randint(2, k_hi)
    pal = _palette(k, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    factors = [(h, w) for h in range(1, q + 1) for w in range(1, q + 1) if h * w == count]
    if factors and random.random() < CONTROL_RECT_PROB:
        h, w = random.choice(factors)
        r0 = random.randint(0, q - h)
        c0 = random.randint(0, q - w)
        tile[r0:r0 + h, c0:c0 + w] = _random_color_array((h, w), pal)
    else:
        mode = random.choice(["rect", "compact", "rows"])
        coords = _ordered_coords(q, mode)[:count]
        for r, c in coords:
            tile[r, c] = random.choice(pal)
    return tile


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Oracle for the combo rule."""
    q = int(fill_tile.shape[0])
    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != BLACK):
        for r in range(out_h):
            for c in range(out_w):
                out[r, c] = int(fill_tile[r % q, c % q])
        return out

    row_sequence = _extract_row_gradient(fill_tile)
    for r in range(out_h):
        stripe_color = int(row_sequence[r % q])
        for c in range(out_w):
            value = _mirrored_stencil_lookup(stencil, r, c)
            if value == BLACK:
                value = stripe_color
            out[r, c] = value
    return out


def _original_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Original f931b4a8 output, used only for degeneration checks."""
    q = int(fill_tile.shape[0])
    rr, cc = np.mgrid[:out_h, :out_w]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _mirrored_coarse_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """D5-only style output on the same input."""
    q = int(fill_tile.shape[0])
    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            value = _mirrored_stencil_lookup(stencil, r, c)
            if value == BLACK:
                value = int(fill_tile[r // q, c // q])
            out[r, c] = value
    return out


def _periodic_row_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """D6-only style output on the same input."""
    q = int(fill_tile.shape[0])
    rr, cc = np.mgrid[:out_h, :out_w]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    row_sequence = _extract_row_gradient(fill_tile)
    tiled[bg_mask] = row_sequence[rr[bg_mask] % q]
    return tiled


def _variant_metrics(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> dict[str, int]:
    """Quantify how strongly the combo differs from simpler nearby rules."""
    combo = _expected_output(fill_tile, stencil, out_h, out_w)
    original = _original_output(fill_tile, stencil, out_h, out_w)
    mirrored_coarse = _mirrored_coarse_output(fill_tile, stencil, out_h, out_w)
    periodic_row = _periodic_row_output(fill_tile, stencil, out_h, out_w)
    return {
        "diff_vs_original": int(np.count_nonzero(combo != original)),
        "diff_vs_mirrored_coarse": int(np.count_nonzero(combo != mirrored_coarse)),
        "diff_vs_periodic_row": int(np.count_nonzero(combo != periodic_row)),
    }


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid non-degenerate combo sample."""
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB
    if stencil_empty:
        return None

    grad_k_lo = min(max(2, GRADIENT_DISTINCT_RANGE[0]), q)
    grad_k_hi = min(GRADIENT_DISTINCT_RANGE[1], q)
    if grad_k_lo > grad_k_hi:
        return None
    ll_k = random.randint(grad_k_lo, grad_k_hi)
    ll_palette = _palette(ll_k)

    remaining_for_stencil = [c for c in COLORS if c not in ll_palette]
    st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(
        *STENCIL_MULTI_COLOR_RANGE
    )
    if len(remaining_for_stencil) < st_k:
        return None
    stencil_palette = _palette(st_k, remaining_for_stencil)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    lo = max(OUTPUT_DIM_MIN, q + 1)
    hi = min(OUTPUT_DIM_MAX, q * q)
    valid_dims = [
        (out_h, out_w)
        for out_h in range(lo, hi + 1)
        for out_w in range(lo, hi + 1)
        if out_h * out_w <= OUTPUT_AREA_MAX
        and ceil(out_h / q) >= 2
        and ceil(out_w / q) >= 2
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)
    tile_rows = int(ceil(out_h / q))
    tile_cols = int(ceil(out_w / q))

    fill_tile = _make_row_gradient_source(q, ll_palette)
    stencil = _make_stencil(q, stencil_palette, need_v=tile_rows >= 2, need_h=tile_cols >= 2)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    metrics = _variant_metrics(fill_tile, stencil, out_h, out_w)
    if min(metrics.values()) < MIN_VARIANT_DIFF_PIXELS:
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
    expected = _expected_output(fill_tile, stencil, out_h, out_w)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
