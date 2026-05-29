# Variant of f931b4a8
# Dimension: D3 — 颜色源（左下象限）的结构 (src)
# Change: 左下象限作为颜色源瓦片，可以是均匀单色或含有多色行列图案；其像素值在铺贴时按瓦片重复坐标粗粒度填充模板的背景位置... → 颜色源为渐变序列

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color

# Design analysis for the D3 variant
# a) Concrete behavior:
#    - The 2x2 quadrant layout is unchanged.
#    - UL and UR still encode output height/width by counting non-black cells.
#    - LL is no longer a generic 2D color tile. It is now a 1D sequence embedded as
#      either row bands or column bands: every row is solid (row-gradient) or every
#      column is solid (column-gradient), and the colors advance along that axis.
#    - LR is still a stencil. When a stencil cell is black, the old task used the
#      coarse block index (r // Q, c // Q) to fetch one LL cell. The variant instead
#      uses the output position itself: row-gradients fill by sequence[r % Q] and
#      column-gradients fill by sequence[c % Q]. This creates visible progressive
#      banding across the whole output rather than coarse QxQ blocks.
#    - Example with Q=4 and a row-gradient LL colors [2,3,4,6]:
#      old background fill on rows 0..7 would use LL row indices [0,0,0,0,1,1,1,1];
#      new background fill uses [0,1,2,3,0,1,2,3]. The change is therefore local to
#      the fill-color lookup stage, but it propagates to every black hole in LR.
# b) Complexity calibration:
#    - LL is restricted to exactly one visible orientation (rows or cols) so the
#      inference problem remains crisp.
#    - The gradient uses only 2-4 colors, repeated cyclically when Q is larger, so
#      the pattern is structured and supported by repetition rather than arbitrary.
#    - LR keeps simple stencil families (checker/stripes/frame/random) so the moving
#      background is easy to observe through repeated holes.
# c) Hidden-assumption audit for main():
#    - The original branch for empty LR assumed arbitrary 2D tiling from LL. This
#      still works only because the generator now guarantees LL is row-uniform or
#      column-uniform, making direct tiling equivalent to repeating the 1D sequence.
#    - The original background substitution used fill_tile[r // Q, c // Q]; that
#      relied on LL storing coarse block colors. This assumption no longer holds and
#      is replaced by an explicit gradient-axis detector plus position-based lookup.
#    - The solver now assumes LL is unambiguous: exactly one of {all rows uniform,
#      all cols uniform} is the meaningful gradient axis, with at least two colors.
#      The generator enforces that invariant.
# d) Placement legality:
#    - Quadrants remain non-overlapping by construction inside an even NxN square.
#    - LL has no black cells. LR may contain black holes but stays within its
#      quadrant. Control counts are chosen so output dimensions are positive and
#      bounded, so all modulo lookups remain valid.
# e) Visual observability:
#    - Humans can spot whether LL encodes row bands or column bands from a single
#      training example, and can then see the same direction reappear inside LR holes
#      in the output.
#    - Using cyclic 2-4 color sequences avoids invisible color-id arithmetic while
#      still producing a clear sense of progression.
# f) Visual cleanliness:
#    - The four regions are still spatially separated.
#    - LL uses clean stripes, LR uses sparse/structured masks, and control colors are
#      excluded from the output palette, so role confusion is limited.
# g) Color-change inferability:
#    - The output color changes are tied only to visible row/column position and the
#      visible orientation of LL. No sorting or transformation depends on numeric
#      color ids. Repeated holes in LR provide enough evidence to infer the rule.


def _extract_gradient(fill_tile: np.ndarray) -> tuple[str, np.ndarray]:
    """Return ('rows'|'cols', 1D color sequence) for a valid LL gradient source."""
    qh, qw = fill_tile.shape
    assert qh == qw and qh >= 2

    row_uniform = bool(np.all(fill_tile == fill_tile[:, :1]))
    col_uniform = bool(np.all(fill_tile == fill_tile[:1, :]))
    row_seq = fill_tile[:, 0].astype(int)
    col_seq = fill_tile[0, :].astype(int)

    if row_uniform and not col_uniform and len(set(map(int, row_seq))) >= 2:
        return "rows", row_seq
    if col_uniform and not row_uniform and len(set(map(int, col_seq))) >= 2:
        return "cols", col_seq
    raise AssertionError("lower-left quadrant must be a non-uniform row or column gradient")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the two upper control quadrants to size an output whose background follows a row- or column-wise color sequence.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants are control regions.
    - The lower-left quadrant is a non-black gradient source with solid rows or solid columns; the lower-right quadrant is a stencil tile or empty.

    Rule:
    - Output height is the number of non-background pixels in the upper-left quadrant.
    - Output width is the number of non-background pixels in the upper-right quadrant.
    - Read the lower-left quadrant as a 1D color sequence: either one color per row or one color per column.
    - If the lower-right stencil has foreground pixels, tile it over the output; each stencil background pixel is filled from the lower-left sequence using the global output row or column position, matching the source orientation.
    - If the lower-right stencil is empty, repeat the lower-left gradient source directly, which yields full row-wise or column-wise bands in the output.

    Key insight:
    - The top quadrants still control only the output dimensions, but the lower-left quadrant now encodes a directional color progression rather than coarse block colors, so the stencil holes reveal a spatial gradient across the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    bg = int(Color.BLACK)
    qh, qw = h // 2, w // 2
    upper_left = g[:qh, :qw]
    upper_right = g[:qh, qw:]
    fill_tile = g[qh:, :qw]
    stencil = g[qh:, qw:]

    out_h = int(np.count_nonzero(upper_left != bg))
    out_w = int(np.count_nonzero(upper_right != bg))
    assert out_h > 0 and out_w > 0

    axis, sequence = _extract_gradient(fill_tile)

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
                idx = r % qh if axis == "rows" else c % qw
                value = int(sequence[idx])
                assert value != bg
            out[r, c] = value

    return out


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
STENCIL_EMPTY_PROB = 0.10
GRADIENT_DISTINCT_RANGE = (2, 4)
STENCIL_SINGLE_COLOR_PROB = 0.75
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
GEN_MAX_RETRIES = 300


def _palette(k: int, pool=COLORS) -> list[int]:
    """Return k distinct colors sampled from pool."""
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    """Fill an array by drawing each cell independently from palette."""
    return np.array([random.choice(palette) for _ in range(int(np.prod(shape)))],
                    dtype=int).reshape(shape)


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
        return sorted(coords, key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()))
    random.shuffle(coords)
    return coords


def _make_gradient_source(q: int, palette: list[int]) -> np.ndarray:
    """Build LL as either row-wise or column-wise solid-color bands."""
    axis = random.choice(["rows", "cols"])
    k = len(palette)
    offset = random.randrange(k)
    sequence = [palette[(i + offset) % k] for i in range(q)]
    tile = np.empty((q, q), dtype=int)

    if axis == "rows":
        for r, color in enumerate(sequence):
            tile[r, :] = color
    else:
        for c, color in enumerate(sequence):
            tile[:, c] = color
    return tile


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build the stencil tile (black = holes filled from the gradient source)."""
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


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control quadrant with exactly `count` non-black pixels."""
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else \
        random.randint(2, min(max_colors, count, len(pool)))
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


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray,
                     out_h: int, out_w: int) -> np.ndarray:
    """Reproduce the variant solver's logic for oracle validation."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]

    axis, sequence = _extract_gradient(fill_tile)
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    if axis == "rows":
        tiled[bg_mask] = sequence[rr[bg_mask] % q]
    else:
        tiled[bg_mask] = sequence[cc[bg_mask] % q]
    return tiled


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid grid; return None on failure."""
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    grad_k_lo = min(max(2, GRADIENT_DISTINCT_RANGE[0]), q)
    grad_k_hi = min(GRADIENT_DISTINCT_RANGE[1], q)
    if grad_k_lo > grad_k_hi:
        return None
    ll_k = random.randint(grad_k_lo, grad_k_hi)
    ll_palette = _palette(ll_k)

    if stencil_empty:
        stencil_palette = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else \
            random.randint(*STENCIL_MULTI_COLOR_RANGE)
        stencil_palette = _palette(st_k)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    lo = max(OUTPUT_DIM_MIN, q)
    hi = min(OUTPUT_DIM_MAX, q * q)
    valid_dims = [
        (h, w) for h in range(lo, hi + 1) for w in range(lo, hi + 1)
        if h * w <= OUTPUT_AREA_MAX
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    fill_tile = _make_gradient_source(q, ll_palette)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

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
