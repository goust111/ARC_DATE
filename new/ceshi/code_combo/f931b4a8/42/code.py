# Combo variant of f931b4a8 (index 42)
# combo_spec: D3/0+D15/1
# Name: 边界背景填充+单色颜色源

from __future__ import annotations

import random

import numpy as np

from arc_utils import Color


# Design analysis for combo D3/0 + D15/1
# =======================================
# 1. Concrete behavior and trace
#    - The grid is still split into four equal quadrants.
#    - Upper-left and upper-right still encode output height/width by non-black counts.
#    - Lower-left no longer carries a coarse qxq color table in practice: D3/0 collapses it to one
#      uniform non-black color K, so every stencil hole should resolve to the same visible fill color.
#    - Lower-right is still the fine stencil tile.
#    - The changed tiling step is D15/1: only complete qxq stencil repeats are rendered. Any area that
#      would belong to a clipped partial repeat at the output boundary becomes background instead.
#    - Concrete trace with q=4, out_h=10, out_w=11:
#      * original task: render rows 0..9 / cols 0..10 by modulo; the last 2 rows and last 3 cols show
#        cropped stencil fragments, and all holes inside them still become K.
#      * fused task here: center the largest complete 8x8 tiled region inside the 10x11 output. The
#        inner 8x8 area shows full stencil repeats with hole color K; the surrounding 1-row/1-row and
#        1-col/2-col border cells are all black.
#      * difference: D3/0 removes coarse block-to-block color variation, while D15/1 makes the edge
#        treatment itself the key visible signal.
# 2. Complexity calibration
#    - LL is forced to a single solid color, so there is only one fill-color hypothesis to learn.
#    - To keep the combo from becoming too weak, the generator always chooses output dimensions with
#      at least one complete repeat and at least one leftover row and column. This guarantees a visible
#      tiled interior plus a visible black edge frame.
#    - LR is kept non-empty and hole-bearing in accepted samples, so both the stencil foreground and the
#      uniform fill color appear in the interior.
# 3. Implicit-assumption audit against the original solver
#    - Original assumption: `fill_tile[r // q, c // q]` matters because LL may vary by coarse block.
#      Broken by D3/0. The fused solver replaces that with extraction of one unique LL color.
#    - Original assumption: every output cell belongs to some modulo-wrapped tile. Broken by D15/1.
#      The fused solver first checks whether a cell lies inside the centered full-tile window; outside
#      that window it writes background directly.
#    - Original empty-stencil branch tiled LL per pixel by modulo. That is invalid under D15/1. The
#      fused branch now fills only the centered complete-tile window and leaves the edge residue black.
# 4. Placement legality
#    - All four quadrants still stay inside an even square input with no overlap.
#    - LL is a full qxq solid block, so extracting its unique non-black color is safe.
#    - The centered full-tile window always fits because its size is `floor(out_dim / q) * q`.
#    - Control counts stay in 1..q^2, so output dimensions remain representable by the top quadrants.
# 5. Visual observability
#    - A human can see that LL is a plain solid swatch, so any internal output color variation must come
#      from LR rather than from a hidden LL lookup table.
#    - The black edge frame is visually obvious and repeats across samples with different thicknesses,
#      making the boundary rule inferable from 2-4 examples.
# 6. Visual cleanliness
#    - The output has only three visual layers: black border residue, one repeated stencil foreground,
#      and one uniform fill color for holes. That is much cleaner than keeping multi-color coarse blocks.
#    - Control colors are excluded from the output palette, so the top quadrants stay purely symbolic.
# 7. Change inferability
#    - No numeric color-id trick is used. The fill color is literally the only visible color in LL, and
#      the edge rule is visible through geometry, not through hidden ordering.
# 8. Fusion-behavior checks
#    - A sample is accepted only if the fused output differs from the original modulo-cropping output.
#    - The generator also requires a positive black-border area and at least one full interior tile, so
#      the combo cannot silently collapse back to the seed task.
# 9. New solver assumptions and generator support
#    - `main()` assumes LL contains exactly one non-black color and no black cells; `_make_lower_left()`
#      enforces that.
#    - `main()` assumes the intended D15/1 concretization is a centered complete-tile window; the
#      generator validates every sample against the same oracle and rejects degeneracy.


BLACK = int(Color.BLACK)
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
STENCIL_EMPTY_PROB = 0.0
STENCIL_SINGLE_COLOR_PROB = 0.82
STENCIL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_FG_FRAC = (0.30, 0.70)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 400


def _palette(k: int, pool=COLORS) -> list[int]:
    """Return k distinct colors sampled from pool."""
    items = list(pool)
    random.shuffle(items)
    return items[:k]


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
        r0 = random.randint(0, q - 1)
        c0 = random.randint(0, q - 1)
        return [((r + r0) % q, (c + c0) % q) for r, c in coords]
    if mode == "rows":
        rows = list(range(q))
        cols = list(range(q))
        random.shuffle(rows)
        random.shuffle(cols)
        return [(r, c) for r in rows for c in cols]
    if mode == "compact":
        ar = random.randrange(q)
        ac = random.randrange(q)
        random.shuffle(coords)
        return sorted(
            coords,
            key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()),
        )
    random.shuffle(coords)
    return coords


def _extract_fill_color(fill_tile: np.ndarray, bg: int) -> int:
    """Read the unique non-background color from a valid solid LL quadrant."""
    colors = sorted(set(map(int, fill_tile.flatten())))
    assert bg not in colors
    assert len(colors) == 1
    return int(colors[0])


def _centered_window(out_dim: int, q: int) -> tuple[int, int, int]:
    """
    Return (pad_before, full_len, pad_after) for the largest centered multiple-of-q span.

    The full span uses only complete tiles. Any remainder stays outside as background.
    """
    full_len = (out_dim // q) * q
    remainder = out_dim - full_len
    pad_before = remainder // 2
    pad_after = remainder - pad_before
    return int(pad_before), int(full_len), int(pad_after)


def _render_fused(fill_color: int, stencil: np.ndarray, out_h: int, out_w: int, bg: int) -> np.ndarray:
    """Oracle for the fused combo: centered full tiles only, partial boundary area stays bg."""
    q = int(stencil.shape[0])
    top, full_h, _bottom = _centered_window(out_h, q)
    left, full_w, _right = _centered_window(out_w, q)
    has_stencil = bool(np.any(stencil != bg))

    out = np.full((out_h, out_w), bg, dtype=int)
    if full_h == 0 or full_w == 0:
        return out

    for r in range(top, top + full_h):
        local_r = r - top
        for c in range(left, left + full_w):
            local_c = c - left
            if has_stencil:
                value = int(stencil[local_r % q, local_c % q])
                if value == bg:
                    value = fill_color
            else:
                value = fill_color
            out[r, c] = value
    return out


def _render_original(fill_color: int, stencil: np.ndarray, out_h: int, out_w: int, bg: int) -> np.ndarray:
    """Oracle for the original seed behavior restricted to a solid LL source."""
    q = int(stencil.shape[0])
    has_stencil = bool(np.any(stencil != bg))

    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            if has_stencil:
                value = int(stencil[r % q, c % q])
                if value == bg:
                    value = fill_color
            else:
                value = fill_color
            out[r, c] = value
    return out


def _combo_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify whether the generated sample really exhibits the fused behavior."""
    g = np.asarray(grid, dtype=int)
    q = g.shape[0] // 2
    ul = g[:q, :q]
    ur = g[:q, q:]
    fill_tile = g[q:, :q]
    stencil = g[q:, q:]

    out_h = int(np.count_nonzero(ul != BLACK))
    out_w = int(np.count_nonzero(ur != BLACK))
    fill_color = _extract_fill_color(fill_tile, BLACK)
    fused = _render_fused(fill_color, stencil, out_h, out_w, BLACK)
    original = _render_original(fill_color, stencil, out_h, out_w, BLACK)

    top, full_h, bottom = _centered_window(out_h, q)
    left, full_w, right = _centered_window(out_w, q)
    interior = fused[top : top + full_h, left : left + full_w] if full_h and full_w else fused[:0, :0]
    interior_colors = set(map(int, interior.flatten()))
    interior_colors.discard(BLACK)

    return {
        "q": int(q),
        "out_h": int(out_h),
        "out_w": int(out_w),
        "full_h": int(full_h),
        "full_w": int(full_w),
        "pad_top": int(top),
        "pad_bottom": int(bottom),
        "pad_left": int(left),
        "pad_right": int(right),
        "edge_bg_count": int(np.count_nonzero(fused == BLACK)),
        "delta_vs_original": int(np.count_nonzero(fused != original)),
        "interior_fg_colors": int(len(interior_colors)),
        "stencil_holes": int(np.count_nonzero(stencil == BLACK)),
        "stencil_fg": int(np.count_nonzero(stencil != BLACK)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top quadrants to size an output whose interior is tiled from the lower quadrants, while clipped edge repeats vanish into background.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants remain control regions for output height and width.
    - The lower-left quadrant is a solid single-color source, and the lower-right quadrant is a stencil tile that may be non-empty or empty.

    Rule:
    - Output height is the number of non-background pixels in the upper-left quadrant.
    - Output width is the number of non-background pixels in the upper-right quadrant.
    - Read the lower-left quadrant as one unique fill color.
    - Find the largest centered output window whose height and width are whole multiples of the stencil size. Tile the lower-right stencil only inside that window: stencil foreground stays unchanged, and stencil background is replaced by the single lower-left fill color.
    - Any cells outside that centered complete-tile window belong to incomplete boundary repeats and therefore stay black. If the stencil is empty, the centered full-tile window is filled uniformly by the lower-left color instead.

    Key insight:
    - The top half still controls only the output dimensions, but the lower half now splits into two simpler roles: one constant fill color and one fine stencil. The real twist is geometric: complete repeats survive in the interior, while partial repeats at the boundary are discarded to background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    qh, qw = h // 2, w // 2
    assert qh == qw
    q = qh

    upper_left = g[:q, :q]
    upper_right = g[:q, q:]
    fill_tile = g[q:, :q]
    stencil = g[q:, q:]

    out_h = int(np.count_nonzero(upper_left != BLACK))
    out_w = int(np.count_nonzero(upper_right != BLACK))
    assert out_h > 0 and out_w > 0

    fill_color = _extract_fill_color(fill_tile, BLACK)
    top, full_h, _bottom = _centered_window(out_h, q)
    left, full_w, _right = _centered_window(out_w, q)
    has_stencil = bool(np.any(stencil != BLACK))

    out = np.full((out_h, out_w), BLACK, dtype=int)
    for r in range(top, top + full_h):
        local_r = r - top
        for c in range(left, left + full_w):
            local_c = c - left
            if has_stencil:
                value = int(stencil[local_r % q, local_c % q])
                if value == BLACK:
                    value = fill_color
            else:
                value = fill_color
            out[r, c] = value
    return out


def _make_lower_left(q: int, color: int) -> np.ndarray:
    """Build the LL quadrant as one solid non-black color."""
    return np.full((q, q), int(color), dtype=int)


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build a non-empty stencil tile with at least one fg cell and one bg hole."""
    tile = np.zeros((q, q), dtype=int)
    modes = ["checker", "checker", "row_stripe", "col_stripe", "random"]
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
        lo = max(1, int(area * STENCIL_FG_FRAC[0]))
        hi = min(area - 1, max(lo, int(area * STENCIL_FG_FRAC[1])))
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
    else:
        mode = random.choice(["rect", "compact", "rows"])
        coords = _ordered_coords(q, mode)[:count]
        for r, c in coords:
            tile[r, c] = random.choice(palette)
    return tile


def _valid_dims(q: int) -> list[tuple[int, int]]:
    """Output sizes that guarantee both a full-tile interior and a visible boundary residue."""
    lo = max(OUTPUT_DIM_MIN, q + 1)
    hi = min(OUTPUT_DIM_MAX, q * q)
    dims: list[tuple[int, int]] = []
    for out_h in range(lo, hi + 1):
        for out_w in range(lo, hi + 1):
            if out_h * out_w > OUTPUT_AREA_MAX:
                continue
            if out_h % q == 0 or out_w % q == 0:
                continue
            dims.append((out_h, out_w))
    return dims


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid combo sample; return None on failure."""
    n = random.choice(GRID_SIZES)
    q = n // 2
    dims = _valid_dims(q)
    if not dims:
        return None

    out_h, out_w = random.choice(dims)
    fill_color = random.choice(COLORS)
    fill_tile = _make_lower_left(q, fill_color)

    stencil_empty = random.random() < STENCIL_EMPTY_PROB
    if stencil_empty:
        stencil = np.zeros((q, q), dtype=int)
        stencil_palette: list[int] = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(*STENCIL_MULTI_COLOR_RANGE)
        stencil_palette = _palette(st_k, [c for c in COLORS if c != fill_color])
        if not stencil_palette:
            return None
        stencil = _make_stencil(q, stencil_palette)

    output_colors = {fill_color, *stencil_palette}
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    expected = _render_fused(fill_color, stencil, out_h, out_w, BLACK)
    metrics = _combo_metrics(grid)

    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    if metrics["full_h"] <= 0 or metrics["full_w"] <= 0:
        return None
    if metrics["edge_bg_count"] <= 0:
        return None
    if metrics["delta_vs_original"] <= 0:
        return None
    if metrics["stencil_holes"] <= 0 or metrics["stencil_fg"] <= 0:
        return None
    if metrics["interior_fg_colors"] < (1 if stencil_empty else 2):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate one valid input for the fused variant."""
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for f931b4a8 combo D3/0+D15/1")
