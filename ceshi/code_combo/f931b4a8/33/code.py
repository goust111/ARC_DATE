# Combo variant of f931b4a8 (index 33)
# combo_spec: D5/3+D13/2
# Name: 单次放置+裁剪窗口

from __future__ import annotations

from math import ceil
import random

import numpy as np

from arc_utils import Color

# Design analysis
# ----------------
# a) Concrete behavior trace:
#    - Keep the original 4-quadrant input: UL/UR are controls, LL is the color source, LR is the stencil.
#    - The original task first counted UL/UR to get output size, then repeated LR everywhere; each black LR
#      hole exposed one coarse LL color chosen by macro-block coordinates.
#    - This combo still crops by a control-defined window, but it no longer repeats the stencil everywhere.
#      Instead, imagine the original infinite coarse LL background, then keep only one stencil copy placed on
#      the macro-block whose absolute rows are [q, 2q) and cols are [q, 2q).
#    - Example with q=4, row_offset=1, col_offset=2, out_h=6, out_w=5:
#      the crop window is absolute rows [1:7) and cols [2:7). It intersects only the upper-left 3x3 corner
#      of the single stencil copy, so the output is mostly coarse LL fill and only its lower-right area shows
#      the template colors. Under the original/D13 tiled rule, repeated stencil copies would also appear
#      outside that corner.
# b) Complexity calibration:
#    - Only two low-cardinality control signals are added: UL bbox top and UR bbox left.
#    - The single-copy anchor is fixed, not data-dependent, so the solver still needs only count + bbox +
#      absolute-coordinate rendering, rather than object matching or color ranking.
#    - Generator emphasizes partial-visibility and off-center windows so the difference from tiled behavior is
#      visible in a few examples, while avoiding too many disjoint cases.
# c) Hidden assumptions from the original solver that break here:
#    - `stencil[r % q, c % q]` assumed every output cell belongs to some repeated stencil phase. Now most cells
#      are outside the lone stencil box, so stencil lookup must be gated by an absolute-position containment test.
#    - The old non-empty branch implicitly assumed "outside holes == stencil fg". Now "outside the single copy"
#      must fall back to the LL coarse background.
#    - D13's crop offsets mean all rendering must happen in absolute coordinates before cutting out the final
#      window; otherwise the single-copy placement cannot shift toward corners.
# d) Placement legality guaranteed by the generator:
#    - UL foreground is constrained to rows `>= row_offset` with at least one cell on `row_offset`, so bbox top
#      uniquely recovers the crop row start.
#    - UR foreground is constrained to cols `>= col_offset` with at least one cell on `col_offset`, so bbox left
#      uniquely recovers the crop col start.
#    - LL is all non-black, so every coarse background lookup is valid.
#    - LR is usually non-empty and always fits exactly in one q x q placement box; no overlaps exist because the
#      overlay is written only inside that fixed box.
# e) Visual observability:
#    - The crop often misses the single stencil entirely or clips only one side, producing outputs where the
#      template sits in a corner or disappears, which is the intended visible fusion effect.
#    - LL is generated with 2-3 colors and simple row/col/diagonal structure so the coarse background remains
#      readable even when the stencil is absent from the crop.
#    - Samples are rejected if the combo output matches either a periodic-stencil reference or an unshifted
#      single-placement reference, preventing silent collapse back to one constituent rule.
# f) Visual cleanliness:
#    - Control information stays confined to the top half; output colors come only from LL and LR.
#    - The fixed one-copy overlay avoids multiple competing stencil layers, so the output is a clean coarse field
#      with at most one local fine-pattern patch.
# g) Inferability of color behavior:
#    - No color ids are used symbolically. Colors only mark control foreground, LL source blocks, and LR stencil.
#    - The only color rule is spatial: LL colors fill coarse macro-blocks, and LR foreground overwrites those
#      colors only inside the single placed stencil.


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (8, 16)  # q=2 cannot express a visible non-zero crop offset with out_dim >= 3
STENCIL_EMPTY_PROB = 0.04
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.75
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.80
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
SHIFTED_WINDOW_PROB = 0.92
GEN_MAX_RETRIES = 500

VISIBILITY_MODES = ("partial", "none", "full")
VISIBILITY_WEIGHTS = (0.58, 0.22, 0.20)


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


def _control_offsets(upper_left: np.ndarray, upper_right: np.ndarray, bg: int) -> tuple[int, int]:
    """Recover the crop-window origin from the control bboxes."""
    ul_coords = np.argwhere(upper_left != bg)
    ur_coords = np.argwhere(upper_right != bg)
    assert ul_coords.size > 0 and ur_coords.size > 0
    return int(ul_coords[:, 0].min()), int(ur_coords[:, 1].min())


def _base_value(fill_tile: np.ndarray, abs_r: int, abs_c: int) -> int:
    """Read the coarse repeated LL background at an absolute coordinate."""
    q = fill_tile.shape[0]
    return int(fill_tile[(abs_r // q) % q, (abs_c // q) % q])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a shifted crop window from the controls, then cut it from a coarse color plane that contains only one stencil placement.

    Input structure:
    - The input is an even square divided into four equal quadrants.
    - Black is the background in the control regions and inside stencil holes.
    - The upper-left and upper-right quadrants still act only as controls: their foreground counts set output height and width.
    - The lower-left quadrant is a non-black color-source tile, while the lower-right quadrant is a stencil tile that is no longer repeated across the whole output.

    Rule:
    - Output height is the non-black count in the upper-left quadrant, and output width is the non-black count in the upper-right quadrant.
    - The top edge of the upper-left foreground bbox gives the crop row offset; the left edge of the upper-right foreground bbox gives the crop col offset.
    - Build a conceptual infinite canvas whose background is the lower-left tile repeated at coarse q x q macro-block resolution.
    - Place the lower-right stencil exactly once on the absolute box `[q:2q, q:2q)`; inside that box stencil foreground overwrites the background, while stencil black cells leave the background visible. The final output is the `out_h x out_w` crop starting at the decoded offset.

    Key insight:
    - The controls now specify both the crop size and which part of a larger absolute canvas is visible, while the stencil is a lone embedded patch rather than a periodic texture.
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
    assert not np.any(fill_tile == bg)

    row_offset, col_offset = _control_offsets(upper_left, upper_right, bg)
    place_r0 = qh
    place_c0 = qw
    stencil_empty = not np.any(stencil != bg)

    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        abs_r = row_offset + r
        for c in range(out_w):
            abs_c = col_offset + c
            value = _base_value(fill_tile, abs_r, abs_c)
            if not stencil_empty and place_r0 <= abs_r < place_r0 + qh and place_c0 <= abs_c < place_c0 + qw:
                stencil_value = int(stencil[abs_r - place_r0, abs_c - place_c0])
                if stencil_value != bg:
                    value = stencil_value
            out[r, c] = value

    return out


def _make_lower_left(q: int, palette: list[int]) -> np.ndarray:
    """Build the coarse color source tile (must contain no black pixels)."""
    k = len(palette)
    tile = np.empty((q, q), dtype=int)
    mode = random.choice(["rows", "cols", "rows", "cols", "diagonal"])

    if mode == "rows":
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
        coords = [(r, c) for r in range(q) for c in range(q)]
        random.shuffle(coords)
        for color, (r, c) in zip(missing, coords):
            tile[r, c] = color
    return tile


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build a mixed stencil tile (black = transparent holes)."""
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


def _compact_order(coords: list[tuple[int, int]], anchor: tuple[int, int]) -> list[tuple[int, int]]:
    """Sort coords by distance from an anchor with mild randomness."""
    random.shuffle(coords)
    return sorted(
        coords,
        key=lambda rc: (abs(rc[0] - anchor[0]) + abs(rc[1] - anchor[1]), random.random()),
    )


def _make_control(
    q: int,
    count: int,
    max_colors: int,
    pool: list[int],
    *,
    fixed_top: int | None = None,
    fixed_left: int | None = None,
) -> np.ndarray:
    """Build a control quadrant with exact count and a recoverable bbox top/left."""
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(
        2, min(max_colors, count, len(pool))
    )
    pal = _palette(k, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    allowed = [
        (r, c)
        for r in range(q)
        for c in range(q)
        if (fixed_top is None or r >= fixed_top) and (fixed_left is None or c >= fixed_left)
    ]
    assert count <= len(allowed)

    rects: list[tuple[int, int, int, int]] = []
    for h in range(1, q + 1):
        for w in range(1, q + 1):
            if h * w != count:
                continue
            row_starts = [fixed_top] if fixed_top is not None else list(range(q - h + 1))
            col_starts = [fixed_left] if fixed_left is not None else list(range(q - w + 1))
            for r0 in row_starts:
                for c0 in col_starts:
                    if r0 + h <= q and c0 + w <= q:
                        rects.append((r0, c0, h, w))

    if rects and random.random() < CONTROL_RECT_PROB:
        r0, c0, h, w = random.choice(rects)
        tile[r0:r0 + h, c0:c0 + w] = _random_color_array((h, w), pal)
        return tile

    if fixed_top is not None:
        boundary = [(fixed_top, c) for c in range(q)]
        anchor = random.choice(boundary)
    elif fixed_left is not None:
        boundary = [(r, fixed_left) for r in range(q)]
        anchor = random.choice(boundary)
    else:
        anchor = random.choice(allowed)

    ordered = [anchor]
    ordered.extend(_compact_order([coord for coord in allowed if coord != anchor], anchor))
    for r, c in ordered[:count]:
        tile[r, c] = random.choice(pal)
    return tile


def _expected_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
    row_offset: int,
    col_offset: int,
) -> np.ndarray:
    """Reproduce the combo solver logic for oracle validation."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    abs_r = rr + row_offset
    abs_c = cc + col_offset

    out = fill_tile[(abs_r // q) % q, (abs_c // q) % q].copy()
    if not np.any(stencil != BLACK):
        return out

    mask = (q <= abs_r) & (abs_r < 2 * q) & (q <= abs_c) & (abs_c < 2 * q)
    if not np.any(mask):
        return out

    local_r = abs_r[mask] - q
    local_c = abs_c[mask] - q
    stencil_vals = stencil[local_r, local_c]
    fg_mask = stencil_vals != BLACK
    patch = out[mask]
    patch[fg_mask] = stencil_vals[fg_mask]
    out[mask] = patch
    return out


def _tiled_reference_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
    row_offset: int,
    col_offset: int,
) -> np.ndarray:
    """Periodic-stencil reference used only to reject degenerate generator samples."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    abs_r = rr + row_offset
    abs_c = cc + col_offset

    out = fill_tile[(abs_r // q) % q, (abs_c // q) % q].copy()
    if not np.any(stencil != BLACK):
        return out

    tiled = stencil[abs_r % q, abs_c % q]
    fg_mask = tiled != BLACK
    out[fg_mask] = tiled[fg_mask]
    return out


def _axis_overlap_kind(start: int, length: int, q: int) -> str:
    """Classify one-dimensional overlap with the single stencil box [q, 2q)."""
    end = start + length
    overlap = max(0, min(end, 2 * q) - max(start, q))
    if overlap == 0:
        return "none"
    if overlap == q and start <= q and end >= 2 * q:
        return "full"
    return "partial"


def _template_visibility(row_offset: int, col_offset: int, out_h: int, out_w: int, q: int) -> str:
    """Classify how much of the single stencil lies inside the crop window."""
    row_kind = _axis_overlap_kind(row_offset, out_h, q)
    col_kind = _axis_overlap_kind(col_offset, out_w, q)
    if row_kind == "none" or col_kind == "none":
        return "none"
    if row_kind == "full" and col_kind == "full":
        return "full"
    return "partial"


def _valid_window_configs(q: int) -> list[tuple[int, int, int, int]]:
    """Enumerate crop offset/size combinations that fit their control encodings."""
    hi = min(OUTPUT_DIM_MAX, q * q)
    configs: list[tuple[int, int, int, int]] = []
    for row_offset in range(q):
        max_h = min(hi, q * (q - row_offset))
        if max_h < OUTPUT_DIM_MIN:
            continue
        for col_offset in range(q):
            max_w = min(hi, q * (q - col_offset))
            if max_w < OUTPUT_DIM_MIN:
                continue
            for out_h in range(OUTPUT_DIM_MIN, max_h + 1):
                for out_w in range(OUTPUT_DIM_MIN, max_w + 1):
                    if out_h * out_w <= OUTPUT_AREA_MAX:
                        configs.append((row_offset, col_offset, out_h, out_w))
    return configs


def _choose_window_config(q: int) -> tuple[int, int, int, int] | None:
    """Sample a crop config with a preferred stencil-visibility regime."""
    configs = _valid_window_configs(q)
    if not configs:
        return None

    shifted = [cfg for cfg in configs if cfg[0] > 0 or cfg[1] > 0]
    source = shifted if shifted and random.random() < SHIFTED_WINDOW_PROB else configs

    buckets = {mode: [] for mode in VISIBILITY_MODES}
    for cfg in source:
        buckets[_template_visibility(*cfg, q)].append(cfg)

    mode = random.choices(VISIBILITY_MODES, weights=VISIBILITY_WEIGHTS, k=1)[0]
    for fallback in (mode, "partial", "full", "none"):
        if buckets[fallback]:
            return random.choice(buckets[fallback])
    return None


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid non-degenerate combo sample."""
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    ll_k = random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k)
    if stencil_empty:
        stencil_palette: list[int] = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(
            *STENCIL_MULTI_COLOR_RANGE
        )
        stencil_palette = _palette(st_k)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    cfg = _choose_window_config(q)
    if cfg is None:
        return None
    row_offset, col_offset, out_h, out_w = cfg

    fill_tile = _make_lower_left(q, ll_palette)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool, fixed_top=row_offset)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool, fixed_left=col_offset)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    expected = _expected_output(fill_tile, stencil, out_h, out_w, row_offset, col_offset)
    repeated = _tiled_reference_output(fill_tile, stencil, out_h, out_w, row_offset, col_offset)
    unshifted = _expected_output(fill_tile, stencil, out_h, out_w, 0, 0)
    visibility = _template_visibility(row_offset, col_offset, out_h, out_w, q)

    if np.unique(expected).size < 2:
        return None
    if np.array_equal(expected, repeated):
        return None
    if (row_offset > 0 or col_offset > 0) and np.array_equal(expected, unshifted):
        return None
    if visibility == "full" and (out_h <= q and out_w <= q):
        return None

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
