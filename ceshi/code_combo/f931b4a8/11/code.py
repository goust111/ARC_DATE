# Combo variant of f931b4a8 (index 11)
# combo_spec: D5/2+D6/1
# Name: 砖墙式铺贴+行索引循环颜色

from __future__ import annotations

from math import ceil
from pathlib import Path
import random
import sys

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import Color


# Design analysis for the combo variant
# -------------------------------------
# concretization:
#   - The input stays as an even square split into four equal quadrants.
#   - The top quadrants still encode only output height and width by counting
#     non-black pixels.
#   - The lower-left quadrant is still a non-black color source, but for this
#     combo it must visibly expose a row sequence, so generated non-empty
#     samples make each row solid and cycle 2-3 colors down the tile.
#   - The lower-right quadrant is still a stencil with black holes, but its
#     copies no longer align in a strict grid: odd tile rows are shifted right
#     by half a tile width, producing a brick-wall layout.
#   - When a sampled stencil cell is black, the fill color is no longer taken
#     from coarse block coordinates; instead it is the row-sequence color for
#     the absolute output row modulo Q. The output therefore combines staggered
#     stencil placement with horizontal stripe fills.
# behavior_trace:
#   - Example with Q=4, shift=2, row sequence [2, 5, 3, 6], and an odd tile-row
#     sample at output cell (r=4, c=1):
#       original task:
#         stencil source = stencil[0, 1]
#         if black, fill = LL[1, 0]
#       D5-only:
#         stencil source = stencil[0, 3]
#         if black, fill = LL[1, 0]
#       D6-only:
#         stencil source = stencil[0, 1]
#         if black, fill = row_sequence[0] = 2
#       combo:
#         stencil source = stencil[0, 3]
#         if black, fill = row_sequence[0] = 2
#   - If stencil[0, 1] is foreground and stencil[0, 3] is black, then the same
#     output cell changes for both reasons at once: the brick offset chooses a
#     different stencil sample, and the black-hole replacement now follows the
#     absolute output row rather than the coarse block row.
# complexity_calibration:
#   - Empty stencils would hide both fused transforms, so the solver still
#     supports that branch but the generator does not emit it.
#   - The generator forces output height to cover two full tile rows. This makes
#     the odd shifted tile row visible and also creates enough row indices for
#     the stripe cycle to repeat against a different coarse row than the
#     original task would use.
#   - Generated stencils are rejected unless shifting by Q/2 changes them, and
#     unless every stencil row contains both foreground and background. That
#     keeps both the brick offset and the row-cycle fill visible in every modulo
#     row of the output.
# implicit_assumption_audit:
#   - Original `stencil[r % q, c % q]` assumed a single global column phase for
#     all stencil copies. The combo breaks that; odd tile rows need a half-width
#     shift.
#   - Original `fill_tile[r // q, c // q]` assumed stencil holes are colored by
#     coarse output blocks. The combo breaks that too; hole colors now depend on
#     `r % q` through a row sequence.
#   - `main()` now reads the fill sequence from the first column of the
#     lower-left tile. Solver correctness only needs that column to be non-black
#     (always true here), while human readability is improved by generator-side
#     row-solid fill tiles.
#   - The half-tile brick shift needs Q even. The original task family already
#     guarantees Q in {2, 4, 8}, so this new assumption is safe.
# placement_legality:
#   - Quadrant placement is unchanged, so there is no new overlap risk inside
#     the input grid.
#   - Output dimensions remain <= Q*Q, which keeps any coarse fallback lookup in
#     bounds for the empty-stencil branch and for baseline comparison oracles.
# visual_observability:
#   - Lower-left uses row stripes and lower-right keeps visible holes, so the
#     fused output shows a stable separation between stripe background and stencil
#     foreground.
#   - Stencil and fill palettes are kept disjoint in generated samples, reducing
#     ambiguity between "pattern color" and "row-cycle color".
#   - Samples are rejected unless the combo output differs from the original
#     task, from the D5-only baseline, and from the D6-only baseline by a
#     measurable number of cells. This is the quantitative anti-degeneracy check.
# visual_cleanliness_and_inference:
#   - The bottom half still has a clear role split: left = stripe source, right
#     = mask pattern. No hidden color ordering or numeric color arithmetic is
#     introduced.
#   - Color counts stay small and repetitive, which gives a human enough repeated
#     evidence across 2-4 examples to infer both "odd tile rows shift" and
#     "holes follow row index".


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
STENCIL_EMPTY_PROB = 0.0
LL_ROW_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.60
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.72)
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 400


def _palette(k: int, pool=COLORS) -> list[int]:
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


def _ordered_coords(q: int, mode: str) -> list[tuple[int, int]]:
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


def _shifted_stencil_col(c: int, q: int, tile_row: int) -> int:
    shift = (q // 2) if tile_row % 2 == 1 else 0
    return (c - shift) % q


def _make_lower_left_general(
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
        fix_coords = (
            list(required_coords)
            if required_coords is not None
            else [(r, c) for r in range(q) for c in range(q)]
        )
        random.shuffle(fix_coords)
        for color, (r, c) in zip(missing, fix_coords):
            tile[r, c] = color
    return tile


def _make_row_sequence_tile(q: int, palette: list[int]) -> np.ndarray:
    tile = np.empty((q, q), dtype=int)
    offset = random.randrange(len(palette))
    for r in range(q):
        tile[r, :] = palette[(offset + r) % len(palette)]

    if len(set(map(int, tile[:, 0]))) < 2:
        tile[0, :] = palette[0]
        tile[1 % q, :] = palette[min(1, len(palette) - 1)]
    return tile


def _repair_stencil_row_mix(tile: np.ndarray, palette: list[int]) -> np.ndarray:
    q = tile.shape[0]
    for r in range(q):
        if np.all(tile[r, :] == BLACK):
            tile[r, random.randrange(q)] = random.choice(palette)
        if np.all(tile[r, :] != BLACK):
            tile[r, random.randrange(q)] = BLACK
    return tile


def _rows_are_mixed(tile: np.ndarray) -> bool:
    return all(np.any(row == BLACK) and np.any(row != BLACK) for row in tile)


def _rebalance_stencil_density(
    tile: np.ndarray,
    palette: list[int],
    lo: int,
    hi: int,
) -> np.ndarray:
    q = tile.shape[0]
    fg_coords = [(r, c) for r in range(q) for c in range(q) if tile[r, c] != BLACK]
    bg_coords = [(r, c) for r in range(q) for c in range(q) if tile[r, c] == BLACK]
    fg_count = len(fg_coords)

    if fg_count < lo:
        random.shuffle(bg_coords)
        for r, c in bg_coords[: lo - fg_count]:
            tile[r, c] = random.choice(palette)
    elif fg_count > hi:
        random.shuffle(fg_coords)
        for r, c in fg_coords[: fg_count - hi]:
            tile[r, c] = BLACK
    return tile


def _is_offset_sensitive(tile: np.ndarray) -> bool:
    shift = tile.shape[1] // 2
    return not np.array_equal(tile, np.roll(tile, shift=shift, axis=1))


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    area = q * q
    lo = max(2, int(ceil(STENCIL_FG_FRAC[0] * area)))
    hi = min(area - 2, max(lo, int(ceil(STENCIL_FG_FRAC[1] * area))))

    for _ in range(120):
        tile = np.zeros((q, q), dtype=int)
        mode = random.choice(["bars", "steps", "blocks", "scatter"])

        if mode == "bars":
            bar_cols = random.sample(range(q), k=random.randint(1, max(1, q - 1)))
            for c in bar_cols:
                r0 = random.randint(0, q - 1)
                r1 = random.randint(r0, q - 1)
                tile[r0:r1 + 1, c] = _random_color_array((r1 - r0 + 1,), palette)
            arm_r = random.randrange(q)
            arm_c0 = random.randrange(q)
            arm_len = random.randint(1, max(1, q // 2))
            for dc in range(arm_len):
                tile[arm_r, (arm_c0 + dc) % q] = random.choice(palette)

        elif mode == "steps":
            start = random.randrange(q)
            width = random.randint(1, max(1, q - 1))
            drift = random.choice([1, 2]) if q > 2 else 1
            for r in range(q):
                c0 = (start + drift * r) % q
                span = min(q - 1, width if random.random() < 0.7 else width + 1)
                for dc in range(max(1, span)):
                    c = (c0 + dc) % q
                    if dc == 0 or (r + dc) % 2 == 0 or random.random() < 0.35:
                        tile[r, c] = random.choice(palette)

        elif mode == "blocks":
            block_count = random.randint(2, 3)
            for _ in range(block_count):
                bh = random.randint(1, max(1, q // 2))
                bw = random.randint(1, max(1, q // 2))
                r0 = random.randint(0, q - bh)
                c0 = random.randint(0, q - bw)
                tile[r0:r0 + bh, c0:c0 + bw] = _random_color_array((bh, bw), palette)
            notch_r = random.randrange(q)
            notch_c = random.randrange(q)
            tile[notch_r, notch_c] = BLACK

        else:
            fg_count = random.randint(lo, hi)
            coords = [(r, c) for r in range(q) for c in range(q)]
            random.shuffle(coords)
            for r, c in coords[:fg_count]:
                tile[r, c] = random.choice(palette)

        tile = _repair_stencil_row_mix(tile, palette)
        tile = _rebalance_stencil_density(tile, palette, lo, hi)
        tile = _repair_stencil_row_mix(tile, palette)

        if (
            np.any(tile == BLACK)
            and np.any(tile != BLACK)
            and _rows_are_mixed(tile)
            and _is_offset_sensitive(tile)
        ):
            return tile

    raise RuntimeError("failed to build a visible combo stencil")


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
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
        tile[r0:r0 + h, c0:c0 + w] = _random_color_array((h, w), pal)
    else:
        mode = random.choice(["rect", "compact", "rows"])
        coords = _ordered_coords(q, mode)[:count]
        for r, c in coords:
            tile[r, c] = random.choice(pal)
    return tile


def _row_fill_sequence(fill_tile: np.ndarray) -> np.ndarray:
    seq = np.asarray(fill_tile[:, 0], dtype=int)
    assert np.all(seq != BLACK)
    return seq


def _render_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
    *,
    use_brick_offset: bool,
    use_row_cycle: bool,
) -> np.ndarray:
    q = fill_tile.shape[0]
    out = np.empty((out_h, out_w), dtype=int)

    if not np.any(stencil != BLACK):
        rr, cc = np.mgrid[:out_h, :out_w]
        tiled = fill_tile[rr % q, cc % q]
        assert np.all(tiled != BLACK)
        return tiled

    row_sequence = _row_fill_sequence(fill_tile)
    for r in range(out_h):
        tile_row = r // q
        stencil_r = r % q
        stripe_color = int(row_sequence[r % q])
        for c in range(out_w):
            stencil_c = (
                _shifted_stencil_col(c, q, tile_row) if use_brick_offset else c % q
            )
            value = int(stencil[stencil_r, stencil_c])
            if value == BLACK:
                if use_row_cycle:
                    value = stripe_color
                else:
                    fill_r = r // q
                    fill_c = c // q
                    assert fill_r < q and fill_c < q
                    value = int(fill_tile[fill_r, fill_c])
                    assert value != BLACK
            out[r, c] = value
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the top half as size controls and render the lower-half pattern using brick-offset stencil tiling over row-cycled fill colors.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants are control areas whose non-black counts determine the output height and width.
    - The lower-left quadrant is a non-black color tile that provides a row-wise color sequence, while the lower-right quadrant is either an empty tile or a stencil with black holes.

    Rule:
    - Output height is the number of non-black pixels in the upper-left quadrant.
    - Output width is the number of non-black pixels in the upper-right quadrant.
    - If the lower-right quadrant is empty, repeat the lower-left tile directly across the output.
    - Otherwise, tile the stencil across the output in brick rows: copies in odd tile rows are shifted right by half a tile width. Whenever the sampled stencil cell is black, replace it with the lower-left row-sequence color for output row r modulo Q.

    Key insight:
    - The top half still encodes only the output rectangle size, but the bottom half now carries two coupled clues: the right quadrant controls a staggered brick-like stencil phase, and the left quadrant supplies a cyclic row palette that paints every stencil hole as a horizontal stripe.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    bg = int(Color.BLACK)
    qh, qw = h // 2, w // 2
    assert qh == qw and qh % 2 == 0

    upper_left = g[:qh, :qw]
    upper_right = g[:qh, qw:]
    fill_tile = g[qh:, :qw]
    stencil = g[qh:, qw:]

    out_h = int(np.count_nonzero(upper_left != bg))
    out_w = int(np.count_nonzero(upper_right != bg))
    assert out_h > 0 and out_w > 0

    return _render_output(
        fill_tile,
        stencil,
        out_h,
        out_w,
        use_brick_offset=True,
        use_row_cycle=True,
    )


def _expected_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    return _render_output(
        fill_tile,
        stencil,
        out_h,
        out_w,
        use_brick_offset=True,
        use_row_cycle=True,
    )


def _expected_output_original(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    return _render_output(
        fill_tile,
        stencil,
        out_h,
        out_w,
        use_brick_offset=False,
        use_row_cycle=False,
    )


def _expected_output_d5_only(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    return _render_output(
        fill_tile,
        stencil,
        out_h,
        out_w,
        use_brick_offset=True,
        use_row_cycle=False,
    )


def _expected_output_d6_only(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    return _render_output(
        fill_tile,
        stencil,
        out_h,
        out_w,
        use_brick_offset=False,
        use_row_cycle=True,
    )


def _fusion_metrics(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> dict[str, int]:
    combo = _expected_output(fill_tile, stencil, out_h, out_w)
    original = _expected_output_original(fill_tile, stencil, out_h, out_w)
    d5_only = _expected_output_d5_only(fill_tile, stencil, out_h, out_w)
    d6_only = _expected_output_d6_only(fill_tile, stencil, out_h, out_w)
    return {
        "diff_vs_original": int(np.count_nonzero(combo != original)),
        "shift_affected_cells": int(np.count_nonzero(combo != d6_only)),
        "rowcycle_affected_cells": int(np.count_nonzero(combo != d5_only)),
    }


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    if stencil_empty:
        ll_k = 1 if random.random() < 0.70 else random.randint(2, min(3, q))
        ll_palette = _palette(ll_k)
        stencil_palette: list[int] = []
    else:
        ll_k = random.randint(LL_ROW_COLOR_RANGE[0], min(LL_ROW_COLOR_RANGE[1], q))
        ll_palette = _palette(ll_k)
        stencil_pool = [c for c in COLORS if c not in ll_palette]
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(
            *STENCIL_MULTI_COLOR_RANGE
        )
        st_k = min(st_k, len(stencil_pool))
        stencil_palette = _palette(st_k, stencil_pool)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    hi = min(OUTPUT_DIM_MAX, q * q)
    min_h = max(OUTPUT_DIM_MIN, q if stencil_empty else 2 * q)
    min_w = max(OUTPUT_DIM_MIN, q if stencil_empty else q + (q // 2))
    valid_dims = [
        (out_h, out_w)
        for out_h in range(min_h, hi + 1)
        for out_w in range(min_w, hi + 1)
        if out_h * out_w <= OUTPUT_AREA_MAX
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    if stencil_empty:
        used_fill = [(r, c) for r in range(ceil(out_h / q)) for c in range(ceil(out_w / q))]
        fill_tile = _make_lower_left_general(q, ll_palette, used_fill)
        stencil = np.zeros((q, q), dtype=int)
    else:
        fill_tile = _make_row_sequence_tile(q, ll_palette)
        stencil = _make_stencil(q, stencil_palette)

    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    expected = _expected_output(fill_tile, stencil, out_h, out_w)
    metrics = _fusion_metrics(fill_tile, stencil, out_h, out_w)
    if not stencil_empty:
        if metrics["shift_affected_cells"] < max(2, q // 2):
            return None
        if metrics["rowcycle_affected_cells"] < max(4, q):
            return None
        if metrics["diff_vs_original"] < max(6, q):
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
