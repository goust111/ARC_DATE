# Combo variant of f931b4a8 (index 40)
# combo_spec: D10/1+D12/1
# Name: 模板色由控制位置决定+多色控制通道

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior and trace:
#    - The 4-quadrant layout is preserved: UL/UR are controls, LL is the coarse fill
#      tile, LR is the stencil tile.
#    - D10/1 contributes a two-color control code in each top quadrant:
#        * more frequent color -> output size on that axis
#        * less frequent color -> phase offset on that axis
#    - D12/1 is fused into the stencil coloring itself instead of being an unrelated
#      extra step. The stencil does not keep its foreground color in the output.
#      Instead, the stencil foreground color only selects which control quadrant to
#      sample from:
#        * stencil color == UL minority color -> sample the UL control tile
#        * stencil color == UR minority color -> sample the UR control tile
#      The sampled control cell at the corresponding phase-shifted modular position
#      becomes the true output color.
#    - Small q=4 example:
#        * UL has 6 red + 2 yellow => out_h=6, row_phase=2
#        * UR has 5 green + 1 blue => out_w=5, col_phase=1
#        * LR uses only yellow / blue / black
#        * At output cell (r,c), look at stencil[(r+2)%4, (c+1)%4]:
#            - black  -> use LL[r//4, c//4]
#            - yellow -> use UL[(r+2)%4, (c+1)%4], so output becomes red or yellow
#            - blue   -> use UR[(r+2)%4, (c+1)%4], so output becomes green or blue
#      Compared with the base task, top colors now matter twice: their counts set size
#      and phase, and their positions decide the colors of stencil foreground pixels.
#
# b) Complexity calibration:
#    - Each top control quadrant is restricted to exactly two non-black colors, so the
#      multi-channel rule is visible without exploding into many color roles.
#    - The stencil is restricted to exactly the two minority control colors plus black.
#      This gives a clean, repeated cue for "which control channel should be queried".
#    - The generator forces each stencil channel to hit both the majority and minority
#      regions of its control tile when possible, so the recoloring is visibly dynamic
#      rather than collapsing to a constant color.
#
# c) Original assumptions that break:
#    - `np.count_nonzero(control != bg)` no longer gives the output dimension. The
#      solver must separate majority vs minority counts inside each control quadrant.
#    - The original code assumed stencil foreground values were already the final
#      output colors. That is false here: foreground values are only channel tags.
#    - The original zero-phase modular lookup is invalid. The minority counts now shift
#      which stencil/control coordinates are consulted, so both the pattern placement
#      and the sampled control colors depend on phase.
#
# d) Placement legality:
#    - The generator constructs each control quadrant inside its own qxq area with
#      `dim_count + phase_count <= q*q`, so no overlap or overflow is possible.
#    - LL stays entirely non-black, matching the original coarse fill assumption.
#    - LR foreground coordinates are chosen only from positions where the referenced
#      control quadrant is non-black, so `main()` never samples black for a stencil hit.
#    - LR still leaves at least one black cell, so coarse LL fill remains visible.
#
# e) Visual observability:
#    - The top colors reappear in the output, while the raw stencil ink does not stay
#      fixed. This makes the D12 behavior visible directly.
#    - Because the stencil colors are exactly the two minority control colors, a human
#      can cross-reference "yellow pixels in the stencil" with the top quadrant where
#      yellow appears, then notice that output colors follow the arrangement of that
#      control quadrant instead of preserving stencil yellow itself.
#    - The phase offsets are also visible because the generator rejects samples where
#      the fused output equals the same control-sampling rule without phase shift.
#
# f) Visual cleanliness:
#    - There are still only four clean regions with no overlays: controls on top,
#      source/stencil on bottom.
#    - Control layouts are biased toward rectangles / compact blobs rather than noisy
#      scatter, and LL keeps simple stripe/diagonal patterns, so the scene remains
#      readable even though multiple color roles are active.
#
# g) Change inferability:
#    - No logic depends on numeric color IDs. The solver uses only counts, spatial
#      positions, and visible color equality between stencil tags and control-channel
#      colors.
#    - The generator enforces both fused mechanisms quantitatively:
#        * D10-side phase must change the fused output relative to a no-phase version.
#        * D12-side recoloring must change the fused output relative to a phase-shifted
#          version that merely preserves stencil colors.


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
LL_SOLID_PROB = 0.45
LL_MULTI_COLOR_RANGE = (2, 3)
CONTROL_RECT_PROB = 0.85
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 500


def _palette(k: int, pool=COLORS) -> list[int]:
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


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


def _order_subset(coords: list[tuple[int, int]], q: int, mode: str) -> list[tuple[int, int]]:
    coords = list(coords)
    if not coords:
        return coords
    if mode == "compact":
        ar, ac = random.choice(coords)
        random.shuffle(coords)
        coords.sort(key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()))
        return coords
    if mode == "rows":
        rows, cols = list(range(q)), list(range(q))
        random.shuffle(rows)
        random.shuffle(cols)
        rank = {(r, c): i for i, (r, c) in enumerate((rc for r in rows for c in cols for rc in [(r, c)]))}
        coords.sort(key=lambda rc: rank[rc])
        return coords
    random.shuffle(coords)
    return coords


def _make_lower_left(
    q: int,
    palette: list[int],
    required_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Build the coarse fill tile; all cells must stay non-black."""
    k = len(palette)
    tile = np.empty((q, q), dtype=int)
    mode = "solid" if k == 1 and random.random() < LL_SOLID_PROB else random.choice(["rows", "cols", "rows", "cols", "diagonal", "solid"])
    if k > 1 and mode == "solid":
        mode = random.choice(["rows", "cols", "diagonal"])

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
    assert np.all(tile != BLACK)
    return tile


def _make_control(
    q: int,
    dim_count: int,
    phase_count: int,
    major_color: int,
    minor_color: int,
) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    """
    Build a two-color control quadrant.

    Final majority-color count = dim_count.
    Final minority-color count = phase_count.
    Total active cells = dim_count + phase_count.
    """
    assert 1 <= phase_count < dim_count
    assert dim_count + phase_count <= q * q
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
        tile[r, c] = major_color

    minor_coords = _order_subset(active_coords, q, random.choice(["compact", "rows", "random"]))[:phase_count]
    for r, c in minor_coords:
        tile[r, c] = minor_color

    active_set = set(active_coords)
    minor_set = set(minor_coords)
    assert np.count_nonzero(tile == major_color) == dim_count
    assert np.count_nonzero(tile == minor_color) == phase_count
    return tile, active_set, minor_set


def _decode_control(control: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    fg = control[control != bg]
    colors, counts = np.unique(fg, return_counts=True)
    assert len(colors) == 2
    count_a, count_b = int(counts[0]), int(counts[1])
    assert count_a != count_b
    if count_a > count_b:
        return count_a, count_b, int(colors[0]), int(colors[1])
    return count_b, count_a, int(colors[1]), int(colors[0])


def _select_channel_coords(
    q: int,
    allowed: set[tuple[int, int]],
    minor_coords: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    extra_limit: int,
) -> list[tuple[int, int]] | None:
    available = [rc for rc in allowed if rc not in blocked]
    if not available:
        return None

    majors = [rc for rc in available if rc not in minor_coords]
    minors = [rc for rc in available if rc in minor_coords]
    chosen: list[tuple[int, int]] = []

    if majors and minors:
        chosen.append(random.choice(majors))
        pick_minor = random.choice(minors)
        if pick_minor not in chosen:
            chosen.append(pick_minor)
    else:
        chosen.append(random.choice(available))

    ordered = _order_subset([rc for rc in available if rc not in chosen], q, random.choice(["compact", "rows", "random"]))
    max_extra = min(extra_limit, len(ordered))
    extra_count = random.randint(0, max_extra)
    chosen.extend(ordered[:extra_count])
    return chosen


def _make_stencil(
    q: int,
    row_tag: int,
    col_tag: int,
    row_active: set[tuple[int, int]],
    row_minor: set[tuple[int, int]],
    col_active: set[tuple[int, int]],
    col_minor: set[tuple[int, int]],
) -> np.ndarray | None:
    tile = np.zeros((q, q), dtype=int)
    area = q * q
    row_extra_limit = max(0, min(len(row_active), max(1, area // 3)) - 1)
    row_coords = _select_channel_coords(q, row_active, row_minor, set(), row_extra_limit)
    if not row_coords:
        return None

    col_extra_limit = max(0, min(len(col_active), max(1, area // 3)) - 1)
    col_coords = _select_channel_coords(q, col_active, col_minor, set(row_coords), col_extra_limit)
    if not col_coords:
        return None

    used = set(row_coords) | set(col_coords)
    if len(used) >= area:
        return None

    for r, c in row_coords:
        tile[r, c] = row_tag
    for r, c in col_coords:
        tile[r, c] = col_tag

    assert np.any(tile == BLACK)
    assert np.any(tile == row_tag)
    assert np.any(tile == col_tag)
    return tile


def _phase_preserving_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
    row_phase: int,
    col_phase: int,
) -> np.ndarray:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    sr = (rr + row_phase) % q
    sc = (cc + col_phase) % q
    tiled = stencil[sr, sc].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _no_phase_fused_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    upper_left: np.ndarray,
    upper_right: np.ndarray,
    out_h: int,
    out_w: int,
    row_tag: int,
    col_tag: int,
) -> np.ndarray:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    sr = rr % q
    sc = cc % q
    stencil_view = stencil[sr, sc]
    out = fill_tile[rr // q, cc // q].copy()
    row_mask = stencil_view == row_tag
    col_mask = stencil_view == col_tag
    out[row_mask] = upper_left[sr[row_mask], sc[row_mask]]
    out[col_mask] = upper_right[sr[col_mask], sc[col_mask]]
    return out


def _expected_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    upper_left: np.ndarray,
    upper_right: np.ndarray,
    out_h: int,
    out_w: int,
    row_phase: int,
    col_phase: int,
    row_tag: int,
    col_tag: int,
) -> np.ndarray:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    sr = (rr + row_phase) % q
    sc = (cc + col_phase) % q
    stencil_view = stencil[sr, sc]
    out = fill_tile[rr // q, cc // q].copy()

    row_mask = stencil_view == row_tag
    col_mask = stencil_view == col_tag
    other_mask = ~(row_mask | col_mask | (stencil_view == BLACK))
    assert not np.any(other_mask)

    out[row_mask] = upper_left[sr[row_mask], sc[row_mask]]
    out[col_mask] = upper_right[sr[col_mask], sc[col_mask]]
    return out


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    q = g.shape[0] // 2
    upper_left = g[:q, :q]
    upper_right = g[:q, q:]
    fill_tile = g[q:, :q]
    stencil = g[q:, q:]
    bg = int(Color.BLACK)

    out_h, row_phase, _, row_tag = _decode_control(upper_left, bg)
    out_w, col_phase, _, col_tag = _decode_control(upper_right, bg)
    fused = _expected_output(fill_tile, stencil, upper_left, upper_right, out_h, out_w, row_phase, col_phase, row_tag, col_tag)
    phase_only = _phase_preserving_output(fill_tile, stencil, out_h, out_w, row_phase, col_phase)
    no_phase = _no_phase_fused_output(fill_tile, stencil, upper_left, upper_right, out_h, out_w, row_tag, col_tag)
    return {
        "phase_delta": int(np.count_nonzero(fused != no_phase)),
        "recolor_delta": int(np.count_nonzero(fused != phase_only)),
        "row_tag_hits": int(np.count_nonzero(_phase_preserving_output(np.full_like(fill_tile, 1), stencil, out_h, out_w, row_phase, col_phase) == row_tag)),
        "col_tag_hits": int(np.count_nonzero(_phase_preserving_output(np.full_like(fill_tile, 1), stencil, out_h, out_w, row_phase, col_phase) == col_tag)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode two-color control quadrants, phase-shift the lower pattern, and recolor stencil hits by sampling the controls.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants are two-color control regions.
    - The lower-left quadrant is a non-black coarse fill tile.
    - The lower-right quadrant is a stencil tile whose only non-black colors are the two minority control colors.

    Rule:
    - In each upper control quadrant, the more frequent non-black color sets the output size on that axis.
    - In each upper control quadrant, the less frequent non-black color sets the phase offset on that axis.
    - Build the repeated lower pattern using those row/column phases.
    - Where the phase-shifted stencil is black, copy the coarse lower-left color at (r // Q, c // Q).
    - Where the phase-shifted stencil equals the upper-left minority color, output the color from the upper-left control at the same phase-shifted modular position.
    - Where the phase-shifted stencil equals the upper-right minority color, output the color from the upper-right control at the same phase-shifted modular position.

    Key insight:
    - The top colors now carry two intertwined meanings: their counts still parameterize the layout, but their spatial arrangement is also sampled directly to determine the visible colors of stencil foreground pixels.
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

    out_h, row_phase, _, row_tag = _decode_control(upper_left, bg)
    out_w, col_phase, _, col_tag = _decode_control(upper_right, bg)
    assert out_h > 0 and out_w > 0
    assert np.all(fill_tile != bg)
    assert row_tag != col_tag

    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            sr = (r + row_phase) % qh
            sc = (c + col_phase) % qh
            stencil_value = int(stencil[sr, sc])
            if stencil_value == bg:
                fill_r = r // qh
                fill_c = c // qh
                assert fill_r < qh and fill_c < qh
                value = int(fill_tile[fill_r, fill_c])
                assert value != bg
            elif stencil_value == row_tag:
                value = int(upper_left[sr, sc])
                assert value != bg
            else:
                assert stencil_value == col_tag
                value = int(upper_right[sr, sc])
                assert value != bg
            out[r, c] = value

    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2

    row_phase = random.randint(1, q - 1)
    col_phase = random.randint(1, q - 1)

    lo = max(OUTPUT_DIM_MIN, q)
    max_h = min(OUTPUT_DIM_MAX, q * q - row_phase)
    max_w = min(OUTPUT_DIM_MAX, q * q - col_phase)
    valid_dims = [
        (hh, ww)
        for hh in range(lo, max_h + 1)
        for ww in range(lo, max_w + 1)
        if hh * ww <= OUTPUT_AREA_MAX
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    coarse_rows = ceil(out_h / q)
    coarse_cols = ceil(out_w / q)
    used_fill = [(r, c) for r in range(coarse_rows) for c in range(coarse_cols)]
    max_fill_colors = min(LL_MULTI_COLOR_RANGE[1], len(used_fill), len(COLORS) - 4)
    if max_fill_colors < 1:
        return None
    if max_fill_colors == 1:
        ll_k = 1
    elif random.random() < LL_SOLID_PROB:
        ll_k = 1
    else:
        ll_k = random.randint(LL_MULTI_COLOR_RANGE[0], max_fill_colors)

    fill_palette = _palette(ll_k)
    control_pool = [c for c in COLORS if c not in fill_palette]
    if len(control_pool) < 4:
        return None
    row_major, row_tag, col_major, col_tag = random.sample(control_pool, 4)

    fill_tile = _make_lower_left(q, fill_palette, used_fill)
    used_fill_colors = {int(fill_tile[r, c]) for r, c in used_fill}
    if used_fill_colors != set(fill_palette):
        return None

    upper_left, row_active, row_minor = _make_control(q, out_h, row_phase, row_major, row_tag)
    upper_right, col_active, col_minor = _make_control(q, out_w, col_phase, col_major, col_tag)

    stencil = _make_stencil(q, row_tag, col_tag, row_active, row_minor, col_active, col_minor)
    if stencil is None:
        return None

    expected = _expected_output(fill_tile, stencil, upper_left, upper_right, out_h, out_w, row_phase, col_phase, row_tag, col_tag)
    no_phase = _no_phase_fused_output(fill_tile, stencil, upper_left, upper_right, out_h, out_w, row_tag, col_tag)
    phase_only = _phase_preserving_output(fill_tile, stencil, out_h, out_w, row_phase, col_phase)
    if np.array_equal(expected, no_phase):
        return None
    if np.array_equal(expected, phase_only):
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

    metrics = _behavior_metrics(grid)
    if metrics["phase_delta"] <= 0 or metrics["recolor_delta"] <= 0:
        return None
    if metrics["row_tag_hits"] <= 0 or metrics["col_tag_hits"] <= 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
