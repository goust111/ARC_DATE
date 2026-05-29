# Combo variant of cb2d8a2c (index 21)
# combo_spec: D14/0+D17/2
# Name: 切比雪夫距离+路径填充禁区
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior trace
#    - Source task in canonical top-start view:
#      a single green pixel starts on the top edge and a 1-pixel route goes downward
#      around horizontal wall-attached bars. Each bar uses its blue-count b as a
#      Manhattan keepout radius, so the forbidden shape is a diamond-like thickening.
#    - D14/0 change:
#      replace Manhattan distance with Chebyshev distance. For a horizontal bar on row r
#      spanning cols [c0, c1], the keepout becomes the axis-aligned rectangle
#      rows [r-b, r+b], cols [c0-b, c1+b], clipped to the grid. Because each bar touches
#      one wall, this rectangle also touches that wall.
#    - D17/2 change in this task:
#      the route is no longer rendered as a single lattice line. Once the path chooses the
#      only open side of a bar, it fills the whole free corridor on that side until the next
#      turning event. In canonical view that means large green rectangles bounded by:
#        1. the square/rectangular Chebyshev keepout edge of the current bar,
#        2. the board edge on the open side,
#        3. the turn row just above the bar, and
#        4. the next turn row below it.
#    - Small concrete example:
#      left-attached bar with blue-count 2 and length 7 in a 16-column grid:
#        original task: go down in one column, move right on one row, go down in one column.
#        fused task: keepout is rows [r-2, r+2], cols [0, 8]; the output fills the whole
#        right corridor cols [9, 15] from the turn row above the bar to the exit row below it.
#      The difference is visually obvious: the output contains a wide green block rather
#      than a thin elbow.
#
# b) Complexity calibration
#    - Blue-counts stay in the same small range 1..4, so the radius cue is still learnable.
#    - The generator requires at least one bar with blue-count >= 2 and at least one open
#      corridor of width >= 4, so the Chebyshev square keepout and the wide-fill effect are
#      both visible in every sample.
#    - Bar count remains 1..4, giving repeated evidence without making the grid cluttered.
#
# c) Hidden-assumption review of the original solver
#    - Original assumption: a 4-neighbor BFS from each bar is the right way to build the
#      forbidden mask. Broken by D14/0; the new keepout is a rectangle, so the solver must
#      reason in Chebyshev geometry instead of Manhattan geometry.
#    - Original assumption: a shortest 1-pixel path is the final output. Broken by D17/2;
#      we now solve the route as ordered corridor switches and paint full slabs, not just
#      the skeleton.
#    - Original assumption: Dijkstra tie-breaking is needed to decide exact turns. With
#      wall-attached alternating bars and full-corridor painting, the canonical geometry is
#      deterministic, so the solver is rewritten explicitly after canonicalization.
#
# d) Placement legality enforced by the generator
#    - Every obstacle component is a single horizontal bar in canonical orientation and
#      touches exactly one side wall.
#    - Consecutive bars alternate walls, guaranteeing that each currently filled corridor is
#      blocked by the next bar and therefore forces another switch.
#    - Main-axis gaps satisfy row_i - row_{i-1} >= b_{i-1} + b_i + 2, which keeps the turn
#      rows outside neighboring Chebyshev keepouts.
#    - The open side width is constrained to be at least 3 cells, and at least one bar has
#      open width >= 4, so filled corridors are legal and visually broad.
#
# e) Observability
#    - Across 2-4 samples a human can see that more blue pixels create larger square-ish
#      no-go rectangles, and the green output occupies whole side corridors rather than a
#      thin trace.
#    - No invisible color-id ordering is used. The only control signal remains the visible
#      blue-count on each bar.
#
# f) Visual cleanliness
#    - Green is painted only on background cells, so bars remain legible and never merge
#      with the fill.
#    - Because the filled areas are monotone rectangles/corridors aligned to bar keepouts,
#      the outputs look structured rather than noisy.
#
# g) New solver assumptions and generator support
#    - Solver assumptions:
#      exactly one green border pixel; canonical bars are single-row components; every bar
#      touches one side wall; bars are ordered from the start side; the current filled
#      corridor intersects each next bar keepout.
#    - Generator support:
#      it constructs exactly that family of instances and validates each candidate by
#      running `main()` itself before accepting it.

import itertools
import random

import numpy as np

from arc_utils import components_mask


BG = 8
BLUE = 1
RED = 2
GREEN = 3

# --- Tunable parameters (CFG) ---
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 1, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
OPEN_WIDTH_MIN = 3
OPEN_WIDTH_STRONG = 4
MAX_GEN_RETRIES = 500


def _to_canonical(grid: np.ndarray) -> tuple[np.ndarray, str]:
    """Rotate/flip so the unique green start sits on the top edge."""
    start_pts = np.argwhere(grid == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])
    h, w = grid.shape
    if sr == 0:
        return np.array(grid, copy=True), "top"
    if sr == h - 1:
        return np.flipud(grid).copy(), "bottom"
    if sc == 0:
        return grid.T.copy(), "left"
    if sc == w - 1:
        return np.fliplr(grid.T).copy(), "right"
    raise AssertionError("green start must be on a border edge")


def _from_canonical(grid: np.ndarray, edge: str) -> np.ndarray:
    """Undo `_to_canonical`."""
    if edge == "top":
        return np.array(grid, copy=True)
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid).T.copy()  # edge == "right"


def _paint_rect_on_bg(out: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> None:
    """Paint an inclusive rectangle green on background cells only."""
    rr0 = max(0, int(r0))
    rr1 = min(out.shape[0] - 1, int(r1))
    cc0 = max(0, int(c0))
    cc1 = min(out.shape[1] - 1, int(c1))
    if rr0 > rr1 or cc0 > cc1:
        return
    view = out[rr0 : rr1 + 1, cc0 : cc1 + 1]
    view[view == BG] = GREEN


def _extract_bars(grid: np.ndarray) -> list[dict[str, int]]:
    """Read horizontal wall-attached bars from a canonical grid."""
    h, w = grid.shape
    comps = components_mask((grid == BLUE) | (grid == RED), connectivity=4)
    bars: list[dict[str, int]] = []
    for comp in comps:
        rows = comp[:, 0]
        cols = comp[:, 1]
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())
        assert r0 == r1, "canonical bars must be horizontal"
        assert c0 == 0 or c1 == w - 1, "each bar must touch one side wall"
        side = 0 if c0 == 0 else 1  # 0 = left-attached, 1 = right-attached
        blue_count = int(np.sum(grid[comp[:, 0], comp[:, 1]] == BLUE))
        left = max(0, c0 - blue_count)
        right = min(w - 1, c1 + blue_count)
        bars.append(
            {
                "row": r0,
                "c0": c0,
                "c1": c1,
                "blue": blue_count,
                "side": side,
                "top": r0 - blue_count,
                "bottom": r0 + blue_count,
                "left": left,
                "right": right,
            }
        )
    bars.sort(key=lambda bar: bar["row"])
    return bars


def _corridor_after_bar(bar: dict[str, int], width: int) -> tuple[int, int]:
    """Return the full open-side corridor interval after clearing one bar."""
    if bar["side"] == 0:
        lo = bar["right"] + 1
        hi = width - 1
    else:
        lo = 0
        hi = bar["left"] - 1
    assert 0 <= lo <= hi < width, "bar must leave a non-empty open corridor"
    return int(lo), int(hi)


def _solve_canonical(grid: np.ndarray) -> np.ndarray:
    """Solve one canonical top-start instance using Chebyshev keepouts + corridor filling."""
    g = np.array(grid, copy=True)
    h, w = g.shape

    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])
    assert sr == 0, "canonical start must lie on the top edge"

    bars = _extract_bars(g)
    assert bars, "grid must contain at least one obstacle bar"

    out = g.copy()
    cur_r = sr
    cur_lo = sc
    cur_hi = sc
    prev_side = None

    for bar in bars:
        top = int(bar["top"])
        bottom = int(bar["bottom"])
        left = int(bar["left"])
        right = int(bar["right"])
        assert top >= 1, "first bar must leave a visible turn row above its keepout"

        if prev_side is not None:
            assert int(bar["side"]) != prev_side, "bars must alternate wall attachment"
        prev_side = int(bar["side"])

        # If the current filled corridor hits the keepout columns, the route must switch sides here.
        assert not (cur_hi < left or cur_lo > right), "current corridor must intersect each next bar keepout"

        turn_row = top - 1
        exit_row = bottom + 1
        assert cur_r <= turn_row, "bar spacing must keep slabs outside the next keepout"

        # Continue straight down in the current corridor until just above the keepout.
        _paint_rect_on_bg(out, cur_r, turn_row, cur_lo, cur_hi)

        new_lo, new_hi = _corridor_after_bar(bar, w)

        # Turn above the keepout, crossing from the old corridor into the new one.
        bridge_lo = min(cur_lo, new_lo)
        bridge_hi = max(cur_hi, new_hi)
        _paint_rect_on_bg(out, turn_row, turn_row, bridge_lo, bridge_hi)

        # D17/2: fill the whole open corridor alongside and just beyond this keepout.
        _paint_rect_on_bg(out, turn_row, exit_row, new_lo, new_hi)

        cur_r = exit_row
        cur_lo, cur_hi = new_lo, new_hi

    _paint_rect_on_bg(out, cur_r, h - 1, cur_lo, cur_hi)
    out[out == BLUE] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route a border-start path around side-attached bars, using square Chebyshev keepouts and filling the full bypass corridor in green.

    Input structure:
    - The background is azure (8), with exactly one green (3) start pixel on a border edge.
    - Each obstacle is a 1-cell-thick red/blue bar (2/1) perpendicular to the travel direction and attached to one side wall.
    - The number of blue pixels on a bar is the distance parameter attached to that bar.

    Rule:
    - Normalize to a canonical top-start view. For each bar, replace the old Manhattan halo with a Chebyshev keepout: every cell at Chebyshev distance <= blue-count from that bar is forbidden, which makes the keepout axis-aligned and square/rectangular.
    - Move from the start edge toward the opposite edge, and whenever the current corridor intersects a bar's keepout, switch to the only open side above that keepout.
    - Instead of painting a 1-pixel route, fill the entire open corridor used to bypass that keepout, producing broad green slabs. After all bypasses are drawn, convert blue pixels to red.

    Key insight: Blue pixels still encode how much clearance each bar demands, but under the fused rule that clearance is read in Chebyshev geometry and the solver must paint the whole safe side corridor around the resulting rectangular keepout, not just a thin shortest path.
    """
    canonical, edge = _to_canonical(np.asarray(grid, dtype=int))
    solved = _solve_canonical(canonical)
    return _from_canonical(solved, edge)


def _min_main_len(blues: tuple[int, ...]) -> int:
    """Minimum canonical main-axis size for the strengthened top/between/end spacing."""
    last_bar = blues[0] + 2
    for a, b in zip(blues, blues[1:]):
        last_bar += a + b + 2
    return last_bar + 2


def _random_composition(total: int, parts: int) -> list[int]:
    """Uniform non-negative composition of `total` into `parts` slots."""
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _choose_blues(count: int, main_hi: int) -> tuple[int, ...]:
    """Pick blue-counts that fit and guarantee a visible Chebyshev effect."""
    choices = [
        combo
        for combo in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count)
        if max(combo) >= 2 and _min_main_len(combo) <= main_hi
    ]
    return random.choice(choices)


def _length_candidates(cross_size: int, blue_count: int) -> list[int]:
    """Allowed bar lengths that preserve a visibly wide open-side corridor."""
    lo = max(
        BAR_LEN_MIN,
        2 * blue_count - 1,
        (BAR_CROSS_PCT_LO * cross_size + 99) // 100,
    )
    hi = min(
        BAR_LEN_MAX,
        cross_size - blue_count - OPEN_WIDTH_MIN,
        (BAR_CROSS_PCT_HI * cross_size) // 100,
    )
    if lo > hi:
        return []
    return list(range(lo, hi + 1))


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, blue_count: int) -> None:
    """Paint one horizontal bar on `row`, attached to the chosen side wall."""
    w = grid.shape[1]
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    if side == 0:
        grid[row, :length] = colors
    else:
        grid[row, w - length :] = colors[::-1]


def _canonical_grid(
    main_size: int,
    cross_size: int,
    blues: tuple[int, ...],
    first_side: int,
) -> np.ndarray | None:
    """Build one canonical top-start instance with alternating wall-attached bars."""
    count = len(blues)
    slack = main_size - _min_main_len(blues)
    if slack < 0:
        return None

    extras = _random_composition(slack, count + 1)
    row = blues[0] + 2 + extras[0]
    rows = [row]
    for i in range(1, count):
        row += blues[i - 1] + blues[i] + 2 + extras[i]
        rows.append(row)

    side_seq = [(first_side + i) % 2 for i in range(count)]
    length_options = [_length_candidates(cross_size, blue) for blue in blues]
    if any(not opts for opts in length_options):
        return None

    lengths = [random.choice(opts) for opts in length_options]
    open_widths = [cross_size - length - blue for length, blue in zip(lengths, blues)]
    if max(open_widths) < OPEN_WIDTH_STRONG:
        strong_choices = [i for i, opts in enumerate(length_options) if any(cross_size - L - blues[i] >= OPEN_WIDTH_STRONG for L in opts)]
        if not strong_choices:
            return None
        idx = random.choice(strong_choices)
        strong_opts = [L for L in length_options[idx] if cross_size - L - blues[idx] >= OPEN_WIDTH_STRONG]
        lengths[idx] = random.choice(strong_opts)
        open_widths[idx] = cross_size - lengths[idx] - blues[idx]

    grid = np.full((main_size, cross_size), BG, dtype=int)
    for row, blue_count, side, length in zip(rows, blues, side_seq, lengths):
        _paint_bar(grid, row, side, length, blue_count)

    first_len = lengths[0]
    if first_side == 0:
        start_col = random.randint(1, first_len - 1)
    else:
        start_col = random.randint(cross_size - first_len, cross_size - 2)
    grid[0, start_col] = GREEN
    return grid


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    """Rotate canonical (top-start) input so the start edge matches `edge`."""
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()  # edge == "right"


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    if edge == "bottom":
        return bool(np.any(out[0, :] == GREEN))
    if edge == "left":
        return bool(np.any(out[:, -1] == GREEN))
    return bool(np.any(out[:, 0] == GREEN))


def _has_visible_fusion(inp: np.ndarray, out: np.ndarray) -> bool:
    """Reject accidental near-degenerate samples with only a thin-looking output."""
    green_in = int(np.sum(inp == GREEN))
    green_out = int(np.sum(out == GREEN))
    extra_green = green_out - green_in
    canonical_inp, _ = _to_canonical(inp)
    bars = _extract_bars(canonical_inp)
    corridor_widths = [(_corridor_after_bar(bar, canonical_inp.shape[1])[1] - _corridor_after_bar(bar, canonical_inp.shape[1])[0] + 1) for bar in bars]
    return extra_green >= 12 and max(corridor_widths) >= OPEN_WIDTH_STRONG


def _gen_attempt() -> np.ndarray | None:
    """One generation attempt; validate by solving the candidate."""
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    count = random.randint(BAR_COUNT_LO, BAR_COUNT_HI)
    blues = _choose_blues(count, main_hi)
    main_size = random.randint(max(main_lo, _min_main_len(blues)), main_hi)
    cross_size = random.randint(cross_lo, cross_hi)
    first_side = random.randrange(2)

    canonical = _canonical_grid(main_size, cross_size, blues, first_side)
    if canonical is None:
        return None
    inp = _orient_grid(canonical, edge)

    try:
        out = main(inp)
        if not _reaches_opposite_edge(out, edge):
            return None
        if not _has_visible_fusion(inp, out):
            return None
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c combo input")
