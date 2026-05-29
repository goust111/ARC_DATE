# Variant of cb2d8a2c
# Dimension: D17 — 绿色路径的宽度 (meta)
# Change: 路径宽度为1像素（单像素宽的线条）... → 路径宽度等于蓝色像素数

# Design analysis for this D17 variant
# -----------------------------------
# source_task:
#   none
# concrete_variant:
#   canonical_view:
#     The grid is normalized so the lone green start pixel is on the top edge and the
#     route progresses downward around horizontal obstacle bars.
#   unchanged_structure:
#     Azure(8) is background, blue(1)+red(2) form 1-cell-thick bars attached to alternating
#     side walls, and blue cells still count how many "units" the bar encodes.
#   D17 change:
#     Each bar still enforces the original detour distance, but now the green route is not
#     uniformly 1-pixel wide. For bar i with blue-count b_i:
#     1. the horizontal bypass segment directly above that bar is painted as a strip of
#        height b_i, extending upward away from the bar;
#     2. the vertical segment that slides down along the bar's open side is painted as a
#        strip of width b_i, extending farther into the open gap away from the bar.
#     All other connector segments remain 1-pixel wide. This keeps the old route skeleton
#     recognizable while making each obstacle's local bypass visibly thicker.
# behavior_trace:
#   original_step:
#     For a left-attached horizontal bar with blue-count 3, the old solver finds a single
#     safe column just to the right of the keep-out halo and paints a 1-pixel L-shape there.
#   new_step:
#     The same safe skeleton is used, but the horizontal piece above the bar becomes a
#     3-row green band, and the descending side piece becomes a 3-column green band.
#   difference:
#     The control variable (blue-count) now affects both how far the path stays from the bar
#     and how much area the path fills while bypassing that bar.
# complexity_calibration:
#   Width only takes the already-visible values 1..4 from the blue-counts, so the rule is
#   low-entropy and repeated across bars. To avoid silent regression to the original task,
#   generation always includes at least one bar with blue-count >= 2.
# implicit_assumptions_checked:
#   - The original paint loop assumed every path cell was a single lattice point. That is
#     no longer harmless: blindly coloring around every path point would either clip width
#     or overwrite bars. The new solver paints only bar-owned rectangles anchored away from
#     the obstacle side.
#   - The original spacing formula only guaranteed a 1-pixel route. Once bypass strips can
#     extend upward or sideways, bars need more free space. The generator therefore enforces
#     larger open gaps and larger main-axis separations.
#   - The original shortest-path search tolerated many safe columns in the open gap. Here we
#     need deterministic width placement, so the solver explicitly uses the nearest legal
#     column/row implied by each bar's geometry.
# placement_legality:
#   cross_axis:
#     For a bar with blue-count b, the uncovered side gap must be at least 2*b cells so the
#     width-b vertical strip fits entirely on the open side without touching the bar.
#   main_axis:
#     The first bar row must satisfy row >= 2*b so the width-b horizontal strip above it
#     stays in-bounds. For later bars, row deltas must satisfy both the old detour spacing
#     and the extra vertical room required by the thicker strip.
# observability_and_cleanliness:
#   The output still contains the simple alternating bars from the input, but now every bar
#   leaves a local green "ribbon" whose thickness matches the number of blue pixels on that
#   same bar. Because widths are anchored away from obstacles, green, red, and remaining
#   azure regions stay visually separable instead of collapsing into overlaps.
# generator_guarantees_for_main:
#   - exactly one border green start pixel
#   - after canonicalization, every obstacle component is a single horizontal bar touching
#     exactly one side wall
#   - rows are sorted by traversal order and satisfy the strengthened spacing inequalities
#   - at least one bar has blue-count >= 2, so D17 behavior is present in every sample

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
MAX_GEN_RETRIES = 400


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
        return np.fliplr(grid).T.copy(), "right"
    raise AssertionError("green start must be on a border edge")


def _from_canonical(grid: np.ndarray, edge: str) -> np.ndarray:
    """Undo _to_canonical."""
    if edge == "top":
        return np.array(grid, copy=True)
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()  # edge == "right"


def _extract_bars(grid: np.ndarray) -> list[dict[str, int]]:
    """Read horizontal bar components from a canonical grid."""
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
        assert c0 == 0 or c1 == w - 1, "each bar must touch exactly one side wall"
        blue_count = sum(1 for r, c in comp if grid[r, c] == BLUE)
        side = 0 if c0 == 0 else 1
        bars.append(
            {
                "row": r0,
                "length": c1 - c0 + 1,
                "blue": int(blue_count),
                "side": side,
            }
        )
    bars.sort(key=lambda bar: bar["row"])
    return bars


def _paint_rect_on_bg(out: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> None:
    """Paint a rectangle green, but only on background cells."""
    rr0 = max(0, int(r0))
    rr1 = min(out.shape[0] - 1, int(r1))
    cc0 = max(0, int(c0))
    cc1 = min(out.shape[1] - 1, int(c1))
    if rr0 > rr1 or cc0 > cc1:
        return
    view = out[rr0 : rr1 + 1, cc0 : cc1 + 1]
    view[view == BG] = GREEN


def _iter_line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """Inclusive axis-aligned lattice segment from a to b."""
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("route segments must be axis-aligned")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a green route from one border to the opposite border, making each obstacle's bypass as thick as that obstacle's blue count.

    Input structure:
    - bg = azure (8). Mixed blue/red bars (1/2) are 1-cell-thick wall-attached obstacles.
    - Exactly one green (3) pixel lies on a border edge and marks the route start.

    Rule:
    - For each bar, its blue pixel count still determines how far the route skeleton must detour from the bar.
    - That same count also sets the width of the local bypass: the strip above the bar and the strip descending along its open side are both thickened to that value.
    - After the route is painted, every blue pixel becomes red.

    Key insight:
    - The blue cells now encode two linked quantities at once: clearance and ribbon thickness. Bigger blue counts create both a wider berth and a fatter green bypass around that specific bar.
    """
    canonical, edge = _to_canonical(np.asarray(grid))
    g = np.array(canonical, copy=True)
    h, w = g.shape

    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])
    assert sr == 0, "canonical start must be on the top edge"

    bars = _extract_bars(g)
    assert bars, "grid must contain at least one obstacle bar"

    out = g.copy()
    path_cells: list[tuple[int, int]] = [(sr, sc)]
    current = (sr, sc)

    for bar in bars:
        row = int(bar["row"])
        blue = int(bar["blue"])
        length = int(bar["length"])
        side = int(bar["side"])

        top_row = row - blue - 1
        bottom_row = row + blue + 1
        assert top_row >= 0 and bottom_row < h, "bar spacing must keep the route in bounds"

        if side == 0:
            safe_col = length + blue
        else:
            safe_col = w - length - blue - 1
        assert 0 <= safe_col < w, "open side must contain a legal safe column"

        above = (top_row, current[1])
        across = (top_row, safe_col)
        below = (bottom_row, safe_col)

        path_cells.extend(_iter_line(current, above)[1:])
        path_cells.extend(_iter_line(above, across)[1:])
        path_cells.extend(_iter_line(across, below)[1:])

        c0, c1 = sorted((current[1], safe_col))
        _paint_rect_on_bg(out, top_row - blue + 1, top_row, c0, c1)
        if side == 0:
            _paint_rect_on_bg(out, top_row, bottom_row, safe_col, safe_col + blue - 1)
        else:
            _paint_rect_on_bg(out, top_row, bottom_row, safe_col - blue + 1, safe_col)

        current = below

    goal = (h - 1, current[1])
    path_cells.extend(_iter_line(current, goal)[1:])

    for r, c in path_cells:
        if out[r, c] == BG:
            out[r, c] = GREEN

    out[out == BLUE] = RED
    return _from_canonical(out, edge)


def _min_main_len(blues: tuple[int, ...]) -> int:
    """Minimum canonical main-axis size that supports the thicker bypass strips."""
    row = 2 * blues[0]
    for prev, cur in zip(blues, blues[1:]):
        row += max(prev + cur + 2, 2 * cur + 1)
    return row + blues[-1] + 2


def _random_composition(total: int, parts: int) -> list[int]:
    """Uniform non-negative composition of total into parts slots."""
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
    """Pick blue-counts that fit vertically and exhibit the D17 thickening."""
    choices = [
        combo
        for combo in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count)
        if max(combo) >= 2 and _min_main_len(combo) <= main_hi
    ]
    return random.choice(choices)


def _length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    """Valid bar lengths when the open side must fit a width-blue_count strip."""
    lo = max(BAR_LEN_MIN, 2 * blue_count - 1, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, cross_size - 2 * blue_count, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, blue_count: int) -> None:
    """Paint one canonical horizontal bar attached to the left or right wall."""
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
    """Build a canonical top-start grid that exposes the width-per-bar rule."""
    count = len(blues)
    slack = main_size - _min_main_len(blues)
    if slack < 0:
        return None

    extras = _random_composition(slack, count + 1)
    row = 2 * blues[0] + extras[0]
    rows = [row]
    for i in range(1, count):
        row += max(blues[i - 1] + blues[i] + 2, 2 * blues[i] + 1) + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    lengths: list[int] = []
    for row, blue_count, side in zip(rows, blues, ((first_side + i) % 2 for i in range(count))):
        bounds = _length_bounds(cross_size, blue_count)
        if bounds is None:
            return None
        length = random.randint(*bounds)
        lengths.append(length)
        _paint_bar(grid, row, side, length, blue_count)

    first_len = lengths[0]
    if first_side == 0:
        start_col = random.randint(1, first_len - 1)
    else:
        start_col = random.randint(cross_size - first_len, cross_size - 2)
    grid[0, start_col] = GREEN
    return grid


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    """Rotate a canonical top-start grid so the start edge matches edge."""
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()  # edge == "right"


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    """Check that the solved output touches the edge opposite the start."""
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    if edge == "bottom":
        return bool(np.any(out[0, :] == GREEN))
    if edge == "left":
        return bool(np.any(out[:, -1] == GREEN))
    return bool(np.any(out[:, 0] == GREEN))


def _variant_score(inp: np.ndarray, out: np.ndarray) -> int:
    """Count how many bars actually create a thickness > 1 in the output."""
    canonical_inp, edge = _to_canonical(inp)
    if edge == "top":
        canonical_out = np.array(out, copy=True)
    elif edge == "bottom":
        canonical_out = np.flipud(out).copy()
    elif edge == "left":
        canonical_out = out.T.copy()
    else:
        canonical_out = np.fliplr(out).T.copy()
    bars = _extract_bars(canonical_inp)
    score = 0
    for bar in bars:
        blue = int(bar["blue"])
        if blue <= 1:
            continue
        row = int(bar["row"])
        length = int(bar["length"])
        side = int(bar["side"])
        top_row = row - blue - 1
        if side == 0:
            safe_col = length + blue
            patch = canonical_out[top_row : row + blue + 2, safe_col : safe_col + blue]
        else:
            safe_col = canonical_out.shape[1] - length - blue - 1
            patch = canonical_out[top_row : row + blue + 2, safe_col - blue + 1 : safe_col + 1]
        if patch.shape[0] > 0 and patch.shape[1] == blue and np.all(patch == GREEN):
            score += 1
    return score


def _gen_attempt() -> np.ndarray | None:
    """Build one candidate input and validate it with the solver."""
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
        if _variant_score(inp, out) == 0:
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
    raise RuntimeError("failed to generate a valid cb2d8a2c D17 input")
