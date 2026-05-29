# Variant of cb2d8a2c
# Dimension: D2 — 路径转弯规则 (path)
# Change: 遇到障碍条时在距离约束处直角转弯，转弯方向由障碍条位置和路径当前方向决定，路径总是绕向障碍条的远端继续前进... → 路径分叉（遇到障碍两侧同时绕行）

"""
Design analysis for the D2 variant.

a) Concretization and behavior trace
   - Canonical view: rotate the grid so the unique green start sits on the top edge and the path
     moves downward. In this view every obstacle component is a 1-cell-thick horizontal bar.
   - Original behavior at one bar:
     the path reaches the keep-out boundary, turns to one lateral side, slips around the bar's open
     end, then turns back to resume a single vertical trunk.
   - Variant behavior at one bar:
     the trunk reaches the top safe row of the bar, splits across that entire top safe segment,
     descends along both the left and right safe columns, then rejoins on the bottom safe segment
     before continuing downward. The output therefore shows a closed rectangular loop that encloses
     the bar at distance blue_count + 1.
   - Supporting adaptation needed for visibility:
     a wall-attached bar cannot produce a visible two-sided ring because one side is fused into the
     border. The generator therefore keeps bars perpendicular to travel and keeps blue-count encoding,
     but makes each bar float with a gap on both ends. This is the minimum non-D2 adjustment needed
     to make the new routing rule both legal and observable.
   - Small trace:
     if a bar occupies row 6, cols 4..10, and has 2 blue cells, then its safe loop is the rectangle
     with top row 3, bottom row 9, left col 1, right col 13. The trunk at the shared spine column
     reaches (3, spine), the path fans out along row 3, goes down both side columns, rejoins on row
     9, and only then continues vertically.

b) Complexity calibration
   - Blue counts remain 1..4, so the keep-out distance still has only a few repeated values.
   - Bar count remains 1..4, preserving the original object vocabulary.
   - Every generated sample contains at least one full two-sided loop; most contain several. This
     makes the D2 change obvious instead of looking like a minor path wiggle.
   - The loop shape is always rectangular at the minimal legal offset. This keeps the rule crisp and
     avoids too many visually unrelated branch styles.

c) Implicit assumption audit against the original main()
   - The original Dijkstra state stores one predecessor per state. That is valid only when the output
     is a single best path; it loses information as soon as one encounter must paint two branches.
   - The original goal search minimizes turns, which actively suppresses the second branch we now want.
   - The original generator encodes bar placement with a wall side and chooses start columns inside the
     first wall-attached bar span. The new solver instead assumes one shared spine column that lies
     inside every bar span, so the generator must guarantee that.
   - The original blocked/forbidden computation was general, but the variant needs a structured union
     of safe-loop perimeters, not a shortest route through free space. Reusing the old pathfinder would
     be logically wrong even if it still found some valid route.

d) Placement legality
   - In canonical orientation each bar must be contiguous, horizontal, and entirely on one row.
   - For a bar with blue count b and span [c0, c1], the loop uses rows r-(b+1), r+(b+1) and cols
     c0-(b+1), c1+(b+1). The generator therefore enforces left and right bar gaps of at least b+1.
   - Consecutive bar rows differ by at least b_i + b_{i+1} + 3, so adjacent loops do not overlap and
     the trunk can connect from one loop to the next.
   - The start spine column lies inside every bar span, so every loop can split from and rejoin to the
     same trunk without diagonal moves or ambiguity.

e) Visual observability
   - A human can see that each obstacle causes a symmetric enclosure instead of a one-sided bypass.
   - The blue-to-red conversion is still present, so the bar itself stays visible after solving while
     the new green loop wraps around it.
   - Because multiple samples vary bar lengths and blue counts, viewers can correlate larger blue counts
     with larger loop offsets.

f) Visual cleanliness
   - Only three semantic layers remain: azure background, red obstacle bars, green route.
   - Green never overlaps the bar cells; it sits on the safe perimeter around each bar.
   - The single shared trunk plus rectangular loops yields a tidy ladder/ring appearance instead of a
     tangled graph.

g) Color-change inference
   - No new colors are introduced.
   - Blue pixels still act only as a count signal and still convert to red in the output.
   - The generator keeps the blue/red alternation on each bar, now anchored to the smaller coordinate
     end in canonical orientation because the wall-attached end no longer exists.

h) Variant-specific behavior metric used after implementation
   - For each bar we can test whether the solved output paints both safe side columns and both safe
     horizontal connectors. A bar passing this test is a realized two-sided loop.
   - The generator should realize this for every bar in every accepted sample; otherwise it has drifted
     back toward the original one-sided behavior.

i) New solver assumptions to re-check after edits
   - Exactly one green start pixel exists and lies on a border.
   - After canonical rotation, every bar is horizontal and the bars are sorted strictly by row.
   - The shared spine column belongs to every bar span.
   - Every bar has enough side clearance for its loop, and loop rows stay in bounds.
   - The generator below constructs inputs that satisfy all of these assumptions before validation.
"""

import itertools
import random

import numpy as np

from arc_utils import components_mask

BG = 8
BLUE = 1
RED = 2
GREEN = 3


def _to_canonical(grid: np.ndarray) -> tuple[np.ndarray, str]:
    """Rotate/flip so the green start is on the top edge."""
    g = np.array(grid, copy=True)
    h, w = g.shape
    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])

    if sr == 0:
        return g.copy(), "top"
    if sr == h - 1:
        return np.flipud(g).copy(), "bottom"
    if sc == 0:
        return g.T.copy(), "left"
    if sc == w - 1:
        return np.fliplr(g).T.copy(), "right"
    raise AssertionError("green start must be on a border edge")


def _from_canonical(grid: np.ndarray, edge: str) -> np.ndarray:
    """Undo `_to_canonical`."""
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _to_canonical_by_edge(grid: np.ndarray, edge: str) -> np.ndarray:
    """Canonicalize a grid when the start edge is already known from the input."""
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid).T.copy()


def _bar_infos(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return bars as (row, c0, c1, blue_count) in canonical orientation."""
    bar_mask = (grid == BLUE) | (grid == RED)
    comps = components_mask(bar_mask, connectivity=4)
    infos: list[tuple[int, int, int, int]] = []

    for comp in comps:
        rows = sorted({int(r) for r, _ in comp})
        cols = sorted(int(c) for _, c in comp)
        assert len(rows) == 1, "canonical bars must be horizontal"
        row = rows[0]
        c0, c1 = cols[0], cols[-1]
        assert len(comp) == c1 - c0 + 1, "bars must be contiguous segments"
        assert np.all(bar_mask[row, c0 : c1 + 1]), "bars cannot contain holes"
        blue_count = int(np.count_nonzero(grid[row, c0 : c1 + 1] == BLUE))
        assert blue_count >= 1, "each bar must encode a positive keep-out distance"
        infos.append((row, c0, c1, blue_count))

    infos.sort()
    return infos


def _paint_hline(out: np.ndarray, row: int, c0: int, c1: int) -> None:
    """Paint an inclusive horizontal green segment on background/green cells."""
    for col in range(c0, c1 + 1):
        assert out[row, col] in (BG, GREEN), "loop geometry must not overlap a bar"
        out[row, col] = GREEN


def _paint_vline(out: np.ndarray, col: int, r0: int, r1: int) -> None:
    """Paint an inclusive vertical green segment on background/green cells."""
    for row in range(r0, r1 + 1):
        assert out[row, col] in (BG, GREEN), "trunk geometry must not overlap a bar"
        out[row, col] = GREEN


def _count_realized_loops(inp: np.ndarray, out: np.ndarray) -> int:
    """Count bars whose full two-sided loop appears in the output."""
    can_inp, edge = _to_canonical(inp)
    can_out = _to_canonical_by_edge(out, edge)
    count = 0
    for row, c0, c1, blue_count in _bar_infos(can_inp):
        top = row - blue_count - 1
        bottom = row + blue_count + 1
        left = c0 - blue_count - 1
        right = c1 + blue_count + 1
        if (
            np.all(can_out[top, left : right + 1] == GREEN)
            and np.all(can_out[bottom, left : right + 1] == GREEN)
            and np.all(can_out[top : bottom + 1, left] == GREEN)
            and np.all(can_out[top : bottom + 1, right] == GREEN)
        ):
            count += 1
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the border-start green route to the opposite edge by wrapping every obstacle bar with a split loop.

    Input structure:
    - bg = azure (8). Red/blue bars (colors 2/1) are still the only obstacles, and a single green (3) pixel
      on a border edge marks the route start.
    - Each bar is a 1-cell-thick segment perpendicular to the travel direction; blue cells remain embedded in
      the bar and encode how far the route must stay away from it.

    Rule:
    - Rotate to a canonical top-to-bottom view. For each bar, let its keep-out distance be its blue-cell count.
    - When the vertical trunk reaches that bar's top safe row, it splits into left and right branches, both
      travel around the bar at distance blue_count + 1, and they reunite on the opposite safe row before the
      trunk continues toward the far edge.
    - After all loops are drawn, blue cells convert to red just as in the original task.

    Key insight: The blue pixels no longer choose a single detour direction; they size a full two-sided loop,
    so every obstacle becomes enclosed by a green ring at the minimal legal offset.
    """
    canonical, edge = _to_canonical(grid)
    out = canonical.copy()
    h, w = out.shape

    start_pts = np.argwhere(canonical == GREEN)
    sr, sc = map(int, start_pts[0])
    assert sr == 0, "canonical start must be on the top edge"

    bars = _bar_infos(canonical)
    assert bars, "grid must contain at least one obstacle bar"

    current_row = sr
    prev_bar_row = -1
    prev_blue = 0

    for idx, (row, c0, c1, blue_count) in enumerate(bars):
        assert row > prev_bar_row, "bars must be strictly ordered along the travel axis"
        assert c0 <= sc <= c1, "the shared trunk must intersect every bar span"

        top = row - blue_count - 1
        bottom = row + blue_count + 1
        left = c0 - blue_count - 1
        right = c1 + blue_count + 1

        assert 0 <= top < bottom < h, "loop rows must stay in bounds"
        assert 0 <= left < right < w, "loop cols must stay in bounds"
        assert current_row <= top, "bars must leave room for the incoming trunk"
        if idx > 0:
            assert row - prev_bar_row >= prev_blue + blue_count + 3, "loops must not overlap vertically"

        _paint_vline(out, sc, current_row, top)
        _paint_hline(out, top, left, right)
        _paint_vline(out, left, top, bottom)
        _paint_vline(out, right, top, bottom)
        _paint_hline(out, bottom, left, right)

        current_row = bottom
        prev_bar_row = row
        prev_blue = blue_count

    _paint_vline(out, sc, current_row, h - 1)
    out[out == BLUE] = RED
    return _from_canonical(out, edge)


# --------------- generator (o5x) ---------------

# --- Tunable parameters (CFG) ---
# Grid size bounds (rows x cols from spec: 12-29 x 13-26)
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
# Bar counts and blue-per-bar range
BAR_COUNT_LO, BAR_COUNT_HI = 1, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
# Bar length bounds: absolute min/max and cross-axis percentage bounds
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
# Generation retry limit
MAX_GEN_RETRIES = 300


def _min_main_len(blues: tuple[int, ...]) -> int:
    """Minimum main-axis size needed for non-overlapping two-sided loops."""
    row = blues[0] + 2
    for prev_blue, next_blue in zip(blues, blues[1:]):
        row += prev_blue + next_blue + 3
    return row + blues[-1] + 3


def _random_composition(total: int, parts: int) -> list[int]:
    """Distribute `total` into `parts` non-negative integers uniformly."""
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    """Valid floating-bar length range with enough margin for a full loop."""
    lo = max(
        BAR_LEN_MIN,
        2 * blue_count - 1,
        (BAR_CROSS_PCT_LO * cross_size + 99) // 100,
    )
    hi = min(
        BAR_LEN_MAX,
        cross_size - 2 * (blue_count + 1),
        (BAR_CROSS_PCT_HI * cross_size) // 100,
    )
    if lo > hi:
        return None
    return lo, hi


def _choose_blues(count: int, main_hi: int, cross_size: int) -> tuple[int, ...] | None:
    """Pick blue counts that fit both vertically and horizontally."""
    choices = [
        combo
        for combo in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count)
        if _min_main_len(combo) <= main_hi and all(_length_bounds(cross_size, b) is not None for b in combo)
    ]
    if not choices:
        return None
    return random.choice(choices)


def _paint_bar(grid: np.ndarray, row: int, c0: int, length: int, blue_count: int) -> None:
    """Paint one floating bar, alternating blue/red from the smaller-coordinate end."""
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    grid[row, c0 : c0 + length] = colors


def _place_bar_start(cross_size: int, spine: int, length: int, blue_count: int) -> int:
    """Choose the left endpoint of a floating bar that contains the shared spine."""
    lo = max(blue_count + 1, spine - length + 1)
    hi = min(spine, cross_size - blue_count - length - 1)
    assert lo <= hi, "spine-aware floating-bar placement must be feasible"
    return random.randint(lo, hi)


def _canonical_grid(main_size: int, cross_size: int, blues: tuple[int, ...]) -> np.ndarray | None:
    """Build the grid in canonical orientation (start edge = top, path goes down)."""
    slack = main_size - _min_main_len(blues)
    if slack < 0:
        return None

    max_blue = max(blues)
    spine_lo = max_blue + 1
    spine_hi = cross_size - max_blue - 2
    if spine_lo > spine_hi:
        return None
    spine = random.randint(spine_lo, spine_hi)

    extras = _random_composition(slack, len(blues) + 1)
    row = blues[0] + 2 + extras[0]
    rows = [row]
    for idx in range(1, len(blues)):
        row += blues[idx - 1] + blues[idx] + 3 + extras[idx]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    for row, blue_count in zip(rows, blues):
        bounds = _length_bounds(cross_size, blue_count)
        if bounds is None:
            return None
        length = random.randint(*bounds)
        c0 = _place_bar_start(cross_size, spine, length, blue_count)
        _paint_bar(grid, row, c0, length, blue_count)

    grid[0, spine] = GREEN
    return grid


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    """Rotate canonical (top-start) grid so the start edge matches `edge`."""
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    """Check whether the solved output has green on the edge opposite to `edge`."""
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    if edge == "bottom":
        return bool(np.any(out[0, :] == GREEN))
    if edge == "left":
        return bool(np.any(out[:, -1] == GREEN))
    return bool(np.any(out[:, 0] == GREEN))


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: pick parameters, build grid, validate via solver."""
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    count = random.randint(BAR_COUNT_LO, BAR_COUNT_HI)
    cross_size = random.randint(cross_lo, cross_hi)
    blues = _choose_blues(count, main_hi, cross_size)
    if blues is None:
        return None

    main_size = random.randint(max(main_lo, _min_main_len(blues)), main_hi)
    canonical = _canonical_grid(main_size, cross_size, blues)
    if canonical is None:
        return None
    inp = _orient_grid(canonical, edge)

    try:
        out = main(inp)
        if not _reaches_opposite_edge(out, edge):
            return None
        if _count_realized_loops(inp, out) != len(_bar_infos(canonical)):
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
    raise RuntimeError("failed to generate a valid cb2d8a2c input")
