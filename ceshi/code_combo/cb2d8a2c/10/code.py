# Combo variant of cb2d8a2c (index 10)
# combo_spec: D2/3+D5/0
# Name: 多路径起点+路径分叉

"""
Design analysis for the D2/3 + D5/0 combo variant.

a) Concretization and behavior trace
   - Canonical view: rotate/flip the grid so all green start pixels lie on the top edge and the
     routes travel downward. In this view every obstacle component is a floating horizontal bar.
   - Original task at one obstacle:
     one green border pixel generates one shortest legal route, and each bar is bypassed on only one
     side according to the best turn sequence.
   - D2/3 alone changes that local event:
     when a route reaches a bar, it no longer chooses one side. It paints the whole minimal safe
     rectangle around the bar: top bridge, both side rails, and bottom bridge.
   - D5/0 alone changes the source configuration:
     several green pixels on the same edge each launch their own straight trunk toward the opposite
     side.
   - In the fused version:
     every trunk meets every bar, every encounter expands into the same full rectangle, and the
     trunks temporarily merge onto the rectangle's top/bottom bridges before separating again along
     their own columns. A small concrete trace is:
     three starts at cols 6, 8, 10 approach a bar on row 7 spanning cols 4..12 with blue-count 2.
     The safe rectangle is rows 4..10 and cols 1..15. All three trunks descend to row 4, fan into
     the shared top bridge, both side rails are painted, the bottom bridge appears on row 10, and
     then all three trunks continue downward from cols 6, 8, 10. The difference from the original is
     both multiplicity of trunks and explicit two-sided enclosure.

b) Complexity calibration
   - Start count is restricted to 2..4 and bar count to 2..4. That is enough to make the output a
     genuine network while keeping the structure visually parseable from a few examples.
   - Blue counts remain 1..4, preserving the original discrete keep-out encoding.
   - Bars remain 1-cell-thick and axis-aligned, so the learner still sees one coherent family of
     objects rather than several unrelated obstacle types.
   - Every accepted sample realizes multi-start branching on every bar, so the fusion is visible in
     all examples instead of appearing only in rare corner cases.

c) Implicit assumption audit against the original main()
   - The original `assert start_pts.shape[0] == 1` is invalid: the combo requires several starts.
   - The original initial-direction logic assumes one state; with several starts it would need one
     state per trunk.
   - The original Dijkstra predecessor map stores only one predecessor per state, which cannot encode
     a branch that must paint both sides simultaneously.
   - The original shortest-path objective would actively delete the extra branch structure even if
     multiple starts were added.
   - The combo therefore replaces the pathfinder with direct geometry on canonical bars. That new
     logic introduces different assumptions: one shared start edge, horizontal floating bars after
     canonicalization, enough margin for every loop, and every bar covering every start column. The
     generator below enforces each of these assumptions explicitly.

d) Placement legality
   - For a bar on row `r`, span `[c0, c1]`, and blue-count `b`, the branch loop uses rows
     `r-(b+1)` and `r+(b+1)` plus cols `c0-(b+1)` and `c1+(b+1)`. The generator therefore keeps
     bars away from both side walls by at least `b+1`.
   - Consecutive bar rows differ by at least `b_i + b_{i+1} + 3`, so neighboring rectangles do not
     overlap vertically.
   - The chosen start columns lie inside every bar span, guaranteeing that each trunk reaches every
     obstacle and that the fused behavior cannot collapse back to "some starts do nothing".

e) Visual observability
   - Multiple green seeds are already visible in the input, so the learner can anticipate multiple
     descendants in the output.
   - Each obstacle produces a symmetric loop instead of a one-sided kink, making the D2/3 component
     obvious.
   - Because the same bar is contacted by several trunks, the output shows visible merges on the loop
     bridges and visible separations below them. That makes the interaction between the two dimensions
     legible in just a few examples.

f) Visual cleanliness
   - Only azure background, red obstacles, and green routes remain after solving.
   - Green never overwrites the obstacle cells themselves; bars stay readable after blue->red
     conversion.
   - Starts are kept in a compact horizontal band, so the network reads like a multi-rail ladder
     rather than a noisy tangle.

g) Color-change inference
   - No new colors are introduced.
   - Blue still has a single role: it encodes keep-out radius and then turns red in the output.
   - The route behavior is conveyed by shape, not by adding another color mapping that would need
     separate inference.

h) Variant-specific behavior metric used after implementation
   - `_count_realized_branch_bars(inp, out)` counts how many bars show a full two-sided loop and
     computes a branch score equal to `realized_bars * number_of_starts`.
   - Requiring branch score to hit that maximum for accepted samples ensures that both combo halves
     are actually active: there must be multiple starts, and every start must participate in every
     split loop.

i) New solver assumptions to re-check after edits
   - All green input pixels lie on one border edge and there are at least two of them.
   - After canonicalization, the starts occupy only the top row.
   - Every obstacle component is a contiguous horizontal bar with positive blue-count.
   - Every bar covers every start column and has enough clearance for its loop.
   - The generator validates these assumptions by constructing only such inputs and by re-running the
     solver plus the branch metric before accepting a sample.
"""

import itertools
import random

import numpy as np

from arc_utils import components_mask

BG = 8
BLUE = 1
RED = 2
GREEN = 3


def _detect_start_edge(grid: np.ndarray) -> str:
    """Return the unique border edge that contains all green start pixels."""
    g = np.asarray(grid)
    h, w = g.shape
    starts = np.argwhere(g == GREEN)
    assert starts.shape[0] >= 2, "combo variant requires at least two green start pixels"

    if np.all(starts[:, 0] == 0):
        return "top"
    if np.all(starts[:, 0] == h - 1):
        return "bottom"
    if np.all(starts[:, 1] == 0):
        return "left"
    if np.all(starts[:, 1] == w - 1):
        return "right"
    raise AssertionError("all green start pixels must lie on one border edge")


def _to_canonical(grid: np.ndarray) -> tuple[np.ndarray, str]:
    """Rotate/flip so the green starts lie on the top edge."""
    g = np.array(grid, copy=True)
    edge = _detect_start_edge(g)
    if edge == "top":
        return g.copy(), edge
    if edge == "bottom":
        return np.flipud(g).copy(), edge
    if edge == "left":
        return g.T.copy(), edge
    return np.fliplr(g).T.copy(), edge


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
    """Canonicalize a grid using a known start edge."""
    if edge == "top":
        return np.array(grid, copy=True)
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid).T.copy()


def _start_columns(canonical: np.ndarray) -> list[int]:
    """Return sorted start columns in canonical orientation."""
    starts = np.argwhere(canonical == GREEN)
    assert starts.shape[0] >= 2, "canonical input must retain multiple starts"
    assert np.all(starts[:, 0] == 0), "canonical green starts must be on the top row"
    cols = sorted(int(c) for _, c in starts)
    assert len(cols) == len(set(cols)), "start columns must be unique"
    return cols


def _bar_infos(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return bars as `(row, c0, c1, blue_count)` in canonical orientation."""
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
    """Paint an inclusive horizontal green segment."""
    for col in range(c0, c1 + 1):
        assert out[row, col] in (BG, GREEN), "green bridges must not overlap bars"
        out[row, col] = GREEN


def _paint_vline(out: np.ndarray, col: int, r0: int, r1: int) -> None:
    """Paint an inclusive vertical green segment."""
    for row in range(r0, r1 + 1):
        assert out[row, col] in (BG, GREEN), "green trunks must not overlap bars"
        out[row, col] = GREEN


def _count_realized_branch_bars(inp: np.ndarray, out: np.ndarray) -> tuple[int, int]:
    """Return `(realized_bar_count, branch_score)` for the combo behavior."""
    can_inp, edge = _to_canonical(inp)
    can_out = _to_canonical_by_edge(out, edge)
    start_cols = _start_columns(can_inp)

    realized = 0
    branch_score = 0
    for row, c0, c1, blue_count in _bar_infos(can_inp):
        top = row - blue_count - 1
        bottom = row + blue_count + 1
        left = c0 - blue_count - 1
        right = c1 + blue_count + 1

        if not (
            np.all(can_out[top, left : right + 1] == GREEN)
            and np.all(can_out[bottom, left : right + 1] == GREEN)
            and np.all(can_out[top : bottom + 1, left] == GREEN)
            and np.all(can_out[top : bottom + 1, right] == GREEN)
        ):
            continue
        if not all(can_out[top, col] == GREEN and can_out[bottom, col] == GREEN for col in start_cols):
            continue
        realized += 1
        branch_score += len(start_cols)

    return realized, branch_score


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow several border-start green trunks into a shared split-loop network that reaches the opposite edge.

    Input structure:
    - bg = azure (8). Red/blue bars (colors 2/1) remain the only obstacle objects.
    - Multiple green (3) pixels lie on one border edge and mark several simultaneous path starts.
    - Each obstacle is a 1-cell-thick floating bar perpendicular to travel; its blue pixels still encode the
      required keep-out distance.

    Rule:
    - Rotate to a canonical top-to-bottom view. Every green start emits a vertical trunk downward.
    - For a bar with blue-count `b`, the legal split loop is the minimal rectangle at offset `b + 1` from the
      bar: top bridge, both side rails, and bottom bridge.
    - Whenever the trunks reach a bar, all of them branch across that rectangle, share its bridges, descend
      along both side rails, then continue downward from their own original columns after the lower bridge.
    - After all bars are processed, blue pixels convert to red.

    Key insight: The blue count still sizes the safe offset, but now it controls a full two-sided enclosure for
    every obstacle, and several independent starts feed into the same loop geometry to form a visible network.
    """
    canonical, edge = _to_canonical(grid)
    out = canonical.copy()
    h, w = out.shape

    start_cols = _start_columns(canonical)
    bars = _bar_infos(canonical)
    assert bars, "grid must contain at least one obstacle bar"

    current_rows = {col: 0 for col in start_cols}
    prev_bar_row = -1
    prev_blue = 0

    for idx, (row, c0, c1, blue_count) in enumerate(bars):
        assert row > prev_bar_row, "bars must be strictly ordered along the travel axis"
        assert all(c0 <= col <= c1 for col in start_cols), "every bar must cover every start column"

        top = row - blue_count - 1
        bottom = row + blue_count + 1
        left = c0 - blue_count - 1
        right = c1 + blue_count + 1

        assert 0 <= top < bottom < h, "loop rows must stay in bounds"
        assert 0 <= left < right < w, "loop cols must stay in bounds"
        if idx > 0:
            assert row - prev_bar_row >= prev_blue + blue_count + 3, "loops must not overlap vertically"

        for col in start_cols:
            assert current_rows[col] <= top, "incoming trunks must reach the next loop from above"
            _paint_vline(out, col, current_rows[col], top)

        _paint_hline(out, top, left, right)
        _paint_vline(out, left, top, bottom)
        _paint_vline(out, right, top, bottom)
        _paint_hline(out, bottom, left, right)

        for col in start_cols:
            current_rows[col] = bottom

        prev_bar_row = row
        prev_blue = blue_count

    for col, row0 in current_rows.items():
        _paint_vline(out, col, row0, h - 1)

    out[out == BLUE] = RED
    return _from_canonical(out, edge)


# --------------- generator (o5x) ---------------

# --- Tunable parameters (CFG) ---
# Grid size bounds (rows x cols from spec: 12-29 x 13-26)
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
# Multi-start count
START_COUNT_LO, START_COUNT_HI = 2, 4
# Bar counts and blue-per-bar range
BAR_COUNT_LO, BAR_COUNT_HI = 2, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
# Bar length bounds: absolute min/max and cross-axis percentage bounds
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
# Generation retry limit
MAX_GEN_RETRIES = 400


def _min_main_len(blues: tuple[int, ...]) -> int:
    """Minimum main-axis size needed for non-overlapping split loops."""
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
    """Pick blue counts that fit both the main axis and the cross axis."""
    choices = [
        combo
        for combo in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count)
        if _min_main_len(combo) <= main_hi and all(_length_bounds(cross_size, b) is not None for b in combo)
    ]
    if not choices:
        return None
    return random.choice(choices)


def _choose_start_cols(cross_size: int, max_blue: int) -> list[int] | None:
    """Pick 2..4 nearby start columns that leave room for every loop side rail."""
    inner_lo = max_blue + 1
    inner_hi = cross_size - max_blue - 2
    if inner_lo > inner_hi:
        return None

    available = inner_hi - inner_lo + 1
    max_count = min(START_COUNT_HI, available)
    if max_count < START_COUNT_LO:
        return None

    count = random.randint(START_COUNT_LO, max_count)
    band_width = random.randint(count, min(available, count + 4))
    band_lo = random.randint(inner_lo, inner_hi - band_width + 1)
    candidates = list(range(band_lo, band_lo + band_width))
    return sorted(random.sample(candidates, count))


def _bar_placement_options(
    cross_size: int,
    blue_count: int,
    start_lo: int,
    start_hi: int,
) -> list[tuple[int, int, int]]:
    """Return feasible `(length, c0_lo, c0_hi)` choices for a bar covering all starts."""
    bounds = _length_bounds(cross_size, blue_count)
    if bounds is None:
        return []

    lo_len, hi_len = bounds
    options: list[tuple[int, int, int]] = []
    for length in range(lo_len, hi_len + 1):
        c0_lo = max(blue_count + 1, start_hi - length + 1)
        c0_hi = min(start_lo, cross_size - blue_count - length - 1)
        if c0_lo <= c0_hi:
            options.append((length, c0_lo, c0_hi))
    return options


def _paint_bar(grid: np.ndarray, row: int, c0: int, length: int, blue_count: int) -> None:
    """Paint one floating bar, alternating blue/red from the smaller-coordinate end."""
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    grid[row, c0 : c0 + length] = colors


def _canonical_grid(main_size: int, cross_size: int, blues: tuple[int, ...], start_cols: list[int]) -> np.ndarray | None:
    """Build the grid in canonical orientation (start edge = top, paths go down)."""
    slack = main_size - _min_main_len(blues)
    if slack < 0:
        return None

    start_lo = start_cols[0]
    start_hi = start_cols[-1]

    extras = _random_composition(slack, len(blues) + 1)
    row = blues[0] + 2 + extras[0]
    rows = [row]
    for idx in range(1, len(blues)):
        row += blues[idx - 1] + blues[idx] + 3 + extras[idx]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    for row, blue_count in zip(rows, blues):
        options = _bar_placement_options(cross_size, blue_count, start_lo, start_hi)
        if not options:
            return None
        length, c0_lo, c0_hi = random.choice(options)
        c0 = random.randint(c0_lo, c0_hi)
        _paint_bar(grid, row, c0, length, blue_count)

    grid[0, start_cols] = GREEN
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


def _opposite_edge_green_count(out: np.ndarray, edge: str) -> int:
    """Count green cells on the edge opposite to `edge`."""
    if edge == "top":
        return int(np.count_nonzero(out[-1, :] == GREEN))
    if edge == "bottom":
        return int(np.count_nonzero(out[0, :] == GREEN))
    if edge == "left":
        return int(np.count_nonzero(out[:, -1] == GREEN))
    return int(np.count_nonzero(out[:, 0] == GREEN))


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

    start_cols = _choose_start_cols(cross_size, max(blues))
    if start_cols is None:
        return None

    main_size = random.randint(max(main_lo, _min_main_len(blues)), main_hi)
    canonical = _canonical_grid(main_size, cross_size, blues, start_cols)
    if canonical is None:
        return None
    inp = _orient_grid(canonical, edge)

    try:
        out = main(inp)
        if _opposite_edge_green_count(out, edge) != len(start_cols):
            return None
        realized_bars, branch_score = _count_realized_branch_bars(inp, out)
        if realized_bars != len(blues):
            return None
        if branch_score != len(blues) * len(start_cols):
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
