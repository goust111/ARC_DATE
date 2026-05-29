# Combo variant of cb2d8a2c (index 29)
# combo_spec: D2/3+D5/0+D10/0
# Name: 路径分叉+多起点+迷宫布局

"""
Design analysis for the combo variant.

a) Concretization and behavior trace
   - Canonical view: rotate the grid so every green source pixel lies on the top edge and all
     obstacle components are horizontal bars. The output is then a downward-growing green network.
   - Original task behavior:
     one border start launches one path; each obstacle is bypassed on only one side; obstacles are
     sparse enough that the path can be treated as one shortest route.
   - Fused behavior here:
     2-3 top-edge green starts launch 2-3 parallel trunks. Every bar contains that whole start band,
     so each trunk reaches the bar's top safe row. At that row the route does not pick one side:
     it paints the full top connector, descends on both safe side columns, paints the full bottom
     connector, and then the same trunks continue downward. Repeating this on a densely packed stack
     of laterally shifted bars creates a woven ladder/maze rather than a single snake.
   - Small trace:
     if starts are at cols 7 and 10 and a bar sits on row 8, cols 5..12 with blue_count = 2, then
     the solver paints both trunks down to row 5, fills row 5 from col 2 to 15, descends on cols 2
     and 15 to row 11, fills row 11 from col 2 to 15, then both trunks resume at cols 7 and 10.
     The next bar can be only one row-gap below and laterally shifted, so these rectangles visually
     interlock into a dense maze.

b) Complexity calibration
   - Start count is limited to 2 or 3. That is enough to make multi-source behavior obvious without
     turning the output into a solid block of green.
   - Bar count is limited to 3 or 4, and blue counts stay in 1..3. Those few repeated radii keep the
     keep-out rule learnable while still allowing several stacked loops in one sample.
   - Bar spans all cover the start band but their left endpoints deliberately zigzag. This makes the
     D10 change visible: the obstacle field is dense and corridor-like, not just a sparse set of bars.

c) Implicit assumption audit against the original main()
   - The original `start_pts.shape[0] == 1` fails immediately: we now require multiple green border
     cells on the same source edge.
   - The original Dijkstra predecessor map can only recover one best path. That discards the D2 split
     branches, so it cannot represent the fused output.
   - The original generator assumed one start column had to lie inside the first bar only. The fused
     solver assumes every start column lies inside every bar span, otherwise some trunks would miss
     later obstacles and the behavior would become mixed/ambiguous.
   - The original "some valid shortest route exists" assumption is replaced by a stronger structured
     assumption: bars are sorted by row, horizontal, non-overlapping vertically, and have enough side
     clearance for full rectangular split loops. The generator must guarantee all of that explicitly.

d) Placement legality
   - Each canonical bar is a contiguous horizontal segment on one row.
   - For a bar `(row, c0..c1, blue_count=b)`, the visible split loop uses rows `row-(b+1)` and
     `row+(b+1)` plus side columns `c0-(b+1)` and `c1+(b+1)`. The generator therefore leaves at
     least `b+1` empty columns on both sides of every bar.
   - Consecutive bars satisfy `next_row - prev_row >= prev_b + next_b + 3`, so loop rectangles never
     overlap vertically. The generator biases this gap toward the minimum, producing the intended maze
     density without illegal collisions.
   - The full start band is contained in every bar span. That makes every start interact with every
     obstacle and ensures the multi-source rule is visible in every accepted sample.

e) Visual observability
   - Humans can read the input directly: multiple green source pixels on one edge, a dense stack of
     red/blue bars, and no hidden ordering cue based on color ids or metadata.
   - Humans can read the output directly: each source grows downward; each bar creates a symmetric
     two-sided enclosure; the bars are packed tightly enough that the green rails look maze-like.
   - Because all bars cover the whole source band, every sample shows all three fused ingredients at
     once rather than isolating them into separate unrelated subcases.

f) Visual cleanliness
   - Only the original semantic colors remain: azure background, blue/red obstacle encoding, green
     paths. No extra marker colors or labels are introduced.
   - Green overlaps only other green. It never paints over red/blue bar cells, so obstacle identity
     remains legible after solving.
   - The generator avoids degenerate spans where every bar is aligned identically; some lateral shift
     is mandatory so the obstacle stack reads as a dense corridor system instead of a trivial ladder.

g) Color-change inference
   - Blue still means "count the keep-out radius" and still turns red in the output.
   - No new color mapping has to be inferred; the novelty is purely in multiplicity of starts and in
     the topology of the green network.

h) Variant-specific behavior metrics used after implementation
   - `realized_loops`: how many bars show the full split rectangle in the solved output. This should
     equal the bar count.
   - `start_count`: how many source pixels appear on the source edge. This should be 2 or 3.
   - `tight_pairs`: how many consecutive loop rectangles are separated by at most one blank row.
     Large values indicate the intended maze density rather than a sparse fallback.
   - `maze_shift`: `max(bar_lefts) - min(bar_lefts)`. If this is too small, the layout has collapsed
     back to a simple aligned stack and no longer exhibits the D10 lateral crowding cue.

i) New solver assumptions to re-check after edits
   - All green input pixels lie on one border edge and there are no interior green pixels in the input.
   - After canonicalization, every obstacle component is one contiguous horizontal bar.
   - Every bar contains the full start band and leaves enough margin for its split loop.
   - Bars are sorted strictly by row and vertically spaced to keep loop rectangles disjoint.
   - `generate_input()` below enforces these conditions and rejects any sample whose measured behavior
     does not satisfy the fused metrics.
"""

import itertools
import random

import numpy as np

from arc_utils import components_mask

BG = 8
BLUE = 1
RED = 2
GREEN = 3

# --- Tunable parameters (CFG) ---
ROWS_LO, ROWS_HI = 18, 29
COLS_LO, COLS_HI = 18, 26
START_COUNT_LO, START_COUNT_HI = 2, 3
BAR_COUNT_LO, BAR_COUNT_HI = 3, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 3
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 40, 82
MAX_EXTRA_MAIN = 6
MAX_GEN_RETRIES = 500


def _to_canonical(grid: np.ndarray) -> tuple[np.ndarray, str]:
    """Rotate/flip so that all green sources lie on the top edge."""
    g = np.array(grid, copy=True)
    h, w = g.shape
    green_pts = [tuple(map(int, rc)) for rc in np.argwhere(g == GREEN)]
    assert len(green_pts) >= 2, "grid must contain multiple green source pixels"

    all_top = all(r == 0 for r, _ in green_pts)
    all_bottom = all(r == h - 1 for r, _ in green_pts)
    all_left = all(c == 0 for _, c in green_pts)
    all_right = all(c == w - 1 for _, c in green_pts)
    flags = [all_top, all_bottom, all_left, all_right]
    assert sum(flags) == 1, "all green source pixels must lie on the same border edge"

    if all_top:
        return g.copy(), "top"
    if all_bottom:
        return np.flipud(g).copy(), "bottom"
    if all_left:
        return g.T.copy(), "left"
    return np.fliplr(g).T.copy(), "right"


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
    """Canonicalize a grid when the source edge is already known."""
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid).T.copy()


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
    """Paint an inclusive horizontal green segment onto background/green cells."""
    for col in range(c0, c1 + 1):
        assert out[row, col] in (BG, GREEN), "loop geometry must not overlap a bar"
        out[row, col] = GREEN


def _paint_vline(out: np.ndarray, col: int, r0: int, r1: int) -> None:
    """Paint an inclusive vertical green segment onto background/green cells."""
    for row in range(r0, r1 + 1):
        assert out[row, col] in (BG, GREEN), "trunk geometry must not overlap a bar"
        out[row, col] = GREEN


def _count_green_on_edge(grid: np.ndarray, edge: str) -> int:
    if edge == "top":
        return int(np.count_nonzero(grid[0, :] == GREEN))
    if edge == "bottom":
        return int(np.count_nonzero(grid[-1, :] == GREEN))
    if edge == "left":
        return int(np.count_nonzero(grid[:, 0] == GREEN))
    return int(np.count_nonzero(grid[:, -1] == GREEN))


def _source_columns(canonical: np.ndarray) -> list[int]:
    """Return sorted top-edge green source columns in canonical orientation."""
    start_cols = sorted(int(c) for c in np.where(canonical[0] == GREEN)[0])
    assert START_COUNT_LO <= len(start_cols) <= START_COUNT_HI, "invalid number of source pixels"
    assert int(np.count_nonzero(canonical == GREEN)) == len(start_cols), "input green must stay on the source edge"
    return start_cols


def _loop_box(bar: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    row, c0, c1, blue_count = bar
    return (
        row - blue_count - 1,
        c0 - blue_count - 1,
        row + blue_count + 1,
        c1 + blue_count + 1,
    )


def _realized_loop_count(inp: np.ndarray, out: np.ndarray) -> int:
    """Count bars whose full split rectangle appears in the solved output."""
    can_inp, edge = _to_canonical(inp)
    can_out = _to_canonical_by_edge(out, edge)
    count = 0
    for bar in _bar_infos(can_inp):
        top, left, bottom, right = _loop_box(bar)
        if (
            np.all(can_out[top, left : right + 1] == GREEN)
            and np.all(can_out[bottom, left : right + 1] == GREEN)
            and np.all(can_out[top : bottom + 1, left] == GREEN)
            and np.all(can_out[top : bottom + 1, right] == GREEN)
        ):
            count += 1
    return count


def fusion_metrics(inp: np.ndarray, out: np.ndarray | None = None) -> dict[str, int]:
    """Measure the fused behavior for a sample."""
    can_inp, edge = _to_canonical(inp)
    bars = _bar_infos(can_inp)
    starts = _source_columns(can_inp)
    band_center2 = starts[0] + starts[-1]

    tight_pairs = 0
    for prev_bar, next_bar in zip(bars, bars[1:]):
        prev_bottom = prev_bar[0] + prev_bar[3] + 1
        next_top = next_bar[0] - next_bar[3] - 1
        if next_top - prev_bottom - 1 <= 1:
            tight_pairs += 1

    bar_lefts = [bar[1] for bar in bars]
    maze_shift = max(bar_lefts) - min(bar_lefts)
    zigzag_pairs = 0
    center_sides = []
    for _row, c0, c1, _b in bars:
        center_sides.append(np.sign(c0 + c1 - band_center2))
    for a, b in zip(center_sides, center_sides[1:]):
        if a != 0 and b != 0 and a != b:
            zigzag_pairs += 1

    metrics = {
        "start_count": len(starts),
        "bar_count": len(bars),
        "tight_pairs": tight_pairs,
        "maze_shift": maze_shift,
        "zigzag_pairs": zigzag_pairs,
    }

    if out is not None:
        can_out = _to_canonical_by_edge(out, edge)
        metrics["realized_loops"] = _realized_loop_count(inp, out)
        metrics["exit_count"] = int(np.count_nonzero(can_out[-1, :] == GREEN))
    return metrics


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a multi-source green network across a dense bar maze by splitting around every obstacle on both sides.

    Input structure:
    - bg = azure (8). Red/blue obstacle bars (2/1) are still 1-cell-thick horizontal strips after canonical rotation.
    - Multiple green (3) source pixels sit on one border edge; each one launches its own downward trunk.
    - Blue cells embedded in each bar still encode that bar's keep-out distance.

    Rule:
    - Rotate to a canonical top-start view. For each bar, let `blue_count` be the number of blue cells in that bar.
    - Every trunk descends to the bar's top safe row, then the route splits symmetrically: paint the full top safe
      connector, both side safe columns, and the full bottom safe connector at offset `blue_count + 1`.
    - After the split rectangle is drawn, the same trunks continue downward from their original columns toward the
      opposite edge. When all bars have been processed, blue cells turn red.

    Key insight: The green pixels are no longer one shortest path seed. They define several simultaneous trunks, and
    each bar forces a full two-sided enclosure, so dense laterally shifted bars turn the output into a woven maze of
    shared rails and repeated split loops.
    """
    canonical, edge = _to_canonical(grid)
    out = canonical.copy()
    h, w = out.shape

    start_cols = _source_columns(canonical)
    bars = _bar_infos(canonical)
    assert bars, "grid must contain at least one obstacle bar"

    band0, band1 = start_cols[0], start_cols[-1]
    current_row = 0
    prev_bar_row = -1
    prev_blue = 0

    for idx, bar in enumerate(bars):
        row, c0, c1, blue_count = bar
        assert row > prev_bar_row, "bars must be strictly ordered along the travel axis"
        assert c0 <= band0 and band1 <= c1, "every bar must cover the whole source band"

        top, left, bottom, right = _loop_box(bar)
        assert 0 <= top < bottom < h, "loop rows must stay in bounds"
        assert 0 <= left < right < w, "loop cols must stay in bounds"
        assert current_row <= top, "bars must leave room for the incoming trunks"
        if idx > 0:
            assert row - prev_bar_row >= prev_blue + blue_count + 3, "loop rectangles must not overlap vertically"

        for col in start_cols:
            _paint_vline(out, col, current_row, top)
        _paint_hline(out, top, left, right)
        _paint_vline(out, left, top, bottom)
        _paint_vline(out, right, top, bottom)
        _paint_hline(out, bottom, left, right)

        current_row = bottom
        prev_bar_row = row
        prev_blue = blue_count

    for col in start_cols:
        _paint_vline(out, col, current_row, h - 1)

    out[out == BLUE] = RED
    return _from_canonical(out, edge)


def _min_main_len(blues: tuple[int, ...]) -> int:
    """Minimum main-axis size for non-overlapping split rectangles."""
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


def _length_bounds(cross_size: int, blue_count: int, band_width: int) -> tuple[int, int] | None:
    """Valid floating-bar length range containing the full source band."""
    lo = max(
        BAR_LEN_MIN,
        2 * blue_count - 1,
        band_width,
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


def _choose_blues(count: int, main_hi: int, cross_size: int, band_width: int) -> tuple[int, ...] | None:
    """Pick blue counts that fit the dense vertical stack and the cross-axis band."""
    choices = [
        combo
        for combo in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count)
        if _min_main_len(combo) <= main_hi
        and all(_length_bounds(cross_size, blue_count, band_width) is not None for blue_count in combo)
    ]
    if not choices:
        return None
    return random.choice(choices)


def _pick_band(cross_size: int, max_blue: int, start_count: int) -> tuple[list[int], int, int] | None:
    """Choose a visible top-edge source band and the exact source columns inside it."""
    band_span = start_count + random.randint(1, 2)
    min_c0 = max_blue + 2
    max_c0 = cross_size - max_blue - 2 - band_span
    if min_c0 > max_c0:
        return None

    target_center = cross_size // 2 - band_span // 2
    lo = max(min_c0, target_center - 2)
    hi = min(max_c0, target_center + 2)
    if lo > hi:
        lo, hi = min_c0, max_c0
    band0 = random.randint(lo, hi)
    band1 = band0 + band_span - 1

    if start_count == 2:
        starts = [band0, band1]
    else:
        starts = [band0, band0 + band_span // 2, band1]
    return starts, band0, band1


def _biased_pick(lo: int, hi: int, side: int) -> int:
    """Pick near the left or right side of a feasible integer interval."""
    assert lo <= hi
    if lo == hi:
        return lo
    width = hi - lo
    span = max(1, width // 3)
    if side == 0:
        return random.randint(lo, min(hi, lo + span))
    return random.randint(max(lo, hi - span), hi)


def _choose_spans(
    cross_size: int,
    blues: tuple[int, ...],
    band0: int,
    band1: int,
) -> list[tuple[int, int]] | None:
    """Choose laterally shifted floating bars that all contain the source band."""
    spans: list[tuple[int, int]] = []
    side = random.randrange(2)
    for blue_count in blues:
        bounds = _length_bounds(cross_size, int(blue_count), band1 - band0 + 1)
        if bounds is None:
            return None
        lo_len, hi_len = bounds
        length = random.randint(max(lo_len, hi_len - 4), hi_len)

        lo_c0 = max(blue_count + 1, band1 - length + 1)
        hi_c0 = min(band0, cross_size - blue_count - length - 1)
        if lo_c0 > hi_c0:
            return None
        c0 = _biased_pick(lo_c0, hi_c0, side)
        spans.append((c0, c0 + length - 1))
        side = 1 - side
    return spans


def _paint_bar(grid: np.ndarray, row: int, c0: int, c1: int, blue_count: int) -> None:
    """Paint one floating bar with alternating blue/red prefix from the smaller-coordinate end."""
    length = c1 - c0 + 1
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    grid[row, c0 : c1 + 1] = colors


def _canonical_grid(
    main_size: int,
    cross_size: int,
    blues: tuple[int, ...],
    start_count: int,
) -> np.ndarray | None:
    """Build the grid in canonical orientation: multi-start on top, travel downward."""
    slack = main_size - _min_main_len(blues)
    if slack < 0:
        return None

    band_data = _pick_band(cross_size, max(blues), start_count)
    if band_data is None:
        return None
    starts, band0, band1 = band_data

    spans = _choose_spans(cross_size, blues, band0, band1)
    if spans is None:
        return None

    lefts = [c0 for c0, _c1 in spans]
    if max(lefts) - min(lefts) < 2:
        return None

    extras = _random_composition(slack, len(blues) + 1)
    row = blues[0] + 2 + extras[0]
    rows = [row]
    for idx in range(1, len(blues)):
        row += blues[idx - 1] + blues[idx] + 3 + extras[idx]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    for row, blue_count, (c0, c1) in zip(rows, blues, spans):
        _paint_bar(grid, row, c0, c1, int(blue_count))
    for col in starts:
        grid[0, col] = GREEN
    return grid


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    """Rotate canonical top-start layout to the requested source edge."""
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt, validated against the fused solver and metrics."""
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    start_count = random.randint(START_COUNT_LO, START_COUNT_HI)
    count = random.randint(BAR_COUNT_LO, BAR_COUNT_HI)
    cross_size = random.randint(cross_lo, cross_hi)

    # Visible multi-start spacing needs a few columns beyond the strict minimum.
    band_width = start_count + 1
    blues = _choose_blues(count, main_hi, cross_size, band_width)
    if blues is None:
        return None

    min_main = max(main_lo, _min_main_len(blues))
    max_main = min(main_hi, _min_main_len(blues) + MAX_EXTRA_MAIN)
    if min_main > max_main:
        return None
    main_size = random.randint(min_main, max_main)

    canonical = _canonical_grid(main_size, cross_size, blues, start_count)
    if canonical is None:
        return None
    inp = _orient_grid(canonical, edge)

    try:
        out = main(inp)
        metrics = fusion_metrics(inp, out)
        if metrics["realized_loops"] != metrics["bar_count"]:
            return None
        if metrics["start_count"] != metrics["exit_count"]:
            return None
        if metrics["tight_pairs"] < metrics["bar_count"] - 2:
            return None
        if metrics["maze_shift"] < 2:
            return None
        if metrics["zigzag_pairs"] < 1:
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
