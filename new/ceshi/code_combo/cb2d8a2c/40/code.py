# Combo variant of cb2d8a2c (index 40)
# combo_spec: D3/3+D20/2
# Name: 蓝色数编码行为模式+GCD间距约束

import random

import numpy as np

from arc_utils import components_mask


BG = 8
BLUE = 1
RED = 2
GREEN = 3

# --- Tunable parameters (kept close to the seed task where still compatible) ---
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 2, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 500


# Design analysis for the fused variant
# ------------------------------------
# 1. Concrete behavior:
#    The seed task used wall-attached bars plus a keep-out radius. That layout
#    cannot express "1=left detour / 2=right detour" because a wall-attached bar
#    only leaves one bypass side. The fused task therefore switches to floating
#    horizontal bars in canonical orientation (top-start, path moving downward).
#    Blue count now has two simultaneous roles:
#      - behavior code: 1->detour to the left end, 2->detour to the right end,
#        3->go straight through the bar, 4->terminate after this bar
#      - spacing code: while crossing that bar's local row-band, green marks are
#        placed every `blue_count` rows instead of continuously
#    Example: if the current path column hits a bar with 2 blue cells, the path
#    reaches row `bar_row-2`, shifts horizontally to `bar_right+1`, then marks
#    only rows `bar_row-2, bar_row, bar_row+2` on that bypass column.
#
# 2. Complexity calibration / observability:
#    Only four behavior codes exist and each remains tied to the visible blue
#    count, so the hypothesis space stays small. Counts 2/3/4 create dotted
#    vertical passages with visibly different spacing, while counts 1/2 also
#    differ by turning left vs right. The generator avoids one-bar instances and
#    forces at least one count > 1 so the fused change is visible.
#
# 3. Hidden assumptions broken from the seed solver:
#    - Bars are no longer wall-attached.
#    - The path no longer minimizes a shortest route; it is reconstructed
#      deterministically from per-bar behavior codes.
#    - A bar with four blue cells can stop the path before the opposite edge.
#    - Count 3 may overwrite a bar cell with green because "through" literally
#      crosses the obstacle instead of avoiding it.
#
# 4. Generator guarantees for those new assumptions:
#    - After canonicalization every obstacle component is a single horizontal
#      1-cell-thick bar, sorted by row.
#    - The current path column lies inside every active bar span, so each bar
#      truly controls the trajectory.
#    - For count 1/2 there is free space on the chosen bypass side, and the new
#      path column stays away from the outer border to keep later bars feasible.
#    - Count 4 only appears on the last bar, preventing unused clutter below a
#      forced stop.
#    - Row spacing satisfies `next_row - next_b >= prev_row + prev_b + 2`, so
#      the stepped bands do not overlap and remain visually readable.


def _start_edge(grid: np.ndarray) -> tuple[str, int, int]:
    pts = np.argwhere(grid == GREEN)
    assert pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    r, c = map(int, pts[0])
    h, w = grid.shape
    if r == 0:
        return "top", r, c
    if r == h - 1:
        return "bottom", r, c
    if c == 0:
        return "left", r, c
    if c == w - 1:
        return "right", r, c
    raise AssertionError("green start must lie on a border edge")


def _to_canonical(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return np.array(grid, copy=True)
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid).T.copy()  # edge == "right"


def _from_canonical(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return np.array(grid, copy=True)
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()  # edge == "right"


def _extract_bars(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    bar_mask = (grid == BLUE) | (grid == RED)
    comps = components_mask(bar_mask, connectivity=4)
    bars: list[tuple[int, int, int, int]] = []
    for comp in comps:
        rows = {int(r) for r, _ in comp}
        assert len(rows) == 1, "each obstacle must be a single horizontal bar"
        row = rows.pop()
        cols = sorted(int(c) for _, c in comp)
        c0, c1 = cols[0], cols[-1]
        blue_count = int(np.sum(grid[row, c0 : c1 + 1] == BLUE))
        assert 1 <= blue_count <= 4, "blue count must encode one of four behaviors"
        bars.append((row, c0, c1, blue_count))
    bars.sort()
    return bars


def _paint_vertical(out: np.ndarray, col: int, r0: int, r1: int, step: int = 1) -> None:
    a, b = sorted((int(r0), int(r1)))
    for r in range(a, b + 1, int(step)):
        out[r, int(col)] = GREEN


def _paint_horizontal(out: np.ndarray, row: int, c0: int, c1: int) -> None:
    a, b = sorted((int(c0), int(c1)))
    out[int(row), a : b + 1] = GREEN


def _solve_canonical(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, _ = g.shape

    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1, "canonical grid must keep one start pixel"
    sr, sc = map(int, start_pts[0])
    assert sr == 0, "canonical orientation expects the start on the top edge"

    bars = _extract_bars(g)
    out = g.copy()

    cur_r, cur_c = sr, sc
    terminated = False

    for idx, (bar_r, c0, c1, blue_count) in enumerate(bars):
        assert c0 <= cur_c <= c1, "generator must place each active bar across the current path column"
        entry_r = bar_r - blue_count
        exit_r = bar_r + blue_count
        assert entry_r >= cur_r, "bars must be ordered along the main axis"

        _paint_vertical(out, cur_c, cur_r, entry_r, step=1)

        if blue_count == 1:
            bypass_c = c0 - 1
            assert bypass_c >= 0
            _paint_horizontal(out, entry_r, cur_c, bypass_c)
            _paint_vertical(out, bypass_c, entry_r, exit_r, step=1)
            cur_r, cur_c = exit_r, bypass_c
        elif blue_count == 2:
            bypass_c = c1 + 1
            assert bypass_c < out.shape[1]
            _paint_horizontal(out, entry_r, cur_c, bypass_c)
            _paint_vertical(out, bypass_c, entry_r, exit_r, step=2)
            cur_r, cur_c = exit_r, bypass_c
        elif blue_count == 3:
            _paint_vertical(out, cur_c, entry_r, exit_r, step=3)
            cur_r = exit_r
        else:
            assert idx == len(bars) - 1, "a terminating bar must be the last active bar"
            _paint_vertical(out, cur_c, entry_r, exit_r, step=4)
            terminated = True
            break

    if not terminated:
        _paint_vertical(out, cur_c, cur_r, h - 1, step=1)

    out[out == BLUE] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the border start through a sequence of floating bars, where each bar's blue count simultaneously chooses the local action and the spacing of the green marks through that bar's row-band.

    Input structure:
    - bg = azure (8). Each obstacle is a 1-cell-thick red/blue bar (colors 2/1), and a single green (3) start pixel sits on one border edge.
    - Bars are ordered along the start-to-opposite direction; after rotating to a canonical top-start view they are horizontal strips that all cross the current path column.
    - The number of blue cells on a bar is always 1, 2, 3, or 4.

    Rule:
    - Blue count 1 means detour around the bar's left side; 2 means detour around its right side; 3 means pass straight through the bar; 4 means stop after traversing that bar.
    - While crossing a bar's local row-band from `row-count` to `row+count`, green is placed every `count` rows instead of continuously, so counts 2/3/4 create dotted progressions.
    - Outside those local bands the path is continuous, and any remaining blue obstacle cells are converted to red at the end.

    Key insight: the blue tally on each bar is a fused code word: it tells you both which side behavior to execute and what periodic spacing to use while the path advances through that bar's zone.
    """
    g = np.array(grid, copy=True)
    edge, _, _ = _start_edge(g)
    canonical = _to_canonical(g, edge)
    solved = _solve_canonical(canonical)
    return _from_canonical(solved, edge)


def _min_main_len(blues: tuple[int, ...]) -> int:
    row = blues[0] + 1
    for prev_b, next_b in zip(blues, blues[1:]):
        row += prev_b + next_b + 2
    return row + blues[-1] + 2


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _choose_blues(main_hi: int) -> tuple[int, ...]:
    for _ in range(200):
        count = random.randint(BAR_COUNT_LO, BAR_COUNT_HI)
        blues = [random.choice((1, 2, 3)) for _ in range(count)]
        if random.random() < 0.35:
            blues[-1] = 4
        if all(v == 1 for v in blues):
            blues[random.randrange(len(blues))] = random.choice((2, 3, 4))
        if len(set(blues)) == 1 and len(blues) >= 3:
            idx = random.randrange(len(blues) - (1 if blues[-1] == 4 else 0))
            blues[idx] = random.choice([v for v in (1, 2, 3) if v != blues[idx]])
        combo = tuple(blues)
        if _min_main_len(combo) <= main_hi:
            return combo
    raise RuntimeError("failed to choose a feasible behavior sequence")


def _length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, 2 * blue_count - 1, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, cross_size - 2, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _interval_candidates(cross_size: int, current_col: int, blue_count: int) -> list[tuple[float, int, int, int]]:
    bounds = _length_bounds(cross_size, blue_count)
    if bounds is None:
        return []

    lo_len, hi_len = bounds
    center = (cross_size - 1) / 2.0
    candidates: list[tuple[float, int, int, int]] = []

    for length in range(lo_len, hi_len + 1):
        for c0 in range(1, cross_size - length):
            c1 = c0 + length - 1
            if not (c0 <= current_col <= c1):
                continue

            if blue_count == 1:
                new_col = c0 - 1
                if new_col < 2 or current_col - new_col < 2:
                    continue
            elif blue_count == 2:
                new_col = c1 + 1
                if new_col > cross_size - 3 or new_col - current_col < 2:
                    continue
            else:
                new_col = current_col
                if current_col in (c0, c1):
                    continue

            score = abs(new_col - center) + 0.3 * abs((c0 + c1) / 2.0 - center)
            candidates.append((score, c0, c1, new_col))

    candidates.sort(key=lambda item: item[0])
    return candidates


def _paint_bar(grid: np.ndarray, row: int, c0: int, c1: int, blue_count: int) -> None:
    length = c1 - c0 + 1
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    grid[row, c0 : c1 + 1] = colors


def _canonical_grid(main_size: int, cross_size: int, blues: tuple[int, ...]) -> np.ndarray | None:
    slack = main_size - _min_main_len(blues)
    if slack < 0:
        return None

    extras = _random_composition(slack, len(blues) + 1)
    row = blues[0] + 1 + extras[0]
    rows = [row]
    for i in range(1, len(blues)):
        row += blues[i - 1] + blues[i] + 2 + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    start_col = random.randint(4, cross_size - 5)
    current_col = start_col

    for row, blue_count in zip(rows, blues):
        candidates = _interval_candidates(cross_size, current_col, blue_count)
        if not candidates:
            return None
        pool = candidates[: min(10, len(candidates))]
        _, c0, c1, new_col = random.choice(pool)
        _paint_bar(grid, row, c0, c1, blue_count)
        if blue_count in (1, 2):
            current_col = new_col

    grid[0, start_col] = GREEN
    return grid


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()  # edge == "right"


def _fusion_signature(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    edge, _, _ = _start_edge(inp)
    cin = _to_canonical(inp, edge)
    cout = _to_canonical(out, edge)
    bars = _extract_bars(cin)

    left_turns = 0
    right_turns = 0
    through_hits = 0
    stop_bars = 0
    stepped_marks = 0

    for row, c0, c1, blue_count in bars:
        band = cout[max(0, row - blue_count) : min(cout.shape[0], row + blue_count + 1), :]
        stepped_marks += int(np.sum(band == GREEN))
        if blue_count == 1 and np.any(cout[row - 1 : row + 2, c0 - 1] == GREEN):
            left_turns += 1
        elif blue_count == 2 and np.any(cout[row - 2 : row + 3, c1 + 1] == GREEN):
            right_turns += 1
        elif blue_count == 3 and cout[row, c0 : c1 + 1].tolist().count(GREEN) > 0:
            through_hits += 1
        elif blue_count == 4:
            stop_bars += 1

    reached_bottom = int(np.any(cout[-1, :] == GREEN))
    return {
        "bars": len(bars),
        "left_turns": left_turns,
        "right_turns": right_turns,
        "through_hits": through_hits,
        "stop_bars": stop_bars,
        "reached_bottom": reached_bottom,
        "stepped_marks": stepped_marks,
    }


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    blues = _choose_blues(main_hi)
    main_size = random.randint(max(main_lo, _min_main_len(blues)), main_hi)
    cross_size = random.randint(cross_lo, cross_hi)

    canonical = _canonical_grid(main_size, cross_size, blues)
    if canonical is None:
        return None
    inp = _orient_grid(canonical, edge)

    try:
        out = main(inp)
        if not np.array_equal(main(inp), out):
            return None
        sig = _fusion_signature(inp, out)
        if sig["bars"] < 2:
            return None
        if sum(v > 0 for v in (sig["left_turns"], sig["right_turns"], sig["through_hits"], sig["stop_bars"])) < 1:
            return None
        if blues[-1] == 4:
            if sig["reached_bottom"] != 0:
                return None
        elif sig["reached_bottom"] != 1:
            return None
        if max(blues) > 1 and sig["stepped_marks"] <= len(blues):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused cb2d8a2c input")
