# Combo variant of cb2d8a2c (index 11)
# combo_spec: D6/1+D12/0
# Name: 步数限制+逐条处理

import heapq
import itertools
import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4

BG = 8
BLUE = 1
RED = 2
GREEN = 3

# Each blue pixel contributes a small, fixed amount of travel budget.
# Using a clean multiplier keeps the D6 change visible while still leaving
# enough room for the D12 stage-by-stage processing to actually occur.
STEP_MULTIPLIER = 4

# Canonical solver orientation: green starts on the top edge and travels downward.
_DIRS_DRL = [(1, 0), (0, 1), (0, -1)]
_DOWN_DIR = 0


# ---------------------------------------------------------------------------
# Design analysis for the fused variant
#
# a) Concrete behavior trace:
#    Original task: compute keep-out zones for every bar once, then find a
#    single path all the way to the opposite edge; finally all blue cells turn
#    red. Fused task: first count all blue cells to obtain a global budget
#    `4 * total_blue`. Then process the nearest remaining bar only: route just
#    far enough to get below that bar's keep-out zone, spend the required
#    number of steps, recolor only that bar's blue cells to red, and remove the
#    bar from later blocking. If the budget ends in the middle of a stage, the
#    path stops there and every later bar keeps its original blue markers.
#
# b) Complexity calibration:
#    A raw 1:1 budget would usually be too short to even engage the first detour
#    on grids of the original size range, making D12 visually invisible. The
#    fixed multiplier keeps the encoded parameter simple and countable while
#    preserving visible partial-processing cases such as "first two bars are
#    cleared, third still has blue markers".
#
# c) Implicit assumptions checked against the original solver:
#    - "Goal is always opposite edge" no longer holds; the path may stop early.
#    - "All bars block simultaneously" no longer holds; only the current stage's
#      target bar defines the active keep-out zone.
#    - "All blue becomes red" no longer holds; only fully processed bars change.
#    - "Output is idempotent as a new input" no longer holds because the output
#      contains a multi-cell green trail instead of a unique border seed.
#
# d) Placement legality:
#    The generator preserves non-overlap, wall attachment, alternating sides,
#    and enough row/column slack so each stage target exists strictly below the
#    current bar's keep-out zone.
#
# e/f/g) Observability and cleanliness:
#    Human-visible cues now align cleanly: local blue count still controls the
#    clearance around each individual bar, while the global blue count controls
#    how far the entire green trace can go. Processed bars lose blue markers,
#    untouched bars keep them, so the sequential nature is visible without
#    hidden state or color-ID tricks.
# ---------------------------------------------------------------------------


def _to_canonical(grid: np.ndarray) -> tuple[np.ndarray, str]:
    """Rotate/flip the puzzle so the single green seed sits on the top edge."""
    pts = np.argwhere(grid == GREEN)
    assert pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, pts[0])
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
    """Undo `_to_canonical`."""
    if edge == "top":
        return np.array(grid, copy=True)
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()  # edge == "right"


def _extract_bars(grid: np.ndarray) -> list[dict]:
    """Extract horizontal obstacle bars from a canonical grid."""
    bar_mask = (grid == BLUE) | (grid == RED)
    comps = components_mask(bar_mask, connectivity=4)
    bars = []
    for comp in comps:
        rows = {int(r) for r, _ in comp}
        cols = [int(c) for _, c in comp]
        assert len(rows) == 1, "canonical bars must be horizontal 1-cell lines"
        row = next(iter(rows))
        c0, c1 = min(cols), max(cols)
        side = 0 if c0 == 0 else 1
        assert c0 == 0 or c1 == grid.shape[1] - 1, "each bar must touch a side wall"
        blue_cells = [(int(r), int(c)) for r, c in comp if int(grid[int(r), int(c)]) == BLUE]
        bars.append(
            {
                "row": row,
                "c0": c0,
                "c1": c1,
                "side": side,
                "cells": [(int(r), int(c)) for r, c in comp],
                "blue_cells": blue_cells,
                "blue_count": len(blue_cells),
            }
        )
    bars.sort(key=lambda bar: bar["row"])
    return bars


def _build_keepout(shape: tuple[int, int], cells: list[tuple[int, int]], radius: int) -> np.ndarray:
    """Manhattan keep-out mask for a single active bar."""
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)
    dist = np.full((h, w), -1, dtype=np.int16)
    q = deque()
    for r, c in cells:
        dist[r, c] = 0
        q.append((r, c))
    while q:
        r, c = q.popleft()
        d = int(dist[r, c])
        blocked[r, c] = True
        if d == radius:
            continue
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] == -1:
                dist[rr, cc] = d + 1
                q.append((rr, cc))
    return blocked


def _shortest_stage_path(
    blocked: np.ndarray,
    start: tuple[int, int],
    start_dir: int,
    goal_row: int,
) -> list[tuple[int, int]]:
    """Shortest monotone-down path to any cell on `goal_row`, minimizing turns second."""
    h, w = blocked.shape
    sr, sc = start
    assert 0 <= goal_row < h

    inf = (10**9, 10**9)
    best: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, start_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, start_dir)]

    goal_state: tuple[int, int, int] | None = None
    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best.get((r, c, dprev), inf):
            continue
        if r == goal_row and (r, c) != start:
            goal_state = (r, c, dprev)
            break
        for nd, (dr, dc) in enumerate(_DIRS_DRL):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if blocked[rr, cc]:
                continue
            ncost = (steps + 1, turns + (0 if nd == dprev else 1))
            key = (rr, cc, nd)
            if ncost < best.get(key, inf):
                best[key] = ncost
                prev[key] = (r, c, dprev)
                heapq.heappush(heap, (ncost, rr, cc, nd))

    assert goal_state is not None, "stage path must exist"

    path: list[tuple[int, int]] = []
    state = goal_state
    while True:
        r, c, _ = state
        path.append((r, c))
        if (r, c) == start:
            break
        state = prev[state]
    path.reverse()
    return path


def _path_dir(a: tuple[int, int], b: tuple[int, int]) -> int:
    dr, dc = int(b[0] - a[0]), int(b[1] - a[1])
    if (dr, dc) == (1, 0):
        return 0
    if (dr, dc) == (0, 1):
        return 1
    if (dr, dc) == (0, -1):
        return 2
    raise AssertionError("unexpected step direction")


def _simulate_canonical(grid: np.ndarray, *, with_metrics: bool = False):
    """
    Solve a canonical top-start grid using the fused rules.

    Returns the solved canonical grid, and optionally a metrics dict used by
    the generator to reject degenerate samples.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape

    pts = np.argwhere(g == GREEN)
    assert pts.shape[0] == 1, "canonical grid must contain one green seed"
    sr, sc = map(int, pts[0])
    assert sr == 0, "canonical grid must start on the top edge"

    bars = _extract_bars(g)
    assert bars, "at least one bar is required"
    total_blue = sum(bar["blue_count"] for bar in bars)
    step_budget = STEP_MULTIPLIER * total_blue

    out = g.copy()
    current = (sr, sc)
    current_dir = _DOWN_DIR
    steps_left = step_budget
    steps_used = 0
    processed = [False] * len(bars)
    processed_indices: list[int] = []
    visited: list[tuple[int, int]] = []

    while steps_left > 0:
        next_idx = next((i for i, done in enumerate(processed) if not done), None)
        if next_idx is None:
            goal_row = h - 1
            if current[0] == goal_row:
                break
            blocked = np.zeros((h, w), dtype=bool)
        else:
            bar = bars[next_idx]
            goal_row = bar["row"] + bar["blue_count"] + 1
            assert goal_row < h, "generator must leave room below every bar keep-out"
            blocked = _build_keepout((h, w), bar["cells"], bar["blue_count"])
            blocked[current] = False

        stage_path = _shortest_stage_path(blocked, current, current_dir, goal_row)
        full_stage_steps = len(stage_path) - 1
        take = min(steps_left, full_stage_steps)

        for idx in range(1, take + 1):
            visited.append(stage_path[idx])
        if take > 0:
            current_dir = _path_dir(stage_path[take - 1], stage_path[take])
            current = stage_path[take]
        steps_left -= take
        steps_used += take

        if take < full_stage_steps:
            break
        if next_idx is None:
            break

        processed[next_idx] = True
        processed_indices.append(next_idx)
        for r, c in bars[next_idx]["blue_cells"]:
            out[r, c] = RED

    for r, c in visited:
        out[r, c] = GREEN

    if not with_metrics:
        return out

    metrics = {
        "step_budget": step_budget,
        "steps_used": steps_used,
        "total_blue": total_blue,
        "bar_count": len(bars),
        "processed_count": len(processed_indices),
        "processed_indices": tuple(processed_indices),
        "reached_bottom": current[0] == h - 1,
        "remaining_blue_bars": len(bars) - len(processed_indices),
        "green_cells_added": int(np.count_nonzero(out == GREEN) - 1),
    }
    return out, metrics


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a green route from the border, but cap the total travel by a blue-count budget while clearing obstacles one at a time.

    Input structure:
    - bg = azure (8). A single green (3) seed sits on one border edge.
    - Red/blue 1-cell-thick bars (2/1) lie across the travel direction and alternate side-wall attachment.

    Rule:
    - Each bar's own blue count still sets its Manhattan keep-out radius, but only for the nearest unprocessed bar.
    - First compute the total number of blue cells across all bars; the path may take at most `4 * total_blue` steps.
    - The path routes just far enough to get past the current bar, then that bar's blue cells turn red and the bar stops blocking later stages.
    - If the budget runs out mid-stage, the path stops immediately and all later bars keep their original blue markers.

    Key insight: Blue carries two scales of information at once — locally it says how widely to avoid one bar, globally it says how much total path can be afforded.
    """
    canonical, edge = _to_canonical(np.array(grid, copy=True))
    solved = _simulate_canonical(canonical, with_metrics=False)
    return _from_canonical(solved, edge)


# --------------- generator ---------------

# Grid size bounds; kept within the original task range but tightened so the
# step-budget effect remains visible in a few examples.
ROWS_LO, ROWS_HI = 12, 18
COLS_LO, COLS_HI = 13, 20

BAR_COUNT_LO, BAR_COUNT_HI = 2, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 2, 4

BAR_LEN_MIN, BAR_LEN_MAX = 6, 12
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 38, 78
MAX_GEN_RETRIES = 800


def _min_main_len(blues: tuple[int, ...]) -> int:
    """Minimum main-axis size with room below every bar's keep-out zone."""
    last_bar_row = blues[0] + 2
    for a, b in zip(blues, blues[1:]):
        last_bar_row += a + b + 2
    return last_bar_row + blues[-1] + 2


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


def _choose_blues(count: int, main_hi: int, mode: str) -> tuple[int, ...]:
    """Pick per-bar blue counts while biasing the resulting stage profile."""
    any_fit = []
    choices = []
    for combo in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count):
        if _min_main_len(combo) > main_hi:
            continue
        any_fit.append(combo)
        total = sum(combo)
        if mode == "tight" and total >= 6:
            choices.append(combo)
        if mode == "generous" and 5 <= total <= 6:
            choices.append(combo)
    if choices:
        return random.choice(choices)
    assert any_fit
    return random.choice(any_fit)


def _length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, 2 * blue_count - 1, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, cross_size - blue_count - 1, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, blue_count: int) -> None:
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    if side == 0:
        grid[row, :length] = colors
    else:
        grid[row, grid.shape[1] - length :] = colors[::-1]


def _pick_length(bounds: tuple[int, int], mode: str) -> int:
    lo, hi = bounds
    if mode == "tight":
        return random.randint(max(lo, hi - 2), hi)
    return random.randint(lo, min(hi, lo + 2))


def _stage_extras(slack: int, count: int, mode: str) -> list[int]:
    """
    Distribute row slack among [before first, between..., after last].

    generous: keep early rows tight and push spare space to the very bottom.
    tight: spend more slack before / between bars to make the budget work harder.
    """
    if slack == 0:
        return [0] * (count + 1)
    if mode == "generous":
        out = [0] * (count + 1)
        out[-1] = slack
        return out
    if count == 1:
        return [slack, 0]
    extras = [0] * (count + 1)
    focus = _random_composition(slack, count)
    for i, val in enumerate(focus):
        extras[i] = val
    return extras


def _choose_start_col(first_side: int, first_len: int, cross_size: int, mode: str) -> int:
    if first_side == 0:
        if mode == "tight":
            return random.randint(1, min(first_len - 1, 2))
        return random.randint(max(1, first_len - 3), first_len - 1)
    if mode == "tight":
        return random.randint(max(cross_size - first_len, cross_size - 3), cross_size - 2)
    return random.randint(cross_size - first_len, min(cross_size - 2, cross_size - first_len + 2))


def _canonical_grid(
    main_size: int,
    cross_size: int,
    blues: tuple[int, ...],
    first_side: int,
    mode: str,
) -> np.ndarray | None:
    count = len(blues)
    slack = main_size - _min_main_len(blues)
    if slack < 0:
        return None

    extras = _stage_extras(slack, count, mode)
    row = blues[0] + 2 + extras[0]
    rows = [row]
    for i in range(1, count):
        row += blues[i - 1] + blues[i] + 2 + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    lengths: list[int] = []
    for row, blue_count, side in zip(rows, blues, ((first_side + i) % 2 for i in range(count))):
        bounds = _length_bounds(cross_size, blue_count)
        if bounds is None:
            return None
        length = _pick_length(bounds, mode)
        lengths.append(length)
        _paint_bar(grid, row, side, length, blue_count)

    start_col = _choose_start_col(first_side, lengths[0], cross_size, mode)
    grid[0, start_col] = GREEN
    return grid


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    return _from_canonical(grid, edge)


def _gen_attempt(mode: str | None = None) -> np.ndarray | None:
    mode = random.choice(("tight", "generous")) if mode is None else mode
    edge = random.choice(("top", "bottom", "left", "right"))

    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    if mode == "tight":
        wanted_counts = range(3, BAR_COUNT_HI + 1)
        cross_size = random.randint(max(cross_lo, 15), cross_hi)
        feasible_counts = [
            count
            for count in wanted_counts
            if _min_main_len((BLUE_PER_BAR_LO,) * count) <= main_hi
        ]
        if not feasible_counts:
            return None
        count = random.choice(feasible_counts)
        blues = _choose_blues(count, main_hi, mode)
        main_min = max(main_lo, _min_main_len(blues))
        main_max = min(main_hi, main_min + 4)
        if main_min > main_max:
            return None
        main_size = random.randint(main_min, main_max)
    else:
        count = 2
        blues = random.choice(((2, 3), (2, 4)))
        base_min = _min_main_len(blues)
        if base_min + 1 > main_hi:
            return None
        cross_size = random.randint(max(cross_lo, 18), cross_hi)
        main_min = max(main_lo, base_min + 1)
        main_max = min(main_hi, base_min + 2)
        if main_min > main_max:
            return None
        main_size = random.randint(main_min, main_max)

    first_side = random.randrange(2)

    canonical = _canonical_grid(main_size, cross_size, blues, first_side, mode)
    if canonical is None:
        return None

    solved_canonical, metrics = _simulate_canonical(canonical, with_metrics=True)
    if metrics["steps_used"] != metrics["step_budget"]:
        return None
    if metrics["processed_count"] < 1:
        return None
    if mode == "tight" and metrics["processed_count"] >= metrics["bar_count"]:
        return None
    if mode == "generous" and (
        metrics["processed_count"] != metrics["bar_count"] or metrics["reached_bottom"]
    ):
        return None

    inp = _orient_grid(canonical, edge)
    out = _orient_grid(solved_canonical, edge)
    try:
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(got, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    target_mode = random.choice(("tight", "generous"))
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(target_mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused cb2d8a2c input")
