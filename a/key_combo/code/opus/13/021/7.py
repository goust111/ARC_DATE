from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
ORANGE = 7
RANK_COLORS = (BLUE, YELLOW, RED)

GEN_HEIGHT_RANGE = (18, 23)
GEN_WIDTH_RANGE = (17, 21)
GEN_MIN_BAR_GAP = 1
GEN_BAR_LENGTHS = tuple(range(4, 8))
GEN_MAX_ATTEMPTS = 400
GEN_EXTRA_OBSTACLE_PROB = 0.55


def _parse_bars(grid: np.ndarray) -> list[dict]:
    bars: list[dict] = []
    bar_rows = []
    for r in range(grid.shape[0]):
        if np.any(grid[r] == GREY):
            bar_rows.append(r)
    assert bar_rows == [0, 1, 2], "expected exactly one gray bar in each of rows 0, 1, 2"

    for r in bar_rows:
        cols = np.flatnonzero(grid[r] == GREY)
        assert cols.size >= 2
        left = int(cols.min())
        right = int(cols.max())
        span = grid[r, left : right + 1]
        gap_offsets = np.flatnonzero(span == BG)
        assert gap_offsets.size == 1, "each gray bar must have exactly one gap"
        gap = left + int(gap_offsets[0])
        assert left < gap < right, "gap must be internal"
        assert np.all(np.isin(span, (BG, GREY))), "bar span may contain only background and gray"
        assert int(np.sum(span == GREY)) == right - left, "all non-gap cells in the span must be gray"
        bars.append(
            {
                "row": int(r),
                "left": left,
                "right": right,
                "gap": gap,
                "length": right - left + 1,
            }
        )

    lengths = [bar["length"] for bar in bars]
    assert len(set(lengths)) == 3, "bar lengths must be distinct"
    ordered_ranges = sorted((bar["left"], bar["right"]) for bar in bars)
    for (_l0, r0), (l1, _r1) in zip(ordered_ranges, ordered_ranges[1:]):
        assert r0 < l1, "bar column ranges must not overlap"
    return bars


def _ranked_bars(bars: list[dict]) -> list[tuple[dict, int]]:
    ranked = sorted(bars, key=lambda bar: bar["length"], reverse=True)
    return list(zip(ranked, RANK_COLORS))


def _row_interval(bar: dict, row: int) -> tuple[int, int] | None:
    depth = row - int(bar["row"])
    if depth <= 0:
        return None
    radius = depth - 1
    left = max(int(bar["left"]), int(bar["gap"]) - radius)
    right = min(int(bar["right"]), int(bar["gap"]) + radius)
    return left, right


def _paint_wedge(src: np.ndarray, out: np.ndarray, bar: dict, color: int) -> None:
    h = src.shape[0]
    gap = int(bar["gap"])

    for r in range(int(bar["row"]) + 1, h):
        interval = _row_interval(bar, r)
        assert interval is not None
        left, right = interval
        cells: list[int] = []

        if int(src[r, gap]) == ORANGE:
            break

        cells.append(gap)

        for c in range(gap - 1, left - 1, -1):
            if int(src[r, c]) == ORANGE:
                break
            cells.append(c)

        for c in range(gap + 1, right + 1):
            if int(src[r, c]) == ORANGE:
                break
            cells.append(c)

        if not cells:
            break
        out[r, cells] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the ranked gray bars, then grow centered wedges downward with row-wise obstacle clipping.

    Input structure:
    - The background is black, with exactly three horizontal gray bars in rows 0, 1, and 2.
    - Each bar occupies one continuous column span, has exactly one internal background gap, and the three spans have distinct lengths and non-overlapping column ranges.
    - From row 4 downward there are scattered orange single-pixel obstacles.

    Rule:
    - Measure each gray bar by its full span length including the gap, then recolor longest to blue, middle to yellow, and shortest to red while leaving the gap black.
    - For each bar, use the gap column as the wedge center and grow one row downward at a time; at depth n the intended row covers the center plus n-1 cells to each side, clipped to the bar's original left and right ends.
    - In every target row, orange obstacles clip the left and right sides independently: an obstacle cell is not painted, and cells farther from the center on that same side are skipped for that row only; if the center itself is blocked, that wedge stops on that row. Orange obstacles stay unchanged.

    Key insight:
    - The final shape is determined by three coupled cues at once: relative bar length sets the color, each gap fixes the wedge center, and orange blockers trim each row locally instead of globally.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bars = _parse_bars(g)

    out = g.copy()
    for bar, color in _ranked_bars(bars):
        row = int(bar["row"])
        left = int(bar["left"])
        right = int(bar["right"])
        gap = int(bar["gap"])
        for c in range(left, right + 1):
            if c != gap:
                out[row, c] = color
        _paint_wedge(g, out, bar, color)

    return out


def _random_composition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _choose_bar_layout(width: int) -> list[dict] | None:
    options = []
    for a in GEN_BAR_LENGTHS:
        for b in GEN_BAR_LENGTHS:
            for c in GEN_BAR_LENGTHS:
                lengths = [a, b, c]
                if len(set(lengths)) != 3:
                    continue
                need = sum(lengths) + GEN_MIN_BAR_GAP * 2
                if need <= width:
                    options.append(lengths)
    if not options:
        return None

    lengths = random.choice(options)
    random.shuffle(lengths)
    spare = width - (sum(lengths) + GEN_MIN_BAR_GAP * 2)
    margins = _random_composition(spare, 4)
    c = margins[0]
    bars = []
    for row, length in enumerate(lengths):
        left = c
        right = c + length - 1
        gap = random.randint(left + 1, right - 1)
        bars.append({"row": row, "left": left, "right": right, "gap": gap, "length": length})
        c = right + 1
        if row < 2:
            c += GEN_MIN_BAR_GAP + margins[row + 1]
    return bars


def _can_place_obstacle(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    if not (4 <= r < h and 0 <= c < w):
        return False
    if int(grid[r, c]) != BG:
        return False
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == ORANGE:
            return False
    return True


def _candidate_rows(bar: dict, height: int, require_left: bool, require_right: bool) -> list[int]:
    rows = []
    for r in range(max(4, int(bar["row"]) + 1), height - 1):
        left, right = _row_interval(bar, r)
        if require_left and left >= int(bar["gap"]):
            continue
        if require_right and right <= int(bar["gap"]):
            continue
        rows.append(r)
    return rows


def _place_event(grid: np.ndarray, bar: dict, event: str) -> bool:
    if event == "left":
        rows = _candidate_rows(bar, grid.shape[0], require_left=True, require_right=False)
    elif event == "right":
        rows = _candidate_rows(bar, grid.shape[0], require_left=False, require_right=True)
    elif event == "double":
        rows = _candidate_rows(bar, grid.shape[0], require_left=True, require_right=True)
    else:
        rows = _candidate_rows(bar, grid.shape[0], require_left=False, require_right=False)

    random.shuffle(rows)
    for r in rows:
        left, right = _row_interval(bar, r)
        gap = int(bar["gap"])

        if event == "left":
            cols = list(range(left, gap))
            random.shuffle(cols)
            for c in cols:
                if _can_place_obstacle(grid, r, c):
                    grid[r, c] = ORANGE
                    return True

        elif event == "right":
            cols = list(range(gap + 1, right + 1))
            random.shuffle(cols)
            for c in cols:
                if _can_place_obstacle(grid, r, c):
                    grid[r, c] = ORANGE
                    return True

        elif event == "double":
            left_cols = list(range(left, gap))
            right_cols = list(range(gap + 1, right + 1))
            random.shuffle(left_cols)
            random.shuffle(right_cols)
            for lc in left_cols:
                if not _can_place_obstacle(grid, r, lc):
                    continue
                grid[r, lc] = ORANGE
                placed = False
                for rc in right_cols:
                    if _can_place_obstacle(grid, r, rc):
                        grid[r, rc] = ORANGE
                        placed = True
                        break
                if placed:
                    return True
                grid[r, lc] = BG

        else:
            if _can_place_obstacle(grid, r, gap):
                grid[r, gap] = ORANGE
                return True

    return False


def _add_extra_obstacle(grid: np.ndarray, bars: list[dict]) -> None:
    h, w = grid.shape
    wedge_cols = set()
    for bar in bars:
        wedge_cols.update(range(int(bar["left"]), int(bar["right"]) + 1))

    cells = [(r, c) for r in range(4, h) for c in range(w) if c not in wedge_cols]
    random.shuffle(cells)
    for r, c in cells:
        if _can_place_obstacle(grid, r, c):
            grid[r, c] = ORANGE
            return


def _bar_obstacle_hits(grid: np.ndarray, bar: dict) -> int:
    hits = 0
    gap = int(bar["gap"])
    for r in range(int(bar["row"]) + 1, grid.shape[0]):
        interval = _row_interval(bar, r)
        assert interval is not None
        left, right = interval
        row_slice = grid[r, left : right + 1]
        if np.any(row_slice == ORANGE):
            hits += 1
        if int(grid[r, gap]) == ORANGE:
            break
    return hits


def _generate_once() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    bars = _choose_bar_layout(width)
    if bars is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for bar in bars:
        row = int(bar["row"])
        left = int(bar["left"])
        right = int(bar["right"])
        gap = int(bar["gap"])
        grid[row, left : right + 1] = GREY
        grid[row, gap] = BG

    event_types = ["left", "right", random.choice(["double", "center_stop"])]
    random.shuffle(event_types)
    for bar, event in zip(bars, event_types):
        if not _place_event(grid, bar, event):
            return None
        if event != "center_stop" and random.random() < 0.5:
            extra = random.choice(["left", "right", "double"])
            _place_event(grid, bar, extra)

    if random.random() < GEN_EXTRA_OBSTACLE_PROB:
        for _ in range(random.randint(1, 3)):
            _add_extra_obstacle(grid, bars)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    if len({bar["length"] for bar in bars}) != 3:
        return None
    if any(_bar_obstacle_hits(grid, bar) == 0 for bar in bars):
        return None
    if not {BLUE, YELLOW, RED}.issubset(set(map(int, np.unique(out)))):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
