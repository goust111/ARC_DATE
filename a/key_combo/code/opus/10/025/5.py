import random

import numpy as np


BG = 0
CONTROL = 1
YELLOW = 4
MAROON = 9

BAR_FILL_COLORS = (2, 3, 5, 6, 7, 8)

GRID_HEIGHT_RANGE = (17, 24)
GRID_WIDTH_RANGE = (20, 26)
BAR_COUNT_CHOICES = (3, 4)
BAR_LENGTH_RANGE = (2, 8)
MAX_GENERATION_ATTEMPTS = 1000


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _draw_frame(grid: np.ndarray, top: int, left: int) -> None:
    grid[top : top + 3, left : left + 3] = YELLOW
    grid[top + 1, left + 1] = CONTROL


def _draw_corner_rays(grid: np.ndarray, top: int, left: int) -> None:
    h, w = grid.shape
    corners = [
        (top, left, -1, -1),
        (top, left + 2, -1, 1),
        (top + 2, left, 1, -1),
        (top + 2, left + 2, 1, 1),
    ]
    for r, c, dr, dc in corners:
        rr = r + dr
        cc = c + dc
        while 0 <= rr < h and 0 <= cc < w:
            grid[rr, cc] = YELLOW
            rr += dr
            cc += dc


def _ray_lengths_ok(top: int, left: int, shape: tuple[int, int]) -> bool:
    h, w = shape
    return (
        min(top, left) >= 3
        and min(top, w - 1 - (left + 2)) >= 3
        and min(h - 1 - (top + 2), left) >= 3
        and min(h - 1 - (top + 2), w - 1 - (left + 2)) >= 3
    )


def _find_legend_height(grid: np.ndarray) -> int:
    h, w = grid.shape
    maroon = grid == MAROON
    seen = np.zeros((h, w), dtype=bool)
    legends = []
    for r in range(h):
        for c in range(w):
            if not maroon[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and maroon[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if any(rr == 0 for rr, _ in comp):
                legends.append(comp)
    assert len(legends) == 1, "expected exactly one legend component touching the top edge"
    comp = legends[0]
    cols = {c for _, c in comp}
    rows = [r for r, _ in comp]
    assert len(cols) == 1, "legend must be a single vertical maroon block"
    return max(rows) - min(rows) + 1


def _find_bars(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    bars = []
    for c in range(w):
        r = 0
        while r < h:
            if int(grid[r, c]) == BG:
                r += 1
                continue
            r0 = r
            while r + 1 < h and int(grid[r + 1, c]) != BG:
                r += 1
            r1 = r
            seg = grid[r0 : r1 + 1, c]
            if (
                len(seg) >= 3
                and int(seg[0]) == MAROON
                and int(seg[-1]) == MAROON
                and len(set(map(int, seg[1:-1]))) == 1
                and int(seg[1]) not in (BG, MAROON)
            ):
                bars.append(
                    {
                        "top": int(r0),
                        "bottom": int(r1),
                        "col": int(c),
                        "fill": int(seg[1]),
                        "length": int(r1 - r0 - 1),
                    }
                )
            r += 1
    assert 3 <= len(bars) <= 4, "expected 3-4 vertical bars"
    lengths = [bar["length"] for bar in bars]
    assert len(set(lengths)) == len(lengths), "bar interior lengths must be unique"
    return bars


def _find_frame_top_left(grid: np.ndarray) -> tuple[int, int]:
    target = np.array(
        [
            [YELLOW, YELLOW, YELLOW],
            [YELLOW, CONTROL, YELLOW],
            [YELLOW, YELLOW, YELLOW],
        ],
        dtype=int,
    )
    h, w = grid.shape
    hits = []
    for r in range(h - 2):
        for c in range(w - 2):
            if np.array_equal(grid[r : r + 3, c : c + 3], target):
                hits.append((r, c))
    assert len(hits) == 1, "expected exactly one 3x3 yellow frame"
    return hits[0]


def _ranked_bars(bars: list[dict]) -> list[dict]:
    return sorted(bars, key=lambda bar: (-bar["length"], bar["col"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select a vertical bar by legend-driven length rank, then move the yellow frame to that bar's encoded position and emit diagonal corner rays.

    Input structure:
    - The background is black.
    - A single maroon legend block touches the top edge; its height k is the desired length rank.
    - The left side contains 3-4 one-column bars with maroon cap pixels at top and bottom and a solid non-maroon interior color.
    - The right side contains one 3x3 yellow frame with a blue center.

    Rule:
    - Rank the maroon-capped vertical bars by interior length in descending order.
    - Use the legend height k to choose the k-th ranked bar.
    - Compute the translation from the input frame's top-left corner to the chosen bar's top cap row and bar column.
    - Fill the output with blue, redraw the yellow frame at the translated location, and extend yellow diagonal rays from the frame's four corners to the grid edges.

    Key insight:
    - The legend does not describe a size directly; it selects one bar by rank, and that bar's spatial position becomes the frame's displacement code.
    """
    g = np.array(grid, dtype=int)
    k = _find_legend_height(g)
    bars = _ranked_bars(_find_bars(g))
    assert 1 <= k <= len(bars), "legend rank must select an existing bar"
    selected = bars[k - 1]

    frame_top, frame_left = _find_frame_top_left(g)
    dr = selected["top"] - frame_top
    dc = selected["col"] - frame_left
    new_top = frame_top + dr
    new_left = frame_left + dc

    out = np.full_like(g, CONTROL)
    _draw_frame(out, new_top, new_left)
    _draw_corner_rays(out, new_top, new_left)
    return out


def _sample_bar_columns(width: int, n_bars: int) -> list[int] | None:
    candidates = list(range(3, width - 9))
    if len(candidates) < n_bars:
        return None
    for _ in range(200):
        cols = sorted(random.sample(candidates, n_bars))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _is_strict_monotone(values: list[int]) -> bool:
    return all(a < b for a, b in zip(values, values[1:])) or all(a > b for a, b in zip(values, values[1:]))


def _selected_bar_anti_shortcut_ok(bars: list[dict], k: int) -> bool:
    ranked = _ranked_bars(bars)
    selected = ranked[k - 1]

    by_col = sorted(bars, key=lambda bar: bar["col"])
    by_row = sorted(bars, key=lambda bar: bar["top"])
    col_rank = 1 + by_col.index(selected)
    row_rank = 1 + by_row.index(selected)
    if col_rank == k or row_rank == k:
        return False

    if _is_strict_monotone([bar["length"] for bar in by_col]):
        return False
    if _is_strict_monotone([bar["top"] for bar in ranked]):
        return False
    return True


def _sample_bars(height: int, width: int, n_bars: int, k: int) -> list[dict] | None:
    cols = _sample_bar_columns(width, n_bars)
    if cols is None:
        return None

    max_len = min(BAR_LENGTH_RANGE[1], height - 6)
    if max_len < BAR_LENGTH_RANGE[0] or max_len - BAR_LENGTH_RANGE[0] + 1 < n_bars:
        return None
    lengths_desc = sorted(random.sample(range(BAR_LENGTH_RANGE[0], max_len + 1), n_bars), reverse=True)
    fills = random.sample(BAR_FILL_COLORS, n_bars)

    for _ in range(400):
        assigned_lengths = lengths_desc[:]
        random.shuffle(assigned_lengths)
        if _is_strict_monotone(assigned_lengths):
            continue

        top_rows = []
        used_rows = set()
        ok = True
        for length in assigned_lengths:
            lo = 3
            hi = min(height - 6, height - length - 2)
            choices = [r for r in range(lo, hi + 1) if r not in used_rows]
            if not choices:
                ok = False
                break
            row = random.choice(choices)
            used_rows.add(row)
            top_rows.append(row)
        if not ok:
            continue

        bars = []
        for col, top, length, fill in zip(cols, top_rows, assigned_lengths, fills):
            if not _ray_lengths_ok(top, col, (height, width)):
                ok = False
                break
            bars.append(
                {
                    "top": top,
                    "bottom": top + length + 1,
                    "col": col,
                    "fill": fill,
                    "length": length,
                }
            )
        if not ok:
            continue
        if _selected_bar_anti_shortcut_ok(bars, k):
            return bars
    return None


def _sample_frame_top_left(height: int, width: int, bars: list[dict], k: int) -> tuple[int, int] | None:
    selected = _ranked_bars(bars)[k - 1]
    max_bar_col = max(bar["col"] for bar in bars)

    col_choices = list(range(max_bar_col + 4, width - 3))
    if not col_choices:
        return None

    above = [r for r in range(1, selected["top"] - 1)]
    below = [r for r in range(selected["top"] + 2, height - 3)]
    row_pools = [pool for pool in (above, below) if pool]
    if not row_pools:
        return None

    row_choices = random.choice(row_pools)
    return random.choice(row_choices), random.choice(col_choices)


def _render_input(height: int, width: int, bars: list[dict], frame_top: int, frame_left: int, k: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[:k, 0] = MAROON
    for bar in bars:
        c = bar["col"]
        top = bar["top"]
        bottom = bar["bottom"]
        grid[top, c] = MAROON
        grid[top + 1 : bottom, c] = bar["fill"]
        grid[bottom, c] = MAROON
    _draw_frame(grid, frame_top, frame_left)
    return grid


def _expected_output_from_parsed(grid: np.ndarray) -> np.ndarray:
    bars = _ranked_bars(_find_bars(grid))
    k = _find_legend_height(grid)
    selected = bars[k - 1]
    out = np.full_like(grid, CONTROL)
    _draw_frame(out, selected["top"], selected["col"])
    _draw_corner_rays(out, selected["top"], selected["col"])
    return out


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        k = _find_legend_height(grid)
        bars = _find_bars(grid)
        frame_top, frame_left = _find_frame_top_left(grid)
    except AssertionError:
        return False

    if not (1 <= k <= len(bars)):
        return False
    if sum(int(grid[r, c] == CONTROL) for r in range(grid.shape[0]) for c in range(grid.shape[1])) != 1:
        return False
    if frame_left <= max(bar["col"] for bar in bars):
        return False

    ranked = _ranked_bars(bars)
    selected = ranked[k - 1]
    if abs(frame_top - selected["top"]) < 2:
        return False
    if not _ray_lengths_ok(selected["top"], selected["col"], grid.shape):
        return False
    if len({bar["top"] for bar in bars}) != len(bars):
        return False
    if len({bar["col"] for bar in bars}) != len(bars):
        return False
    if not _selected_bar_anti_shortcut_ok(bars, k):
        return False

    expected = _expected_output_from_parsed(grid)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if set(map(int, np.unique(actual))) - {CONTROL, YELLOW}:
        return False
    return True


def _generate_once() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    n_bars = random.choice(BAR_COUNT_CHOICES)
    k = random.randint(1, n_bars)

    bars = _sample_bars(height, width, n_bars, k)
    if bars is None:
        return None

    frame_pos = _sample_frame_top_left(height, width, bars, k)
    if frame_pos is None:
        return None
    frame_top, frame_left = frame_pos

    grid = _render_input(height, width, bars, frame_top, frame_left, k)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
