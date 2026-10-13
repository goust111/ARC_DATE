from __future__ import annotations

import random

import numpy as np


BG = 0
END = 2
START = 3
PATH_COLOR_POOL = (1, 4, 5, 6, 7, 8, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _legend(grid: np.ndarray) -> list[int]:
    legend = [int(grid[0, c]) for c in range(grid.shape[1]) if int(grid[0, c]) != BG]
    assert 2 <= len(legend) <= 4, "legend must list 2-4 colors on the top row"
    assert len(set(legend)) == len(legend), "legend colors must be distinct"
    return legend


def _find_unique(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == color)
    assert coords.shape[0] == 1, f"expected one marker of color {color}"
    return tuple(map(int, coords[0]))


def _non_bg_neighbors(grid: np.ndarray, pos: tuple[int, int]) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    r, c = pos
    out: list[tuple[int, int, int]] = []
    for d, (dr, dc) in enumerate(DIRS):
        nr, nc = r + dr, c + dc
        if _in_bounds(h, w, nr, nc) and int(grid[nr, nc]) != BG:
            out.append((d, nr, nc))
    return out


def _trace_colors(grid: np.ndarray) -> list[int]:
    legend = _legend(grid)
    priority = {color: idx for idx, color in enumerate(legend)}
    path_colors = set(legend)

    start = _find_unique(grid, START)
    end = _find_unique(grid, END)
    assert start[1] == 0, "start marker must be on the left edge"
    assert end[1] == grid.shape[1] - 1, "end marker must be on the right edge"

    start_moves = [
        (d, nr, nc)
        for d, nr, nc in _non_bg_neighbors(grid, start)
        if int(grid[nr, nc]) in path_colors
    ]
    assert len(start_moves) == 1, "start must touch exactly one path segment"
    direction, nr, nc = start_moves[0]
    colors = [int(grid[nr, nc])]
    pos = start
    current_color = colors[-1]

    for _ in range(grid.shape[0] * grid.shape[1] * 4):
        dr, dc = DIRS[direction]
        nr, nc = pos[0] + dr, pos[1] + dc
        assert _in_bounds(grid.shape[0], grid.shape[1], nr, nc), "path left the grid unexpectedly"
        cell = int(grid[nr, nc])
        assert cell != BG, "path broke on background"
        pos = (nr, nc)

        if pos == end:
            return colors

        options: list[tuple[int, int, int]] = []
        for turn_dir in ((direction - 1) % 4, (direction + 1) % 4):
            tr, tc = pos[0] + DIRS[turn_dir][0], pos[1] + DIRS[turn_dir][1]
            if not _in_bounds(grid.shape[0], grid.shape[1], tr, tc):
                continue
            turn_color = int(grid[tr, tc])
            if turn_color in path_colors and turn_color != current_color:
                options.append((priority[turn_color], turn_dir, turn_color))

        if options:
            options.sort()
            _priority, direction, current_color = options[0]
            colors.append(current_color)
            continue

        forward_r, forward_c = pos[0] + dr, pos[1] + dc
        assert _in_bounds(grid.shape[0], grid.shape[1], forward_r, forward_c), "segment terminated before end"
        assert int(grid[forward_r, forward_c]) != BG, "segment terminated before end"

    raise AssertionError("path trace exceeded safety budget")


def main(grid: np.ndarray) -> np.ndarray:
    """Follow the colored signal path and write the traversed segment colors as a stripe at the end marker."""
    g = np.asarray(grid, dtype=int)
    colors = _trace_colors(g)
    end_r, end_c = _find_unique(g, END)
    assert end_r + len(colors) <= g.shape[0], "not enough room below the end marker"

    out = g.copy()
    for i, color in enumerate(colors):
        out[end_r + i, end_c] = int(color)
    return out


def _paint_line(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    r0, c0 = a
    r1, c1 = b
    assert r0 == r1 or c0 == c1, "segments must be axis-aligned"
    rr0, rr1 = sorted((r0, r1))
    cc0, cc1 = sorted((c0, c1))
    if r0 == r1:
        cells = [(r0, c) for c in range(cc0, cc1 + 1)]
    else:
        cells = [(r, c0) for r in range(rr0, rr1 + 1)]
    for r, c in cells:
        if int(grid[r, c]) == BG:
            grid[r, c] = int(color)
        else:
            grid[r, c] = min(int(grid[r, c]), int(color))


def _sample_spaced_values(count: int, low: int, high: int, min_gap: int) -> list[int] | None:
    if count == 0:
        return []
    if high < low:
        return None
    span = high - low + 1
    needed = count + (count - 1) * min_gap
    if span < needed:
        return None
    compressed = span - (count - 1) * min_gap
    picks = sorted(random.sample(range(compressed), count))
    return [low + picks[i] + i * min_gap for i in range(count)]


def _vertical_limit(row: int, used_rows: set[int], height: int, sign: int) -> int:
    assert sign in (-1, 1)
    if sign < 0:
        blocker = max((r for r in used_rows if r < row), default=1)
        return row - (blocker + 1)
    blocker = min((r for r in used_rows if r > row), default=height)
    return (blocker - 1) - row


def _left_limit(col: int, used_cols: set[int]) -> int:
    blocker = max((c for c in used_cols if c < col), default=0)
    return col - (blocker + 1)


def _assign_colors(
    legend: list[int],
    ambiguous: list[bool],
) -> tuple[list[int], list[int | None]] | None:
    priority = {color: idx for idx, color in enumerate(legend)}
    n_segments = len(ambiguous) + 1

    for _ in range(300):
        seg_colors = [random.choice(legend)]
        decoy_colors: list[int | None] = []
        ok = True

        for amb in ambiguous:
            current = seg_colors[-1]
            next_candidates = [c for c in legend if c != current]
            if not next_candidates:
                ok = False
                break

            if amb:
                pairs = [
                    (active, decoy)
                    for active in next_candidates
                    for decoy in next_candidates
                    if active != decoy and priority[active] < priority[decoy]
                ]
                if not pairs:
                    ok = False
                    break
                active, decoy = random.choice(pairs)
                seg_colors.append(active)
                decoy_colors.append(decoy)
            else:
                seg_colors.append(random.choice(next_candidates))
                decoy_colors.append(None)

        if not ok:
            continue
        used = set(seg_colors) | {color for color in decoy_colors if color is not None}
        if used == set(legend) and len(seg_colors) == n_segments:
            return seg_colors, decoy_colors
    return None


def _build_nodes(rows: list[int], cols: list[int], width: int) -> list[tuple[int, int]]:
    nodes = [(rows[0], 0)]
    for i, col in enumerate(cols):
        nodes.append((rows[i], col))
        nodes.append((rows[i + 1], col))
    nodes.append((rows[-1], width - 1))
    return nodes


def _expected_output(grid: np.ndarray, end: tuple[int, int], colors: list[int]) -> np.ndarray:
    out = grid.copy()
    r, c = end
    for i, color in enumerate(colors):
        out[r + i, c] = int(color)
    return out


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(16, 24)
    width = random.randint(20, 30)

    n_segments = random.choice((3, 5))
    n_junctions = n_segments - 1
    n_vertical = (n_segments - 1) // 2
    n_rows = n_vertical + 1

    has_priority_branch = random.random() < 0.5
    if has_priority_branch:
        legend_len = random.choice((3, 4))
    else:
        legend_len = random.choice((2, 3))

    slot_count = n_segments
    if has_priority_branch:
        slot_count += random.randint(1, n_junctions)
    if legend_len > slot_count:
        return None

    row_choices = _sample_spaced_values(n_rows, 2, height - n_segments, 3)
    col_choices = _sample_spaced_values(n_vertical, 4, width - 6, 4)
    if row_choices is None or col_choices is None:
        return None
    random.shuffle(row_choices)
    cols = col_choices

    ambiguous = [False] * n_junctions
    if has_priority_branch:
        count = random.randint(1, n_junctions)
        picks = random.sample(range(n_junctions), count)
        for idx in picks:
            ambiguous[idx] = True

    legend = random.sample(list(PATH_COLOR_POOL), legend_len)
    colors = _assign_colors(legend, ambiguous)
    if colors is None:
        return None
    seg_colors, decoy_colors = colors

    used_rows = set(row_choices)
    used_cols = set(cols)
    nodes = _build_nodes(row_choices, cols, width)

    grid = np.full((height, width), BG, dtype=int)
    legend_cols = sorted(random.sample(range(1, width - 1), legend_len))
    for c, color in zip(legend_cols, legend):
        grid[0, c] = int(color)

    for i in range(n_segments):
        _paint_line(grid, nodes[i], nodes[i + 1], seg_colors[i])

    for j, decoy in enumerate(decoy_colors):
        if decoy is None:
            continue
        junction = nodes[j + 1]
        jr, jc = junction
        if j % 2 == 0:
            active_sign = 1 if nodes[j + 2][0] > jr else -1
            decoy_sign = -active_sign
            limit = _vertical_limit(jr, used_rows, height, decoy_sign)
            if limit < 1:
                return None
            length = random.randint(1, min(4, limit))
            endpoint = (jr + decoy_sign * length, jc)
        else:
            limit = _left_limit(jc, used_cols)
            if limit < 1:
                return None
            length = random.randint(1, min(4, limit))
            endpoint = (jr, jc - length)
        _paint_line(grid, junction, endpoint, int(decoy))

    start = nodes[0]
    end = nodes[-1]
    grid[start] = START
    grid[end] = END

    expected = _expected_output(grid, end, seg_colors)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None

    if not has_priority_branch and legend_len == 3:
        if len(set(seg_colors)) != legend_len:
            return None

    if any(color not in grid for color in legend):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-routing puzzle")
