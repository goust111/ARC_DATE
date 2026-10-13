import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400
LOGICAL_GENERATION_ATTEMPTS = 200


def _bbox_from_mask(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _line_group(mask: np.ndarray, axis: int):
    groups = {}
    outer = mask.shape[0] if axis == 0 else mask.shape[1]
    for index in range(outer):
        line = mask[index, :] if axis == 0 else mask[:, index]
        positions = np.flatnonzero(line)
        if positions.size < 3:
            continue
        if positions[-1] - positions[0] + 1 != positions.size:
            continue
        key = (int(positions[0]), int(positions[-1]))
        groups.setdefault(key, []).append(index)

    candidates = []
    for (start, end), lines in groups.items():
        lines = sorted(lines)
        if len(lines) < 3:
            continue
        diffs = np.diff(lines)
        if diffs.size == 0:
            continue
        step = int(diffs[0])
        if step < 2 or not np.all(diffs == step):
            continue
        candidates.append((len(lines), end - start, lines, start, end, step))

    assert candidates, "failed to detect a regular grey lattice"
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _count, _span, lines, start, end, step = candidates[0]
    return [int(v) for v in lines], int(start), int(end), int(step)


def _extract_board(grid: np.ndarray):
    grey_mask = grid == GREY
    line_rows, row_c0, row_c1, _row_step = _line_group(grey_mask, axis=0)
    line_cols, col_r0, col_r1, _col_step = _line_group(grey_mask, axis=1)

    assert row_c0 == line_cols[0] and row_c1 == line_cols[-1]
    assert col_r0 == line_rows[0] and col_r1 == line_rows[-1]

    cells = np.zeros((len(line_rows) - 1, len(line_cols) - 1), dtype=int)
    for rr in range(cells.shape[0]):
        for cc in range(cells.shape[1]):
            block = grid[
                line_rows[rr] + 1 : line_rows[rr + 1],
                line_cols[cc] + 1 : line_cols[cc + 1],
            ]
            assert block.size > 0
            colors = np.unique(block)
            assert GREY not in colors and len(colors) == 1
            cells[rr, cc] = int(colors[0])

    board_bbox = (line_rows[0], line_cols[0], line_rows[-1], line_cols[-1])
    return cells, line_rows, line_cols, board_bbox


def _extract_priority(grid: np.ndarray, board_bbox):
    r0, c0, r1, c1 = board_bbox
    outside = np.ones(grid.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    legend_mask = outside & (grid != BLACK) & (grid != GREY)
    legend_bbox = _bbox_from_mask(legend_mask)
    assert legend_bbox is not None, "missing priority legend"

    lr0, lc0, lr1, lc1 = legend_bbox
    assert lr0 > r1 and lc0 > c1, "legend must sit southeast of the board"

    priority = []
    for rr in range(lr0, lr1 + 1):
        colors = sorted(set(map(int, grid[rr, lc0 : lc1 + 1])) - {BLACK, GREY})
        assert len(colors) == 1
        priority.append(colors[0])
    assert len(priority) >= 2
    assert len(set(priority)) == len(priority)
    return priority


def _solve_cells(cells: np.ndarray, priority: list[int]) -> np.ndarray:
    ranks = {int(color): index for index, color in enumerate(priority)}
    used = set(map(int, np.unique(cells))) - {BLACK}
    assert used.issubset(ranks), "every board color must appear in the legend"

    out = cells.copy()
    rows, cols = cells.shape
    for rr in range(rows):
        for cc in range(cols):
            if int(cells[rr, cc]) == BLACK:
                continue
            neighbors = []
            for dr, dc in DIR4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and int(cells[nr, nc]) != BLACK:
                    neighbors.append(int(cells[nr, nc]))
            if not neighbors:
                continue
            counts = Counter(neighbors)
            best = max(counts.values())
            winners = [color for color, count in counts.items() if count == best]
            if len(winners) == 1:
                out[rr, cc] = int(winners[0])
            else:
                out[rr, cc] = int(min(winners, key=lambda color: ranks[color]))
    return out


def _paint_cells(grid: np.ndarray, line_rows: list[int], line_cols: list[int], cells: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for rr in range(cells.shape[0]):
        for cc in range(cells.shape[1]):
            out[
                line_rows[rr] + 1 : line_rows[rr + 1],
                line_cols[cc] + 1 : line_cols[cc + 1],
            ] = int(cells[rr, cc])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every non-empty square-cell by voting among its orthogonal non-empty neighbors.

    Grey lattice lines define the logical cell grid, while a small southeast color
    legend orders the tie-break priority from top to bottom. Empty black cells stay
    empty, isolated colored cells stay unchanged, and all updates are synchronous.
    """
    g = np.asarray(grid, dtype=int)
    cells, line_rows, line_cols, board_bbox = _extract_board(g)
    priority = _extract_priority(g, board_bbox)
    updated = _solve_cells(cells, priority)
    return _paint_cells(g, line_rows, line_cols, updated)


def _summarize_branches(cells: np.ndarray, priority: list[int]):
    updated = _solve_cells(cells, priority)
    summary = {
        "isolated": 0,
        "majority": 0,
        "tie": 0,
        "changed_majority": 0,
        "changed_tie": 0,
        "stable_isolated": 0,
    }
    rows, cols = cells.shape
    for rr in range(rows):
        for cc in range(cols):
            if int(cells[rr, cc]) == BLACK:
                continue
            neighbors = []
            for dr, dc in DIR4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and int(cells[nr, nc]) != BLACK:
                    neighbors.append(int(cells[nr, nc]))
            if not neighbors:
                summary["isolated"] += 1
                if int(updated[rr, cc]) == int(cells[rr, cc]):
                    summary["stable_isolated"] += 1
                continue
            counts = Counter(neighbors)
            best = max(counts.values())
            winners = [color for color, count in counts.items() if count == best]
            if len(winners) == 1:
                summary["majority"] += 1
                if int(updated[rr, cc]) != int(cells[rr, cc]):
                    summary["changed_majority"] += 1
            else:
                summary["tie"] += 1
                if int(updated[rr, cc]) != int(cells[rr, cc]):
                    summary["changed_tie"] += 1
    return summary, updated


def _can_place(cells: np.ndarray, reserved: np.ndarray, rel_cells: list[tuple[int, int, int]], origin_r: int, origin_c: int):
    coords = []
    seen = set()
    for dr, dc, color in rel_cells:
        rr = origin_r + dr
        cc = origin_c + dc
        if not (0 <= rr < cells.shape[0] and 0 <= cc < cells.shape[1]):
            return None
        if (rr, cc) in seen or reserved[rr, cc]:
            return None
        seen.add((rr, cc))
        coords.append((rr, cc, int(color)))
    return coords


def _stamp_pattern(cells: np.ndarray, reserved: np.ndarray, coords: list[tuple[int, int, int]]):
    for rr, cc, color in coords:
        cells[rr, cc] = int(color)
    for rr, cc, _color in coords:
        reserved[rr, cc] = True
        for dr, dc in DIR4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < cells.shape[0] and 0 <= nc < cells.shape[1]:
                reserved[nr, nc] = True


def _place_random_pattern(cells: np.ndarray, reserved: np.ndarray, rel_cells: list[tuple[int, int, int]], attempts: int = 80):
    drs = [dr for dr, _dc, _color in rel_cells]
    dcs = [dc for _dr, dc, _color in rel_cells]
    min_r, max_r = min(drs), max(drs)
    min_c, max_c = min(dcs), max(dcs)
    lo_r = -min_r
    hi_r = cells.shape[0] - 1 - max_r
    lo_c = -min_c
    hi_c = cells.shape[1] - 1 - max_c
    if lo_r > hi_r or lo_c > hi_c:
        return False

    for _ in range(attempts):
        origin_r = random.randint(lo_r, hi_r)
        origin_c = random.randint(lo_c, hi_c)
        coords = _can_place(cells, reserved, rel_cells, origin_r, origin_c)
        if coords is None:
            continue
        _stamp_pattern(cells, reserved, coords)
        return True
    return False


def _make_isolated_pattern(colors: list[int]):
    return [(0, 0, random.choice(colors))]


def _make_unique_pattern(colors: list[int]):
    majority = random.choice(colors)
    others = [color for color in colors if color != majority]
    center = random.choice(others if others else colors)
    neighbor_count = random.choice([3, 4])
    dirs = random.sample(DIR4, neighbor_count)

    if neighbor_count == 3:
        minority = random.choice(others if others else colors)
        neighbor_colors = [majority, majority, minority]
    else:
        if len(others) >= 2 and random.random() < 0.5:
            extra1, extra2 = random.sample(others, 2)
            neighbor_colors = [majority, majority, extra1, extra2]
        else:
            minority = random.choice(others if others else colors)
            neighbor_colors = [majority, majority, majority, minority]
    random.shuffle(neighbor_colors)

    rel_cells = [(0, 0, center)]
    for (dr, dc), color in zip(dirs, neighbor_colors):
        rel_cells.append((dr, dc, color))
    return rel_cells


def _make_tie_pattern(colors: list[int], priority: list[int]):
    ranks = {color: index for index, color in enumerate(priority)}
    mode = random.choice(["two", "four", "three"] if len(colors) >= 3 else ["two", "four"])

    if mode == "three":
        tied = random.sample(colors, 3)
        dirs = random.sample(DIR4, 3)
        neighbor_colors = tied[:]
    else:
        tied = random.sample(colors, 2)
        if mode == "two":
            dirs = random.sample(DIR4, 2)
            neighbor_colors = tied[:]
        else:
            dirs = DIR4[:]
            neighbor_colors = [tied[0], tied[0], tied[1], tied[1]]
            random.shuffle(neighbor_colors)

    winner = min(set(neighbor_colors), key=lambda color: ranks[color])
    center_choices = [color for color in colors if color != winner]
    center = random.choice(center_choices if center_choices else colors)

    rel_cells = [(0, 0, center)]
    for (dr, dc), color in zip(dirs, neighbor_colors):
        rel_cells.append((dr, dc, color))
    return rel_cells


def _make_extra_pattern(colors: list[int]):
    if random.random() < 0.6:
        return _make_isolated_pattern(colors)
    color = random.choice(colors)
    if random.random() < 0.5:
        return [(0, 0, color), (0, 1, color)]
    return [(0, 0, color), (1, 0, color)]


def _build_cells(rows: int, cols: int, colors: list[int], priority: list[int]):
    for _ in range(LOGICAL_GENERATION_ATTEMPTS):
        cells = np.zeros((rows, cols), dtype=int)
        reserved = np.zeros((rows, cols), dtype=bool)

        patterns = [
            _make_isolated_pattern(colors),
            _make_unique_pattern(colors),
            _make_tie_pattern(colors, priority),
        ]
        random.shuffle(patterns)
        ok = True
        for pattern in patterns:
            if not _place_random_pattern(cells, reserved, pattern):
                ok = False
                break
        if not ok:
            continue

        used = set(map(int, np.unique(cells))) - {BLACK}
        missing = [color for color in colors if color not in used]
        for color in missing:
            if not _place_random_pattern(cells, reserved, [(0, 0, color)], attempts=120):
                ok = False
                break
        if not ok:
            continue

        for _extra in range(random.randint(0, 3)):
            _place_random_pattern(cells, reserved, _make_extra_pattern(colors), attempts=40)

        summary, updated = _summarize_branches(cells, priority)
        if summary["isolated"] < 1 or summary["majority"] < 1 or summary["tie"] < 1:
            continue
        if summary["stable_isolated"] < 1 or summary["changed_majority"] < 1 or summary["changed_tie"] < 1:
            continue
        if int(np.sum(updated != cells)) < 2:
            continue
        return cells
    return None


def _render_instance(cells: np.ndarray, priority: list[int]):
    cell_size = random.randint(1, 3)
    rows, cols = cells.shape
    board_h = rows * (cell_size + 1) + 1
    board_w = cols * (cell_size + 1) + 1

    top = random.randint(1, 3)
    left = random.randint(1, 3)
    gap_r = random.randint(1, 3)
    gap_c = random.randint(2, 4)
    legend_r0 = top + board_h + gap_r
    legend_c0 = left + board_w + gap_c

    canvas_h = legend_r0 + len(priority) + random.randint(1, 3)
    canvas_w = legend_c0 + 1 + random.randint(1, 3)
    grid = np.zeros((canvas_h, canvas_w), dtype=int)

    line_rows = [top + step * (cell_size + 1) for step in range(rows + 1)]
    line_cols = [left + step * (cell_size + 1) for step in range(cols + 1)]
    for rr in line_rows:
        grid[rr, left : left + board_w] = GREY
    for cc in line_cols:
        grid[top : top + board_h, cc] = GREY

    grid = _paint_cells(grid, line_rows, line_cols, cells)
    for offset, color in enumerate(priority):
        grid[legend_r0 + offset, legend_c0] = int(color)

    return grid, line_rows, line_cols


def _gen_attempt():
    rows = random.randint(6, 9)
    cols = random.randint(6, 9)
    color_count = random.randint(3, min(5, len(PALETTE)))
    colors = random.sample(PALETTE, color_count)
    priority = random.sample(colors, len(colors))

    cells = _build_cells(rows, cols, colors, priority)
    if cells is None:
        return None

    grid, line_rows, line_cols = _render_instance(cells, priority)
    expected = _paint_cells(grid, line_rows, line_cols, _solve_cells(cells, priority))
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
