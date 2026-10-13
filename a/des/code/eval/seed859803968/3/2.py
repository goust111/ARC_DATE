import random

import numpy as np


BLACK = 0
YELLOW = 4
ORANGE = 7
STRUCTURAL = {BLACK, YELLOW, ORANGE}
PATH_COLORS = [1, 2, 3, 5, 6, 8, 9]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _is_path(value: int) -> bool:
    return int(value) not in STRUCTURAL


def _find_marker_rect(grid: np.ndarray, end: tuple[int, int]) -> tuple[int, int, int] | None:
    h, w = grid.shape
    er, ec = end
    candidates = []
    for dc in (1, -1):
        c0 = ec + dc
        if not (0 <= c0 < w):
            continue
        for r0 in (er - 1, er, er - 2):
            if not (0 <= r0 and r0 + 2 < h):
                continue
            width = 0
            c = c0
            while 0 <= c < w and np.all(grid[r0 : r0 + 3, c] == BLACK):
                width += 1
                c += dc
            if width >= 2:
                left = c0 if dc > 0 else c0 - width + 1
                candidates.append((r0, left, width))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
    return candidates[0]


def _trace(grid: np.ndarray, start: tuple[int, int]) -> tuple[tuple[int, int], int]:
    h, w = grid.shape
    previous: tuple[int, int] | None = None
    current = start
    switches = 0
    seen: set[tuple[tuple[int, int] | None, tuple[int, int]]] = set()

    while True:
        state = (previous, current)
        if state in seen:
            return current, switches
        seen.add(state)

        cr, cc = current
        color = int(grid[cr, cc])
        candidates = [
            pos
            for pos in _neighbors(cr, cc, h, w)
            if pos != previous and _is_path(int(grid[pos]))
        ]
        if not candidates:
            return current, switches

        candidates.sort(key=lambda pos: (int(grid[pos]) == color, pos[0], pos[1]))
        nxt = candidates[0]
        if int(grid[nxt]) != color:
            switches += 1
        previous, current = current, nxt


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()
    h, w = g.shape

    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != YELLOW:
                continue
            starts = [
                pos for pos in _neighbors(r, c, h, w) if _is_path(int(g[pos]))
            ]
            if not starts:
                continue
            start = sorted(starts)[0]
            end, count = _trace(g, start)
            rect = _find_marker_rect(g, end)
            if rect is None:
                continue
            r0, c0, width = rect
            fill = min(count, width)
            out[r0 + 1, c0 : c0 + fill] = ORANGE
    return out


def _draw_polyline(grid: np.ndarray, cells: list[tuple[int, int]], colors: list[int]) -> None:
    for (r, c), color in zip(cells, colors):
        if grid[r, c] not in (BLACK, color):
            raise ValueError("path collision")
        grid[r, c] = color


def _make_path(
    h: int,
    w: int,
    base_r: int,
    start_c: int,
    max_end_c: int,
    colors: list[int],
    switch_count: int,
) -> tuple[list[tuple[int, int]], list[int]]:
    cells: list[tuple[int, int]] = []
    cell_colors: list[int] = []
    r = base_r
    c = start_c
    color_idx = 0

    for step in range(switch_count + 1):
        run = random.randint(3, 5)
        for _ in range(run):
            if c > max_end_c:
                break
            cells.append((r, c))
            cell_colors.append(colors[color_idx])
            c += 1
        if step == switch_count:
            break
        direction = random.choice([-1, 1])
        if not (2 <= r + direction <= h - 3):
            direction *= -1
        r += direction
        cells.append((r, c - 1))
        color_idx = (color_idx + 1) % len(colors)
        cell_colors.append(colors[color_idx])

    return cells, cell_colors


def generate_input() -> np.ndarray:
    for _ in range(300):
        h = random.randint(18, 28)
        w = random.randint(24, 34)
        grid = np.zeros((h, w), dtype=int)
        path_count = random.randint(1, 3)
        rows = random.sample(range(3, h - 3), path_count)
        rows.sort()
        ok = True

        for row in rows:
            switch_count = random.randint(1, 5)
            rect_w = max(3, switch_count + random.randint(1, 4))
            end_limit = w - rect_w - 3
            if end_limit < 10:
                ok = False
                break
            start_c = 1
            grid[row, 0] = YELLOW
            colors = random.sample(PATH_COLORS, random.randint(2, min(4, len(PATH_COLORS))))
            cells, cell_colors = _make_path(h, w, row, start_c, end_limit, colors, switch_count)
            if len(cells) < 4:
                ok = False
                break
            end_r, end_c = cells[-1]
            if not (1 <= end_r <= h - 2 and end_c + rect_w < w):
                ok = False
                break
            try:
                _draw_polyline(grid, cells, cell_colors)
            except ValueError:
                ok = False
                break
            rect_r = end_r - 1
            if np.any(grid[rect_r : rect_r + 3, end_c + 1 : end_c + 1 + rect_w] != BLACK):
                ok = False
                break

        if ok and np.count_nonzero(grid == YELLOW) == path_count:
            out = main(grid)
            if isinstance(out, np.ndarray) and out.shape == grid.shape and np.any(out == ORANGE):
                return grid
    raise RuntimeError("failed to generate valid input")
