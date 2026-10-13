import random
from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _inside(h, w, r, c):
    return 0 <= r < h and 0 <= c < w


def _find_legend(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    best = None
    best_dist = None

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or grid[sr, sc] in (BLACK, YELLOW):
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            comp = []
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for dr, dc in DIRS:
                    nr, nc = r + dr, c + dc
                    if (
                        _inside(h, w, nr, nc)
                        and not seen[nr, nc]
                        and grid[nr, nc] not in (BLACK, YELLOW)
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))

            if not (2 <= len(comp) <= 4):
                continue
            rows = [p[0] for p in comp]
            cols = [p[1] for p in comp]
            if len(set(cols)) != 1:
                continue
            if sorted(rows) != list(range(min(rows), max(rows) + 1)):
                continue

            comp_set = set(comp)
            touches_other = False
            for r, c in comp:
                for dr, dc in DIRS:
                    nr, nc = r + dr, c + dc
                    if (
                        _inside(h, w, nr, nc)
                        and (nr, nc) not in comp_set
                        and grid[nr, nc] != BLACK
                    ):
                        touches_other = True
                        break
                if touches_other:
                    break
            if touches_other:
                continue

            c = cols[0]
            r0, r1 = min(rows), max(rows)
            dist = min(r0 + c, r0 + (w - 1 - c), (h - 1 - r1) + c, (h - 1 - r1) + (w - 1 - c))
            if dist > 8:
                continue
            if best is None or dist < best_dist:
                ordered = [(r, c) for r in sorted(rows)]
                best = ([int(grid[r, c]) for r, c in ordered], set(ordered))
                best_dist = dist

    if best is None:
        return [], set()
    return best


def _source_paths(grid, legend_cells, priority):
    h, w = grid.shape
    source_info = {}
    source_order = []

    for r in range(h):
        for c in range(w):
            if grid[r, c] != YELLOW:
                continue
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if (
                    _inside(h, w, nr, nc)
                    and (nr, nc) not in legend_cells
                    and grid[nr, nc] not in (BLACK, YELLOW)
                ):
                    color = int(grid[nr, nc])
                    if color not in source_info:
                        source_info[color] = ((nr, nc), (dr, dc))
                        source_order.append(color)
                    break

    rank = {color: i for i, color in enumerate(priority)}
    for color in source_order:
        if color not in rank:
            rank[color] = len(rank)

    def contains_color(r, c, color, direction):
        if not _inside(h, w, r, c) or (r, c) in legend_cells:
            return False
        val = int(grid[r, c])
        if val == color:
            return True
        if val in (BLACK, YELLOW):
            return False
        dr, dc = direction
        br, bc = r + dr, c + dc
        return (
            _inside(h, w, br, bc)
            and (br, bc) not in legend_cells
            and int(grid[br, bc]) == color
        )

    paths = {}
    directions = {}
    for color in source_order:
        start, direction = source_info[color]
        path = [start]
        r, c = start
        dr, dc = direction
        guard = h * w
        while guard:
            guard -= 1
            nr, nc = r + dr, c + dc
            if not contains_color(nr, nc, color, direction):
                break
            path.append((nr, nc))
            r, c = nr, nc
        if len(path) >= 1:
            paths[color] = path
            directions[color] = direction

    return paths, directions, rank, source_order


def _place_block(base, out, terminal, direction, color):
    h, w = out.shape
    r, c = terminal
    dr, dc = direction
    preferred = []
    if (dr, dc) == (0, 1):
        preferred = [(r, c + 1), (r - 1, c + 1)]
    elif (dr, dc) == (0, -1):
        preferred = [(r, c - 2), (r - 1, c - 2)]
    elif (dr, dc) == (1, 0):
        preferred = [(r + 1, c), (r + 1, c - 1)]
    elif (dr, dc) == (-1, 0):
        preferred = [(r - 2, c), (r - 2, c - 1)]

    candidates = list(preferred)
    for tr in range(r - 2, r + 2):
        for tc in range(c - 2, c + 2):
            cells = [(tr + rr, tc + cc) for rr in range(2) for cc in range(2)]
            if (r, c) in cells:
                continue
            if any(abs(ar - r) + abs(ac - c) == 1 for ar, ac in cells):
                if (tr, tc) not in candidates:
                    candidates.append((tr, tc))

    for tr, tc in candidates:
        cells = [(tr + rr, tc + cc) for rr in range(2) for cc in range(2)]
        if not all(_inside(h, w, ar, ac) for ar, ac in cells):
            continue
        if all(base[ar, ac] == BLACK and out[ar, ac] in (BLACK, color) for ar, ac in cells):
            for ar, ac in cells:
                out[ar, ac] = color
            return


def main(grid: np.ndarray) -> np.ndarray:
    """Propagate edge signals through colored pipe crossings and mark their final terminals.

    Input structure:
    The grid has a black background, isolated yellow source pixels on an outer edge,
    one-pixel-wide horizontal or vertical colored pipes, and a small isolated vertical
    legend of 2-4 pipe colors near a corner. Each yellow source touches the endpoint
    of one colored pipe; different colored pipes may cross at shared pixels, where
    only one of the two colors is visible.

    Rule:
    Read the legend from top to bottom as high-to-low pipe priority. Reconstruct each
    sourced pipe as the straight run extending away from its yellow source, allowing
    a crossing pixel of another color when the same pipe color resumes immediately
    after it. A signal starts from every yellow source. At each crossing, if the
    crossed pipe has higher priority than the current pipe, the signal switches to
    that pipe and continues in that pipe's source-to-terminal direction; otherwise it
    keeps moving straight. When the signal reaches a free terminal, draw a 2x2 block
    of the final pipe color in adjacent black cells beyond that terminal.

    Key insight:
    The visible color stack is not decoration: it gives the priority order needed to
    decide whether each crossing is a turn or a straight-through move.
    """
    grid = np.array(grid, copy=True)
    out = grid.copy()
    priority, legend_cells = _find_legend(grid)
    paths, directions, rank, source_order = _source_paths(grid, legend_cells, priority)

    cell_colors = {}
    index = {}
    for color, path in paths.items():
        index[color] = {}
        for i, pos in enumerate(path):
            index[color][pos] = i
            cell_colors.setdefault(pos, []).append(color)

    finals = []
    for start_color in source_order:
        if start_color not in paths:
            continue
        color = start_color
        i = 0
        seen = set()
        while True:
            path = paths[color]
            if i >= len(path):
                break
            pos = path[i]
            state = (color, i)
            if state in seen:
                break
            seen.add(state)

            higher = [
                other
                for other in cell_colors.get(pos, [])
                if other != color and rank.get(other, 999) < rank.get(color, 999)
            ]
            if higher:
                new_color = min(higher, key=lambda x: rank.get(x, 999))
                color = new_color
                i = index[new_color][pos]
                continue

            if i == len(path) - 1:
                finals.append((pos, directions[color], color))
                break
            i += 1

    used = set()
    for terminal, direction, color in finals:
        key = (terminal, direction, color)
        if key in used:
            continue
        used.add(key)
        _place_block(grid, out, terminal, direction, color)
    return out


def _sample_spaced(lo, hi, n, gap):
    values = list(range(lo, hi + 1))
    random.shuffle(values)
    chosen = []
    for value in values:
        if all(abs(value - other) >= gap for other in chosen):
            chosen.append(value)
            if len(chosen) == n:
                return chosen
    return sorted(random.sample(range(lo, hi + 1), n))


def generate_input() -> np.ndarray:
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    k = random.randint(2, 4)
    colors = random.sample([1, 2, 3, 6, 7, 8, 9], k)
    random.shuffle(colors)

    h_count = random.randint(1, k - 1)
    v_count = k - h_count
    shuffled = colors[:]
    random.shuffle(shuffled)
    h_colors = shuffled[:h_count]
    v_colors = shuffled[h_count:]

    rows = _sample_spaced(5, h - 6, h_count, 3)
    cols = _sample_spaced(5, w - 6, v_count, 3)
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    pipes = {}
    for color, row in zip(h_colors, rows):
        if random.choice([True, False]):
            source = (row, 0)
            start = (row, 1)
            end_col = random.randint(max_col + 1, w - 3)
            direction = (0, 1)
        else:
            source = (row, w - 1)
            start = (row, w - 2)
            end_col = random.randint(2, min_col - 1)
            direction = (0, -1)
        c0, c1 = sorted([start[1], end_col])
        path = [(row, c) for c in range(c0, c1 + 1)]
        if direction == (0, -1):
            path.reverse()
        pipes[color] = {"source": source, "path": path}

    for color, col in zip(v_colors, cols):
        if random.choice([True, False]):
            source = (0, col)
            start = (1, col)
            end_row = random.randint(max_row + 1, h - 3)
            direction = (1, 0)
        else:
            source = (h - 1, col)
            start = (h - 2, col)
            end_row = random.randint(2, min_row - 1)
            direction = (-1, 0)
        r0, r1 = sorted([start[0], end_row])
        path = [(r, col) for r in range(r0, r1 + 1)]
        if direction == (-1, 0):
            path.reverse()
        pipes[color] = {"source": source, "path": path}

    grid = np.zeros((h, w), dtype=int)

    occ = {}
    for color, data in pipes.items():
        for pos in data["path"]:
            occ.setdefault(pos, []).append(color)
    for pos, cell_colors in occ.items():
        grid[pos] = random.choice(cell_colors)
    for data in pipes.values():
        grid[data["source"]] = YELLOW

    legend_col = 1
    legend_start = 1
    for i, color in enumerate(colors):
        grid[legend_start + i, legend_col] = color

    return grid
