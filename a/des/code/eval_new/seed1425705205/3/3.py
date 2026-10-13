from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PAYLOAD_COLORS = (1, 2, 3, 4, 6, 7, 8)

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
VEC_TO_DIR = {value: key for key, value in DIRS.items()}
COLOR_PRIORITY = {color: rank for rank, color in enumerate((0, 1, 2, 3, 4, 5, 6, 7, 8, 9))}

MIN_CELL_SIZE = 5
MAX_CELL_SIZE = 8
MIN_GRID_CELLS = 2
MAX_GRID_CELLS = 4
MAX_ATTEMPTS = 400


def _grid_neighbors(cell: tuple[int, int], n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < n_rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < n_cols:
        out.append((r, c + 1))
    return out


def _parse_table(grid: np.ndarray) -> tuple[list[int], list[int], int, int, int, int]:
    h, w = grid.shape
    sep_rows = [int(r) for r in range(h) if np.all(grid[r, :] == GREY)]
    sep_cols = [int(c) for c in range(w) if np.all(grid[:, c] == GREY)]

    assert len(sep_rows) >= 2 and len(sep_cols) >= 2, "expected grey separator lines"
    assert sep_rows[0] == 0 and sep_rows[-1] == h - 1, "expected outer grey border rows"
    assert sep_cols[0] == 0 and sep_cols[-1] == w - 1, "expected outer grey border cols"

    row_steps = [b - a for a, b in zip(sep_rows, sep_rows[1:], strict=False)]
    col_steps = [b - a for a, b in zip(sep_cols, sep_cols[1:], strict=False)]
    assert len(set(row_steps)) == 1 and row_steps[0] > 1, "row separators must be evenly spaced"
    assert len(set(col_steps)) == 1 and col_steps[0] > 1, "col separators must be evenly spaced"

    cell_h = row_steps[0] - 1
    cell_w = col_steps[0] - 1
    n_rows = len(sep_rows) - 1
    n_cols = len(sep_cols) - 1
    return sep_rows, sep_cols, n_rows, n_cols, cell_h, cell_w


def _pointer_direction(cell: np.ndarray) -> str:
    coords = np.argwhere(cell == MAROON)
    assert coords.shape == (3, 2), "each region must contain exactly one 3-pixel maroon pointer"

    rows = sorted({int(r) for r, _c in coords})
    cols = sorted({int(c) for _r, c in coords})
    h, w = cell.shape

    if len(rows) == 1:
        ordered = sorted(int(c) for _r, c in coords)
        assert ordered == list(range(ordered[0], ordered[0] + 3)), "horizontal pointer must be contiguous"
        left_gap = ordered[0]
        right_gap = w - 1 - ordered[-1]
        assert left_gap != right_gap, "horizontal pointer must visibly favor one side"
        return "left" if left_gap < right_gap else "right"

    if len(cols) == 1:
        ordered = sorted(int(r) for r, _c in coords)
        assert ordered == list(range(ordered[0], ordered[0] + 3)), "vertical pointer must be contiguous"
        top_gap = ordered[0]
        bottom_gap = h - 1 - ordered[-1]
        assert top_gap != bottom_gap, "vertical pointer must visibly favor one side"
        return "up" if top_gap < bottom_gap else "down"

    raise AssertionError("pointer must be a straight 3-pixel segment")


def _collect_cells(
    grid: np.ndarray,
    sep_rows: list[int],
    sep_cols: list[int],
) -> tuple[dict[tuple[int, int], str], dict[tuple[int, int], list[tuple[int, int, int]]]]:
    directions: dict[tuple[int, int], str] = {}
    payloads: dict[tuple[int, int], list[tuple[int, int, int]]] = {}

    for cell_r in range(len(sep_rows) - 1):
        for cell_c in range(len(sep_cols) - 1):
            r0 = sep_rows[cell_r] + 1
            r1 = sep_rows[cell_r + 1]
            c0 = sep_cols[cell_c] + 1
            c1 = sep_cols[cell_c + 1]
            cell = grid[r0:r1, c0:c1]

            directions[(cell_r, cell_c)] = _pointer_direction(cell)
            coords = np.argwhere((cell != BLACK) & (cell != GREY) & (cell != MAROON))
            payloads[(cell_r, cell_c)] = [
                (int(rr), int(cc), int(cell[rr, cc]))
                for rr, cc in coords
            ]

    return directions, payloads


def _follow_sink(
    cell: tuple[int, int],
    directions: dict[tuple[int, int], str],
    n_rows: int,
    n_cols: int,
    state: dict[tuple[int, int], int],
    memo: dict[tuple[int, int], tuple[int, int]],
) -> tuple[int, int]:
    if cell in memo:
        return memo[cell]
    assert state.get(cell, 0) != 1, "pointer chains must not contain cycles"
    state[cell] = 1

    dr, dc = DIRS[directions[cell]]
    nr, nc = cell[0] + dr, cell[1] + dc
    if 0 <= nr < n_rows and 0 <= nc < n_cols:
        sink = _follow_sink((nr, nc), directions, n_rows, n_cols, state, memo)
    else:
        sink = cell

    state[cell] = 2
    memo[cell] = sink
    return sink


def _route_payloads(
    n_rows: int,
    n_cols: int,
    cell_h: int,
    cell_w: int,
    directions: dict[tuple[int, int], str],
    payloads: dict[tuple[int, int], list[tuple[int, int, int]]],
) -> np.ndarray:
    sinks: dict[tuple[int, int], tuple[int, int]] = {}
    state: dict[tuple[int, int], int] = {}
    for cell_r in range(n_rows):
        for cell_c in range(n_cols):
            cell = (cell_r, cell_c)
            sinks[cell] = _follow_sink(cell, directions, n_rows, n_cols, state, sinks)

    out = np.full((n_rows * cell_h, n_cols * cell_w), BLACK, dtype=int)
    for source, entries in payloads.items():
        sink_r, sink_c = sinks[source]
        base_r = sink_r * cell_h
        base_c = sink_c * cell_w
        for off_r, off_c, color in entries:
            rr = base_r + off_r
            cc = base_c + off_c
            if COLOR_PRIORITY[color] > COLOR_PRIORITY[int(out[rr, cc])]:
                out[rr, cc] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each region's payload pixels along the pointer chain and merge them in the terminal region.

    Input structure:
    - The background is black, and full grey separator rows and columns form a rectangular table with equal-sized cells.
    - Each cell contains one maroon 3-pixel pointer segment aligned horizontally or vertically, placed closer to the side it points toward.
    - Cells may also contain zero or more non-black payload pixels whose colors are neither grey nor maroon.

    Rule:
    - Recover the table geometry from the grey separators, then read one pointer direction per cell.
    - Follow each cell's pointer through neighboring cells until the chain exits the table; that last in-bounds cell is the terminal region for that chain.
    - Remove separators and pointers, and draw every payload pixel only in its terminal region at the same within-cell offset as in its source cell.
    - When multiple routed payloads land on the same output coordinate, keep the higher-priority ARC color in the fixed order black < blue < red < green < yellow < grey < magenta < orange < azure < maroon.

    Key insight:
    - The pointer network must be resolved globally: a cell's payload belongs to the terminal region reached by repeatedly following arrows, not merely to the immediate neighbor it points at.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_rows, sep_cols, n_rows, n_cols, cell_h, cell_w = _parse_table(g)
    directions, payloads = _collect_cells(g, sep_rows, sep_cols)
    return _route_payloads(n_rows, n_cols, cell_h, cell_w, directions, payloads)


def _sample_layout() -> tuple[int, int, int, int] | None:
    n_rows = random.randint(MIN_GRID_CELLS, MAX_GRID_CELLS)
    n_cols = random.randint(MIN_GRID_CELLS, MAX_GRID_CELLS)

    max_cell_h = min(MAX_CELL_SIZE, (30 - (n_rows + 1)) // n_rows)
    max_cell_w = min(MAX_CELL_SIZE, (30 - (n_cols + 1)) // n_cols)
    if max_cell_h < MIN_CELL_SIZE or max_cell_w < MIN_CELL_SIZE:
        return None

    cell_h = random.randint(MIN_CELL_SIZE, max_cell_h)
    cell_w = random.randint(MIN_CELL_SIZE, max_cell_w)
    return n_rows, n_cols, cell_h, cell_w


def _edge_cells(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    out = []
    for r in range(n_rows):
        for c in range(n_cols):
            if r in (0, n_rows - 1) or c in (0, n_cols - 1):
                out.append((r, c))
    return out


def _terminal_direction(cell: tuple[int, int], n_rows: int, n_cols: int) -> str:
    r, c = cell
    options = []
    if r == 0:
        options.append("up")
    if r == n_rows - 1:
        options.append("down")
    if c == 0:
        options.append("left")
    if c == n_cols - 1:
        options.append("right")
    return random.choice(options)


def _build_forest(
    n_rows: int,
    n_cols: int,
) -> tuple[
    dict[tuple[int, int], str],
    dict[tuple[int, int], tuple[int, int]],
    int,
    bool,
]:
    cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    edge = _edge_cells(n_rows, n_cols)
    total = len(cells)
    max_terminals = min(3, len(edge), total - 1)
    terminal_count = random.randint(1, max_terminals)
    roots = random.sample(edge, terminal_count)

    owner: dict[tuple[int, int], int] = {}
    for idx, cell in enumerate(roots):
        owner[cell] = idx
    queue = roots[:]
    random.shuffle(queue)

    head = 0
    while head < len(queue):
        cell = queue[head]
        head += 1
        neighbors = _grid_neighbors(cell, n_rows, n_cols)
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor in owner:
                continue
            owner[neighbor] = owner[cell]
            queue.append(neighbor)

    groups: dict[int, set[tuple[int, int]]] = {idx: {root} for idx, root in enumerate(roots)}
    for cell, idx in owner.items():
        groups[idx].add(cell)

    directions: dict[tuple[int, int], str] = {}
    sinks: dict[tuple[int, int], tuple[int, int]] = {}
    max_depth = 0
    child_counts: defaultdict[tuple[int, int], int] = defaultdict(int)

    for idx, root in enumerate(roots):
        group = groups[idx]
        dist = {root: 0}
        bfs = deque([root])
        while bfs:
            cell = bfs.popleft()
            neighbors = _grid_neighbors(cell, n_rows, n_cols)
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in group or neighbor in dist:
                    continue
                dist[neighbor] = dist[cell] + 1
                bfs.append(neighbor)

        directions[root] = _terminal_direction(root, n_rows, n_cols)
        sinks[root] = root
        max_depth = max(max_depth, max(dist.values()))

        nodes = [cell for cell in group if cell != root]
        nodes.sort(key=lambda cell: dist[cell], reverse=True)
        for cell in nodes:
            candidates = [
                neighbor
                for neighbor in _grid_neighbors(cell, n_rows, n_cols)
                if neighbor in group and dist[neighbor] == dist[cell] - 1
            ]
            parent = random.choice(candidates)
            dr = parent[0] - cell[0]
            dc = parent[1] - cell[1]
            directions[cell] = VEC_TO_DIR[(dr, dc)]
            sinks[cell] = root
            child_counts[parent] += 1

    has_branch = any(count >= 2 for count in child_counts.values())
    return directions, sinks, max_depth, has_branch


def _arrow_cells(direction: str, cell_h: int, cell_w: int) -> set[tuple[int, int]]:
    if direction == "left":
        row = random.randint(1, cell_h - 2)
        return {(row, 0), (row, 1), (row, 2)}
    if direction == "right":
        row = random.randint(1, cell_h - 2)
        return {(row, cell_w - 3), (row, cell_w - 2), (row, cell_w - 1)}
    if direction == "up":
        col = random.randint(1, cell_w - 2)
        return {(0, col), (1, col), (2, col)}
    col = random.randint(1, cell_w - 2)
    return {(cell_h - 3, col), (cell_h - 2, col), (cell_h - 1, col)}


def _free_positions(cell_h: int, cell_w: int, arrow: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(cell_h)
        for c in range(cell_w)
        if (r, c) not in arrow
    }


def _add_payload(
    payload_map: dict[tuple[int, int], dict[tuple[int, int], int]],
    cell: tuple[int, int],
    pos: tuple[int, int],
    color: int,
) -> None:
    payload_map[cell][pos] = color


def _sample_payload_map(
    n_rows: int,
    n_cols: int,
    cell_h: int,
    cell_w: int,
    directions: dict[tuple[int, int], str],
    sinks: dict[tuple[int, int], tuple[int, int]],
) -> tuple[
    dict[tuple[int, int], dict[tuple[int, int], int]],
    dict[tuple[int, int], set[tuple[int, int]]],
]:
    cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    arrows = {cell: _arrow_cells(directions[cell], cell_h, cell_w) for cell in cells}
    free = {cell: _free_positions(cell_h, cell_w, arrows[cell]) for cell in cells}
    payload_map = {cell: {} for cell in cells}

    palette_size = random.randint(3, min(6, len(PAYLOAD_COLORS)))
    palette = random.sample(PAYLOAD_COLORS, palette_size)
    groups: defaultdict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for cell, sink in sinks.items():
        groups[sink].append(cell)

    if random.random() < 0.5:
        overlap_choices: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], list[tuple[int, int]]]] = []
        for contributors in groups.values():
            if len(contributors) < 2:
                continue
            for i in range(len(contributors)):
                for j in range(i + 1, len(contributors)):
                    a = contributors[i]
                    b = contributors[j]
                    common = sorted(free[a] & free[b])
                    if common:
                        overlap_choices.append((a, b, sinks[a], common))
        if overlap_choices:
            a, b, _sink, common = random.choice(overlap_choices)
            pos = random.choice(common)
            color_a, color_b = random.sample(palette, 2)
            _add_payload(payload_map, a, pos, color_a)
            _add_payload(payload_map, b, pos, color_b)

    for cell in cells:
        remaining = sorted(free[cell] - set(payload_map[cell]))
        if not remaining:
            continue
        desired = random.choices((0, 1, 2, 3), weights=(3, 4, 2, 1))[0]
        desired = min(desired, len(remaining))
        if desired == 0:
            continue
        for pos in random.sample(remaining, desired):
            _add_payload(payload_map, cell, pos, random.choice(palette))

    non_roots = [cell for cell in cells if sinks[cell] != cell]
    if not any(payload_map[cell] for cell in non_roots):
        cell = random.choice(non_roots)
        pos = random.choice(sorted(free[cell] - set(payload_map[cell])))
        _add_payload(payload_map, cell, pos, random.choice(palette))

    if sum(len(entries) for entries in payload_map.values()) < 2:
        choices = [cell for cell in cells if free[cell] - set(payload_map[cell])]
        for cell in random.sample(choices, min(2, len(choices))):
            pos = random.choice(sorted(free[cell] - set(payload_map[cell])))
            _add_payload(payload_map, cell, pos, random.choice(palette))

    return payload_map, arrows


def _render_grid(
    n_rows: int,
    n_cols: int,
    cell_h: int,
    cell_w: int,
    directions: dict[tuple[int, int], str],
    payload_map: dict[tuple[int, int], dict[tuple[int, int], int]],
    arrows: dict[tuple[int, int], set[tuple[int, int]]],
) -> np.ndarray:
    height = n_rows * cell_h + (n_rows + 1)
    width = n_cols * cell_w + (n_cols + 1)
    grid = np.full((height, width), BLACK, dtype=int)

    sep_rows = [idx * (cell_h + 1) for idx in range(n_rows + 1)]
    sep_cols = [idx * (cell_w + 1) for idx in range(n_cols + 1)]
    grid[sep_rows, :] = GREY
    grid[:, sep_cols] = GREY

    for cell_r in range(n_rows):
        for cell_c in range(n_cols):
            base_r = sep_rows[cell_r] + 1
            base_c = sep_cols[cell_c] + 1
            cell = (cell_r, cell_c)

            for off_r, off_c in arrows[cell]:
                grid[base_r + off_r, base_c + off_c] = MAROON
            for (off_r, off_c), color in payload_map[cell].items():
                assert grid[base_r + off_r, base_c + off_c] == BLACK
                grid[base_r + off_r, base_c + off_c] = int(color)

    return grid


def _payload_lists(
    payload_map: dict[tuple[int, int], dict[tuple[int, int], int]]
) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    return {
        cell: [(r, c, color) for (r, c), color in entries.items()]
        for cell, entries in payload_map.items()
    }


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    n_rows, n_cols, cell_h, cell_w = layout

    directions, sinks, max_depth, has_branch = _build_forest(n_rows, n_cols)
    if max_depth < 2:
        return None
    if n_rows * n_cols >= 6 and not has_branch:
        return None

    payload_map, arrows = _sample_payload_map(n_rows, n_cols, cell_h, cell_w, directions, sinks)
    payloads = _payload_lists(payload_map)
    expected = _route_payloads(n_rows, n_cols, cell_h, cell_w, directions, payloads)

    grid = _render_grid(n_rows, n_cols, cell_h, cell_w, directions, payload_map, arrows)
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
