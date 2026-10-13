import random
from collections import Counter, defaultdict, deque

import numpy as np


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MIXED_COLOR = 9  # ARC maroon

# Generator palette.
GEN_BG = 0
GEN_PIPE = 5
GEN_SIGNAL_COLORS = [1, 2, 3, 4, 6, 7, 8]
GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.flat))
    return int(max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _analyze_network(
    grid: np.ndarray,
) -> tuple[int, int, dict[tuple[int, int], int], dict[tuple[int, int], int], dict[tuple[int, int], list[tuple[int, int]]], list[tuple[tuple[int, int], int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    counts = Counter(map(int, g.flat))
    non_bg = [int(c) for c in counts if int(c) != bg]
    assert non_bg, "grid must contain a pipe network"

    pipe = int(max(non_bg, key=lambda c: (counts[c], -c)))

    sources: dict[tuple[int, int], int] = {}
    gates: dict[tuple[int, int], int] = {}
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or color == pipe:
                continue
            right_is_pipe = c + 1 < w and int(g[r, c + 1]) == pipe
            non_bg_neighbors = 0
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) != bg:
                    non_bg_neighbors += 1
            if right_is_pipe and non_bg_neighbors == 1:
                sources[(r, c)] = color
            else:
                gates[(r, c)] = color

    nodes = {(int(r), int(c)) for r, c in np.argwhere(g == pipe)}
    nodes.update(gates)
    assert nodes, "network must contain pipe cells"

    adj: dict[tuple[int, int], list[tuple[int, int]]] = {pos: [] for pos in nodes}
    for r, c in nodes:
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if (nr, nc) in nodes:
                adj[(r, c)].append((nr, nc))

    entries: list[tuple[tuple[int, int], int]] = []
    for (r, c), color in sorted(sources.items()):
        port = (r, c + 1)
        assert port in nodes and int(g[port]) == pipe, "source must feed a grey entry port"
        entries.append((port, int(color)))
    assert entries, "expected at least one entry source"

    exits: list[tuple[int, int]] = []
    for r, c in nodes:
        if int(g[r, c]) != pipe:
            continue
        if c == 0 or c + 1 < w and (r, c + 1) in nodes:
            continue
        if (r, c - 1) not in nodes:
            continue
        if len(adj[(r, c)]) != 1:
            continue
        exits.append((r, c))

    exits.sort(key=lambda pos: (pos[0], pos[1]))
    assert exits, "expected at least one right-side exit"
    return bg, pipe, sources, gates, adj, entries, exits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace colored signals through a grey pipe network with color-matching gates.

    Entry colors start from colored pixels immediately left of grey endpoints.
    Embedded colored gate cells pass only the matching signal color.
    Exit ports are the grey right endpoints; each output cell reports the set of
    colors that can reach that exit: one color -> that color, many -> maroon,
    none -> background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, pipe, _, gates, adj, entries, exits = _analyze_network(g)
    exit_set = set(exits)
    reached: dict[tuple[int, int], set[int]] = {pos: set() for pos in exits}

    for start, signal_color in entries:
        queue = deque([(None, start)])
        seen: set[tuple[tuple[int, int] | None, tuple[int, int]]] = set()
        while queue:
            prev, cur = queue.popleft()
            state = (prev, cur)
            if state in seen:
                continue
            seen.add(state)

            gate_color = gates.get(cur)
            if gate_color is not None and int(gate_color) != int(signal_color):
                continue

            forward = [nxt for nxt in adj[cur] if nxt != prev]
            if not forward:
                if cur in exit_set:
                    reached[cur].add(int(signal_color))
                continue

            for nxt in forward:
                queue.append((cur, nxt))

    out = np.full((len(exits), 1), bg, dtype=int)
    for idx, pos in enumerate(exits):
        colors = reached[pos]
        if len(colors) == 1:
            out[idx, 0] = next(iter(colors))
        elif len(colors) >= 2:
            out[idx, 0] = MIXED_COLOR
    return out


def _paint_line(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        for c in range(lo, hi + 1):
            existing = int(grid[r0, c])
            assert existing in (GEN_BG, color), "unexpected overlap while drawing"
            grid[r0, c] = int(color)
        return
    assert c0 == c1, "lines must be axis-aligned"
    lo, hi = sorted((r0, r1))
    for r in range(lo, hi + 1):
        existing = int(grid[r, c0])
        assert existing in (GEN_BG, color), "unexpected overlap while drawing"
        grid[r, c0] = int(color)


def _sample_rows(
    row_pool: list[int],
    bus_row: int,
    count: int,
    include_bus: bool,
    forbidden: set[int],
) -> list[int]:
    choices = [r for r in row_pool if r not in forbidden and (include_bus or r != bus_row)]
    selected: list[int] = []
    if include_bus:
        choices = [r for r in row_pool if r not in forbidden]
        assert bus_row in choices
        selected.append(bus_row)

    remaining = count - len(selected)
    above = [r for r in choices if r < bus_row and r not in selected]
    below = [r for r in choices if r > bus_row and r not in selected]

    if remaining >= 2 and above and below:
        pick = random.choice(above)
        selected.append(pick)
        above.remove(pick)
        remaining -= 1

        pick = random.choice(below)
        selected.append(pick)
        below.remove(pick)
        remaining -= 1

    leftovers = [r for r in choices if r not in selected]
    while remaining > 0:
        pick = random.choice(leftovers)
        selected.append(pick)
        leftovers.remove(pick)
        remaining -= 1

    selected = sorted(set(selected))
    assert len(selected) == count
    return selected


def _expected_from_statuses(
    exit_rows: list[int],
    status_map: dict[int, tuple[str, int | None]],
) -> np.ndarray:
    values = []
    for row in sorted(exit_rows):
        status, color = status_map[row]
        if status == "single":
            values.append(int(color))
        elif status == "multi":
            values.append(MIXED_COLOR)
        else:
            values.append(GEN_BG)
    return np.array(values, dtype=int).reshape(-1, 1)


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    n_inputs = random.randint(2, 4)
    n_exits = random.randint(3, 5)
    min_h = max(15, 2 * (n_inputs + n_exits + 1) + 1)
    h = random.randint(min_h, min_h + 6)
    w = random.randint(18, 30)
    row_pool = list(range(1, h - 1, 2))
    if len(row_pool) < n_inputs + n_exits:
        return None

    bus_row = random.choice(row_pool[1:-1])
    cross_mode = random.choice(["entry", "exit", "none"])

    entry_rows = _sample_rows(row_pool, bus_row, n_inputs, cross_mode == "entry", set())
    exit_rows = _sample_rows(
        row_pool,
        bus_row,
        n_exits,
        cross_mode == "exit",
        set(entry_rows),
    )

    c1_hi = max(5, w // 3)
    c1 = random.randint(4, c1_hi)
    c2_low = max(c1 + 5, w // 2)
    c2_high = w - 4
    if c2_low > c2_high:
        return None
    c2 = random.randint(c2_low, c2_high)

    entry_col = 1
    exit_col = w - 2
    if c1 <= entry_col + 1 or c2 >= exit_col:
        return None

    signal_colors = random.sample(GEN_SIGNAL_COLORS, n_inputs)
    blocker_choices = [c for c in GEN_SIGNAL_COLORS if c not in signal_colors]
    blocker_color = random.choice(blocker_choices)

    statuses = ["single", "multi", "none"]
    while len(statuses) < n_exits:
        statuses.append(random.choice(["single", "multi", "none"]))
    random.shuffle(statuses)

    status_map: dict[int, tuple[str, int | None]] = {}
    for row, status in zip(exit_rows, statuses):
        if status == "single":
            status_map[row] = ("single", random.choice(signal_colors))
        elif status == "multi":
            status_map[row] = ("multi", None)
        else:
            status_map[row] = ("none", None)

    grid = np.full((h, w), GEN_BG, dtype=int)

    for row in entry_rows:
        _paint_line(grid, (row, entry_col), (row, c1), GEN_PIPE)
    _paint_line(grid, (min(entry_rows + [bus_row]), c1), (max(entry_rows + [bus_row]), c1), GEN_PIPE)
    _paint_line(grid, (bus_row, c1), (bus_row, c2), GEN_PIPE)
    _paint_line(grid, (min(exit_rows + [bus_row]), c2), (max(exit_rows + [bus_row]), c2), GEN_PIPE)
    for row in exit_rows:
        _paint_line(grid, (row, c2), (row, exit_col), GEN_PIPE)

    for row, color in zip(entry_rows, signal_colors):
        grid[row, entry_col - 1] = int(color)

        if c1 - entry_col >= 2 and random.random() < 0.6:
            gate_c = random.randint(entry_col + 1, c1 - 1)
            grid[row, gate_c] = int(color)

    for row in exit_rows:
        gate_c = random.randint(c2 + 1, exit_col - 1)
        status, color = status_map[row]
        if status == "single":
            grid[row, gate_c] = int(color)
        elif status == "none":
            grid[row, gate_c] = int(blocker_color)

    expected = _expected_from_statuses(exit_rows, status_map)
    return grid, expected


def _is_valid_generated(grid: np.ndarray, expected: np.ndarray) -> bool:
    counts = Counter(map(int, grid.flat))
    if counts[GEN_BG] <= counts[GEN_PIPE]:
        return False
    other_non_bg = [count for color, count in counts.items() if color not in (GEN_BG, GEN_PIPE)]
    if other_non_bg and counts[GEN_PIPE] <= max(other_non_bg):
        return False

    try:
        bg, pipe, sources, _, _, entries, exits = _analyze_network(grid)
    except AssertionError:
        return False

    if bg != GEN_BG or pipe != GEN_PIPE:
        return False
    if len(sources) != len(entries):
        return False
    if len(exits) != expected.shape[0]:
        return False
    if any(port[1] != 1 for port, _ in entries):
        return False
    if int(np.unique(expected).size) < 3:
        return False

    actual = main(grid)
    return actual.shape == expected.shape and np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_candidate()
        if built is None:
            continue
        grid, expected = built
        if _is_valid_generated(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid pipe-and-gates puzzle")
