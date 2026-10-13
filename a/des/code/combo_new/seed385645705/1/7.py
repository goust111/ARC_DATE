import random

import numpy as np


GREY = 5
DIRS: dict[str, tuple[int, int]] = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
PERPS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "U": ((0, -1), (0, 1)),
    "D": ((0, -1), (0, 1)),
    "L": ((-1, 0), (1, 0)),
    "R": ((-1, 0), (1, 0)),
}


def _background(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    max_count = int(counts.max())
    winners = [int(v) for v, n in zip(vals, counts) if int(n) == max_count]
    assert len(winners) == 1, "expected a unique dominant background color"
    return winners[0]


def _inside(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _arrow_cells(center: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    r, c = center
    dr, dc = DIRS[direction]
    cells = [(r, c), (r + dr, c + dc)]
    for pr, pc in PERPS[direction]:
        cells.append((r + pr, c + pc))
    return cells


def _candidate_direction(g: np.ndarray, r: int, c: int, bg: int) -> str | None:
    h, w = g.shape
    if int(g[r, c]) == bg:
        return None

    matches: list[str] = []
    for name, (dr, dc) in DIRS.items():
        needed = [(r + dr, c + dc)]
        needed.extend((r + pr, c + pc) for pr, pc in PERPS[name])
        opp = (r - dr, c - dc)
        if not all(_inside(h, w, rr, cc) and int(g[rr, cc]) == GREY for rr, cc in needed):
            continue
        if not _inside(h, w, opp[0], opp[1]):
            continue
        if int(g[opp]) == GREY:
            continue
        matches.append(name)

    assert len(matches) <= 1, "a T arrow center must encode exactly one direction"
    return matches[0] if matches else None


def _find_arrows(g: np.ndarray, bg: int) -> dict[tuple[int, int], str]:
    arrows: dict[tuple[int, int], str] = {}
    for r, c in np.argwhere(g != bg):
        direction = _candidate_direction(g, int(r), int(c), bg)
        if direction is not None:
            arrows[(int(r), int(c))] = direction

    assert arrows, "expected at least one T arrow"

    covered: set[tuple[int, int]] = set()
    h, w = g.shape
    for center, direction in arrows.items():
        for rr, cc in _arrow_cells(center, direction):
            assert _inside(h, w, rr, cc), "arrow cells must stay inside the grid"
            covered.add((rr, cc))

    for r, c in np.argwhere(g != bg):
        assert (int(r), int(c)) in covered, "every foreground cell must belong to a T arrow"

    colored = {
        (int(r), int(c))
        for r, c in np.argwhere((g != bg) & (g != GREY))
    }
    assert colored <= set(arrows), "colored pixels may only appear at arrow centers"
    return arrows


def _next_center(
    start: tuple[int, int],
    direction: str,
    centers: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = shape
    dr, dc = DIRS[direction]
    r, c = start
    while True:
        r += dr
        c += dc
        if not _inside(h, w, r, c):
            return None
        if (r, c) in centers:
            return (r, c)


def _trace_endpoint(
    start: tuple[int, int],
    arrows: dict[tuple[int, int], str],
    shape: tuple[int, int],
) -> tuple[int, int]:
    centers = set(arrows)
    cur = start
    seen: set[tuple[int, int]] = set()
    while True:
        assert cur not in seen, "arrow chains must not cycle"
        seen.add(cur)
        nxt = _next_center(cur, arrows[cur], centers, shape)
        if nxt is None:
            return cur
        cur = nxt


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace colored signals through T-shaped grey arrows and mark each chain endpoint with the source color.

    Input structure:
    - The background is the dominant color and grey pixels form scattered 3x3 T-shaped arrow elements.
    - Each arrow has a center cell, one grey stem cell in its output direction, and two grey crossbar cells perpendicular to that stem.
    - Some arrow centers are colored signal sources; all other arrow centers and all arrow arms are grey.

    Rule:
    - For every colored source arrow, follow its stem direction to the nearest arrow center on that row or column.
    - At each reached arrow, continue in that arrow's encoded output direction until no further arrow center lies ahead.
    - Paint the terminal arrow center with the original source color, unless that terminal center is itself a colored source, in which case it remains unchanged.

    Key insight:
    - The missing arm of each grey T identifies the opposite of the output direction, so solving requires decoding the local T shape and then tracing the global chain of nearest arrow centers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background(g)
    assert bg != GREY, "grey arrows must differ from the background"

    arrows = _find_arrows(g, bg)
    sources = [
        (pos, int(g[pos]))
        for pos in arrows
        if int(g[pos]) not in (bg, GREY)
    ]
    assert sources, "expected at least one colored signal source"

    source_positions = {pos for pos, _color in sources}
    paints: dict[tuple[int, int], int] = {}
    for pos, color in sources:
        endpoint = _trace_endpoint(pos, arrows, g.shape)
        if endpoint in source_positions:
            continue
        assert int(g[endpoint]) == GREY, "non-source endpoints must have grey centers"
        previous = paints.get(endpoint)
        assert previous is None or previous == color, "different sources may not target the same endpoint"
        paints[endpoint] = color

    out = g.copy()
    for (r, c), color in paints.items():
        out[r, c] = int(color)
    return out


# --------------- generator ---------------

GRID_MIN, GRID_MAX = 16, 30
LATTICE_GAP = 4
CHAIN_COUNT_RANGE = (2, 4)
CHAIN_LENGTH_RANGE = (2, 5)
GENERATION_RETRIES = 300


def _axis_positions(size: int, rng: random.Random) -> list[int]:
    start = rng.randint(1, 3)
    return list(range(start, size - 1, LATTICE_GAP))


def _lattice_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in DIRS.values():
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            out.append((rr, cc))
    return out


def _sample_path(
    rng: random.Random,
    rows: int,
    cols: int,
    used: set[tuple[int, int]],
    length: int,
) -> list[tuple[int, int]] | None:
    free = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in used]
    if not free:
        return None

    for _ in range(100):
        start = rng.choice(free)
        path = [start]
        local = {start}
        for _step in range(length - 1):
            choices = [
                cell
                for cell in _lattice_neighbors(path[-1], rows, cols)
                if cell not in used and cell not in local
            ]
            if not choices:
                break
            nxt = rng.choice(choices)
            path.append(nxt)
            local.add(nxt)
        if len(path) == length:
            return path
    return None


def _sample_paths(
    rng: random.Random,
    rows: int,
    cols: int,
    chain_count: int,
) -> list[list[tuple[int, int]]] | None:
    used: set[tuple[int, int]] = set()
    paths: list[list[tuple[int, int]]] = []
    capacity = rows * cols

    for i in range(chain_count):
        remaining = chain_count - i - 1
        max_len = min(CHAIN_LENGTH_RANGE[1], capacity - len(used) - 2 * remaining)
        if max_len < CHAIN_LENGTH_RANGE[0]:
            return None
        length = rng.randint(CHAIN_LENGTH_RANGE[0], max_len)
        path = _sample_path(rng, rows, cols, used, length)
        if path is None:
            return None
        used.update(path)
        paths.append(path)

    return paths


def _direction_between(a: tuple[int, int], b: tuple[int, int]) -> str:
    ar, ac = a
    br, bc = b
    if br == ar - 1 and bc == ac:
        return "U"
    if br == ar + 1 and bc == ac:
        return "D"
    if br == ar and bc == ac - 1:
        return "L"
    if br == ar and bc == ac + 1:
        return "R"
    raise AssertionError("path steps must be lattice-adjacent")


def _empty_directions(node: tuple[int, int], nodes: set[tuple[int, int]]) -> list[str]:
    r, c = node
    choices: list[str] = []
    if not any(rr < r and cc == c for rr, cc in nodes):
        choices.append("U")
    if not any(rr > r and cc == c for rr, cc in nodes):
        choices.append("D")
    if not any(rr == r and cc < c for rr, cc in nodes):
        choices.append("L")
    if not any(rr == r and cc > c for rr, cc in nodes):
        choices.append("R")
    return choices


def _assign_directions(
    rng: random.Random,
    paths: list[list[tuple[int, int]]],
) -> dict[tuple[int, int], str] | None:
    nodes = {cell for path in paths for cell in path}
    directions: dict[tuple[int, int], str] = {}

    for path in paths:
        for a, b in zip(path, path[1:]):
            directions[a] = _direction_between(a, b)

    for path in paths:
        endpoint = path[-1]
        choices = _empty_directions(endpoint, nodes)
        if not choices:
            return None
        directions[endpoint] = rng.choice(choices)

    if len(set(directions.values())) < 2:
        return None
    return directions


def _pixel_node(
    node: tuple[int, int],
    row_positions: list[int],
    col_positions: list[int],
) -> tuple[int, int]:
    return row_positions[node[0]], col_positions[node[1]]


def _render_grid(
    height: int,
    width: int,
    bg: int,
    row_positions: list[int],
    col_positions: list[int],
    directions: dict[tuple[int, int], str],
    source_colors: dict[tuple[int, int], int],
) -> np.ndarray | None:
    grid = np.full((height, width), bg, dtype=int)

    for node, direction in directions.items():
        center = _pixel_node(node, row_positions, col_positions)
        cells = _arrow_cells(center, direction)
        if any(not _inside(height, width, r, c) or int(grid[r, c]) != bg for r, c in cells):
            return None

        cr, cc = center
        grid[cr, cc] = int(source_colors.get(node, GREY))
        for r, c in cells[1:]:
            grid[r, c] = GREY

    return grid


def _expected_output(
    inp: np.ndarray,
    paths: list[list[tuple[int, int]]],
    row_positions: list[int],
    col_positions: list[int],
    source_colors: dict[tuple[int, int], int],
) -> np.ndarray:
    out = inp.copy()
    source_nodes = set(source_colors)
    for path in paths:
        start = path[0]
        endpoint = path[-1]
        if endpoint in source_nodes:
            continue
        r, c = _pixel_node(endpoint, row_positions, col_positions)
        out[r, c] = int(source_colors[start])
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    row_positions = _axis_positions(height, rng)
    col_positions = _axis_positions(width, rng)
    if len(row_positions) < 3 or len(col_positions) < 3:
        return None

    capacity = len(row_positions) * len(col_positions)
    max_chains = min(CHAIN_COUNT_RANGE[1], capacity // 2)
    if max_chains < CHAIN_COUNT_RANGE[0]:
        return None
    chain_count = rng.randint(CHAIN_COUNT_RANGE[0], max_chains)

    paths = _sample_paths(rng, len(row_positions), len(col_positions), chain_count)
    if paths is None:
        return None

    terminal_source_count = 0
    if chain_count > 1 and rng.random() < 0.5:
        terminal_source_count = rng.randint(1, chain_count - 1)
    terminal_source_indices = set(rng.sample(range(chain_count), terminal_source_count))

    source_nodes: list[tuple[int, int]] = [path[0] for path in paths]
    source_nodes.extend(paths[i][-1] for i in sorted(terminal_source_indices))

    bg = rng.choice([0, 1, 2, 3, 4, 6, 7, 8, 9])
    palette = [c for c in range(10) if c not in (bg, GREY)]
    if len(source_nodes) > len(palette):
        return None
    colors = rng.sample(palette, len(source_nodes))
    source_colors = dict(zip(source_nodes, colors))

    directions = _assign_directions(rng, paths)
    if directions is None:
        return None

    inp = _render_grid(height, width, bg, row_positions, col_positions, directions, source_colors)
    if inp is None:
        return None

    expected = _expected_output(inp, paths, row_positions, col_positions, source_colors)
    if np.array_equal(inp, expected):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_RETRIES} attempts")
