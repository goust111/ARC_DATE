import random
from collections import defaultdict, deque

import numpy as np


BG = 0
PIPE = 5
MAROON = 9
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SOURCE_COLORS = (1, 2, 3, 4, 6, 7, 8)

# --------------- solver ---------------


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _build_adjacency(pipe_mask: np.ndarray) -> dict[tuple[int, int], tuple[tuple[int, int], ...]]:
    h, w = pipe_mask.shape
    coords = [tuple(map(int, rc)) for rc in np.argwhere(pipe_mask)]
    assert coords, "expected at least one pipe pixel"

    adj: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    for r, c in coords:
        neighbors = tuple((nr, nc) for nr, nc in _neighbors4(r, c, h, w) if bool(pipe_mask[nr, nc]))
        adj[(r, c)] = neighbors

    start = coords[0]
    seen = {start}
    q = deque([start])
    while q:
        cell = q.popleft()
        for nbr in adj[cell]:
            if nbr not in seen:
                seen.add(nbr)
                q.append(nbr)
    assert len(seen) == len(coords), "pipe network must be 4-connected"
    return adj


def _endpoint_outside(endpoint: tuple[int, int], adj: dict[tuple[int, int], tuple[tuple[int, int], ...]]) -> tuple[int, int]:
    neighbors = adj[endpoint]
    assert len(neighbors) == 1, "endpoint must have degree 1"
    r, c = endpoint
    nr, nc = neighbors[0]
    return r + (r - nr), c + (c - nc)


def _analyze_grid(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    pipe_mask = g == PIPE
    adj = _build_adjacency(pipe_mask)
    assert any(len(neighbors) >= 3 for neighbors in adj.values()), "expected at least one junction"

    h, w = g.shape
    endpoints = [cell for cell, neighbors in adj.items() if len(neighbors) == 1]
    assert len(endpoints) >= 3, "expected multiple pipe endpoints"

    sources: dict[tuple[int, int], int] = {}
    terminals: dict[tuple[int, int], tuple[int, int]] = {}
    source_cells: set[tuple[int, int]] = set()

    for endpoint in endpoints:
        er, ec = _endpoint_outside(endpoint, adj)
        assert 0 <= er < h and 0 <= ec < w, "endpoint exterior must stay in bounds"
        pipe_neighbors = [(nr, nc) for nr, nc in _neighbors4(er, ec, h, w) if bool(pipe_mask[nr, nc])]
        assert pipe_neighbors == [endpoint] or pipe_neighbors == [tuple(endpoint)], "exterior cell must touch only its endpoint"

        value = int(g[er, ec])
        if value == BG:
            terminals[endpoint] = (er, ec)
        else:
            assert value not in (PIPE, MAROON), "source pixel must use a non-pipe, non-maroon color"
            sources[endpoint] = value
            source_cells.add((er, ec))

    assert sources, "expected at least one signal source"
    assert terminals, "expected at least one terminal"

    extras = {
        (int(r), int(c))
        for r, c in np.argwhere((g != BG) & (g != PIPE))
    }
    assert extras == source_cells, "all colored non-pipe cells must be source pixels"

    return g, adj, sources, terminals


def _simulate(adj: dict[tuple[int, int], tuple[tuple[int, int], ...]], sources: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
    claimed: dict[tuple[int, int], int] = {}
    proposals: dict[tuple[int, int], set[int]] = defaultdict(set)
    for cell, color in sources.items():
        proposals[cell].add(int(color))

    while proposals:
        expanders: list[tuple[tuple[int, int], int]] = []
        for cell, colors in proposals.items():
            if cell in claimed:
                continue
            if len(colors) == 1:
                color = next(iter(colors))
                claimed[cell] = color
                expanders.append((cell, color))
            else:
                claimed[cell] = MAROON

        next_proposals: dict[tuple[int, int], set[int]] = defaultdict(set)
        for cell, color in expanders:
            for nbr in adj[cell]:
                if nbr not in claimed:
                    next_proposals[nbr].add(color)
        proposals = next_proposals

    return claimed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate colored sources through a grey pipe network and color terminal cells.

    Input structure:
    - bg = black.
    - Grey pixels form a connected, single-pixel-wide pipe network using only 4-neighbor steps.
    - Some degree-1 pipe endpoints have one colored pixel immediately outside them; these are signal sources.
    - Other degree-1 pipe endpoints have no outside source pixel; those outside cells are terminals.

    Rule:
    - Each source injects its color into its adjacent endpoint and then spreads through the pipe one cell per step.
    - Propagation is synchronous; signals split through junctions.
    - If different colors reach the same previously unclaimed pipe cell in the same step, that pipe cell becomes maroon and blocks further propagation.
    - Reached terminal outside-cells inherit the color of their endpoint; blocked terminals stay black.
    """
    g, adj, sources, terminals = _analyze_grid(grid)
    claimed = _simulate(adj, sources)

    out = g.copy()
    for (r, c), color in claimed.items():
        out[r, c] = int(color)

    for endpoint, (tr, tc) in terminals.items():
        terminal_color = claimed.get(endpoint)
        out[tr, tc] = int(terminal_color) if terminal_color not in (None, MAROON) else BG

    return out


# --------------- generator ---------------

GRID_H_RANGE = (13, 22)
GRID_W_RANGE = (13, 22)
TRUNK_LEFT_RANGE = (2, 4)
TRUNK_RIGHT_MARGIN_RANGE = (3, 5)
TRUNK_MIN_LEN = 7
BRANCH_COUNT_CHOICES = (2, 3)
BRANCH_GAP = 3
SPUR_PROB = 0.55
GEN_RETRIES = 400
ASSIGN_RETRIES = 250


def _draw_h(cells: set[tuple[int, int]], r: int, c0: int, c1: int) -> None:
    lo, hi = sorted((int(c0), int(c1)))
    for c in range(lo, hi + 1):
        cells.add((int(r), c))


def _draw_v(cells: set[tuple[int, int]], c: int, r0: int, r1: int) -> None:
    lo, hi = sorted((int(r0), int(r1)))
    for r in range(lo, hi + 1):
        cells.add((r, int(c)))


def _sample_spaced(lo: int, hi: int, count: int, gap: int, rng: random.Random) -> list[int] | None:
    vals = list(range(lo, hi + 1))
    if len(vals) < count:
        return None
    for _ in range(80):
        picks = sorted(rng.sample(vals, count))
        if all(b - a >= gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _max_spur_len(pipe: set[tuple[int, int]], r: int, c: int, dc: int, limit: int) -> int:
    best = 0
    step = 1
    while True:
        cc = c + dc * step
        outside = c + dc * (step + 1)
        if not (0 <= cc < limit and 0 <= outside < limit):
            break
        if (r, cc) in pipe or (r, outside) in pipe:
            break
        best = step
        step += 1
    return best


def _validate_pipe(pipe: set[tuple[int, int]], h: int, w: int) -> bool:
    if not pipe:
        return False

    mask = np.zeros((h, w), dtype=bool)
    for r, c in pipe:
        if not (0 <= r < h and 0 <= c < w):
            return False
        mask[r, c] = True

    for r in range(h - 1):
        for c in range(w - 1):
            if mask[r : r + 2, c : c + 2].all():
                return False

    try:
        adj = _build_adjacency(mask)
    except AssertionError:
        return False

    if not any(len(neighbors) >= 3 for neighbors in adj.values()):
        return False

    endpoints = [cell for cell, neighbors in adj.items() if len(neighbors) == 1]
    if len(endpoints) < 4:
        return False

    seen_exteriors: set[tuple[int, int]] = set()
    for endpoint in endpoints:
        er, ec = _endpoint_outside(endpoint, adj)
        if not (0 <= er < h and 0 <= ec < w):
            return False
        if (er, ec) in pipe or (er, ec) in seen_exteriors:
            return False
        seen_exteriors.add((er, ec))
        pipe_neighbors = [(nr, nc) for nr, nc in _neighbors4(er, ec, h, w) if mask[nr, nc]]
        if pipe_neighbors != [endpoint] and pipe_neighbors != [tuple(endpoint)]:
            return False

    return True


def _build_horizontal_pipe(h: int, w: int, rng: random.Random) -> set[tuple[int, int]] | None:
    if h < 13 or w < 13:
        return None

    pipe: set[tuple[int, int]] = set()
    row = rng.randint(3, h - 4)
    left = rng.randint(*TRUNK_LEFT_RANGE)
    min_right = max(left + TRUNK_MIN_LEN, w - TRUNK_RIGHT_MARGIN_RANGE[1])
    max_right = w - TRUNK_RIGHT_MARGIN_RANGE[0]
    if min_right > max_right:
        return None
    right = rng.randint(min_right, max_right)
    _draw_h(pipe, row, left, right)

    n_branches = rng.choice(BRANCH_COUNT_CHOICES)
    cols = _sample_spaced(left + 2, right - 2, n_branches, BRANCH_GAP, rng)
    if cols is None:
        return None

    max_up = row - 2
    max_down = h - row - 3
    if max(max_up, max_down) < 2:
        return None

    force_cross = bool(max_up >= 2 and max_down >= 2 and rng.random() < 0.45)
    styles: list[str] = []
    for _ in range(60):
        styles = []
        for _col in cols:
            opts: list[str] = []
            if max_up >= 2:
                opts.extend(["up", "up"])
            if max_down >= 2:
                opts.extend(["down", "down"])
            if max_up >= 2 and max_down >= 2:
                opts.append("cross")
            styles.append(rng.choice(opts))
        if force_cross and "cross" not in styles:
            continue
        if not any(style in ("up", "cross") for style in styles):
            continue
        if not any(style in ("down", "cross") for style in styles):
            continue
        break
    else:
        return None

    arms: list[tuple[int, int, int]] = []
    for col, style in zip(cols, styles):
        if style in ("up", "cross"):
            up_len = rng.randint(2, max_up)
            _draw_v(pipe, col, row - up_len, row)
            arms.append((col, row - up_len, row))
        if style in ("down", "cross"):
            down_len = rng.randint(2, max_down)
            _draw_v(pipe, col, row, row + down_len)
            arms.append((col, row, row + down_len))

    if arms and rng.random() < SPUR_PROB:
        rng.shuffle(arms)
        for col, r0, r1 in arms:
            lo, hi = sorted((r0, r1))
            attach_rows = [rr for rr in range(lo + 1, hi) if abs(rr - row) >= 2]
            if not attach_rows:
                continue
            attach_row = rng.choice(attach_rows)
            dirs = [-1, 1]
            rng.shuffle(dirs)
            for dc in dirs:
                max_len = _max_spur_len(pipe, attach_row, col, dc, w)
                if max_len >= 2:
                    length = rng.randint(2, max_len)
                    _draw_h(pipe, attach_row, col, col + dc * length)
                    return pipe if _validate_pipe(pipe, h, w) else None

    return pipe if _validate_pipe(pipe, h, w) else None


def _generate_pipe(h: int, w: int, rng: random.Random) -> set[tuple[int, int]] | None:
    if rng.random() < 0.5:
        return _build_horizontal_pipe(h, w, rng)

    transposed = _build_horizontal_pipe(w, h, rng)
    if transposed is None:
        return None
    pipe = {(c, r) for r, c in transposed}
    return pipe if _validate_pipe(pipe, h, w) else None


def _assignment_ok(mode: str, claimed: dict[tuple[int, int], int], terminals: list[tuple[int, int]]) -> bool:
    maroon_count = sum(color == MAROON for color in claimed.values())
    colored = sum(1 for endpoint in terminals if claimed.get(endpoint) not in (None, MAROON))
    blocked = len(terminals) - colored
    if mode == "no_conflict":
        return maroon_count == 0 and colored >= 2
    if mode == "partial_block":
        return maroon_count > 0 and colored >= 1 and blocked >= 1
    if mode == "heavy_block":
        return maroon_count > 0 and blocked >= 2
    raise ValueError(mode)


def _assign_sources(
    pipe: set[tuple[int, int]], h: int, w: int, rng: random.Random
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int]]] | None:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in pipe:
        mask[r, c] = True
    adj = _build_adjacency(mask)
    endpoints = [cell for cell, neighbors in adj.items() if len(neighbors) == 1]
    exteriors = {endpoint: _endpoint_outside(endpoint, adj) for endpoint in endpoints}

    modes = ["no_conflict", "partial_block", "heavy_block"]
    preferred = rng.choice(modes)
    ordered_modes = [preferred] + [mode for mode in modes if mode != preferred]

    for mode in ordered_modes:
        for _ in range(ASSIGN_RETRIES):
            max_sources = min(3, len(endpoints) - 1, len(SOURCE_COLORS))
            if max_sources < 2:
                break
            source_count = rng.randint(2, max_sources)
            source_endpoints = rng.sample(endpoints, source_count)
            source_colors = rng.sample(list(SOURCE_COLORS), source_count)
            sources = {endpoint: color for endpoint, color in zip(source_endpoints, source_colors)}
            claimed = _simulate(adj, sources)
            terminal_endpoints = [endpoint for endpoint in endpoints if endpoint not in sources]
            if _assignment_ok(mode, claimed, terminal_endpoints):
                terminals = {endpoint: exteriors[endpoint] for endpoint in terminal_endpoints}
                return sources, terminals

    return None


def _render_grid(
    h: int,
    w: int,
    pipe: set[tuple[int, int]],
    sources: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for r, c in pipe:
        grid[r, c] = PIPE

    mask = grid == PIPE
    adj = _build_adjacency(mask)
    for endpoint, color in sources.items():
        er, ec = _endpoint_outside(endpoint, adj)
        grid[er, ec] = int(color)

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    pipe = _generate_pipe(h, w, rng)
    if pipe is None:
        return None

    assignment = _assign_sources(pipe, h, w, rng)
    if assignment is None:
        return None
    sources, terminals = assignment
    if len(terminals) < 1:
        return None

    grid = _render_grid(h, w, pipe, sources)
    try:
        out = main(grid)
    except AssertionError:
        return None

    terminal_colors = [int(out[r, c]) for r, c in terminals.values()]
    if all(color == BG for color in terminal_colors):
        return None
    if np.array_equal(grid, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe puzzle instance")
