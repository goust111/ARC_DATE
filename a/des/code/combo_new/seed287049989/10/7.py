from __future__ import annotations

import numpy as np


BG = 0
PATH_COLOR = 5
GATE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MARGIN = 3
MIN_SEG_LEN = 3
MAX_GENERATION_ATTEMPTS = 300


def _orth_neighbors(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _edge_key(point: tuple[int, int], h: int, w: int) -> tuple[int, int]:
    r, c = point
    if r == 0:
        return (0, c)
    if c == 0:
        return (1, r)
    if r == h - 1:
        return (2, c)
    if c == w - 1:
        return (3, r)
    return (4, r + c)


def _ordered_path(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a path"

    endpoints: list[tuple[int, int]] = []
    for r, c in coords:
        r = int(r)
        c = int(c)
        nbrs = [(rr, cc) for rr, cc in _orth_neighbors(r, c, h, w) if bool(mask[rr, cc])]
        assert len(nbrs) in (1, 2), "path must be a simple 4-connected chain"
        if len(nbrs) == 1:
            assert r in (0, h - 1) or c in (0, w - 1), "path endpoints must lie on the border"
            endpoints.append((r, c))

    assert len(endpoints) == 2, f"expected two border endpoints, got {endpoints}"
    start = min(endpoints, key=lambda pt: _edge_key(pt, h, w))

    ordered = [start]
    visited = {start}
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        next_steps = []
        for nxt in _orth_neighbors(cur[0], cur[1], h, w):
            if not bool(mask[nxt[0], nxt[1]]):
                continue
            if nxt == prev:
                continue
            next_steps.append(nxt)
        assert len(next_steps) <= 1, "path must not branch"
        if not next_steps:
            break
        prev, cur = cur, next_steps[0]
        assert cur not in visited, "path must not self-intersect"
        ordered.append(cur)
        visited.add(cur)

    assert len(visited) == int(mask.sum()), "path traversal missed some cells"
    return ordered


def _line_length(g: np.ndarray, start: tuple[int, int], step: tuple[int, int], color: int) -> int:
    h, w = map(int, g.shape)
    r, c = start
    dr, dc = step
    length = 0
    while 0 <= r < h and 0 <= c < w and int(g[r, c]) == int(color):
        length += 1
        r += dr
        c += dc
    return length


def _gate_color_at(g: np.ndarray, path: list[tuple[int, int]], idx: int) -> int | None:
    if idx <= 0 or idx >= len(path) - 1:
        return None

    pr, pc = path[idx - 1]
    r, c = path[idx]
    nr, nc = path[idx + 1]

    if pr == r == nr:
        if r == 0 or r == g.shape[0] - 1:
            return None
        up = int(g[r - 1, c])
        down = int(g[r + 1, c])
        if up != down or up in (BG, PATH_COLOR):
            return None
        up_len = _line_length(g, (r - 1, c), (-1, 0), up)
        down_len = _line_length(g, (r + 1, c), (1, 0), up)
        if up_len in (2, 3) and down_len in (2, 3):
            return up
        return None

    if pc == c == nc:
        if c == 0 or c == g.shape[1] - 1:
            return None
        left = int(g[r, c - 1])
        right = int(g[r, c + 1])
        if left != right or left in (BG, PATH_COLOR):
            return None
        left_len = _line_length(g, (r, c - 1), (0, -1), left)
        right_len = _line_length(g, (r, c + 1), (0, 1), left)
        if left_len in (2, 3) and right_len in (2, 3):
            return left
        return None

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Trace the gray path and recolor each later segment with the color of the last gate it crossed.

    Input structure:
    - Black background surrounds one continuous one-pixel-wide gray path.
    - The path starts on the top or left edge and ends on the bottom or right edge.
    - Each gate is a same-colored pair of 2-3 pixel straight posts placed on opposite sides of one straight path cell and perpendicular to the path.

    Rule:
    - Traverse the path from its top/left border endpoint toward its bottom/right border endpoint.
    - The path is gray before the first gate.
    - When a path cell lies between the two posts of a gate, keep that crossing cell in the current color, then recolor every later path cell to the gate's color until the next gate.
    - Leave the colored gate posts unchanged.

    Key insight:
    - A gate colors the segment after the crossing, so the solver must first recover the path direction and gate order before repainting the path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    path = _ordered_path(g == PATH_COLOR)
    out = g.copy()

    current_color = PATH_COLOR
    for idx, (r, c) in enumerate(path):
        out[r, c] = current_color
        gate_color = _gate_color_at(g, path, idx)
        if gate_color is not None:
            current_color = gate_color

    return out


def _split_total(total: int, parts: int, min_each: int, rng: np.random.Generator) -> list[int]:
    assert total >= parts * min_each
    values = np.full(parts, min_each, dtype=int)
    extra = int(total - parts * min_each)
    if extra > 0:
        values += rng.multinomial(extra, np.full(parts, 1.0 / parts))
    return [int(v) for v in values]


def _build_path(start: tuple[int, int], runs: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    r, c = start
    path = [(int(r), int(c))]
    for dr, dc, length in runs:
        for _ in range(int(length)):
            r += int(dr)
            c += int(dc)
            path.append((int(r), int(c)))
    return path


def _make_path_attempt(rng: np.random.Generator) -> tuple[tuple[int, int], list[tuple[int, int, int]], tuple[int, int]] | None:
    case = int(rng.integers(4))
    turns = int(rng.integers(1, 4))

    if case == 0:  # top -> bottom
        south_runs = turns + 1
        east_runs = turns
        south_total_lo = max(south_runs * MIN_SEG_LEN, 12)
        south_total_hi = 24
        start_col = int(rng.integers(MARGIN, 7))
        east_total_hi = 26 - start_col
        east_total_lo = max(east_runs * MIN_SEG_LEN, 8)
        if east_total_lo > east_total_hi:
            return None
        south_total = int(rng.integers(south_total_lo, south_total_hi + 1))
        east_total = int(rng.integers(east_total_lo, east_total_hi + 1))
        right_margin = int(rng.integers(MARGIN, min(9, 30 - (start_col + east_total)) ))
        h = south_total + 1
        w = start_col + east_total + right_margin + 1
        south_parts = _split_total(south_total, south_runs, MIN_SEG_LEN, rng)
        east_parts = _split_total(east_total, east_runs, MIN_SEG_LEN, rng)
        runs: list[tuple[int, int, int]] = []
        for i in range(south_runs):
            runs.append((1, 0, south_parts[i]))
            if i < east_runs:
                runs.append((0, 1, east_parts[i]))
        return (0, start_col), runs, (h, w)

    if case == 1:  # top -> right
        run_count = turns
        south_total_lo = max(run_count * MIN_SEG_LEN, 8)
        east_total_lo = max(run_count * MIN_SEG_LEN, 10)
        start_col = int(rng.integers(MARGIN, 7))
        south_total = int(rng.integers(south_total_lo, 19))
        east_total_hi = 29 - start_col
        if east_total_lo > east_total_hi:
            return None
        east_total = int(rng.integers(east_total_lo, east_total_hi + 1))
        bottom_margin_hi = 29 - south_total
        if bottom_margin_hi < MARGIN:
            return None
        bottom_margin = int(rng.integers(MARGIN, min(9, bottom_margin_hi + 1)))
        h = south_total + bottom_margin + 1
        w = start_col + east_total + 1
        south_parts = _split_total(south_total, run_count, MIN_SEG_LEN, rng)
        east_parts = _split_total(east_total, run_count, MIN_SEG_LEN, rng)
        runs = []
        for i in range(run_count):
            runs.append((1, 0, south_parts[i]))
            runs.append((0, 1, east_parts[i]))
        return (0, start_col), runs, (h, w)

    if case == 2:  # left -> bottom
        run_count = turns
        south_total_lo = max(run_count * MIN_SEG_LEN, 10)
        east_total_lo = max(run_count * MIN_SEG_LEN, 8)
        start_row = int(rng.integers(MARGIN, 7))
        east_total = int(rng.integers(east_total_lo, 19))
        south_total_hi = 29 - start_row
        if south_total_lo > south_total_hi:
            return None
        south_total = int(rng.integers(south_total_lo, south_total_hi + 1))
        right_margin_hi = 29 - east_total
        if right_margin_hi < MARGIN:
            return None
        right_margin = int(rng.integers(MARGIN, min(9, right_margin_hi + 1)))
        h = start_row + south_total + 1
        w = east_total + right_margin + 1
        south_parts = _split_total(south_total, run_count, MIN_SEG_LEN, rng)
        east_parts = _split_total(east_total, run_count, MIN_SEG_LEN, rng)
        runs = []
        for i in range(run_count):
            runs.append((0, 1, east_parts[i]))
            runs.append((1, 0, south_parts[i]))
        return (start_row, 0), runs, (h, w)

    south_runs = turns
    east_runs = turns + 1
    south_total_lo = max(south_runs * MIN_SEG_LEN, 8)
    east_total_lo = max(east_runs * MIN_SEG_LEN, 12)
    start_row = int(rng.integers(MARGIN, 7))
    south_total_hi = 26 - start_row
    if south_total_lo > south_total_hi:
        return None
    south_total = int(rng.integers(south_total_lo, south_total_hi + 1))
    east_total = int(rng.integers(east_total_lo, 25))
    bottom_margin_hi = 29 - (start_row + south_total)
    if bottom_margin_hi < MARGIN:
        return None
    bottom_margin = int(rng.integers(MARGIN, min(9, bottom_margin_hi + 1)))
    h = start_row + south_total + bottom_margin + 1
    w = east_total + 1
    south_parts = _split_total(south_total, south_runs, MIN_SEG_LEN, rng)
    east_parts = _split_total(east_total, east_runs, MIN_SEG_LEN, rng)
    runs = []
    for i in range(east_runs):
        runs.append((0, 1, east_parts[i]))
        if i < south_runs:
            runs.append((1, 0, south_parts[i]))
    return (start_row, 0), runs, (h, w)


def _gate_cells(path: list[tuple[int, int]], idx: int, length: int) -> list[tuple[int, int]] | None:
    if idx <= 0 or idx >= len(path) - 1:
        return None
    pr, pc = path[idx - 1]
    r, c = path[idx]
    nr, nc = path[idx + 1]

    if pr == r == nr:
        return [(r - d, c) for d in range(1, length + 1)] + [(r + d, c) for d in range(1, length + 1)]
    if pc == c == nc:
        return [(r, c - d) for d in range(1, length + 1)] + [(r, c + d) for d in range(1, length + 1)]
    return None


def _gate_options(path: list[tuple[int, int]], idx: int, shape: tuple[int, int]) -> list[int]:
    h, w = shape
    path_set = set(path)
    options: list[int] = []
    for length in (2, 3):
        cells = _gate_cells(path, idx, length)
        if cells is None:
            continue
        ok = True
        for r, c in cells:
            if not (0 <= r < h and 0 <= c < w):
                ok = False
                break
            if (r, c) in path_set:
                ok = False
                break
        if ok:
            options.append(length)
    return options


def _select_gate_indices(
    candidates: list[tuple[int, list[int]]],
    gate_count: int,
    path_len: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    usable = [(idx, opts) for idx, opts in candidates if 2 <= idx <= path_len - 3]
    if len(usable) < gate_count:
        return None

    for _ in range(100):
        order = list(range(len(usable)))
        rng.shuffle(order)
        chosen: list[tuple[int, int]] = []
        chosen_idx: list[int] = []
        for pos in order:
            idx, opts = usable[pos]
            if any(abs(idx - prev_idx) < 3 for prev_idx in chosen_idx):
                continue
            chosen_idx.append(idx)
            chosen.append((idx, int(rng.choice(opts))))
            if len(chosen) == gate_count:
                chosen.sort()
                return chosen
    return None


def _render_input(shape: tuple[int, int], path: list[tuple[int, int]], gates: list[dict[str, int]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for r, c in path:
        grid[r, c] = PATH_COLOR
    for gate in gates:
        for r, c in _gate_cells(path, int(gate["idx"]), int(gate["length"])) or []:
            grid[r, c] = int(gate["color"])
    return grid


def _expected_output(inp: np.ndarray, path: list[tuple[int, int]], gates: list[dict[str, int]]) -> np.ndarray:
    out = np.array(inp, copy=True)
    gate_by_idx = {int(gate["idx"]): int(gate["color"]) for gate in gates}
    current_color = PATH_COLOR
    for idx, (r, c) in enumerate(path):
        out[r, c] = current_color
        if idx in gate_by_idx:
            current_color = gate_by_idx[idx]
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    path_info = _make_path_attempt(rng)
    if path_info is None:
        return None
    start, runs, shape = path_info
    path = _build_path(start, runs)

    h, w = shape
    if max(h, w) > 30:
        return None
    if len(path) < 12:
        return None

    candidates = []
    for idx in range(1, len(path) - 1):
        options = _gate_options(path, idx, shape)
        if options:
            candidates.append((idx, options))
    if len(candidates) < 2:
        return None

    gate_count = int(rng.integers(2, 5))
    chosen = _select_gate_indices(candidates, gate_count, len(path), rng)
    if chosen is None:
        return None

    colors = [int(c) for c in rng.choice(GATE_COLORS, size=gate_count, replace=False)]
    gates = [
        {"idx": int(idx), "length": int(length), "color": int(color)}
        for (idx, length), color in zip(chosen, colors)
    ]

    inp = _render_input(shape, path, gates)
    expected = _expected_output(inp, path, gates)
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gate-path puzzle input")
