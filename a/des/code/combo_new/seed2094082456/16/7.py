from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREEN = 3
RED = 2
BODY_COLORS = [1, 4, 5, 6, 7, 8, 9]

PATH_COUNT_RANGE = (1, 3)
RUN_COUNT_RANGE = (2, 6)
RUN_LENGTH_RANGE = (1, 2)
GRID_HEIGHT_RANGE = (10, 22)
GRID_WIDTH_RANGE = (10, 22)
GEN_RETRIES = 400
PLACE_RETRIES = 80


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
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


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _compress_runs(colors: list[int]) -> list[int]:
    compressed: list[int] = []
    for color in colors:
        if not compressed or compressed[-1] != color:
            compressed.append(int(color))
    return compressed


def _trace_path(grid: np.ndarray, component: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int], list[int]]:
    h, w = grid.shape
    cells = set(component)
    greens = [pos for pos in component if int(grid[pos]) == GREEN]
    reds = [pos for pos in component if int(grid[pos]) == RED]
    assert len(greens) == 1, "each path component must contain exactly one green endpoint"
    assert len(reds) == 1, "each path component must contain exactly one red endpoint"
    start = greens[0]
    end = reds[0]

    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for pos in component:
        r, c = pos
        adjacency[pos] = [nbr for nbr in _neighbors4(r, c, h, w) if nbr in cells]

    for pos, nbrs in adjacency.items():
        expected_degree = 1 if pos in {start, end} else 2
        assert len(nbrs) == expected_degree, "component must be a simple 4-connected path"

    ordered_colors: list[int] = []
    prev: tuple[int, int] | None = None
    cur = start
    visited = {start}
    while cur != end:
        nxts = [nbr for nbr in adjacency[cur] if nbr != prev]
        assert len(nxts) == 1, "path traversal must have a unique next step"
        nxt = nxts[0]
        if nxt != end:
            color = int(grid[nxt])
            assert color not in {BG, GREEN, RED}, "path body must use non-marker foreground colors"
            ordered_colors.append(color)
        prev, cur = cur, nxt
        if cur != end:
            assert cur not in visited, "path must not self-intersect"
            visited.add(cur)

    assert len(visited) + 1 == len(component), "component must be fully covered by the traced path"
    return start, end, ordered_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten each marked color path and compress repeated colors along the walk.

    Input structure:
    - The background is the dominant color.
    - Each non-background 4-connected component is a single simple path with one green endpoint and one red endpoint.
    - The path body between those endpoints is an ordered sequence of colored pixels; the endpoints share a row or a column.

    Rule:
    - For every path, traverse from the green endpoint to the red endpoint and read the body colors in order.
    - Compress adjacent equal colors to one cell, then erase the entire original path.
    - Redraw the path on the straight row or column from the green endpoint toward the red endpoint: keep green at the start, place the compressed colors one cell at a time, and place red immediately after them.

    Key insight:
    - The bends and repeated same-color stretches only encode order and run length; the output keeps the order but discards both the detours and the duplicate neighbors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg = _background_color(g)
    components = _components4(g != bg)
    out = np.full((h, w), bg, dtype=int)

    for component in components:
        start, end, sequence = _trace_path(g, component)
        compressed = _compress_runs(sequence)

        if start[0] == end[0]:
            step = (0, 1 if end[1] > start[1] else -1)
            line_capacity = abs(end[1] - start[1]) - 1
        else:
            assert start[1] == end[1], "endpoints must align horizontally or vertically"
            step = (1 if end[0] > start[0] else -1, 0)
            line_capacity = abs(end[0] - start[0]) - 1

        assert len(compressed) <= line_capacity, "compressed sequence must fit on the endpoint line"
        sr, sc = start
        out[sr, sc] = GREEN

        r, c = sr, sc
        for color in compressed:
            r += step[0]
            c += step[1]
            out[r, c] = int(color)

        r += step[0]
        c += step[1]
        assert 0 <= r < h and 0 <= c < w
        out[r, c] = RED

    return out


def _sample_runs() -> tuple[list[int], list[int]]:
    run_count = random.randint(*RUN_COUNT_RANGE)
    colors: list[int] = []
    prev: int | None = None
    for _ in range(run_count):
        choices = [color for color in BODY_COLORS if color != prev]
        color = random.choice(choices)
        colors.append(color)
        prev = color

    while True:
        repeats = [random.randint(*RUN_LENGTH_RANGE) for _ in range(run_count)]
        extra = sum(length - 1 for length in repeats)
        max_bumps = (run_count + 2) // 2
        if extra > 0 and extra % 2 == 0 and extra // 2 <= max_bumps:
            return colors, repeats


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _can_extend_induced_path(
    nxt: tuple[int, int],
    cur: tuple[int, int],
    end: tuple[int, int],
    visited: set[tuple[int, int]],
    remaining_edges: int,
    h: int,
    w: int,
) -> bool:
    if nxt in visited:
        return False
    next_remaining = remaining_edges - 1
    for adj in _neighbors4(nxt[0], nxt[1], h, w):
        if adj == cur:
            continue
        if adj in visited:
            return False
        if nxt != end and adj == end and next_remaining != 1:
            return False
    if nxt == end:
        return next_remaining == 0
    dist = _manhattan(nxt, end)
    return dist <= next_remaining and dist % 2 == next_remaining % 2


def _find_induced_path(
    h: int,
    w: int,
    start: tuple[int, int],
    end: tuple[int, int],
    edge_count: int,
) -> np.ndarray | None:
    path = [start]
    visited = {start}

    def dfs(cur: tuple[int, int], remaining_edges: int) -> bool:
        if remaining_edges == 0:
            return cur == end

        candidates = _neighbors4(cur[0], cur[1], h, w)
        random.shuffle(candidates)
        for nxt in candidates:
            if not _can_extend_induced_path(nxt, cur, end, visited, remaining_edges, h, w):
                continue

            path.append(nxt)
            if nxt == end:
                if dfs(nxt, remaining_edges - 1):
                    return True
                path.pop()
                continue

            visited.add(nxt)
            if dfs(nxt, remaining_edges - 1):
                return True
            visited.remove(nxt)
            path.pop()
        return False

    if dfs(start, edge_count):
        return np.asarray(path, dtype=int)
    return None


def _reflect_coords(coords: np.ndarray, horizontal: bool) -> np.ndarray:
    reflected = coords.copy()
    if horizontal:
        width = int(reflected[:, 1].max()) + 1
        reflected[:, 1] = width - 1 - reflected[:, 1]
    else:
        height = int(reflected[:, 0].max()) + 1
        reflected[:, 0] = height - 1 - reflected[:, 0]
    return reflected


def _build_patch() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(120):
        colors, repeats = _sample_runs()
        run_count = len(colors)
        body_len = sum(repeats)
        direct_gap = run_count + 1
        edge_count = body_len + 1
        extra = body_len - run_count

        horizontal = bool(random.getrandbits(1))
        forward = bool(random.getrandbits(1))
        parallel_pad_before = random.randint(0, 2)
        parallel_pad_after = random.randint(0, 2)
        orthogonal_bonus = random.randint(0, 2)

        if horizontal:
            height = 3 + extra + orthogonal_bonus
            width = direct_gap + 1 + parallel_pad_before + parallel_pad_after
            row = random.randint(1, height - 2)
            start = (row, parallel_pad_before)
            end = (row, parallel_pad_before + direct_gap)
        else:
            height = direct_gap + 1 + parallel_pad_before + parallel_pad_after
            width = 3 + extra + orthogonal_bonus
            col = random.randint(1, width - 2)
            start = (parallel_pad_before, col)
            end = (parallel_pad_before + direct_gap, col)

        coords = _find_induced_path(height, width, start, end, edge_count)
        if coords is None:
            continue
        if not forward:
            coords = _reflect_coords(coords, horizontal=horizontal)
        break
    else:
        raise RuntimeError("failed to build a valid patch")

    height = int(coords[:, 0].max()) + 1
    width = int(coords[:, 1].max()) + 1
    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)

    expanded: list[int] = []
    for color, length in zip(colors, repeats):
        expanded.extend([int(color)] * int(length))

    start = tuple(map(int, coords[0]))
    end = tuple(map(int, coords[-1]))
    body = [tuple(map(int, pos)) for pos in coords[1:-1]]
    assert len(body) == len(expanded)

    inp[start] = GREEN
    inp[end] = RED
    for pos, color in zip(body, expanded):
        inp[pos] = int(color)

    out[start] = GREEN
    if start[0] == end[0]:
        step = (0, 1 if end[1] > start[1] else -1)
    else:
        step = (1 if end[0] > start[0] else -1, 0)

    r, c = start
    for color in colors:
        r += step[0]
        c += step[1]
        out[r, c] = int(color)
    r += step[0]
    c += step[1]
    out[r, c] = RED
    assert (r, c) == end

    return inp, out


def _can_place_box(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    hh, ww = occupied.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(hh, r0 + h + 1)
    cc1 = min(ww, c0 + w + 1)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _paste_non_bg(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    rows, cols = np.nonzero(patch != BG)
    dst[top + rows, left + cols] = patch[rows, cols]


def _scene_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    path_count = random.randint(*PATH_COUNT_RANGE)

    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(path_count):
        patches.append(_build_patch())

    input_grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    for inp_patch, out_patch in sorted(patches, key=lambda item: item[0].size, reverse=True):
        ph, pw = inp_patch.shape
        if ph > h or pw > w:
            return None

        placed = False
        for _ in range(PLACE_RETRIES):
            r0 = random.randint(0, h - ph)
            c0 = random.randint(0, w - pw)
            if not _can_place_box(occupied, r0, c0, ph, pw):
                continue
            _paste_non_bg(input_grid, inp_patch, r0, c0)
            _paste_non_bg(expected, out_patch, r0, c0)
            occupied[r0:r0 + ph, c0:c0 + pw] = True
            placed = True
            break
        if not placed:
            return None

    if _background_color(input_grid) != BG:
        return None
    try:
        actual = main(input_grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _scene_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
