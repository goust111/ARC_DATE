from __future__ import annotations

import random

import numpy as np


BG = 0
PATH_COLOR = 3
START_COLOR = 9
GATE_PALETTE = (1, 2, 4, 5, 6, 7, 8)

Coord = tuple[int, int]

MAX_GENERATION_ATTEMPTS = 400
TRACK_CHOICES = (1, 3, 5)
TRACK_GAPS = (2, 3, 4)
MIN_SPAN = 9
MAX_SPAN = 20
MAX_GATES = 4


def _neighbors4(coord: Coord, shape: tuple[int, int]) -> list[Coord]:
    r, c = coord
    h, w = shape
    out: list[Coord] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _neighbors8(coord: Coord, shape: tuple[int, int]) -> list[Coord]:
    r, c = coord
    h, w = shape
    out: list[Coord] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _path_neighbors(mask: np.ndarray, coord: Coord) -> list[Coord]:
    return [nb for nb in _neighbors4(coord, mask.shape) if bool(mask[nb])]


def _walk_path(mask: np.ndarray, start: Coord) -> list[Coord]:
    path: list[Coord] = []
    prev: Coord | None = None
    cur = start

    while True:
        path.append(cur)
        nxt = [nb for nb in _path_neighbors(mask, cur) if nb != prev]
        if not nxt:
            break
        assert len(nxt) == 1, f"path branches at {cur}"
        prev, cur = cur, nxt[0]

    assert len(path) == int(mask.sum()), "path must be a single connected chain"
    return path


def _detect_gate_at(grid: np.ndarray, coord: Coord) -> tuple[int, tuple[Coord, Coord]] | None:
    h, w = grid.shape
    r, c = coord
    found: list[tuple[int, tuple[Coord, Coord]]] = []

    pairs = []
    if 0 < r < h - 1:
        pairs.append(((r - 1, c), (r + 1, c)))
    if 0 < c < w - 1:
        pairs.append(((r, c - 1), (r, c + 1)))

    for a, b in pairs:
        ca = int(grid[a])
        cb = int(grid[b])
        if ca == cb and ca not in (BG, PATH_COLOR, START_COLOR):
            found.append((ca, (a, b)))

    assert len(found) <= 1, f"ambiguous gate at {coord}"
    return found[0] if found else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a single thin green path by the ordered sequence of filter gates it crosses.

    The maroon marker identifies which endpoint is the start. Each gate is a pair
    of same-colored pixels flanking one path pixel; from that path pixel onward,
    the remaining path takes the gate color until another gate overrides it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    path_mask = g == PATH_COLOR
    assert path_mask.any(), "expected a green path"

    path_coords = [tuple(map(int, rc)) for rc in np.argwhere(path_mask)]
    endpoints: list[Coord] = []
    for coord in path_coords:
        deg = len(_path_neighbors(path_mask, coord))
        assert deg in (1, 2), f"path must be branch-free, got degree {deg} at {coord}"
        if deg == 1:
            endpoints.append(coord)
    assert len(endpoints) == 2, f"expected exactly two path endpoints, got {endpoints}"

    start_marks = [tuple(map(int, rc)) for rc in np.argwhere(g == START_COLOR)]
    assert len(start_marks) == 1, "expected exactly one maroon start marker"
    marker = start_marks[0]

    start_candidates = [ep for ep in endpoints if marker in _neighbors8(ep, g.shape)]
    assert len(start_candidates) == 1, "marker must identify exactly one endpoint"
    start = start_candidates[0]

    ordered_path = _walk_path(path_mask, start)
    assert ordered_path[-1] in endpoints and ordered_path[-1] != start

    gate_cells: set[Coord] = set()
    gate_by_index: dict[int, int] = {}
    for idx, coord in enumerate(ordered_path):
        gate = _detect_gate_at(g, coord)
        if gate is None:
            continue
        color, pair = gate
        gate_by_index[idx] = color
        gate_cells.update(pair)

    extra_colors = {
        tuple(map(int, rc))
        for rc in np.argwhere((g != BG) & (g != PATH_COLOR) & (g != START_COLOR))
    }
    assert extra_colors == gate_cells, "all non-special pixels must belong to gates"
    assert len(set(gate_by_index.values())) == len(gate_by_index), "gate colors must be distinct"

    out = g.copy()
    out[g == START_COLOR] = BG

    current_color = PATH_COLOR
    for idx, (r, c) in enumerate(ordered_path):
        if idx in gate_by_index:
            current_color = gate_by_index[idx]
        out[r, c] = current_color

    return out


def _sample_tracks() -> tuple[int, list[int]]:
    track_count = random.choice(TRACK_CHOICES)
    while True:
        gaps = [random.choice(TRACK_GAPS) for _ in range(track_count - 1)]
        min_axis = 3 + sum(gaps)
        if min_axis <= MAX_SPAN:
            break

    axis_len = random.randint(min_axis, MAX_SPAN)
    start = random.randint(1, axis_len - 2 - sum(gaps))
    tracks = [start]
    for gap in gaps:
        tracks.append(tracks[-1] + gap)
    return axis_len, tracks


def _build_horizontal_snake(height: int, width: int, rows: list[int], start_left: bool) -> list[Coord]:
    c_left = 1
    c_right = width - 2
    assert width >= 4

    if len(rows) == 1:
        row = rows[0]
        if start_left:
            return [(row, c) for c in range(width)]
        return [(row, c) for c in range(width - 1, -1, -1)]

    path: list[Coord] = []
    if start_left:
        path.extend((rows[0], c) for c in range(0, c_right + 1))
        current_col = c_right
    else:
        path.extend((rows[0], c) for c in range(width - 1, c_left - 1, -1))
        current_col = c_left

    for i in range(1, len(rows)):
        prev_row = rows[i - 1]
        row = rows[i]
        path.extend((r, current_col) for r in range(prev_row + 1, row + 1))

        if i == len(rows) - 1:
            target_col = width - 1 if current_col == c_left else 0
        else:
            target_col = c_left if current_col == c_right else c_right

        step = 1 if target_col > current_col else -1
        path.extend((row, c) for c in range(current_col + step, target_col + step, step))
        current_col = target_col

    return path


def _build_random_path() -> tuple[int, int, list[Coord]]:
    orientation = random.choice(("horizontal", "vertical"))
    axis_len, tracks = _sample_tracks()
    other_len = random.randint(MIN_SPAN, MAX_SPAN)
    start_first_side = random.choice((True, False))

    if orientation == "horizontal":
        height = axis_len
        width = other_len
        path = _build_horizontal_snake(height, width, tracks, start_first_side)
        return height, width, path

    width = axis_len
    height = other_len
    transposed = _build_horizontal_snake(width, height, tracks, start_first_side)
    path = [(c, r) for r, c in transposed]
    return height, width, path


def _path_axis(path: list[Coord], idx: int) -> str | None:
    r, c = path[idx]
    prev = path[idx - 1] if idx > 0 else None
    nxt = path[idx + 1] if idx + 1 < len(path) else None

    if prev is None and nxt is None:
        return None
    if prev is None:
        return "h" if nxt[0] == r else "v"
    if nxt is None:
        return "h" if prev[0] == r else "v"
    if prev[0] == r and nxt[0] == r:
        return "h"
    if prev[1] == c and nxt[1] == c:
        return "v"
    return None


def _candidate_gates(path: list[Coord], shape: tuple[int, int]) -> list[tuple[int, tuple[Coord, Coord]]]:
    path_set = set(path)
    candidates: list[tuple[int, tuple[Coord, Coord]]] = []
    for idx, (r, c) in enumerate(path):
        if idx < 3 or idx > len(path) - 4:
            continue
        axis = _path_axis(path, idx)
        if axis == "h":
            pair = ((r - 1, c), (r + 1, c))
        elif axis == "v":
            pair = ((r, c - 1), (r, c + 1))
        else:
            continue

        if any(not (0 <= rr < shape[0] and 0 <= cc < shape[1]) for rr, cc in pair):
            continue
        if any(cell in path_set for cell in pair):
            continue
        candidates.append((idx, pair))
    return candidates


def _select_gates(candidates: list[tuple[int, tuple[Coord, Coord]]]) -> list[tuple[int, tuple[Coord, Coord]]]:
    random.shuffle(candidates)
    target_count = random.randint(1, min(MAX_GATES, len(candidates)))
    chosen: list[tuple[int, tuple[Coord, Coord]]] = []
    for idx, pair in candidates:
        if any(abs(idx - prev_idx) < 3 for prev_idx, _prev_pair in chosen):
            continue
        chosen.append((idx, pair))
        if len(chosen) == target_count:
            break
    if not chosen:
        return []
    return sorted(chosen)


def _choose_marker(endpoint: Coord, occupied: set[Coord], shape: tuple[int, int]) -> Coord | None:
    choices = _neighbors8(endpoint, shape)
    random.shuffle(choices)
    for coord in choices:
        if coord not in occupied:
            return coord
    return None


def _build_input() -> np.ndarray | None:
    height, width, path = _build_random_path()
    shape = (height, width)
    path_set = set(path)

    candidates = _candidate_gates(path, shape)
    if not candidates:
        return None

    gates = _select_gates(candidates)
    if not gates:
        return None

    gate_colors = random.sample(GATE_PALETTE, len(gates))
    gate_pixels: set[Coord] = set()
    for _idx, pair in gates:
        gate_pixels.update(pair)

    start_at_front = random.choice((True, False))
    endpoint = path[0] if start_at_front else path[-1]
    marker = _choose_marker(endpoint, path_set | gate_pixels, shape)
    if marker is None:
        return None

    grid = np.full(shape, BG, dtype=int)
    for r, c in path:
        grid[r, c] = PATH_COLOR
    for color, (_idx, pair) in zip(gate_colors, gates):
        for r, c in pair:
            grid[r, c] = color
    grid[marker] = START_COLOR

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert out.shape == grid.shape
    assert int(np.count_nonzero(grid == START_COLOR)) == 1
    assert int(np.count_nonzero(out == START_COLOR)) == 0
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid filter-gate path puzzle")
