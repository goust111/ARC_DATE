from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
GREY_PATH = 5
AZURE_END = 8
MAROON_START = 9
GATE_AND_TARGET_COLORS = [1, 2, 3, 4, 6, 7]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_SIDE_RANGE = (18, 30)
COARSE_SIDE_RANGE = (4, 8)
COARSE_PATH_LEN_RANGE = (4, 11)
MAX_GATES = 5
GEN_ATTEMPTS = 400
PATH_BUILD_ATTEMPTS = 80


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in NEIGHBORS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _component_from_seed(mask: np.ndarray, seed: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    assert mask[seed], "seed must be inside the component"
    stack = [seed]
    seen = {seed}
    coords: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        coords.append((r, c))
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return coords


def _all_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _trace_simple_path(
    path_cells: set[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    degrees: dict[tuple[int, int], int] = {}
    for cell in path_cells:
        r, c = cell
        deg = sum((nr, nc) in path_cells for nr, nc in _neighbors4(r, c, h, w))
        degrees[cell] = deg

    assert degrees.get(start) == 1, "start must be a path endpoint"
    assert degrees.get(end) == 1, "end must be a path endpoint"
    for cell, deg in degrees.items():
        if cell in {start, end}:
            continue
        assert deg == 2, "path must be a single non-branching chain"

    ordered = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while cur != end:
        r, c = cur
        nxt = [(nr, nc) for nr, nc in _neighbors4(r, c, h, w) if (nr, nc) in path_cells and (nr, nc) != prev]
        assert len(nxt) == 1, "simple path traversal must have exactly one forward choice"
        prev, cur = cur, nxt[0]
        ordered.append(cur)

    assert len(ordered) == len(path_cells), "traced path must cover the whole component"
    return ordered


def _extract_lookup_blocks(grid: np.ndarray, residual_mask: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for comp in _all_components(residual_mask):
        assert len(comp) == 4, "every detached object must be a 2x2 lookup block"
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert r1 == r0 + 1 and c1 == c0 + 1, "lookup block must fill a 2x2 box"
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        index_color = int(patch[0, 1])
        assert index_color != BG_COLOR, "lookup index color must be non-background"
        assert int(patch[1, 0]) == index_color and int(patch[1, 1]) == index_color
        target_color = int(patch[0, 0])
        assert target_color != BG_COLOR and target_color != index_color
        assert index_color not in mapping, "lookup index colors must be unique"
        mapping[index_color] = target_color
    return mapping


def _render_output(
    shape: tuple[int, int],
    ordered_path: list[tuple[int, int]],
    gate_indices: list[int],
    gate_colors: list[int],
    mapping: dict[int, int],
) -> np.ndarray:
    out = np.full(shape, BG_COLOR, dtype=int)
    out[ordered_path[0]] = MAROON_START
    out[ordered_path[-1]] = AZURE_END

    prev_idx = 0
    for gate_idx, gate_color in zip(gate_indices, gate_colors):
        fill_color = int(mapping[gate_color])
        for r, c in ordered_path[prev_idx + 1 : gate_idx + 1]:
            out[r, c] = fill_color
        prev_idx = gate_idx

    tail_color = int(mapping[gate_colors[-1]])
    for r, c in ordered_path[prev_idx + 1 : -1]:
        out[r, c] = tail_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each path segment using the lookup block of the next gate encountered from maroon to azure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    start_coords = np.argwhere(g == MAROON_START)
    end_coords = np.argwhere(g == AZURE_END)
    assert len(start_coords) == 1, "expected exactly one maroon start"
    assert len(end_coords) == 1, "expected exactly one azure end"
    start = tuple(map(int, start_coords[0]))
    end = tuple(map(int, end_coords[0]))

    path_component = _component_from_seed(g != BG_COLOR, start)
    path_cells = set(path_component)
    assert end in path_cells, "azure endpoint must lie on the same path component"
    ordered_path = _trace_simple_path(path_cells, start, end, g.shape)

    gate_indices: list[int] = []
    gate_colors: list[int] = []
    for idx, (r, c) in enumerate(ordered_path[1:-1], start=1):
        color = int(g[r, c])
        if color not in {GREY_PATH, MAROON_START, AZURE_END}:
            gate_indices.append(idx)
            gate_colors.append(color)
    assert gate_colors, "path must contain at least one gate"

    residual_mask = (g != BG_COLOR).copy()
    for r, c in ordered_path:
        residual_mask[r, c] = False
    mapping = _extract_lookup_blocks(g, residual_mask)
    assert set(mapping) == set(gate_colors), "lookup blocks must match the gate colors exactly"

    return _render_output(g.shape, ordered_path, gate_indices, gate_colors, mapping)


def _coarse_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    return _neighbors4(r, c, rows, cols)


def _reachable_room(
    start: tuple[int, int],
    visited: set[tuple[int, int]],
    rows: int,
    cols: int,
) -> int:
    stack = [start]
    seen = {start}
    count = 0
    while stack:
        cell = stack.pop()
        count += 1
        for nxt in _coarse_neighbors(cell, rows, cols):
            if nxt in seen:
                continue
            if nxt == start or nxt not in visited:
                seen.add(nxt)
                stack.append(nxt)
    return count


def _build_coarse_path(
    rows: int,
    cols: int,
    target_len: int,
) -> list[tuple[int, int]] | None:
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]

    def extend(path: list[tuple[int, int]], visited: set[tuple[int, int]]) -> bool:
        if len(path) == target_len:
            return True
        current = path[-1]
        candidates = [cell for cell in _coarse_neighbors(current, rows, cols) if cell not in visited]
        random.shuffle(candidates)
        candidates.sort(
            key=lambda cell: sum(nb not in visited for nb in _coarse_neighbors(cell, rows, cols)),
            reverse=True,
        )
        for nxt in candidates:
            path.append(nxt)
            visited.add(nxt)
            remaining = target_len - len(path)
            if _reachable_room(nxt, visited, rows, cols) >= remaining + 1 and extend(path, visited):
                return True
            visited.remove(nxt)
            path.pop()
        return False

    for _ in range(PATH_BUILD_ATTEMPTS):
        start = random.choice(all_cells)
        path = [start]
        visited = {start}
        if _reachable_room(start, visited, rows, cols) >= target_len and extend(path, visited):
            return path
    return None


def _embed_path(
    coarse_path: list[tuple[int, int]],
    offset_r: int,
    offset_c: int,
) -> list[tuple[int, int]]:
    ordered = [(offset_r + 2 * coarse_path[0][0], offset_c + 2 * coarse_path[0][1])]
    for (r0, c0), (r1, c1) in zip(coarse_path, coarse_path[1:]):
        ar0, ac0 = offset_r + 2 * r0, offset_c + 2 * c0
        ar1, ac1 = offset_r + 2 * r1, offset_c + 2 * c1
        ordered.append(((ar0 + ar1) // 2, (ac0 + ac1) // 2))
        ordered.append((ar1, ac1))
    return ordered


def _sample_gate_sequence(gate_count: int, repeat_branch: bool) -> list[int]:
    if repeat_branch and gate_count >= 2:
        unique_count = random.randint(1, min(gate_count - 1, len(GATE_AND_TARGET_COLORS)))
        palette = random.sample(GATE_AND_TARGET_COLORS, unique_count)
        seq = palette.copy()
        while len(seq) < gate_count:
            seq.append(random.choice(palette))
        random.shuffle(seq)
        return seq
    seq = random.sample(GATE_AND_TARGET_COLORS, gate_count)
    random.shuffle(seq)
    return seq


def _place_lookup_blocks(
    grid: np.ndarray,
    mapping: dict[int, int],
    path_cells: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    for r, c in path_cells:
        blocked[r, c] = True

    items = list(mapping.items())
    random.shuffle(items)
    for index_color, target_color in items:
        candidates = [(r, c) for r in range(h - 1) for c in range(w - 1)]
        random.shuffle(candidates)
        placed = False
        for r, c in candidates:
            r0, r1 = max(0, r - 1), min(h, r + 3)
            c0, c1 = max(0, c - 1), min(w, c + 3)
            if blocked[r0:r1, c0:c1].any():
                continue
            grid[r, c] = target_color
            grid[r, c + 1] = index_color
            grid[r + 1, c] = index_color
            grid[r + 1, c + 1] = index_color
            blocked[r0:r1, c0:c1] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _make_input_grid(
    shape: tuple[int, int],
    ordered_path: list[tuple[int, int]],
    gate_indices: list[int],
    gate_colors: list[int],
    mapping: dict[int, int],
) -> np.ndarray | None:
    grid = np.full(shape, BG_COLOR, dtype=int)
    for r, c in ordered_path[1:-1]:
        grid[r, c] = GREY_PATH
    grid[ordered_path[0]] = MAROON_START
    grid[ordered_path[-1]] = AZURE_END
    for gate_idx, gate_color in zip(gate_indices, gate_colors):
        grid[ordered_path[gate_idx]] = gate_color
    if not _place_lookup_blocks(grid, mapping, set(ordered_path)):
        return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    coarse_rows = random.randint(*COARSE_SIDE_RANGE)
    coarse_cols = random.randint(*COARSE_SIDE_RANGE)
    coarse_len_hi = min(COARSE_PATH_LEN_RANGE[1], coarse_rows * coarse_cols)
    coarse_len_lo = min(COARSE_PATH_LEN_RANGE[0], coarse_len_hi)
    coarse_len = random.randint(coarse_len_lo, coarse_len_hi)
    coarse_path = _build_coarse_path(coarse_rows, coarse_cols, coarse_len)
    if coarse_path is None:
        return None

    bbox_h = 2 * coarse_rows - 1
    bbox_w = 2 * coarse_cols - 1
    grid_h_lo = max(GRID_SIDE_RANGE[0], bbox_h + 2)
    grid_w_lo = max(GRID_SIDE_RANGE[0], bbox_w + 2)
    if grid_h_lo > GRID_SIDE_RANGE[1] or grid_w_lo > GRID_SIDE_RANGE[1]:
        return None
    grid_h = random.randint(grid_h_lo, GRID_SIDE_RANGE[1])
    grid_w = random.randint(grid_w_lo, GRID_SIDE_RANGE[1])

    offset_r = random.randint(1, grid_h - bbox_h - 1)
    offset_c = random.randint(1, grid_w - bbox_w - 1)
    ordered_path = _embed_path(coarse_path, offset_r, offset_c)

    gate_slots = list(range(2, len(ordered_path) - 2, 2))
    if not gate_slots:
        return None
    gate_count = random.randint(1, min(MAX_GATES, len(gate_slots)))
    gate_indices = sorted(random.sample(gate_slots, gate_count))

    repeat_branch = gate_count >= 2 and random.random() < 0.5
    gate_colors = _sample_gate_sequence(gate_count, repeat_branch)

    mapping: dict[int, int] = {}
    for gate_color in set(gate_colors):
        choices = [color for color in GATE_AND_TARGET_COLORS if color != gate_color]
        mapping[gate_color] = random.choice(choices)

    inp = _make_input_grid((grid_h, grid_w), ordered_path, gate_indices, gate_colors, mapping)
    if inp is None:
        return None

    expected = _render_output((grid_h, grid_w), ordered_path, gate_indices, gate_colors, mapping)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
