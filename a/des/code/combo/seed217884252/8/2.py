from __future__ import annotations

from collections import deque

import numpy as np


BG = 7
YELLOW = 4
PALETTE = tuple(color for color in range(10) if color not in (BG, YELLOW))
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(sorted(comp))
    return comps


def _is_simple_component_path(cells: list[tuple[int, int]]) -> bool:
    if not cells:
        return False
    if len(cells) == 1:
        return True
    cell_set = set(cells)
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    degrees = []
    for r, c in cells:
        deg = sum((nr, nc) in cell_set for nr, nc in _neighbors4(r, c, h, w))
        if deg > 2:
            return False
        degrees.append(deg)
    return degrees.count(1) == 2 and all(deg in (1, 2) for deg in degrees)


def _bfs_path(mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    if start == goal:
        return [start]
    h, w = mask.shape
    q = deque([start])
    parent = {start: None}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            nxt = (nr, nc)
            if not mask[nr, nc] or nxt in parent:
                continue
            parent[nxt] = (r, c)
            if nxt == goal:
                path = [goal]
                cur = goal
                while parent[cur] is not None:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                return path
            q.append(nxt)
    raise AssertionError("expected a path inside the visible top path")


def _component_adjacent_cells(component: list[tuple[int, int]], mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    adj = set()
    for r, c in component:
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc]:
                adj.add((nr, nc))
    return sorted(adj)


def _pick_overlap(top_mask: np.ndarray, bottom_components: list[list[tuple[int, int]]]):
    assert len(bottom_components) == 2
    adj_lists = [_component_adjacent_cells(comp, top_mask) for comp in bottom_components]
    candidates = []
    for a in adj_lists[0]:
        for b in adj_lists[1]:
            path = _bfs_path(top_mask, a, b)
            path_set = set(path)
            touches = []
            valid = True
            allowed = {path[0], path[-1]}
            for comp in bottom_components:
                touch = set()
                for r, c in comp:
                    for nr, nc in _neighbors4(r, c, *top_mask.shape):
                        if (nr, nc) in path_set:
                            touch.add((nr, nc))
                if not touch or not touch.issubset(allowed):
                    valid = False
                    break
                touches.append(touch)
            if not valid:
                continue
            if len(path) == 1:
                if not all(path[0] in touch for touch in touches):
                    continue
            else:
                left_right = path[0] in touches[0] and path[-1] in touches[1]
                right_left = path[-1] in touches[0] and path[0] in touches[1]
                if not (left_right or right_left):
                    continue

            remainder = top_mask.copy()
            for r, c in path:
                remainder[r, c] = False
            top_parts = _components(remainder)
            if len(top_parts) != 2:
                continue
            if not all(_is_simple_component_path(part) for part in top_parts):
                continue
            candidates.append((len(path), path, top_parts))

    assert candidates, "failed to reconstruct the shared subpath"
    candidates.sort(key=lambda item: (-item[0], item[1]))
    _, path, top_parts = candidates[0]
    return path, top_parts


def _shorter_component(components: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    assert len(components) == 2
    left, right = components
    assert len(left) != len(right), "generator avoids equal-length arms"
    return left if len(left) < len(right) else right


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark the true shared subpath between two simple paths yellow, then erase the shorter arm on each path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    colors = sorted(int(v) for v in np.unique(g) if int(v) != BG)
    assert len(colors) == 2, "expected exactly two non-background colors"

    comp_map = {color: _components(g == color) for color in colors}
    connected = [color for color in colors if len(comp_map[color]) == 1]
    split = [color for color in colors if len(comp_map[color]) == 2]
    assert len(connected) == 1 and len(split) == 1, "expected one visible intact path and one split path"

    top_color = connected[0]
    bottom_color = split[0]
    top_components = comp_map[top_color]
    bottom_components = comp_map[bottom_color]
    assert _is_simple_component_path(top_components[0])
    assert all(_is_simple_component_path(comp) for comp in bottom_components)

    overlap, top_parts = _pick_overlap(g == top_color, bottom_components)
    shorter_top = _shorter_component(top_parts)
    shorter_bottom = _shorter_component(bottom_components)

    out = g.copy()
    for r, c in shorter_top:
        out[r, c] = BG
    for r, c in shorter_bottom:
        out[r, c] = BG
    for r, c in overlap:
        out[r, c] = YELLOW
    return out


def _step(cell: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    return cell[0] + direction[0], cell[1] + direction[1]


def _ordered_path_is_simple(path: list[tuple[int, int]]) -> bool:
    if len(path) < 3 or len(path) != len(set(path)):
        return False
    for i in range(len(path) - 1):
        if abs(path[i][0] - path[i + 1][0]) + abs(path[i][1] - path[i + 1][1]) != 1:
            return False
    return _is_simple_component_path(path)


def _safe_new_cell(
    cell: tuple[int, int],
    prev: tuple[int, int],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    r, c = cell
    if not (0 <= r < h and 0 <= c < w):
        return False
    if cell in occupied:
        return False
    occupied_neighbors = {
        (nr, nc)
        for nr, nc in _neighbors4(r, c, h, w)
        if (nr, nc) in occupied
    }
    return occupied_neighbors.issubset({prev})


def _grow_arm(
    rng: np.random.Generator,
    anchor: tuple[int, int],
    first_dir: tuple[int, int],
    length: int,
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int]] | None:
    assert length >= 1
    first = _step(anchor, first_dir)
    if not _safe_new_cell(first, anchor, occupied, shape):
        return None

    path = [first]
    local_occupied = set(occupied)
    local_occupied.add(first)

    def dfs(cur: tuple[int, int], remaining: int, last_dir: tuple[int, int]) -> bool:
        if remaining == 0:
            return True
        directions = DIRS4.copy()
        rng.shuffle(directions)
        if rng.random() < 0.7:
            directions.sort(key=lambda d: d != last_dir)
        for direction in directions:
            nxt = _step(cur, direction)
            if not _safe_new_cell(nxt, cur, local_occupied, shape):
                continue
            path.append(nxt)
            local_occupied.add(nxt)
            if dfs(nxt, remaining - 1, direction):
                return True
            local_occupied.remove(nxt)
            path.pop()
        return False

    if dfs(first, length - 1, first_dir):
        return path
    return None


def _sample_distinct_lengths(rng: np.random.Generator) -> tuple[int, int]:
    while True:
        a = int(rng.integers(2, 7))
        b = int(rng.integers(2, 7))
        if a != b:
            return a, b


def _build_path(overlap: list[tuple[int, int]], arm_a: list[tuple[int, int]], arm_b: list[tuple[int, int]]):
    return list(reversed(arm_a)) + overlap + arm_b


def _path_overlap_slice(path: list[tuple[int, int]], overlap: set[tuple[int, int]]):
    idx = [i for i, cell in enumerate(path) if cell in overlap]
    assert idx, "expected a non-empty overlap"
    assert idx == list(range(idx[0], idx[-1] + 1)), "overlap must be contiguous in path order"
    left = path[: idx[0]]
    right = path[idx[-1] + 1 :]
    assert left and right, "generator keeps the overlap away from path endpoints"
    return left, right


def _render_input(
    shape: tuple[int, int],
    path0: list[tuple[int, int]],
    color0: int,
    path1: list[tuple[int, int]],
    color1: int,
    top_index: int,
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    items = [(path0, color0), (path1, color1)]
    for idx in (1 - top_index, top_index):
        path, color = items[idx]
        for r, c in path:
            grid[r, c] = color
    return grid


def _expected_output(
    shape: tuple[int, int],
    path0: list[tuple[int, int]],
    color0: int,
    path1: list[tuple[int, int]],
    color1: int,
) -> np.ndarray:
    overlap = set(path0) & set(path1)
    assert overlap
    out = np.full(shape, BG, dtype=int)
    for path, color in ((path0, color0), (path1, color1)):
        left, right = _path_overlap_slice(path, overlap)
        keep = left if len(left) > len(right) else right
        assert len(left) != len(right), "generator avoids ties"
        for r, c in keep:
            out[r, c] = color
    for r, c in overlap:
        out[r, c] = YELLOW
    return out


def _gen_single_cell_overlap(
    rng: np.random.Generator,
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = shape
    anchor = (int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2)))
    directions = DIRS4.copy()
    rng.shuffle(directions)
    lengths = [_sample_distinct_lengths(rng), _sample_distinct_lengths(rng)]
    occupied = {anchor}

    arm00 = _grow_arm(rng, anchor, directions[0], lengths[0][0], occupied, shape)
    if arm00 is None:
        return None
    occupied.update(arm00)
    arm01 = _grow_arm(rng, anchor, directions[1], lengths[0][1], occupied, shape)
    if arm01 is None:
        return None
    occupied.update(arm01)
    arm10 = _grow_arm(rng, anchor, directions[2], lengths[1][0], occupied, shape)
    if arm10 is None:
        return None
    occupied.update(arm10)
    arm11 = _grow_arm(rng, anchor, directions[3], lengths[1][1], occupied, shape)
    if arm11 is None:
        return None

    path0 = _build_path([anchor], arm00, arm01)
    path1 = _build_path([anchor], arm10, arm11)
    return path0, path1


def _gen_shared_segment(
    rng: np.random.Generator,
    shape: tuple[int, int],
    overlap_len: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = shape
    horizontal = bool(rng.integers(0, 2))
    if horizontal:
        row = int(rng.integers(2, h - 2))
        col0 = int(rng.integers(2, w - overlap_len - 1))
        overlap = [(row, col0 + i) for i in range(overlap_len)]
    else:
        row0 = int(rng.integers(2, h - overlap_len - 1))
        col = int(rng.integers(2, w - 2))
        overlap = [(row0 + i, col) for i in range(overlap_len)]

    end0 = overlap[0]
    end1 = overlap[-1]
    if horizontal:
        options0 = [(-1, 0), (1, 0), (0, -1)]
        options1 = [(-1, 0), (1, 0), (0, 1)]
    else:
        options0 = [(0, -1), (0, 1), (-1, 0)]
        options1 = [(0, -1), (0, 1), (1, 0)]

    top_dir0, bottom_dir0 = rng.choice(len(options0), size=2, replace=False)
    top_dir1, bottom_dir1 = rng.choice(len(options1), size=2, replace=False)
    top_len0, top_len1 = _sample_distinct_lengths(rng)
    bottom_len0, bottom_len1 = _sample_distinct_lengths(rng)

    occupied = set(overlap)
    top_arm0 = _grow_arm(rng, end0, options0[int(top_dir0)], top_len0, occupied, shape)
    if top_arm0 is None:
        return None
    occupied.update(top_arm0)
    top_arm1 = _grow_arm(rng, end1, options1[int(top_dir1)], top_len1, occupied, shape)
    if top_arm1 is None:
        return None
    occupied.update(top_arm1)
    bottom_arm0 = _grow_arm(rng, end0, options0[int(bottom_dir0)], bottom_len0, occupied, shape)
    if bottom_arm0 is None:
        return None
    occupied.update(bottom_arm0)
    bottom_arm1 = _grow_arm(rng, end1, options1[int(bottom_dir1)], bottom_len1, occupied, shape)
    if bottom_arm1 is None:
        return None

    path0 = _build_path(overlap, top_arm0, top_arm1)
    if rng.random() < 0.5:
        path1 = _build_path(overlap, bottom_arm0, bottom_arm1)
    else:
        path1 = _build_path(list(reversed(overlap)), bottom_arm1, bottom_arm0)
    return path0, path1


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(10, 19))
    w = int(rng.integers(10, 19))
    shape = (h, w)
    colors = [int(v) for v in rng.choice(PALETTE, size=2, replace=False)]

    if rng.random() < 0.5:
        paths = _gen_single_cell_overlap(rng, shape)
    else:
        overlap_len = int(rng.integers(2, 6))
        paths = _gen_shared_segment(rng, shape, overlap_len)
    if paths is None:
        return None
    path0, path1 = paths

    if not (_ordered_path_is_simple(path0) and _ordered_path_is_simple(path1)):
        return None
    overlap = set(path0) & set(path1)
    if not overlap:
        return None
    if set(path0) ^ overlap and set(path1) ^ overlap:
        pass
    if (set(path0) | set(path1)) - overlap & overlap:
        return None
    left0, right0 = _path_overlap_slice(path0, overlap)
    left1, right1 = _path_overlap_slice(path1, overlap)
    if len(left0) == len(right0) or len(left1) == len(right1):
        return None

    top_index = int(rng.integers(0, 2))
    grid = _render_input(shape, path0, colors[0], path1, colors[1], top_index)
    expected = _expected_output(shape, path0, colors[0], path1, colors[1])

    visible_colors = sorted(int(v) for v in np.unique(grid) if int(v) != BG)
    if visible_colors != sorted(colors):
        return None
    comp_sizes = sorted(len(_components(grid == color)) for color in colors)
    if comp_sizes != [1, 2]:
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-overlap puzzle")
