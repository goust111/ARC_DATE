from __future__ import annotations

import random

import numpy as np


WARM_COLORS = (2, 4, 7)
COOL_COLORS = (1, 3, 8)
BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _mask_to_coords(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _connected_component(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    start = next(iter(coords))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


def _trace_group(grid: np.ndarray, palette: tuple[int, ...]) -> tuple[list[tuple[int, int]], list[int]]:
    mask = np.isin(grid, palette)
    coords = _mask_to_coords(mask)
    assert coords, f"missing path for palette {palette}"
    assert _connected_component(coords) == coords, "path group must be 4-connected"

    degrees: dict[tuple[int, int], int] = {}
    for cell in coords:
        degree = sum((nbr in coords) for nbr in _neighbors4(*cell))
        degrees[cell] = int(degree)

    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(degree in (1, 2) for degree in degrees.values()), "path must be a single chain without branches"

    start = min(endpoints)
    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        order.append(cur)
        next_cells = [nbr for nbr in _neighbors4(*cur) if nbr in coords and nbr != prev]
        if not next_cells:
            break
        assert len(next_cells) == 1, "chain traversal must be unambiguous"
        prev, cur = cur, next_cells[0]

    assert len(order) == len(coords), "trace must visit the whole chain"
    return order, [int(grid[r, c]) for r, c in order]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten two interwoven snake paths and overlay the aligned columns with length-based priority.

    Input structure:
    - The grid is a black background with exactly two non-black path groups.
    - Warm colors {2,4,7} form one 4-connected simple chain, and cool colors {1,3,8} form another.
    - Each chain has exactly two endpoints, no branches, no shared cells, and length at least 4.

    Rule:
    - Split non-black cells into the warm path and the cool path by color family.
    - For each path, start from the endpoint closest to the top-left corner and trace the chain cell by cell.
    - Record the colors in traversal order to form two column vectors; pad the shorter one with black at the end.
    - Overlay the aligned columns row by row: if both entries are non-black, keep the color from the longer path;
      otherwise keep the only non-black entry, or black if both are black.

    Key insight:
    - The 2D geometry only serves to recover two ordered color sequences; the final decision happens after both paths
      are straightened into 1D and conflicts are resolved by total path length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    valid = set(WARM_COLORS) | set(COOL_COLORS) | {BG}
    colors = set(map(int, np.unique(g)))
    assert colors <= valid, "input contains colors outside the declared warm/cool palettes"

    warm_mask = np.isin(g, WARM_COLORS)
    cool_mask = np.isin(g, COOL_COLORS)
    assert np.any(warm_mask) and np.any(cool_mask), "both paths must be present"
    assert not np.any(warm_mask & cool_mask), "paths may not share cells"

    _, warm_seq = _trace_group(g, WARM_COLORS)
    _, cool_seq = _trace_group(g, COOL_COLORS)
    assert len(warm_seq) >= 4 and len(cool_seq) >= 4, "each path must have length at least 4"
    assert len(warm_seq) != len(cool_seq), "the two paths must have different lengths"

    out_len = max(len(warm_seq), len(cool_seq))
    warm_pad = np.array(warm_seq + [BG] * (out_len - len(warm_seq)), dtype=int)
    cool_pad = np.array(cool_seq + [BG] * (out_len - len(cool_seq)), dtype=int)

    warm_longer = len(warm_seq) > len(cool_seq)
    out = np.zeros(out_len, dtype=int)
    for idx in range(out_len):
        w = int(warm_pad[idx])
        c = int(cool_pad[idx])
        if w != BG and c != BG:
            out[idx] = w if warm_longer else c
        elif w != BG:
            out[idx] = w
        elif c != BG:
            out[idx] = c
    return out.reshape((-1, 1))


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(2, len(path)):
        d1 = (path[i - 1][0] - path[i - 2][0], path[i - 1][1] - path[i - 2][1])
        d2 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        turns += d1 != d2
    return turns


def _span(path: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in path]
    cols = [c for _, c in path]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _candidate_next_cells(
    shape: tuple[int, int],
    cur: tuple[int, int],
    prev: tuple[int, int] | None,
    occupied: set[tuple[int, int]],
    path_set: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    candidates = []
    for nr, nc in _neighbors4(*cur):
        if not _in_bounds(shape, nr, nc):
            continue
        if (nr, nc) in occupied or (nr, nc) in path_set:
            continue
        if sum((nbr in path_set) for nbr in _neighbors4(nr, nc)) != 1:
            continue
        score = random.random()
        if prev is not None:
            prev_dir = (cur[0] - prev[0], cur[1] - prev[1])
            next_dir = (nr - cur[0], nc - cur[1])
            if next_dir != prev_dir:
                score -= 0.35
        if nr in (0, shape[0] - 1) or nc in (0, shape[1] - 1):
            score += 0.2
        candidates.append((score, (nr, nc)))
    candidates.sort(key=lambda item: item[0])
    return [cell for _, cell in candidates]


def _dfs_path(
    shape: tuple[int, int],
    target_len: int,
    occupied: set[tuple[int, int]],
    path: list[tuple[int, int]],
    path_set: set[tuple[int, int]],
) -> bool:
    if len(path) == target_len:
        return True

    prev = path[-2] if len(path) >= 2 else None
    cur = path[-1]
    for nxt in _candidate_next_cells(shape, cur, prev, occupied, path_set):
        path.append(nxt)
        path_set.add(nxt)
        if _dfs_path(shape, target_len, occupied, path, path_set):
            return True
        path.pop()
        path_set.remove(nxt)
    return False


def _build_path(
    shape: tuple[int, int],
    length: int,
    occupied: set[tuple[int, int]],
    prefer_adjacent_to: set[tuple[int, int]] | None,
    require_touch: bool,
) -> list[tuple[int, int]] | None:
    all_cells = [(r, c) for r in range(1, shape[0] - 1) for c in range(1, shape[1] - 1) if (r, c) not in occupied]
    random.shuffle(all_cells)

    if prefer_adjacent_to:
        near = []
        far = []
        for cell in all_cells:
            touching = any(nbr in prefer_adjacent_to for nbr in _neighbors4(*cell))
            (near if touching else far).append(cell)
        starts = near + far if require_touch else far + near
    else:
        starts = all_cells

    for start in starts[:120]:
        path = [start]
        path_set = {start}
        if not _dfs_path(shape, length, occupied, path, path_set):
            continue
        turns = _turn_count(path)
        span_r, span_c = _span(path)
        if turns < 1:
            continue
        if span_r < 2 or span_c < 2:
            continue
        if len(path) >= 8 and turns < 2:
            continue
        if require_touch and prefer_adjacent_to is not None:
            if not any(any(nbr in prefer_adjacent_to for nbr in _neighbors4(r, c)) for r, c in path):
                continue
        return path
    return None


def _partition_length(total: int, parts: int) -> list[int] | None:
    if parts <= 0 or parts > total:
        return None
    lengths = [1] * parts
    for _ in range(total - parts):
        idx = random.randrange(parts)
        lengths[idx] += 1
    random.shuffle(lengths)
    return lengths


def _color_path(path: list[tuple[int, int]], palette: tuple[int, ...]) -> list[int]:
    max_segments = min(5, len(path))
    min_segments = 2 if len(path) >= 4 else 1
    segments = random.randint(min_segments, max_segments)
    segment_lengths = _partition_length(len(path), segments)
    assert segment_lengths is not None

    used_colors = random.sample(list(palette), k=random.randint(2, len(palette)))
    colors = []
    prev_color = None
    for seg_len in segment_lengths:
        choices = [color for color in used_colors if color != prev_color]
        color = random.choice(choices)
        colors.extend([color] * seg_len)
        prev_color = color

    if len(set(colors)) < 2:
        pivot = random.randrange(len(colors))
        replacement = random.choice([color for color in palette if color != colors[pivot]])
        colors[pivot] = replacement
    return colors


def _render_grid(shape: tuple[int, int], warm_path: list[tuple[int, int]], cool_path: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    warm_colors = _color_path(warm_path, WARM_COLORS)
    cool_colors = _color_path(cool_path, COOL_COLORS)
    for (r, c), color in zip(warm_path, warm_colors):
        grid[r, c] = color
    for (r, c), color in zip(cool_path, cool_colors):
        grid[r, c] = color
    return grid


def _touching_between(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    return any(nbr in b for cell in a for nbr in _neighbors4(*cell))


def _validate_instance(grid: np.ndarray, expect_touching: bool) -> bool:
    try:
        warm_order, warm_seq = _trace_group(grid, WARM_COLORS)
        cool_order, cool_seq = _trace_group(grid, COOL_COLORS)
        out = main(grid)
    except AssertionError:
        return False

    if len(warm_seq) == len(cool_seq):
        return False
    if out.shape != (max(len(warm_seq), len(cool_seq)), 1):
        return False
    if len(set(warm_seq)) < 2 or len(set(cool_seq)) < 2:
        return False

    warm_set = set(warm_order)
    cool_set = set(cool_order)
    if expect_touching and not _touching_between(warm_set, cool_set):
        return False
    if not expect_touching and _touching_between(warm_set, cool_set) and random.random() < 0.5:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(10, 16)
    width = random.randint(10, 16)
    shape = (height, width)

    warm_longer = random.random() < 0.5
    longer_len = random.randint(8, 15)
    shorter_len = random.randint(4, longer_len - 1)
    if longer_len - shorter_len > 6 and random.random() < 0.7:
        shorter_len = longer_len - random.randint(1, 5)

    warm_len = longer_len if warm_longer else shorter_len
    cool_len = shorter_len if warm_longer else longer_len
    expect_touching = random.random() < 0.7

    warm_path = _build_path(shape, warm_len, occupied=set(), prefer_adjacent_to=None, require_touch=False)
    if warm_path is None:
        return None
    occupied = set(warm_path)
    cool_path = _build_path(shape, cool_len, occupied=occupied, prefer_adjacent_to=occupied, require_touch=expect_touching)
    if cool_path is None:
        return None

    grid = _render_grid(shape, warm_path, cool_path)
    if not _validate_instance(grid, expect_touching):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
