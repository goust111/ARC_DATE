import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
SIGNAL_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_ATTEMPTS = 400
PATH_NODE_COUNT_RANGE = (5, 11)
COARSE_SIZE_RANGE = (4, 10)
MAX_SEGMENTS = 5
MAX_PATH_SEARCH_ATTEMPTS = 120
MAX_SEGMENT_SEARCH_ATTEMPTS = 120


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _path_neighbor_list(path_mask: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    h, w = path_mask.shape
    return [(rr, cc) for rr, cc in _neighbors4(r, c, h, w) if bool(path_mask[rr, cc])]


def _traverse_path(path_mask: np.ndarray) -> list[tuple[int, int]]:
    cells = np.argwhere(path_mask)
    assert len(cells) >= 3, "expected a nontrivial path"

    endpoints: list[tuple[int, int]] = []
    for r, c in map(tuple, cells):
        degree = len(_path_neighbor_list(path_mask, (int(r), int(c))))
        assert degree in (1, 2), "path must be one-pixel wide and non-branching"
        if degree == 1:
            endpoints.append((int(r), int(c)))

    assert len(endpoints) == 2, "path must have exactly two endpoints"
    start = min(endpoints)

    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        order.append(cur)
        next_steps = [nxt for nxt in _path_neighbor_list(path_mask, cur) if nxt != prev]
        if not next_steps:
            break
        assert len(next_steps) == 1, "path traversal must stay unambiguous"
        prev, cur = cur, next_steps[0]

    assert len(order) == int(path_mask.sum()), "path must be connected"
    return order


def _signal_neighbors(
    grid: np.ndarray, path_mask: np.ndarray, cell: tuple[int, int]
) -> list[tuple[int, int, int]]:
    r, c = cell
    h, w = grid.shape
    found: list[tuple[int, int, int]] = []
    for rr, cc in _neighbors4(r, c, h, w):
        value = int(grid[rr, cc])
        if bool(path_mask[rr, cc]) or value in (BLACK, GREY, MAROON):
            continue
        found.append((rr, cc, value))
    return found


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each grey path segment with the traffic-light color beside that segment's start.

    Input structure:
    - The background is black.
    - A single one-pixel-wide 4-connected path is drawn with grey pixels, and some
      interior path cells are recolored maroon.
    - The grey and maroon cells together form one non-branching path with exactly two
      grey endpoints.
    - Each grey segment start, when the whole path is read from the earlier endpoint
      in row-major order, has exactly one adjacent colored traffic-light pixel off the path.

    Rule:
    - Treat grey and maroon cells as the full path and choose the endpoint with smaller
      row, then smaller column, as the global start.
    - Walk along the path in order; every maroon cell ends the current grey segment, and
      the next grey cell starts a new segment.
    - Recolor every grey cell in a segment with the color of the unique traffic-light
      pixel adjacent to that segment's first grey cell.
    - Keep maroon cells unchanged and erase all traffic-light pixels to black.

    Key insight:
    - Maroon cells are separators rather than repaint targets, so the solver must infer
      the path direction first and then read each segment's color from its own starting cell.
    """
    g = np.asarray(grid, dtype=int)
    path_mask = (g == GREY) | (g == MAROON)
    assert bool(path_mask.any()), "expected a grey/maroon path"

    order = _traverse_path(path_mask)
    assert int(g[order[0]]) == GREY and int(g[order[-1]]) == GREY, "markers cannot be endpoints"

    out = g.copy()
    signal_mask = ~((g == BLACK) | (g == GREY) | (g == MAROON))
    out[signal_mask] = BLACK

    used_signals: set[tuple[int, int]] = set()

    start_signals = _signal_neighbors(g, path_mask, order[0])
    assert len(start_signals) == 1, "first segment start needs exactly one signal"
    current_color = int(start_signals[0][2])
    used_signals.add((int(start_signals[0][0]), int(start_signals[0][1])))

    for idx, (r, c) in enumerate(order):
        if int(g[r, c]) == MAROON:
            out[r, c] = MAROON
            if idx + 1 < len(order):
                nr, nc = order[idx + 1]
                assert int(g[nr, nc]) == GREY, "markers must split grey runs"
                next_signals = _signal_neighbors(g, path_mask, (nr, nc))
                assert len(next_signals) == 1, "every segment start needs exactly one signal"
                current_color = int(next_signals[0][2])
                used_signals.add((int(next_signals[0][0]), int(next_signals[0][1])))
        else:
            out[r, c] = current_color

    all_signals = {tuple(map(int, rc)) for rc in np.argwhere(signal_mask)}
    assert used_signals == all_signals, "every colored off-path pixel must be a used signal"
    return out


def _coarse_neighbors(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    return [(rr, cc) for rr, cc in _neighbors4(r, c, h, w)]


def _random_coarse_path(h: int, w: int, length: int) -> list[tuple[int, int]] | None:
    def dfs(path: list[tuple[int, int]], used: set[tuple[int, int]]) -> bool:
        if len(path) == length:
            return True

        cur = path[-1]
        options = [nxt for nxt in _coarse_neighbors(cur, h, w) if nxt not in used]
        random.shuffle(options)

        if len(path) >= 2:
            prev = path[-2]
            dr = cur[0] - prev[0]
            dc = cur[1] - prev[1]
            options.sort(key=lambda cell: (cell[0] - cur[0] != dr or cell[1] - cur[1] != dc, random.random()))

        for nxt in options:
            used.add(nxt)
            path.append(nxt)
            if dfs(path, used):
                return True
            path.pop()
            used.remove(nxt)
        return False

    for _ in range(MAX_PATH_SEARCH_ATTEMPTS):
        start = (random.randrange(h), random.randrange(w))
        path = [start]
        used = {start}
        if dfs(path, used):
            return path
    return None


def _expand_path(nodes: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    order: list[tuple[int, int]] = [(2 * nodes[0][0], 2 * nodes[0][1])]
    for (r0, c0), (r1, c1) in zip(nodes, nodes[1:]):
        rr0, cc0 = order[-1]
        rr1, cc1 = 2 * r1, 2 * c1
        order.append(((rr0 + rr1) // 2, (cc0 + cc1) // 2))
        order.append((rr1, cc1))
    return order, (2 * max(r for r, _ in nodes) + 1, 2 * max(c for _, c in nodes) + 1)


def _signal_slots(
    cell: tuple[int, int], path_set: set[tuple[int, int]], h: int, w: int
) -> list[tuple[int, int]]:
    r, c = cell
    slots: list[tuple[int, int]] = []
    for rr, cc in _neighbors4(r, c, h, w):
        if (rr, cc) in path_set:
            continue
        touching = [(nr, nc) for nr, nc in _neighbors4(rr, cc, h, w) if (nr, nc) in path_set]
        if len(touching) == 1 and touching[0] == cell:
            slots.append((rr, cc))
    return slots


def _choose_segment_layout(
    order: list[tuple[int, int]], h: int, w: int
) -> tuple[list[int], dict[int, tuple[int, int]]] | None:
    path_set = set(order)
    slot_map = {idx: _signal_slots(cell, path_set, h, w) for idx, cell in enumerate(order)}
    if not slot_map[0]:
        return None

    candidates = [idx for idx in range(2, len(order)) if slot_map[idx]]
    max_segments = min(MAX_SEGMENTS, (len(order) + 1) // 2)
    desired_counts = list(range(2, max_segments + 1))
    random.shuffle(desired_counts)

    for target_count in desired_counts:
        for _ in range(MAX_SEGMENT_SEARCH_ATTEMPTS):
            first_slots = slot_map[0][:]
            random.shuffle(first_slots)
            for first_slot in first_slots:
                used_slots = {first_slot}
                starts = [0]
                signals = {0: first_slot}
                result = _extend_segments(candidates, slot_map, target_count - 1, 2, used_slots, starts, signals)
                if result is not None:
                    return result
    return None


def _extend_segments(
    candidates: list[int],
    slot_map: dict[int, list[tuple[int, int]]],
    remaining: int,
    min_start: int,
    used_slots: set[tuple[int, int]],
    starts: list[int],
    signals: dict[int, tuple[int, int]],
) -> tuple[list[int], dict[int, tuple[int, int]]] | None:
    if remaining == 0:
        return starts[:], dict(signals)

    choices = [idx for idx in candidates if idx >= min_start]
    random.shuffle(choices)
    for idx in choices:
        available = [slot for slot in slot_map[idx] if slot not in used_slots]
        random.shuffle(available)
        for slot in available:
            starts.append(idx)
            signals[idx] = slot
            used_slots.add(slot)
            result = _extend_segments(
                candidates,
                slot_map,
                remaining - 1,
                idx + 2,
                used_slots,
                starts,
                signals,
            )
            if result is not None:
                return result
            used_slots.remove(slot)
            del signals[idx]
            starts.pop()
    return None


def _embed_path(local_order: list[tuple[int, int]], local_shape: tuple[int, int]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    lh, lw = local_shape
    final_h = random.randint(lh + 2, min(30, lh + 10))
    final_w = random.randint(lw + 2, min(30, lw + 10))
    top = random.randint(1, final_h - lh - 1)
    left = random.randint(1, final_w - lw - 1)
    shifted = [(r + top, c + left) for r, c in local_order]
    if shifted[-1] < shifted[0]:
        shifted.reverse()
    return shifted, (final_h, final_w)


def _build_input() -> np.ndarray | None:
    coarse_h = random.randint(*COARSE_SIZE_RANGE)
    coarse_w = random.randint(*COARSE_SIZE_RANGE)
    max_nodes = min(PATH_NODE_COUNT_RANGE[1], coarse_h * coarse_w)
    min_nodes = min(PATH_NODE_COUNT_RANGE[0], max_nodes)
    node_count = random.randint(min_nodes, max_nodes)

    coarse_path = _random_coarse_path(coarse_h, coarse_w, node_count)
    if coarse_path is None:
        return None

    local_order, local_shape = _expand_path(coarse_path)
    full_order, (h, w) = _embed_path(local_order, local_shape)
    layout = _choose_segment_layout(full_order, h, w)
    if layout is None:
        return None

    starts, signals = layout
    segment_colors = random.sample(SIGNAL_COLORS, k=len(starts))
    start_to_color = {start_idx: color for start_idx, color in zip(starts, segment_colors)}

    grid = np.full((h, w), BLACK, dtype=int)
    for r, c in full_order:
        grid[r, c] = GREY

    for start_idx in starts[1:]:
        mr, mc = full_order[start_idx - 1]
        grid[mr, mc] = MAROON

    for start_idx, (sr, sc) in signals.items():
        grid[sr, sc] = start_to_color[start_idx]

    _validate_generated_input(grid)
    return grid


def _validate_generated_input(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    path_mask = (g == GREY) | (g == MAROON)
    order = _traverse_path(path_mask)
    assert int(g[order[0]]) == GREY and int(g[order[-1]]) == GREY

    signal_cells = [tuple(map(int, rc)) for rc in np.argwhere(~((g == BLACK) | (g == GREY) | (g == MAROON)))]
    assert len(signal_cells) >= 2

    segment_starts = [0]
    for idx, cell in enumerate(order[:-1]):
        if int(g[cell]) == MAROON:
            segment_starts.append(idx + 1)
    assert len(segment_starts) == len(signal_cells)

    seen_signals: set[tuple[int, int]] = set()
    for start_idx in segment_starts:
        slot_info = _signal_neighbors(g, path_mask, order[start_idx])
        assert len(slot_info) == 1
        seen_signals.add((int(slot_info[0][0]), int(slot_info[0][1])))
    assert seen_signals == set(signal_cells)

    out = main(g)
    assert out.shape == g.shape
    assert not np.any(~((out == BLACK) | (out == GREY) | (out == MAROON)) & (~path_mask))


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-and-signal ARC input")
