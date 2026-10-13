from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


BG = 0
WALL = 5
SHORT_RECOLOR = 4
LONG_RECOLOR = 7
THRESHOLD = 5
ENDPOINT_COLORS = (1, 2, 3, 6, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


@dataclass(frozen=True)
class PathInfo:
    color: int
    start: tuple[int, int]
    end: tuple[int, int]
    cells: tuple[tuple[int, int], ...]
    length: int


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if _in_bounds(shape, rr, cc):
            yield rr, cc


def _extract_pairs(grid: np.ndarray) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    pairs = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, WALL)):
        coords = np.argwhere(grid == color)
        assert coords.shape == (2, 2), f"color {color} must appear exactly twice"
        a = tuple(int(x) for x in coords[0])
        b = tuple(int(x) for x in coords[1])
        pairs.append((int(color), a, b))
    assert 2 <= len(pairs) <= 3, "expected 2-3 endpoint pairs"
    return pairs


def _shortest_black_path(
    grid: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    shape = grid.shape
    start_sources = [p for p in _neighbors4(shape, *start) if int(grid[p]) == BG]
    end_targets = [p for p in _neighbors4(shape, *end) if int(grid[p]) == BG]
    assert start_sources and end_targets, "each endpoint needs an adjacent black corridor cell"

    dist = np.full(shape, -1, dtype=int)
    count = np.zeros(shape, dtype=int)
    parent_r = np.full(shape, -1, dtype=int)
    parent_c = np.full(shape, -1, dtype=int)
    q: deque[tuple[int, int]] = deque()

    for r, c in start_sources:
        if dist[r, c] == -1:
            dist[r, c] = 1
            count[r, c] = 1
            q.append((r, c))
        else:
            count[r, c] = min(2, count[r, c] + 1)

    while q:
        r, c = q.popleft()
        nd = int(dist[r, c]) + 1
        for rr, cc in _neighbors4(shape, r, c):
            if int(grid[rr, cc]) != BG:
                continue
            if dist[rr, cc] == -1:
                dist[rr, cc] = nd
                count[rr, cc] = count[r, c]
                parent_r[rr, cc] = r
                parent_c[rr, cc] = c
                q.append((rr, cc))
            elif dist[rr, cc] == nd:
                count[rr, cc] = min(2, count[rr, cc] + count[r, c])

    target_distances = [int(dist[r, c]) for r, c in end_targets if dist[r, c] >= 0]
    assert target_distances, "pair must be connected through black cells"
    best = min(target_distances)
    total_paths = 0
    best_targets = []
    for r, c in end_targets:
        if int(dist[r, c]) == best:
            total_paths = min(2, total_paths + int(count[r, c]))
            best_targets.append((r, c))
    assert total_paths == 1, "shortest path must be unique"

    terminal = next((p for p in best_targets if int(count[p]) == 1), None)
    assert terminal is not None
    rev = []
    cur = terminal
    while cur != (-1, -1):
        rev.append(cur)
        pr = int(parent_r[cur])
        pc = int(parent_c[cur])
        cur = (pr, pc) if pr >= 0 else (-1, -1)
    cells = tuple(reversed(rev))
    assert len(cells) == best
    return cells, best, total_paths


def _solve_paths(grid: np.ndarray) -> list[PathInfo]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    infos = []
    occupied = set()
    for color, start, end in _extract_pairs(g):
        cells, length, _ = _shortest_black_path(g, start, end)
        overlap = occupied & set(cells)
        assert not overlap, "different endpoint pairs must not share path cells"
        occupied |= set(cells)
        infos.append(PathInfo(color=color, start=start, end=end, cells=cells, length=length))
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each endpoint pair and its unique maze path according to whether the path is short or long.

    Input structure:
    - A black background contains 2-3 same-color endpoint pairs, each endpoint being a singleton pixel.
    - Grey wall pixels form small maze-like enclosures; the remaining black pixels are the only passable corridor cells.
    - For each color, the two endpoints are connected by a unique 4-connected shortest route through black cells, and different pairs do not share route pixels.

    Rule:
    - Pair the singleton endpoints by their original color.
    - For each pair, find the unique 4-connected shortest path through black corridor cells only.
    - Count the number of black path cells, excluding the two endpoints themselves.
    - If that count is at most 5, recolor both endpoints and every path cell yellow; otherwise recolor them orange.
    - Keep all grey walls and every black cell that is not on one of those shortest paths unchanged.

    Key insight:
    - The endpoint color is only a pairing label; the visible output color is determined entirely by the intermediate shortest-path length threshold.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for info in _solve_paths(g):
        paint = SHORT_RECOLOR if info.length <= THRESHOLD else LONG_RECOLOR
        out[info.start] = paint
        out[info.end] = paint
        for cell in info.cells:
            out[cell] = paint
    return out


def _occupied_neighbor_count(cells: set[tuple[int, int]], coord: tuple[int, int]) -> int:
    r, c = coord
    return sum((rr, cc) in cells for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))


def _candidate_path_step(
    used: set[tuple[int, int]],
    current: tuple[int, int],
    nxt: tuple[int, int],
    shape: tuple[int, int],
) -> bool:
    if not _in_bounds(shape, *nxt) or nxt in used:
        return False
    if _occupied_neighbor_count(used, nxt) != 1:
        return False
    return current in set(_neighbors4(shape, *nxt))


def _remaining_reachable(
    used: set[tuple[int, int]], start: tuple[int, int], shape: tuple[int, int]
) -> int:
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(shape, r, c):
            if (rr, cc) in used or (rr, cc) in seen:
                continue
            seen.add((rr, cc))
            q.append((rr, cc))
    return len(seen)


def _build_main_path(
    inner_h: int, inner_w: int, node_count: int, rng: np.random.Generator
) -> list[tuple[int, int]] | None:
    shape = (inner_h, inner_w)

    def dfs(path: list[tuple[int, int]], used: set[tuple[int, int]]) -> bool:
        if len(path) == node_count:
            return True
        current = path[-1]
        candidates = []
        for rr, cc in _neighbors4(shape, *current):
            if not _candidate_path_step(used, current, (rr, cc), shape):
                continue
            spare = _remaining_reachable(used | {(rr, cc)}, (rr, cc), shape)
            if spare >= node_count - len(path):
                candidates.append((rr, cc))
        if not candidates:
            return False
        rng.shuffle(candidates)
        for nxt in candidates:
            path.append(nxt)
            used.add(nxt)
            if dfs(path, used):
                return True
            used.remove(nxt)
            path.pop()
        return False

    starts = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    rng.shuffle(starts)
    for start in starts:
        path = [start]
        used = {start}
        if dfs(path, used):
            return path
    return None


def _grow_branch(
    occupied: set[tuple[int, int]],
    anchor: tuple[int, int],
    branch_len: int,
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    branch = []
    prev = anchor
    local_used = set(occupied)
    for remaining in range(branch_len, 0, -1):
        candidates = []
        for nxt in _neighbors4(shape, *prev):
            if nxt in local_used:
                continue
            if _occupied_neighbor_count(local_used, nxt) != 1:
                continue
            if prev not in set(_neighbors4(shape, *nxt)):
                continue
            spare = _remaining_reachable(local_used | {nxt}, nxt, shape)
            if spare >= remaining - 1:
                candidates.append(nxt)
        if not candidates:
            return None
        rng.shuffle(candidates)
        nxt = candidates[0]
        branch.append(nxt)
        local_used.add(nxt)
        prev = nxt
    return branch


def _build_chamber(
    length: int, endpoint_color: int, rng: np.random.Generator
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    node_count = length + 2
    for _ in range(200):
        inner_h = int(rng.integers(max(3, int(np.ceil(np.sqrt(node_count))) - 1), 8))
        inner_w = int(rng.integers(max(3, int(np.ceil(node_count / max(1, inner_h))) - 1), 9))
        if inner_h * inner_w < node_count + 2:
            continue
        main_path = _build_main_path(inner_h, inner_w, node_count, rng)
        if main_path is None:
            continue
        occupied = set(main_path)
        branch_cells = []
        branch_count = int(rng.choice([0, 1, 1, 2]))
        anchors = main_path[1:-1].copy()
        rng.shuffle(anchors)
        for anchor in anchors:
            if branch_count == 0:
                break
            branch_len = int(rng.integers(1, 4))
            branch = _grow_branch(occupied, anchor, branch_len, (inner_h, inner_w), rng)
            if branch is None:
                continue
            occupied |= set(branch)
            branch_cells.extend(branch)
            branch_count -= 1

        chamber = np.full((inner_h + 2, inner_w + 2), WALL, dtype=int)
        for r, c in main_path[1:-1]:
            chamber[r + 1, c + 1] = BG
        for r, c in branch_cells:
            chamber[r + 1, c + 1] = BG
        sr, sc = main_path[0]
        er, ec = main_path[-1]
        chamber[sr + 1, sc + 1] = endpoint_color
        chamber[er + 1, ec + 1] = endpoint_color
        return chamber, {(r + 1, c + 1) for r, c in branch_cells}
    raise RuntimeError("failed to build chamber")


def _slot_layout(n_pairs: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    if n_pairs == 2:
        return rng.choice(
            [
                [(0, 0), (0, 1)],
                [(0, 0), (1, 0)],
                [(0, 0), (1, 0)],
                [(0, 0), (0, 1)],
            ]
        ).tolist()
    layouts = [
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
    ]
    return layouts[int(rng.integers(len(layouts)))]


def _place_chambers(
    chambers: list[np.ndarray], rng: np.random.Generator
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[tuple[int, int]]]:
    slots = _slot_layout(len(chambers), rng)
    rows = max(slot[0] for slot in slots) + 1
    cols = max(slot[1] for slot in slots) + 1
    row_heights = [0] * rows
    col_widths = [0] * cols
    for chamber, (sr, sc) in zip(chambers, slots):
        row_heights[sr] = max(row_heights[sr], chamber.shape[0])
        col_widths[sc] = max(col_widths[sc], chamber.shape[1])

    margin_r = int(rng.integers(1, 3))
    margin_c = int(rng.integers(1, 3))
    gap_r = int(rng.integers(2, 5))
    gap_c = int(rng.integers(2, 5))
    H = 2 * margin_r + sum(row_heights) + gap_r * (rows - 1)
    W = 2 * margin_c + sum(col_widths) + gap_c * (cols - 1)
    grid = np.full((H, W), BG, dtype=int)

    row_starts = [margin_r]
    for h in row_heights[:-1]:
        row_starts.append(row_starts[-1] + h + gap_r)
    col_starts = [margin_c]
    for w in col_widths[:-1]:
        col_starts.append(col_starts[-1] + w + gap_c)

    boxes = []
    centers = []
    for chamber, (sr, sc) in zip(chambers, slots):
        slot_h = row_heights[sr]
        slot_w = col_widths[sc]
        extra_r = slot_h - chamber.shape[0]
        extra_c = slot_w - chamber.shape[1]
        r0 = row_starts[sr] + int(rng.integers(extra_r + 1))
        c0 = col_starts[sc] + int(rng.integers(extra_c + 1))
        grid[r0 : r0 + chamber.shape[0], c0 : c0 + chamber.shape[1]] = chamber
        boxes.append((r0, c0, r0 + chamber.shape[0] - 1, c0 + chamber.shape[1] - 1))
        centers.append(((r0 + r0 + chamber.shape[0] - 1) // 2, (c0 + c0 + chamber.shape[1] - 1) // 2))
    return grid, boxes, slots


def _draw_line(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    if a[0] == b[0]:
        r = a[0]
        c0, c1 = sorted((a[1], b[1]))
        grid[r, c0 : c1 + 1] = WALL
        return
    assert a[1] == b[1]
    c = a[1]
    r0, r1 = sorted((a[0], b[0]))
    grid[r0 : r1 + 1, c] = WALL


def _connect_boxes(
    grid: np.ndarray, boxes: list[tuple[int, int, int, int]], slots: list[tuple[int, int]]
) -> None:
    for idx in range(1, len(boxes)):
        prev_box = boxes[idx - 1]
        box = boxes[idx]
        pr0, pc0, pr1, pc1 = prev_box
        r0, c0, r1, c1 = box
        prc = (pr0 + pr1) // 2
        pcc = (pc0 + pc1) // 2
        rc = (r0 + r1) // 2
        cc = (c0 + c1) // 2

        if abs(cc - pcc) >= abs(rc - prc):
            start = (prc, pc1 if pcc < cc else pc0)
            turn = (prc, c0 if pcc < cc else c1)
            end = (rc, c0 if pcc < cc else c1)
        else:
            start = (pr1 if prc < rc else pr0, pcc)
            turn = (r0 if prc < rc else r1, pcc)
            end = (r0 if prc < rc else r1, cc)
        _draw_line(grid, start, turn)
        _draw_line(grid, turn, end)


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        infos = _solve_paths(grid)
    except Exception:
        return False
    lengths = [info.length for info in infos]
    if not any(length <= THRESHOLD for length in lengths):
        return False
    if not any(length > THRESHOLD for length in lengths):
        return False
    if np.count_nonzero(grid == BG) <= np.count_nonzero(grid == WALL):
        return False

    path_sets = [set(info.cells) for info in infos]
    for i in range(len(path_sets)):
        for j in range(i + 1, len(path_sets)):
            if path_sets[i] & path_sets[j]:
                return False

    out = main(grid)
    yellow = sum(info.length <= THRESHOLD for info in infos)
    orange = len(infos) - yellow
    if yellow and not np.any(out == SHORT_RECOLOR):
        return False
    if orange and not np.any(out == LONG_RECOLOR):
        return False

    for info in infos:
        paint = SHORT_RECOLOR if info.length <= THRESHOLD else LONG_RECOLOR
        if int(out[info.start]) != paint or int(out[info.end]) != paint:
            return False
        for cell in info.cells:
            if int(out[cell]) != paint:
                return False
    return True


def _generate_lengths(n_pairs: int, rng: np.random.Generator) -> list[int]:
    lengths = [int(rng.integers(1, 6)), int(rng.integers(6, 13))]
    while len(lengths) < n_pairs:
        if rng.random() < 0.5:
            lengths.append(int(rng.integers(1, 6)))
        else:
            lengths.append(int(rng.integers(6, 13)))
    rng.shuffle(lengths)
    return lengths


def _generate_candidate(rng: np.random.Generator) -> np.ndarray:
    n_pairs = 2 if rng.random() < 0.58 else 3
    colors = list(rng.choice(np.array(ENDPOINT_COLORS, dtype=int), size=n_pairs, replace=False))
    lengths = _generate_lengths(n_pairs, rng)

    chambers = []
    for color, length in zip(colors, lengths):
        chamber, _branch_cells = _build_chamber(length, int(color), rng)
        chambers.append(chamber)

    grid, boxes, slots = _place_chambers(chambers, rng)
    _connect_boxes(grid, boxes, slots)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = np.random.default_rng()
        grid = _generate_candidate(rng)
        if max(grid.shape) > 30:
            continue
        if _semantic_ok(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
