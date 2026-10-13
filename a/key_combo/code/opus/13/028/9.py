from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5

GEN_RETRIES = 400
BLOCK_SIDE_OPTIONS = (4, 5, 6)
TOTAL_MARKERS_RANGE = (6, 10)
MAX_MARKERS_PER_BLOCK = 3


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _block_neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < 3 and 0 <= nc < 3:
            out.append((nr, nc))
    return out


def _is_connected_block_set(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in _block_neighbors(cur):
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _enumerate_reachable_sets() -> list[tuple[tuple[int, int], ...]]:
    cells = [(r, c) for r in range(3) for c in range(3)]
    valid = []
    for size in range(3, 7):
        for subset in combinations(cells, size):
            subset_set = set(subset)
            if (0, 0) not in subset_set:
                continue
            if not _is_connected_block_set(subset_set):
                continue
            valid.append(tuple(sorted(subset)))
    return valid


REACHABLE_SETS = _enumerate_reachable_sets()


def _reachable_pixels(grid: np.ndarray) -> np.ndarray:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    if int(grid[0, 0]) == GREY:
        return seen

    q: deque[tuple[int, int]] = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != GREY:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _coarse_stats(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, grid.shape)
    assert h % 3 == 0 and w % 3 == 0
    block_h = h // 3
    block_w = w // 3

    reached = _reachable_pixels(grid)
    has_marker = np.zeros((3, 3), dtype=bool)
    reachable = np.zeros((3, 3), dtype=bool)
    for br in range(3):
        for bc in range(3):
            rs = slice(br * block_h, (br + 1) * block_h)
            cs = slice(bc * block_w, (bc + 1) * block_w)
            block = grid[rs, cs]
            has_marker[br, bc] = np.any(block == RED)
            reachable[br, bc] = np.any(reached[rs, cs])
    return has_marker, reachable


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the grid into a 3x3 map that keeps only blocks containing red markers and reachable from the top-left corner.

    Input structure:
    - The input is a mostly black grid whose height and width are both divisible by 3.
    - Red pixels are sparse markers, and grey pixels form 1-cell-thick wall segments.
    - The grid is interpreted as a 3x3 array of equal rectangular coarse blocks, with the top-left pixel left open as the flood-fill seed.

    Rule:
    - Split the grid into 3x3 equal blocks.
    - For each block, record whether it contains at least one red marker.
    - Flood-fill from the top-left pixel using 4-connectivity across every non-grey cell, and record whether each block contains at least one reachable pixel.
    - Output a 3x3 grid with green in blocks where both conditions are true, and black elsewhere.

    Key insight:
    - Marker presence alone is insufficient: a coarse block is selected only if a red marker survives the independent reachability test imposed by the grey walls.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert g.shape[0] % 3 == 0 and g.shape[1] % 3 == 0

    has_marker, reachable = _coarse_stats(g)
    out = np.full((3, 3), BLACK, dtype=int)
    out[has_marker & reachable] = GREEN
    return out


def _sample_marked_blocks(reachable_blocks: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    all_blocks = {(r, c) for r in range(3) for c in range(3)}
    unreachable_blocks = all_blocks - reachable_blocks

    reachable_unmarked_n = random.randint(1, min(2, len(reachable_blocks) - 2))
    reachable_marked = set(random.sample(sorted(reachable_blocks), len(reachable_blocks) - reachable_unmarked_n))

    unreachable_marked_n = random.randint(2, min(len(unreachable_blocks), 4))
    unreachable_marked = set(random.sample(sorted(unreachable_blocks), unreachable_marked_n))

    remaining = sorted(all_blocks - reachable_marked - unreachable_marked)
    extra_marked_n = random.randint(0, min(2, len(remaining)))
    extra_marked = set(random.sample(remaining, extra_marked_n)) if extra_marked_n else set()

    occupied = sorted(reachable_marked | unreachable_marked | extra_marked)
    total_markers = random.randint(max(TOTAL_MARKERS_RANGE[0], len(occupied)), TOTAL_MARKERS_RANGE[1])

    counts = {block: 1 for block in occupied}
    remaining_markers = total_markers - len(occupied)
    candidates = [block for block in occupied if counts[block] < MAX_MARKERS_PER_BLOCK]
    while remaining_markers > 0 and candidates:
        block = random.choice(candidates)
        counts[block] += 1
        remaining_markers -= 1
        candidates = [b for b in occupied if counts[b] < MAX_MARKERS_PER_BLOCK]
    return counts


def _paint_boundary_walls(grid: np.ndarray, reachable_blocks: set[tuple[int, int]], block_h: int, block_w: int) -> None:
    for br in range(3):
        for bc in range(2):
            left = (br, bc) in reachable_blocks
            right = (br, bc + 1) in reachable_blocks
            if left != right:
                c = (bc + 1) * block_w
                grid[br * block_h : (br + 1) * block_h, c] = GREY

    for br in range(2):
        for bc in range(3):
            top = (br, bc) in reachable_blocks
            bottom = (br + 1, bc) in reachable_blocks
            if top != bottom:
                r = (br + 1) * block_h
                grid[r, bc * block_w : (bc + 1) * block_w] = GREY


def _add_wall_stubs(grid: np.ndarray, reachable_blocks: set[tuple[int, int]], block_h: int, block_w: int) -> None:
    all_blocks = {(r, c) for r in range(3) for c in range(3)}
    unreachable_blocks = list(sorted(all_blocks - reachable_blocks))
    if not unreachable_blocks:
        return

    stub_count = random.randint(0, 4)
    h, w = grid.shape
    for _ in range(stub_count):
        br, bc = random.choice(unreachable_blocks)
        r0, r1 = br * block_h, (br + 1) * block_h
        c0, c1 = bc * block_w, (bc + 1) * block_w

        if random.random() < 0.5:
            c = random.randint(c0, c1 - 1)
            max_len = min(3, r1 - r0 - 1)
            if max_len <= 0:
                continue
            length = random.randint(1, max_len)
            if br > 0 and grid[r0, c] == GREY:
                start = r0
                end = min(h, r0 + length)
            elif br < 2 and grid[r1 - 1, c] == GREY:
                start = max(r0, r1 - length)
                end = r1
            else:
                rr = random.randint(r0, r1 - length)
                start = rr
                end = rr + length
            grid[start:end, c] = GREY
        else:
            r = random.randint(r0, r1 - 1)
            max_len = min(3, c1 - c0 - 1)
            if max_len <= 0:
                continue
            length = random.randint(1, max_len)
            if bc > 0 and grid[r, c0] == GREY:
                start = c0
                end = min(w, c0 + length)
            elif bc < 2 and grid[r, c1 - 1] == GREY:
                start = max(c0, c1 - length)
                end = c1
            else:
                cc = random.randint(c0, c1 - length)
                start = cc
                end = cc + length
            grid[r, start:end] = GREY


def _place_markers(grid: np.ndarray, marker_counts: dict[tuple[int, int], int], block_h: int, block_w: int) -> bool:
    h, w = grid.shape
    forbidden = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    for (br, bc), count in marker_counts.items():
        r0, r1 = br * block_h, (br + 1) * block_h
        c0, c1 = bc * block_w, (bc + 1) * block_w
        candidates = [
            (r, c)
            for r in range(r0, r1)
            for c in range(c0, c1)
            if int(grid[r, c]) == BLACK and (r, c) not in forbidden
        ]
        if len(candidates) < count:
            return False
        for r, c in random.sample(candidates, count):
            grid[r, c] = RED
    return True


def _gen_attempt() -> np.ndarray | None:
    block_h = random.choice(BLOCK_SIDE_OPTIONS)
    block_w = random.choice(BLOCK_SIDE_OPTIONS)
    h, w = 3 * block_h, 3 * block_w

    reachable_blocks = set(random.choice(REACHABLE_SETS))
    marker_counts = _sample_marked_blocks(reachable_blocks)

    grid = np.full((h, w), BLACK, dtype=int)
    _paint_boundary_walls(grid, reachable_blocks, block_h, block_w)
    if random.random() < 0.65:
        _add_wall_stubs(grid, reachable_blocks, block_h, block_w)

    if int(grid[0, 0]) != BLACK:
        return None
    if not _place_markers(grid, marker_counts, block_h, block_w):
        return None

    has_marker, reachable = _coarse_stats(grid)
    reachable_marked = int(np.sum(has_marker & reachable))
    unreachable_marked = int(np.sum(has_marker & ~reachable))
    reachable_unmarked = int(np.sum(~has_marker & reachable))

    if reachable_marked < 2 or unreachable_marked < 2 or reachable_unmarked < 1:
        return None
    if np.count_nonzero(grid == RED) < TOTAL_MARKERS_RANGE[0]:
        return None
    if np.count_nonzero(grid == GREY) == 0:
        return None

    out = main(grid)
    if np.count_nonzero(out == GREEN) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
