from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RING_OFFSETS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
]

GRID_SIDE_RANGE = (11, 16)
MAX_RADIUS = 4
MAX_GROW_PAIRS = 5
MAX_GENERATION_ATTEMPTS = 400
MAX_DELETE_ATTEMPTS = 250


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _mirror(center: tuple[int, int], cell: tuple[int, int]) -> tuple[int, int]:
    pr, pc = center
    r, c = cell
    return 2 * pr - r, 2 * pc - c


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) == int(mask.sum())


def _unique_anchor(grid: np.ndarray) -> tuple[int, tuple[int, int]]:
    counts = Counter(int(v) for v in grid.flat if int(v) != BLACK)
    unique_colors = [color for color, count in counts.items() if count == 1]
    assert len(unique_colors) == 1, "expected exactly one non-black color with unique frequency"
    color = int(unique_colors[0])
    locs = np.argwhere(grid == color)
    assert len(locs) == 1, "unique-frequency color must appear exactly once"
    return color, tuple(map(int, locs[0]))


def main(grid: np.ndarray) -> np.ndarray:
    """Use the uniquely frequent red anchor to mark the missing rotational partners of the blue object.

    Input structure:
    - The background is black.
    - A single red cell is the only non-black color whose frequency is unique, and it marks the 180-degree rotation center.
    - A 4-connected blue object occupies 8-15 cells and is almost point-symmetric around that red center, except that 1-3 blue partners are missing.

    Rule:
    - Find the uniquely frequent non-black color and use its position as center P.
    - For every blue cell, compute its 180-degree rotated position around P.
    - If that rotated position lies inside the grid and is black, fill it green.
    - Keep the original blue object and the red anchor unchanged.

    Key insight:
    - The red cell is not part of the shape to repair; frequency uniqueness identifies it as the symmetry anchor, and only then can the missing mirrored cells be found.
    """
    g = np.asarray(grid, dtype=int)
    anchor_color, center = _unique_anchor(g)
    assert anchor_color == RED, "expected the unique anchor color to be red"

    out = g.copy()
    for r, c in np.argwhere(g == BLUE):
        mr, mc = _mirror(center, (int(r), int(c)))
        if _in_bounds(g.shape, mr, mc) and int(g[mr, mc]) == BLACK:
            out[mr, mc] = GREEN
    return out


def _canonical_offset(offset: tuple[int, int]) -> tuple[int, int]:
    dr, dc = offset
    mirror = (-dr, -dc)
    return min((dr, dc), mirror)


def _frontier_offsets(full_offsets: set[tuple[int, int]]) -> list[tuple[int, int]]:
    frontier: set[tuple[int, int]] = set()
    for dr, dc in full_offsets:
        for sdr, sdc in DIRS4:
            nxt = (dr + sdr, dc + sdc)
            if nxt == (0, 0):
                continue
            if nxt in full_offsets or (-nxt[0], -nxt[1]) in full_offsets:
                continue
            if max(abs(nxt[0]), abs(nxt[1])) > MAX_RADIUS:
                continue
            if not any((nxt[0] + adr, nxt[1] + adc) in full_offsets for adr, adc in DIRS4):
                continue
            frontier.add(_canonical_offset(nxt))
    return list(frontier)


def _grow_symmetric_offsets(extra_pairs: int) -> set[tuple[int, int]] | None:
    full_offsets = set(RING_OFFSETS)
    added_pairs = 0
    while added_pairs < extra_pairs:
        candidates = _frontier_offsets(full_offsets)
        if not candidates:
            return None
        choice = random.choice(candidates)
        full_offsets.add(choice)
        full_offsets.add((-choice[0], -choice[1]))
        added_pairs += 1
    return full_offsets


def _offsets_to_cells(center: tuple[int, int], offsets: set[tuple[int, int]]) -> set[tuple[int, int]]:
    pr, pc = center
    return {(pr + dr, pc + dc) for dr, dc in offsets}


def _pair_groups(cells: set[tuple[int, int]], center: tuple[int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    groups = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for cell in cells:
        mate = _mirror(center, cell)
        pair = tuple(sorted((cell, mate)))
        if pair in seen:
            continue
        seen.add(pair)
        groups.append((pair[0], pair[1]))
    return groups


def _choose_missing_cells(
    full_cells: set[tuple[int, int]],
    center: tuple[int, int],
    missing_count: int,
    shape: tuple[int, int],
) -> set[tuple[int, int]] | None:
    pairs = _pair_groups(full_cells, center)
    if missing_count > len(pairs):
        return None

    for _ in range(MAX_DELETE_ATTEMPTS):
        chosen_pairs = random.sample(pairs, missing_count)
        deleted = {random.choice(pair) for pair in chosen_pairs}
        if len(deleted) != missing_count:
            continue
        remaining = full_cells - deleted
        mask = np.zeros(shape, dtype=bool)
        for r, c in remaining:
            mask[r, c] = True
        if not _connected(mask):
            continue
        return deleted
    return None


def _build_candidate() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    if height < 2 * MAX_RADIUS + 1 or width < 2 * MAX_RADIUS + 1:
        return None

    center = (
        random.randint(MAX_RADIUS, height - MAX_RADIUS - 1),
        random.randint(MAX_RADIUS, width - MAX_RADIUS - 1),
    )

    missing_count = random.randint(1, 3)
    min_pairs = 2 if missing_count == 3 else 1
    extra_pairs = random.randint(min_pairs, MAX_GROW_PAIRS if missing_count == 3 else 4)

    offsets = _grow_symmetric_offsets(extra_pairs)
    if offsets is None:
        return None

    full_cells = _offsets_to_cells(center, offsets)
    deleted = _choose_missing_cells(full_cells, center, missing_count, (height, width))
    if deleted is None:
        return None

    blue_cells = full_cells - deleted
    if not (8 <= len(blue_cells) <= 15):
        return None

    grid = np.zeros((height, width), dtype=int)
    for r, c in blue_cells:
        grid[r, c] = BLUE
    grid[center] = RED

    if not _validate_example(grid):
        return None
    return grid


def _validate_example(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        if g.ndim != 2 or max(g.shape) > 30:
            return False
        if not set(map(int, np.unique(g))).issubset({BLACK, BLUE, RED}):
            return False

        anchor_color, center = _unique_anchor(g)
        if anchor_color != RED:
            return False
        if int(np.count_nonzero(g == RED)) != 1:
            return False
        blue_mask = g == BLUE
        blue_count = int(blue_mask.sum())
        if not (8 <= blue_count <= 15):
            return False
        if not _connected(blue_mask):
            return False

        out = main(g)
        if out.shape != g.shape:
            return False
        if not np.array_equal(out[g == BLUE], np.full(blue_count, BLUE, dtype=int)):
            return False
        if int(out[center]) != RED:
            return False

        expected_green: set[tuple[int, int]] = set()
        paired_blue_blue = 0
        for r, c in np.argwhere(g == BLUE):
            cell = (int(r), int(c))
            mate = _mirror(center, cell)
            if not _in_bounds(g.shape, *mate):
                return False
            if int(g[mate]) == BLUE:
                paired_blue_blue += 1
            elif int(g[mate]) == BLACK:
                expected_green.add(mate)
            else:
                return False

        actual_green = {tuple(map(int, rc)) for rc in np.argwhere(out == GREEN)}
        if not (1 <= len(actual_green) <= 3):
            return False
        if actual_green != expected_green:
            return False
        if paired_blue_blue == 0:
            return False

        recovered = np.zeros_like(g, dtype=bool)
        recovered[g == BLUE] = True
        recovered[out == GREEN] = True
        for r, c in np.argwhere(recovered):
            mate = _mirror(center, (int(r), int(c)))
            if not _in_bounds(g.shape, *mate) or not recovered[mate]:
                return False
        if int(recovered.sum()) != blue_count + len(actual_green):
            return False
        if not _connected(recovered):
            return False
        if not np.array_equal(out[(g != BLACK)], g[(g != BLACK)]):
            return False
        if np.any(out[(g == BLACK) & ~(out == GREEN)] != BLACK):
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
