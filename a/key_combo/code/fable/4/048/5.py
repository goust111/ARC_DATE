from __future__ import annotations

import random
from collections import Counter

import numpy as np


TEAL = 8
GEN_RETRIES = 200
ALL_COLORS = tuple(range(10))


def _band_starts(length: int, separator_lines: list[int]) -> tuple[list[int], list[int]]:
    points = [-1] + sorted(separator_lines) + [length]
    starts: list[int] = []
    spans: list[int] = []
    for a, b in zip(points, points[1:]):
        gap = b - a - 1
        if gap > 0:
            starts.append(a + 1)
            spans.append(gap)
    return starts, spans


def _find_layout(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    best: tuple[int, list[int], list[int], int] | None = None
    for color in map(int, np.unique(grid)):
        row_lines = [r for r in range(h) if np.all(grid[r, :] == color)]
        col_lines = [c for c in range(w) if np.all(grid[:, c] == color)]
        if not row_lines or not col_lines:
            continue
        row_starts, row_spans = _band_starts(h, row_lines)
        col_starts, col_spans = _band_starts(w, col_lines)
        if len(row_starts) < 2 or len(col_starts) < 2:
            continue
        if len(set(row_spans)) != 1 or len(set(col_spans)) != 1:
            continue
        cell_size = row_spans[0]
        if cell_size != col_spans[0] or cell_size <= 1:
            continue
        n_cells = len(row_starts) * len(col_starts)
        if n_cells < 3:
            continue
        candidate = (n_cells, row_starts, col_starts, cell_size)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None, "failed to detect square cell layout"
    _, row_starts, col_starts, cell_size = best
    return cell_size, row_starts, col_starts


def _rotation_to_canonical(cell: np.ndarray) -> int:
    coords = np.argwhere(cell == TEAL)
    assert coords.shape[0] == 1, "each cell must contain exactly one teal marker"
    r, c = map(int, coords[0])
    last = cell.shape[0] - 1
    mapping = {
        (0, 0): 0,
        (0, last): 1,
        (last, last): 2,
        (last, 0): 3,
    }
    assert (r, c) in mapping, "teal marker must sit in a corner"
    return mapping[(r, c)]


def _majority_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values))
    winner, win_count = counts.most_common(1)[0]
    assert win_count * 2 > len(values), "pixelwise majority must be strict"
    return int(winner)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each rotated damaged panel by registering corner markers and taking a pixelwise majority vote.

    Input structure:
    - A single-color separator lattice divides the canvas into equal square cells.
    - Every cell contains the same underlying pattern in one of the four right-angle rotations.
    - Each copy has exactly one teal corner marker pixel, and that marker is never damaged.
    - A few other pixels in each cell may be corrupted to background or to unrelated colors.

    Rule:
    - Use the teal corner marker to infer each cell's rotation and rotate every copy into a shared orientation.
    - In that aligned coordinate system, recover the true pattern by taking the strict per-pixel majority color across all copies.
    - Rotate the recovered pattern back to each cell's original orientation and replace the full cell with the repaired version.
    - Keep the separator lines unchanged.

    Key insight:
    - Voting must happen after rotation-based registration: corresponding pixels only agree once all damaged copies are aligned by the marker corner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    cell_size, row_starts, col_starts = _find_layout(g)

    rotations: list[int] = []
    aligned_cells: list[np.ndarray] = []
    for r0 in row_starts:
        for c0 in col_starts:
            cell = g[r0 : r0 + cell_size, c0 : c0 + cell_size]
            assert cell.shape == (cell_size, cell_size)
            rot = _rotation_to_canonical(cell)
            rotations.append(rot)
            aligned_cells.append(np.rot90(cell, k=rot))

    stack = np.stack(aligned_cells, axis=0)
    repaired = np.zeros((cell_size, cell_size), dtype=int)
    for r in range(cell_size):
        for c in range(cell_size):
            repaired[r, c] = _majority_color(stack[:, r, c])

    out = g.copy()
    idx = 0
    for r0 in row_starts:
        for c0 in col_starts:
            out[r0 : r0 + cell_size, c0 : c0 + cell_size] = np.rot90(repaired, k=-rotations[idx])
            idx += 1
    return out


def _paint_rectangle(arr: np.ndarray, color: int) -> None:
    h, w = arr.shape
    rh = random.randint(1, max(1, h - 2))
    rw = random.randint(1, max(1, w - 2))
    r0 = random.randint(0, h - rh)
    c0 = random.randint(0, w - rw)
    arr[r0 : r0 + rh, c0 : c0 + rw] = color


def _paint_line(arr: np.ndarray, color: int) -> None:
    h, w = arr.shape
    if random.random() < 0.5:
        r = random.randint(0, h - 1)
        c0 = random.randint(0, w - 2)
        c1 = random.randint(c0 + 1, w - 1)
        arr[r, c0 : c1 + 1] = color
    else:
        c = random.randint(0, w - 1)
        r0 = random.randint(0, h - 2)
        r1 = random.randint(r0 + 1, h - 1)
        arr[r0 : r1 + 1, c] = color


def _paint_l(arr: np.ndarray, color: int) -> None:
    h, w = arr.shape
    r = random.randint(0, h - 2)
    c = random.randint(0, w - 2)
    arm_r = random.randint(2, h - r)
    arm_c = random.randint(2, w - c)
    arr[r : r + arm_r, c] = color
    arr[r, c : c + arm_c] = color


def _make_canonical_pattern(cell_size: int, bg: int, separator: int) -> np.ndarray:
    usable_colors = [c for c in ALL_COLORS if c not in {bg, separator, TEAL}]
    for _ in range(100):
        arr = np.full((cell_size, cell_size), bg, dtype=int)
        n_features = random.randint(3, min(5, len(usable_colors)))
        feature_colors = random.sample(usable_colors, k=n_features)
        painters = [_paint_rectangle, _paint_line, _paint_l]
        for color in feature_colors:
            random.choice(painters)(arr, color)
        arr[0, 0] = TEAL
        non_bg = int(np.sum(arr != bg))
        if non_bg < cell_size + 2:
            continue
        if len(set(map(int, np.unique(arr)))) < 4:
            continue
        if np.sum(arr == TEAL) != 1:
            continue
        return arr
    raise RuntimeError("failed to build a sufficiently varied canonical pattern")


def _assign_damage_positions(
    n_cells: int,
    cell_size: int,
    damage_targets: list[int],
) -> list[list[tuple[int, int]]]:
    positions = [(r, c) for r in range(cell_size) for c in range(cell_size) if (r, c) != (0, 0)]
    random.shuffle(positions)
    capacity = {pos: (n_cells - 1) // 2 for pos in positions}
    assignments: list[list[tuple[int, int]]] = [[] for _ in range(n_cells)]

    for cell_idx, target in enumerate(damage_targets):
        random.shuffle(positions)
        for pos in positions:
            if len(assignments[cell_idx]) >= target:
                break
            if capacity[pos] == 0 or pos in assignments[cell_idx]:
                continue
            assignments[cell_idx].append(pos)
            capacity[pos] -= 1
        if len(assignments[cell_idx]) != target:
            raise RuntimeError("failed to allocate non-degenerate damage positions")
    return assignments


def _damage_color(true_color: int, bg: int, separator: int) -> int:
    pool = [c for c in ALL_COLORS if c not in {true_color, separator, TEAL}]
    if bg in pool and random.random() < 0.5:
        return bg
    return random.choice(pool)


def _gen_attempt() -> np.ndarray | None:
    row_count = random.randint(2, 3)
    col_count = random.randint(2, 3)
    n_cells = row_count * col_count
    if n_cells < 4:
        return None

    cell_size = random.randint(5, 8)
    height = row_count * cell_size + (row_count + 1)
    width = col_count * cell_size + (col_count + 1)
    if max(height, width) > 30:
        return None

    separator = random.choice(list(ALL_COLORS))
    bg_candidates = [c for c in ALL_COLORS if c not in {separator, TEAL}]
    bg = random.choice(bg_candidates)
    canonical = _make_canonical_pattern(cell_size, bg, separator)

    rel_rots = list(range(4))
    while len(rel_rots) < n_cells:
        rel_rots.append(random.randint(0, 3))
    random.shuffle(rel_rots)

    max_damage = min(5, cell_size * cell_size - 2)
    bait_idx = random.randrange(n_cells)
    damage_targets = [random.randint(2, max_damage) for _ in range(n_cells)]
    damage_targets[bait_idx] = 1
    assignments = _assign_damage_positions(n_cells, cell_size, damage_targets)

    cells: list[np.ndarray] = []
    for idx in range(n_cells):
        damaged = canonical.copy()
        for r, c in assignments[idx]:
            damaged[r, c] = _damage_color(int(canonical[r, c]), bg, separator)
        if np.array_equal(damaged, canonical):
            return None
        cells.append(np.rot90(damaged, k=-rel_rots[idx]))

    grid = np.full((height, width), separator, dtype=int)
    idx = 0
    for rr in range(row_count):
        for cc in range(col_count):
            r0 = 1 + rr * (cell_size + 1)
            c0 = 1 + cc * (cell_size + 1)
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = cells[idx]
            idx += 1

    repaired = main(grid)
    for rr in range(row_count):
        for cc in range(col_count):
            r0 = 1 + rr * (cell_size + 1)
            c0 = 1 + cc * (cell_size + 1)
            cell_in = grid[r0 : r0 + cell_size, c0 : c0 + cell_size]
            cell_out = repaired[r0 : r0 + cell_size, c0 : c0 + cell_size]
            if np.array_equal(cell_in, cell_out):
                return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid repaired-majority ARC input")
