from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_edge(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _enclosed_regions(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    return [comp for comp in _components(g != GREY) if not _touches_edge(comp, h, w)]


def _color_stats(grid: np.ndarray, cells: list[tuple[int, int]]) -> tuple[Counter[int], dict[int, tuple[int, int]]]:
    counts: Counter[int] = Counter()
    first_pos: dict[int, tuple[int, int]] = {}

    for r, c in sorted(cells):
        color = int(grid[r, c])
        if color in (BLACK, GREY):
            continue
        counts[color] += 1
        if color not in first_pos:
            first_pos[color] = (r, c)
    return counts, first_pos


def _winning_color(grid: np.ndarray, cells: list[tuple[int, int]]) -> int | None:
    counts, first_pos = _color_stats(grid, cells)
    if not counts:
        return None
    best = max(counts.values())
    tied = [color for color, count in counts.items() if count == best]
    return min(tied, key=lambda color: first_pos[color])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-walled enclosed region with the winning marker color and
    recolor all losing markers in that region.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for cells in _enclosed_regions(g):
        winner = _winning_color(g, cells)
        if winner is None:
            continue
        coords = np.asarray(cells, dtype=int)
        out[coords[:, 0], coords[:, 1]] = int(winner)

    out[g == GREY] = GREY
    return out


def _random_partition(rows: int, cols: int, n_regions: int) -> np.ndarray:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    labels = np.full((rows, cols), -1, dtype=int)
    for label, cell in enumerate(random.sample(cells, n_regions)):
        labels[cell] = label

    unassigned = {cell for cell in cells if int(labels[cell]) == -1}
    while unassigned:
        frontier = []
        for r, c in unassigned:
            choices = {
                int(labels[nr, nc])
                for nr, nc in _neighbors4(r, c)
                if 0 <= nr < rows and 0 <= nc < cols and int(labels[nr, nc]) != -1
            }
            if choices:
                frontier.append(((r, c), tuple(choices)))
        assert frontier, "partition growth must always have a frontier"
        cell, choices = random.choice(frontier)
        labels[cell] = random.choice(choices)
        unassigned.remove(cell)

    return labels


def _axis_starts(sizes: list[int]) -> tuple[list[int], int]:
    starts = []
    pos = 1
    for size in sizes:
        starts.append(pos)
        pos += size + 1
    return starts, pos


def _build_room_grid(labels: np.ndarray, row_sizes: list[int], col_sizes: list[int]) -> np.ndarray:
    rows, cols = labels.shape
    row_starts, height = _axis_starts(row_sizes)
    col_starts, width = _axis_starts(col_sizes)
    grid = np.full((height, width), GREY, dtype=int)

    for r in range(rows):
        r0 = row_starts[r]
        r1 = r0 + row_sizes[r]
        for c in range(cols):
            c0 = col_starts[c]
            c1 = c0 + col_sizes[c]
            grid[r0:r1, c0:c1] = BLACK

            if c + 1 < cols and int(labels[r, c]) == int(labels[r, c + 1]):
                grid[r0:r1, c1] = BLACK
            if r + 1 < rows and int(labels[r, c]) == int(labels[r + 1, c]):
                grid[r1, c0:c1] = BLACK

    return grid


def _pick_scattered_cells(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    pool = list(cells)
    random.shuffle(pool)
    if count == 0:
        return []

    picked = [pool.pop()]
    while len(picked) < count:
        best_score = -1
        best_cells: list[tuple[int, int]] = []
        for cell in pool:
            score = min(abs(cell[0] - pr) + abs(cell[1] - pc) for pr, pc in picked)
            if score > best_score:
                best_score = score
                best_cells = [cell]
            elif score == best_score:
                best_cells.append(cell)
        chosen = random.choice(best_cells)
        pool.remove(chosen)
        picked.append(chosen)
    return picked


def _tie_plan(area: int) -> list[tuple[int, int]] | None:
    max_markers = min(area - 2, 7)
    if max_markers < 2:
        return None

    n_tied = 2
    if max_markers >= 6 and random.random() < 0.35:
        n_tied = 3

    shared_max = min(3, max_markers // n_tied)
    if shared_max < 1:
        return None
    shared = random.randint(1, shared_max)

    counts = [shared] * n_tied
    if shared > 1 and max_markers - sum(counts) >= 1 and random.random() < 0.5:
        counts.append(random.randint(1, min(shared - 1, max_markers - sum(counts))))

    colors = random.sample(PALETTE, len(counts))
    return list(zip(colors, counts, strict=False))


def _majority_plan(area: int) -> list[tuple[int, int]] | None:
    max_markers = min(area - 2, 8)
    if max_markers < 3:
        return None

    n_colors = 2 if max_markers <= 5 or random.random() < 0.65 else 3
    winner_max = min(4, max_markers - (n_colors - 1))
    if winner_max < 2:
        return None
    winner_count = random.randint(2, winner_max)

    loser_counts = [1] * (n_colors - 1)
    spare = max_markers - winner_count - sum(loser_counts)
    while spare > 0:
        choices = [i for i, count in enumerate(loser_counts) if count < winner_count - 1]
        if not choices or random.random() < 0.35:
            break
        idx = random.choice(choices)
        loser_counts[idx] += 1
        spare -= 1

    colors = random.sample(PALETTE, n_colors)
    return [(colors[0], winner_count)] + list(zip(colors[1:], loser_counts, strict=False))


def _paint_region(grid: np.ndarray, cells: list[tuple[int, int]], tie_case: bool) -> bool:
    plan = _tie_plan(len(cells)) if tie_case else _majority_plan(len(cells))
    if plan is None:
        return False

    marker_total = sum(count for _, count in plan)
    if marker_total >= len(cells):
        return False

    chosen_cells = _pick_scattered_cells(cells, marker_total)
    colors = [color for color, count in plan for _ in range(count)]
    random.shuffle(colors)

    for (r, c), color in zip(chosen_cells, colors, strict=False):
        grid[r, c] = color
    return True


def _is_valid_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2:
        return False
    if not np.all(grid[0, :] == GREY) or not np.all(grid[-1, :] == GREY):
        return False
    if not np.all(grid[:, 0] == GREY) or not np.all(grid[:, -1] == GREY):
        return False

    regions = _enclosed_regions(grid)
    if len(regions) < 2:
        return False

    tie_flags = []
    for cells in regions:
        counts, _first_pos = _color_stats(grid, cells)
        if len(counts) < 2:
            return False
        if sum(counts.values()) >= len(cells):
            return False
        best = max(counts.values())
        tie_flags.append(sum(count == best for count in counts.values()) > 1)

    if not any(tie_flags) or all(tie_flags):
        return False

    out = main(grid)
    if not np.array_equal(out[grid == GREY], grid[grid == GREY]):
        return False
    if not np.any(out != grid):
        return False

    for cells in regions:
        winner = _winning_color(grid, cells)
        if winner is None:
            return False
        values = {int(out[r, c]) for r, c in cells}
        if values != {winner}:
            return False

    return True


def _generate_attempt() -> np.ndarray | None:
    coarse_rows = random.randint(2, 4)
    coarse_cols = random.randint(2, 4)
    n_cells = coarse_rows * coarse_cols
    n_regions = random.randint(2, min(5, n_cells))

    row_sizes = [random.randint(3, 5) for _ in range(coarse_rows)]
    col_sizes = [random.randint(3, 5) for _ in range(coarse_cols)]

    labels = _random_partition(coarse_rows, coarse_cols, n_regions)
    grid = _build_room_grid(labels, row_sizes, col_sizes)
    regions = _enclosed_regions(grid)
    if len(regions) != n_regions:
        return None

    tie_count = random.randint(1, n_regions - 1)
    tie_flags = [True] * tie_count + [False] * (n_regions - tie_count)
    random.shuffle(tie_flags)

    random.shuffle(regions)
    for cells, tie_case in zip(regions, tie_flags, strict=False):
        if not _paint_region(grid, cells, tie_case):
            return None

    if not _is_valid_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid enclosed-region voting puzzle")
