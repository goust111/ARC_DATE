from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

RANK_TO_COLOR = {
    1: BLUE,
    2: RED,
    3: GREEN,
    4: YELLOW,
    5: GREY,
    6: PINK,
    7: ORANGE,
    8: TEAL,
    9: MAROON,
}
INPUT_COLOR_POOL = [BLUE, RED, GREEN, YELLOW, PINK, ORANGE, TEAL, MAROON]
GENERATE_RETRIES = 400


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _split_runs(length: int, separators: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = 0
    for stop in separators:
        if start > stop - 1:
            raise AssertionError("separator lines must leave non-empty cell runs")
        runs.append((start, stop - 1))
        start = stop + 1
    if start > length - 1:
        raise AssertionError("separator lines must leave a final cell run")
    runs.append((start, length - 1))
    return runs


def _segment_span(length: int, separators: list[int], band_idx: int) -> tuple[int, int]:
    start = 0 if band_idx == 0 else separators[band_idx - 1]
    end = length - 1 if band_idx == len(separators) else separators[band_idx]
    return start, end


def _parse_layout(grid: np.ndarray) -> tuple[list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    sep_rows = [r for r in range(h) if np.all(grid[r, :] == GREY)]
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    assert len(sep_rows) == 2, "expected exactly two full-width grey separator rows"
    assert len(sep_cols) == 2, "expected exactly two full-height grey separator columns"
    row_runs = _split_runs(h, sep_rows)
    col_runs = _split_runs(w, sep_cols)
    assert len(row_runs) == 3 and len(col_runs) == 3
    return sep_rows, sep_cols, row_runs, col_runs


def _cell_counts(
    grid: np.ndarray,
    row_runs: list[tuple[int, int]],
    col_runs: list[tuple[int, int]],
) -> np.ndarray:
    counts = np.zeros((3, 3), dtype=int)
    for rr, (r0, r1) in enumerate(row_runs):
        for cc, (c0, c1) in enumerate(col_runs):
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            counts[rr, cc] = int(np.count_nonzero((cell != BLACK) & (cell != GREY)))
    return counts


def _rank_grid(counts: np.ndarray) -> np.ndarray:
    flat_counts = [int(v) for v in counts.flat]
    assert len(flat_counts) == 9
    assert len(set(flat_counts)) == 9, "all nine cells must have distinct foreground counts"
    order = sorted(range(9), key=lambda idx: flat_counts[idx], reverse=True)
    ranks = np.zeros((3, 3), dtype=int)
    for rank, flat_idx in enumerate(order, start=1):
        ranks[flat_idx // 3, flat_idx % 3] = rank
    return ranks


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each cell by its foreground-count rank and turn consecutive-rank dividers orange.

    Input structure:
    - Gray full-width rows and full-height columns partition the canvas into a 3x3 array of rectangular cells.
    - Inside each cell, non-black non-gray foreground pixels sit on a black background.
    - The nine cells have pairwise distinct foreground-pixel counts.

    Rule:
    - Count the foreground pixels in each cell, ignoring black background and gray separators.
    - Rank the cells from largest count to smallest count.
    - Recolor every foreground pixel in rank 1..9 with colors blue, red, green, yellow, gray, pink, orange, teal, maroon respectively.
    - For every separator segment between adjacent cells, recolor that whole segment orange when the two cell ranks differ by exactly 1; otherwise leave it gray.

    Key insight:
    - The cell shapes and original colors are distractors: only the per-cell foreground counts matter, and the same ranking controls both the cell recoloring and the divider network.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_rows, sep_cols, row_runs, col_runs = _parse_layout(g)
    ranks = _rank_grid(_cell_counts(g, row_runs, col_runs))
    out = g.copy()

    for rr, (r0, r1) in enumerate(row_runs):
        for cc, (c0, c1) in enumerate(col_runs):
            cell = out[r0 : r1 + 1, c0 : c1 + 1]
            mask = (cell != BLACK) & (cell != GREY)
            cell[mask] = RANK_TO_COLOR[int(ranks[rr, cc])]

    for rr in range(3):
        seg_r0, seg_r1 = _segment_span(g.shape[0], sep_rows, rr)
        for cc in range(2):
            if abs(int(ranks[rr, cc]) - int(ranks[rr, cc + 1])) == 1:
                out[seg_r0 : seg_r1 + 1, sep_cols[cc]] = ORANGE

    for rr in range(2):
        for cc in range(3):
            seg_c0, seg_c1 = _segment_span(g.shape[1], sep_cols, cc)
            if abs(int(ranks[rr, cc]) - int(ranks[rr + 1, cc])) == 1:
                out[sep_rows[rr], seg_c0 : seg_c1 + 1] = ORANGE

    return out


def _all_hamilton_paths() -> list[tuple[tuple[int, int], ...]]:
    neighbors = {
        (r, c): [(nr, nc) for nr, nc in _neighbors4(r, c) if 0 <= nr < 3 and 0 <= nc < 3]
        for r in range(3)
        for c in range(3)
    }
    paths: list[tuple[tuple[int, int], ...]] = []

    def dfs(path: list[tuple[int, int]], seen: set[tuple[int, int]]) -> None:
        if len(path) == 9:
            step_dirs = {
                (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                for i in range(8)
            }
            has_horizontal = any(dc != 0 for _dr, dc in step_dirs)
            has_vertical = any(dr != 0 for dr, _dc in step_dirs)
            if has_horizontal and has_vertical:
                paths.append(tuple(path))
            return
        for nxt in neighbors[path[-1]]:
            if nxt in seen:
                continue
            seen.add(nxt)
            path.append(nxt)
            dfs(path, seen)
            path.pop()
            seen.remove(nxt)

    for start in neighbors:
        dfs([start], {start})
    return paths


HAMILTON_PATHS = _all_hamilton_paths()


def _random_partition3(low: int, high: int) -> list[int]:
    return [random.randint(low, high) for _ in range(3)]


def _random_counts(max_count: int) -> list[int]:
    values = random.sample(range(2, max_count + 1), 9)
    values.sort(reverse=True)
    return values


def _choose_input_colors() -> dict[tuple[int, int], int]:
    dup = random.choice(INPUT_COLOR_POOL)
    non_adjacent_pairs = [
        (a, b)
        for a in [(r, c) for r in range(3) for c in range(3)]
        for b in [(r, c) for r in range(3) for c in range(3)]
        if a < b and abs(a[0] - b[0]) + abs(a[1] - b[1]) >= 2
    ]
    pos_a, pos_b = random.choice(non_adjacent_pairs)
    assignments = {pos_a: dup, pos_b: dup}
    remaining_colors = [color for color in INPUT_COLOR_POOL if color != dup]
    random.shuffle(remaining_colors)
    remaining_positions = [(r, c) for r in range(3) for c in range(3) if (r, c) not in assignments]
    for pos, color in zip(remaining_positions, remaining_colors):
        assignments[pos] = color
    return assignments


def _split_component_sizes(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1))
    prev = 0
    out = []
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    random.shuffle(out)
    return out


def _seed_positions(h: int, w: int, count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    seeds: list[tuple[int, int]] = []
    for r, c in cells:
        if any(abs(r - sr) + abs(c - sc) <= 1 for sr, sc in seeds):
            continue
        seeds.append((r, c))
        if len(seeds) == count:
            return seeds
    return []


def _grow_components(h: int, w: int, total: int, components: int) -> np.ndarray | None:
    seeds = _seed_positions(h, w, components)
    if len(seeds) != components:
        return None

    owners = np.full((h, w), -1, dtype=int)
    for idx, (r, c) in enumerate(seeds):
        owners[r, c] = idx
    target_sizes = _split_component_sizes(total, components)
    sizes = [1] * components

    while sizes != target_sizes:
        progress = False
        indices = list(range(components))
        random.shuffle(indices)
        for idx in indices:
            if sizes[idx] >= target_sizes[idx]:
                continue
            frontier: list[tuple[int, int]] = []
            for r, c in np.argwhere(owners == idx):
                rr, cc = int(r), int(c)
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if owners[nr, nc] != -1:
                        continue
                    touching_other = False
                    for ar, ac in _neighbors4(nr, nc):
                        if 0 <= ar < h and 0 <= ac < w and owners[ar, ac] not in (-1, idx):
                            touching_other = True
                            break
                    if not touching_other:
                        frontier.append((nr, nc))
            if frontier:
                nr, nc = random.choice(frontier)
                owners[nr, nc] = idx
                sizes[idx] += 1
                progress = True
        if not progress:
            return None

    return owners >= 0


def _generate_cell_mask(h: int, w: int, count: int) -> np.ndarray | None:
    for _ in range(120):
        box_h = random.randint(3, h)
        box_w = random.randint(3, w)
        fill_ratio = count / float(box_h * box_w)
        if not (0.18 <= fill_ratio <= 0.82):
            continue
        top = random.randint(0, h - box_h)
        left = random.randint(0, w - box_w)

        max_components = min(3, count)
        if count <= 4:
            component_choices = [1]
        elif count <= 8:
            component_choices = [1, 2]
        else:
            component_choices = [1, 2, 3]
        parts = random.choice(component_choices[:max_components])
        local = _grow_components(box_h, box_w, count, parts)
        if local is None:
            continue
        coords = np.argwhere(local)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if int(r1 - r0) == 0 and int(c1 - c0) == 0:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[top : top + box_h, left : left + box_w] = local
        return mask
    return None


def _build_rank_grid(path: tuple[tuple[int, int], ...]) -> np.ndarray:
    ranks = np.zeros((3, 3), dtype=int)
    for idx, (r, c) in enumerate(path, start=1):
        ranks[r, c] = idx
    return ranks


def _orange_segment_flags(ranks: np.ndarray) -> dict[tuple[str, int, int], bool]:
    flags: dict[tuple[str, int, int], bool] = {}
    for rr in range(3):
        for cc in range(2):
            flags[("v", rr, cc)] = abs(int(ranks[rr, cc]) - int(ranks[rr, cc + 1])) == 1
    for rr in range(2):
        for cc in range(3):
            flags[("h", rr, cc)] = abs(int(ranks[rr, cc]) - int(ranks[rr + 1, cc])) == 1
    return flags


def _validate_output_semantics(grid: np.ndarray, expected_ranks: np.ndarray, expected_counts: np.ndarray) -> bool:
    sep_rows, sep_cols, row_runs, col_runs = _parse_layout(grid)
    counts = _cell_counts(grid, row_runs, col_runs)
    if not np.array_equal(counts, expected_counts):
        return False
    ranks = _rank_grid(counts)
    if not np.array_equal(ranks, expected_ranks):
        return False

    out = main(grid)
    for rr, (r0, r1) in enumerate(row_runs):
        for cc, (c0, c1) in enumerate(col_runs):
            in_cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            out_cell = out[r0 : r1 + 1, c0 : c1 + 1]
            fg = (in_cell != BLACK) & (in_cell != GREY)
            if np.any(out_cell[fg] != RANK_TO_COLOR[int(ranks[rr, cc])]):
                return False
            if np.any(out_cell[~fg] != in_cell[~fg]):
                return False

    expected_flags = _orange_segment_flags(ranks)
    actual_orange = 0

    for rr in range(3):
        seg_r0, seg_r1 = _segment_span(grid.shape[0], sep_rows, rr)
        for cc in range(2):
            segment = out[seg_r0 : seg_r1 + 1, sep_cols[cc]]
            is_orange = bool(np.all(segment == ORANGE))
            if is_orange != expected_flags[("v", rr, cc)]:
                return False
            actual_orange += int(is_orange)

    for rr in range(2):
        for cc in range(3):
            seg_c0, seg_c1 = _segment_span(grid.shape[1], sep_cols, cc)
            segment = out[sep_rows[rr], seg_c0 : seg_c1 + 1]
            is_orange = bool(np.all(segment == ORANGE))
            if is_orange != expected_flags[("h", rr, cc)]:
                return False
            actual_orange += int(is_orange)

    if actual_orange != 8:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    row_heights = _random_partition3(5, 8)
    col_widths = _random_partition3(5, 8)
    h = sum(row_heights) + 2
    w = sum(col_widths) + 2
    if max(h, w) > 30:
        return None

    path = random.choice(HAMILTON_PATHS)
    ranks = _build_rank_grid(path)
    cell_areas = np.array([[row_heights[r] * col_widths[c] for c in range(3)] for r in range(3)], dtype=int)
    max_count = min(15, int(cell_areas.min()) - 1)
    if max_count < 10:
        return None
    rank_counts = _random_counts(max_count)
    expected_counts = np.zeros((3, 3), dtype=int)
    for rr in range(3):
        for cc in range(3):
            expected_counts[rr, cc] = rank_counts[int(ranks[rr, cc]) - 1]

    grid = np.zeros((h, w), dtype=int)
    sep_rows = [row_heights[0], row_heights[0] + 1 + row_heights[1]]
    sep_cols = [col_widths[0], col_widths[0] + 1 + col_widths[1]]
    grid[sep_rows, :] = GREY
    grid[:, sep_cols] = GREY
    row_runs = _split_runs(h, sep_rows)
    col_runs = _split_runs(w, sep_cols)
    input_colors = _choose_input_colors()

    for rr, (r0, r1) in enumerate(row_runs):
        for cc, (c0, c1) in enumerate(col_runs):
            mask = _generate_cell_mask(r1 - r0 + 1, c1 - c0 + 1, int(expected_counts[rr, cc]))
            if mask is None:
                return None
            grid[r0 : r1 + 1, c0 : c1 + 1][mask] = input_colors[(rr, cc)]

    if not _validate_output_semantics(grid, ranks, expected_counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ranked-divider ARC input")
