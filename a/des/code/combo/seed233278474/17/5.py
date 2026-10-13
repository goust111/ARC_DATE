from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np


MARKER_PALETTE = (1, 2, 3, 4, 7)
BACKGROUND_CHOICES = (0, 6, 8, 9)
SEPARATOR_COLOR = 5
LAYOUTS = ((2, 2), (2, 3), (2, 4), (2, 5), (3, 2), (4, 2), (5, 2))
PAIRING_MODES = ("none", "some", "many")
BLOCK_SIZE_MIN = 2
BLOCK_SIZE_MAX = 6
GEN_ATTEMPTS = 400


def _intervals(length: int, separators: Sequence[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        if start <= sep - 1:
            spans.append((start, sep - 1))
        start = sep + 1
    if start <= length - 1:
        spans.append((start, length - 1))
    return spans


def _pick_background(values: np.ndarray) -> int:
    counts = Counter(map(int, values.ravel()))
    return int(max(counts, key=counts.get))


def _extract_structure(
    grid: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], np.ndarray] | None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[
        tuple[int, int, int, list[tuple[int, int]], list[tuple[int, int]], np.ndarray]
    ] = []

    for color in map(int, np.unique(g)):
        full_rows = [r for r in range(h) if np.all(g[r, :] == color)]
        full_cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if not full_rows or not full_cols:
            continue

        row_spans = _intervals(h, full_rows)
        col_spans = _intervals(w, full_cols)
        if not row_spans or not col_spans:
            continue

        interior = g[g != color]
        if interior.size == 0:
            continue
        bg = _pick_background(interior)

        markers = np.zeros((len(row_spans), len(col_spans)), dtype=int)
        color_counts: Counter[int] = Counter()
        valid = True

        for i, (r0, r1) in enumerate(row_spans):
            for j, (c0, c1) in enumerate(col_spans):
                region = g[r0 : r1 + 1, c0 : c1 + 1]
                if np.any(region == color):
                    valid = False
                    break
                ys, xs = np.nonzero(region != bg)
                if len(ys) != 1:
                    valid = False
                    break
                marker = int(region[ys[0], xs[0]])
                if marker in (bg, color):
                    valid = False
                    break
                markers[i, j] = marker
                color_counts[marker] += 1
            if not valid:
                break

        if not valid:
            continue
        if any(count % 2 != 0 for count in color_counts.values()):
            continue

        score = len(row_spans) * len(col_spans) * 10 + len(full_rows) + len(full_cols)
        candidates.append((score, bg, color, row_spans, col_spans, markers))

    if not candidates:
        return None

    _score, bg, sep_color, row_spans, col_spans, markers = max(
        candidates, key=lambda item: item[0]
    )
    return bg, sep_color, row_spans, col_spans, markers


def _separator_between(left: tuple[int, int], right: tuple[int, int]) -> int:
    return left[1] + 1 if left[1] < right[0] else right[1] + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase only those separator segments shared by directly adjacent same-colored regions.

    Input structure
    - The grid is partitioned by full grey separator rows and columns.
    - Each rectangular region contains exactly one colored marker on a background.
    - Marker colors occur in pairs.

    Rule
    - Same-colored markers identify paired regions.
    - Only if a pair lies in directly adjacent regions, erase the shared grey wall
      segment between them; otherwise keep all separators unchanged.
    """
    g = np.asarray(grid, dtype=int)
    parsed = _extract_structure(g)
    if parsed is None:
        return g.copy()

    bg, sep_color, row_spans, col_spans, markers = parsed
    out = g.copy()

    for i, row_span in enumerate(row_spans):
        for j, col_span in enumerate(col_spans):
            if j + 1 < len(col_spans) and int(markers[i, j]) == int(markers[i, j + 1]):
                sep_c = _separator_between(col_span, col_spans[j + 1])
                out[row_span[0] : row_span[1] + 1, sep_c] = bg
            if i + 1 < len(row_spans) and int(markers[i, j]) == int(markers[i + 1, j]):
                sep_r = _separator_between(row_span, row_spans[i + 1])
                out[sep_r, col_span[0] : col_span[1] + 1] = bg

    return out


def _is_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _pairing_targets(mode: str, pair_count: int) -> list[int]:
    if mode == "none":
        return [0]
    if mode == "some":
        return [1]
    return list(range(pair_count, 1, -1))


def _find_pairing(
    cells: list[tuple[int, int]], target_adjacent: int, rng: np.random.Generator
) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    adjacency = {
        (a, b): _is_adjacent(cells[a], cells[b])
        for a in range(len(cells))
        for b in range(a + 1, len(cells))
    }

    def backtrack(
        remaining: tuple[int, ...],
        adjacent_so_far: int,
    ) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
        if not remaining:
            return [] if adjacent_so_far == target_adjacent else None
        pairs_left = len(remaining) // 2
        if adjacent_so_far > target_adjacent:
            return None
        if adjacent_so_far + pairs_left < target_adjacent:
            return None

        first = remaining[0]
        partner_positions = rng.permutation(len(remaining) - 1)
        for pos in partner_positions:
            second = remaining[1 + int(pos)]
            lo, hi = sorted((first, second))
            new_adjacent = adjacent_so_far + int(adjacency[(lo, hi)])
            if new_adjacent > target_adjacent:
                continue
            next_remaining = tuple(idx for idx in remaining[1:] if idx != second)
            tail = backtrack(next_remaining, new_adjacent)
            if tail is not None:
                return [(cells[first], cells[second])] + tail
        return None

    return backtrack(tuple(range(len(cells))), 0)


def _place_separators(
    grid: np.ndarray, sizes: Sequence[int], axis: int
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for index, size in enumerate(sizes):
        spans.append((cursor, cursor + size - 1))
        cursor += size
        if index != len(sizes) - 1:
            if axis == 0:
                grid[cursor, :] = SEPARATOR_COLOR
            else:
                grid[:, cursor] = SEPARATOR_COLOR
            cursor += 1
    return spans


def _build_grid(
    row_sizes: Sequence[int],
    col_sizes: Sequence[int],
    colors: np.ndarray,
    rng: np.random.Generator,
    bg: int,
) -> np.ndarray:
    h = int(sum(row_sizes) + len(row_sizes) - 1)
    w = int(sum(col_sizes) + len(col_sizes) - 1)
    grid = np.full((h, w), bg, dtype=int)
    row_spans = _place_separators(grid, row_sizes, axis=0)
    col_spans = _place_separators(grid, col_sizes, axis=1)

    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            rr = int(rng.integers(r0, r1 + 1))
            cc = int(rng.integers(c0, c1 + 1))
            grid[rr, cc] = int(colors[i, j])
    return grid


def _adjacent_pair_count(colors: np.ndarray) -> int:
    count = 0
    rows, cols = colors.shape
    for i in range(rows):
        for j in range(cols):
            if i + 1 < rows and int(colors[i, j]) == int(colors[i + 1, j]):
                count += 1
            if j + 1 < cols and int(colors[i, j]) == int(colors[i, j + 1]):
                count += 1
    return count


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_rows, n_cols = LAYOUTS[int(rng.integers(len(LAYOUTS)))]
    pair_count = n_rows * n_cols // 2
    mode = PAIRING_MODES[int(rng.integers(len(PAIRING_MODES)))]

    row_sizes = rng.integers(BLOCK_SIZE_MIN, BLOCK_SIZE_MAX, size=n_rows)
    col_sizes = rng.integers(BLOCK_SIZE_MIN, BLOCK_SIZE_MAX, size=n_cols)
    bg = int(BACKGROUND_CHOICES[int(rng.integers(len(BACKGROUND_CHOICES)))])

    cells = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    target_values = _pairing_targets(mode, pair_count)
    rng.shuffle(target_values)

    pairs = None
    target_adjacent = None
    for target in target_values:
        pairs = _find_pairing(cells, int(target), rng)
        if pairs is not None:
            target_adjacent = int(target)
            break
    if pairs is None or target_adjacent is None:
        return None

    chosen_colors = rng.permutation(np.array(MARKER_PALETTE, dtype=int))[:pair_count]
    rng.shuffle(chosen_colors)
    color_grid = np.zeros((n_rows, n_cols), dtype=int)
    for color, (a, b) in zip(chosen_colors, pairs):
        color_grid[a] = int(color)
        color_grid[b] = int(color)

    grid = _build_grid(row_sizes, col_sizes, color_grid, rng, bg)
    parsed = _extract_structure(grid)
    if parsed is None:
        return None

    parsed_bg, sep_color, row_spans, col_spans, markers = parsed
    if parsed_bg != bg or sep_color != SEPARATOR_COLOR:
        return None
    if len(row_spans) != n_rows or len(col_spans) != n_cols:
        return None
    if not np.array_equal(markers, color_grid):
        return None

    adjacent_pairs = _adjacent_pair_count(markers)
    if mode == "none" and adjacent_pairs != 0:
        return None
    if mode == "some" and adjacent_pairs != 1:
        return None
    if mode == "many" and adjacent_pairs < 2:
        return None

    out = main(grid)
    changed = not np.array_equal(grid, out)
    if (adjacent_pairs == 0 and changed) or (adjacent_pairs > 0 and not changed):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid paired-region grid")
