from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
ARC_COLORS = list(range(10))
NON_BLACK_COLORS = list(range(1, 10))


def _uniform_stripes(grid: np.ndarray, axis: int) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    length = grid.shape[axis]
    for idx in range(length):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        color = int(line[0])
        if np.all(line == color):
            groups.setdefault(color, []).append(int(idx))
    return groups


def _choose_divider_color(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    row_groups = _uniform_stripes(grid, axis=0)
    col_groups = _uniform_stripes(grid, axis=1)
    candidates = sorted(set(row_groups) | set(col_groups))
    assert candidates, "expected at least one uniform divider stripe"

    best_color = None
    best_score = None
    for color in candidates:
        rows = row_groups.get(color, [])
        cols = col_groups.get(color, [])
        score = (
            int(bool(rows) and bool(cols)),
            len(rows) + len(cols),
            len(rows) * len(cols),
        )
        if best_score is None or score > best_score:
            best_color = int(color)
            best_score = score

    assert best_color is not None
    return best_color, row_groups.get(best_color, []), col_groups.get(best_color, [])


def _non_divider_spans(length: int, divider_indices: list[int]) -> list[tuple[int, int]]:
    blocked = set(divider_indices)
    spans: list[tuple[int, int]] = []
    start = None
    for idx in range(length):
        if idx in blocked:
            if start is not None:
                spans.append((start, idx - 1))
                start = None
        else:
            if start is None:
                start = idx
    if start is not None:
        spans.append((start, length - 1))
    return spans


def _background_color(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    divider: int,
) -> int:
    local_modes: list[int] = []
    for row_span in row_spans:
        for col_span in col_spans:
            r0, r1 = row_span
            c0, c1 = col_span
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            vals = [int(v) for v in patch.flat if int(v) != divider]
            assert vals, "empty region"
            local_modes.append(Counter(vals).most_common(1)[0][0])
    return int(Counter(local_modes).most_common(1)[0][0])


def _region_counts(
    grid: np.ndarray,
    row_span: tuple[int, int],
    col_span: tuple[int, int],
    bg: int,
    divider: int,
) -> Counter[int]:
    r0, r1 = row_span
    c0, c1 = col_span
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    vals = [int(v) for v in patch.flat if int(v) not in {bg, divider}]
    return Counter(vals)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare every divider-defined region against its point-symmetric partner and color the coarse cell by the unique net-gain color.

    Input structure:
    - Full-row and full-column divider bars split a non-black canvas into a coarse grid of rectangular regions.
    - Each region contains scattered colored pixels on a dominant background color.
    - Point-symmetric partner regions share the same non-background color set, but their per-color counts differ.

    Rule:
    - Detect the divider rows and columns, then compress the input to one output cell per rectangular region.
    - Pair each coarse cell with the cell at the point-symmetric position across the coarse grid center.
    - For every pair, keep the unique color whose count is strictly larger in one region than in its partner; if a center cell pairs with itself, output black there.

    Key insight:
    - The output color is not a local majority: it is the only color that survives a count comparison with the faraway point-symmetric partner region.
    """
    g = np.asarray(grid, dtype=int)
    divider, divider_rows, divider_cols = _choose_divider_color(g)
    row_spans = _non_divider_spans(g.shape[0], divider_rows)
    col_spans = _non_divider_spans(g.shape[1], divider_cols)
    assert row_spans and col_spans, "expected divider-defined regions"

    bg = _background_color(g, row_spans, col_spans, divider)
    out = np.full((len(row_spans), len(col_spans)), BLACK, dtype=int)

    seen: set[tuple[int, int]] = set()
    for r in range(len(row_spans)):
        for c in range(len(col_spans)):
            if (r, c) in seen:
                continue
            pr = len(row_spans) - 1 - r
            pc = len(col_spans) - 1 - c
            seen.add((r, c))
            seen.add((pr, pc))
            if (r, c) == (pr, pc):
                out[r, c] = BLACK
                continue

            counts_a = _region_counts(g, row_spans[r], col_spans[c], bg, divider)
            counts_b = _region_counts(g, row_spans[pr], col_spans[pc], bg, divider)
            palette = set(counts_a) | set(counts_b)
            survive_a = [color for color in palette if counts_a[color] > counts_b[color]]
            survive_b = [color for color in palette if counts_b[color] > counts_a[color]]
            assert len(survive_a) == 1 and len(survive_b) == 1, "expected one surviving color per partnered region"
            out[r, c] = int(survive_a[0])
            out[pr, pc] = int(survive_b[0])
    return out


def _symmetric_sizes(count: int, lo: int, hi: int) -> list[int]:
    left = [random.randint(lo, hi) for _ in range(count // 2)]
    center = [random.randint(lo, hi)] if count % 2 else []
    return left + center + left[::-1]


def _axis_layout(sizes: list[int], divider_thickness: int) -> tuple[list[tuple[int, int]], list[int], int]:
    spans: list[tuple[int, int]] = []
    divider_indices: list[int] = []
    pos = 0
    for idx, size in enumerate(sizes):
        spans.append((pos, pos + size - 1))
        pos += size
        if idx != len(sizes) - 1:
            divider_indices.extend(range(pos, pos + divider_thickness))
            pos += divider_thickness
    return spans, divider_indices, pos


def _place_counts(
    grid: np.ndarray,
    row_span: tuple[int, int],
    col_span: tuple[int, int],
    counts: dict[int, int],
) -> None:
    coords = [
        (r, c)
        for r in range(row_span[0], row_span[1] + 1)
        for c in range(col_span[0], col_span[1] + 1)
    ]
    total = sum(counts.values())
    assert total < len(coords), "leave at least one background cell per region"
    random.shuffle(coords)
    chosen = coords[:total]
    colors: list[int] = []
    for color, count in counts.items():
        colors.extend([int(color)] * int(count))
    random.shuffle(colors)
    for (r, c), color in zip(chosen, colors):
        grid[r, c] = int(color)


def _paired_region_counts(area: int, palette_pool: list[int]) -> tuple[dict[int, int], dict[int, int], int, int]:
    payload_limit = (area - 1) // 2
    max_palette = min(4, payload_limit - 1)
    palette_size = random.randint(2, max_palette)
    palette = random.sample(palette_pool, palette_size)
    win_a, win_b = random.sample(palette, 2)

    counts_a = {color: 1 for color in palette}
    counts_b = {color: 1 for color in palette}

    max_delta = min(3, payload_limit - palette_size)
    counts_a[win_a] += random.randint(1, max_delta)
    counts_b[win_b] += random.randint(1, max_delta)

    shared_room = min(payload_limit - sum(counts_a.values()), payload_limit - sum(counts_b.values()))
    for _ in range(random.randint(0, min(shared_room, palette_size + 1))):
        color = random.choice(palette)
        counts_a[color] += 1
        counts_b[color] += 1

    return counts_a, counts_b, int(win_a), int(win_b)


def _center_region_counts(area: int, palette_pool: list[int]) -> dict[int, int]:
    payload_limit = (area - 1) // 2
    palette_size = random.randint(1, min(3, payload_limit))
    palette = random.sample(palette_pool, palette_size)
    counts = {color: 1 for color in palette}
    room = payload_limit - sum(counts.values())
    for _ in range(random.randint(0, min(room, palette_size + 2))):
        counts[random.choice(palette)] += 1
    return counts


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    if random.random() < 0.45:
        coarse_rows = random.choice([3, 5])
        coarse_cols = random.choice([3, 5])
    else:
        coarse_rows = random.randint(2, 5)
        coarse_cols = random.randint(2, 5)

    row_sizes = _symmetric_sizes(coarse_rows, 3, 5)
    col_sizes = _symmetric_sizes(coarse_cols, 3, 5)
    row_divider = random.randint(1, 2)
    col_divider = random.randint(1, 2)

    row_spans, divider_rows, height = _axis_layout(row_sizes, row_divider)
    col_spans, divider_cols, width = _axis_layout(col_sizes, col_divider)
    if height > 30 or width > 30:
        return None

    bg = random.choice(NON_BLACK_COLORS)
    if random.random() < 0.5:
        divider = BLACK
    else:
        divider = random.choice([color for color in NON_BLACK_COLORS if color != bg])

    palette_pool = [color for color in NON_BLACK_COLORS if color not in {bg, divider}]
    if len(palette_pool) < 2:
        return None

    grid = np.full((height, width), bg, dtype=int)
    if divider_rows:
        grid[divider_rows, :] = divider
    if divider_cols:
        grid[:, divider_cols] = divider

    expected = np.full((coarse_rows, coarse_cols), BLACK, dtype=int)
    seen: set[tuple[int, int]] = set()
    for r in range(coarse_rows):
        for c in range(coarse_cols):
            if (r, c) in seen:
                continue
            pr = coarse_rows - 1 - r
            pc = coarse_cols - 1 - c
            seen.add((r, c))
            seen.add((pr, pc))

            area = (row_spans[r][1] - row_spans[r][0] + 1) * (col_spans[c][1] - col_spans[c][0] + 1)
            if (r, c) == (pr, pc):
                counts = _center_region_counts(area, palette_pool)
                _place_counts(grid, row_spans[r], col_spans[c], counts)
                expected[r, c] = BLACK
                continue

            counts_a, counts_b, win_a, win_b = _paired_region_counts(area, palette_pool)
            _place_counts(grid, row_spans[r], col_spans[c], counts_a)
            _place_counts(grid, row_spans[pr], col_spans[pc], counts_b)
            expected[r, c] = win_a
            expected[pr, pc] = win_b

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(300):
        sample = _generate_attempt()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid input")
