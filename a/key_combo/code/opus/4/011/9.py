from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
TOP_COLORS = [1, 2, 3, 4]
BOTTOM_COLORS = [5, 8]
SEPARATOR = 6
SPIRAL = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (2, 1),
    (2, 0),
    (1, 0),
)

GEN_ATTEMPTS = 800
PANEL_HEIGHT_CHOICES = [5, 6]
MAX_PANEL_WIDTH = 24


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_by_color(grid: np.ndarray, *, bg: int = BG) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": int(len(cells)),
                }
            )
    return components


def _separator_row(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates = []
    for r in range(h):
        row_color = int(grid[r, 0])
        if row_color == BG:
            continue
        if np.all(grid[r, :] == row_color) and int(np.count_nonzero(grid == row_color)) == w:
            candidates.append(r)
    assert len(candidates) == 1, "expected exactly one unique solid separator row"
    sep = int(candidates[0])
    assert sep > 0 and sep < h - 1, "separator must split two non-empty panels"
    return sep


def _top_stats(top: np.ndarray, bottom: np.ndarray) -> list[tuple[int, int, int]]:
    stats = []
    for color in sorted(set(map(int, np.unique(top))) - {BG}):
        mask = top == color
        area = int(mask.sum())
        overlap = int(np.count_nonzero(mask & (bottom != BG)))
        stats.append((area, color, overlap))
    stats.sort(key=lambda item: (item[0], item[1]))
    return stats


def main(grid: np.ndarray) -> np.ndarray:
    """Align the two panels, count per-color overlaps, and encode them as area-sorted spiral gauges.

    Input structure:
    - The grid has black background and one unique solid separator row between two equal-size panels.
    - The top panel contains 2-4 monochrome 4-connected colored regions in distinct colors.
    - The bottom panel contains 2-3 monochrome 4-connected regions whose colors differ from the top panel and act only as foreground masks.

    Rule:
    - Remove the separator row and align the top and bottom panels by position.
    - For each top-panel color C, count how many cells of C lie over any non-black bottom-panel foreground cell, and also count the total area of C in the top panel.
    - Sort the top-panel colors by ascending total area, breaking ties by color id.
    - Output one 3x3 gauge per sorted color, laid out left-to-right with one black spacer column; in each gauge, fill the first O(C) cells of the clockwise outer spiral with color C and leave the rest black.

    Key insight:
    - The overlay step does not produce a visible intersection mask; it produces a separate overlap count for each top color, while the placement order comes from top-panel area instead of overlap amount.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _separator_row(g)
    top = g[:sep, :]
    bottom = g[sep + 1 :, :]
    assert top.shape == bottom.shape, "expected equal-size panels above and below the separator"

    stats = _top_stats(top, bottom)
    assert 2 <= len(stats) <= 4, "expected 2-4 top-panel colors"
    assert all(0 <= overlap <= len(SPIRAL) for _area, _color, overlap in stats)

    out = np.zeros((3, len(stats) * 4 - 1), dtype=int)
    for idx, (_area, color, overlap) in enumerate(stats):
        c0 = idx * 4
        for r, c in SPIRAL[:overlap]:
            out[r, c0 + c] = int(color)
    return out


def _build_component_mask(panel_h: int, area: int, overlap: int) -> np.ndarray | None:
    service_row = panel_h // 2

    for _ in range(200):
        min_width = max(overlap if overlap > 0 else 2, 2, (area + 1) // 2)
        max_width = max(min_width, min(8, area + 2))
        widths = list(range(min_width, max_width + 1))
        random.shuffle(widths)

        for width in widths:
            if overlap > width:
                continue

            if overlap > 0:
                cells = {(service_row, c) for c in range(overlap)}
            else:
                seed_row = random.choice([r for r in range(panel_h) if r != service_row])
                cells = {(seed_row, 0)}

            def allowed(r: int, c: int) -> bool:
                if not (0 <= r < panel_h and 0 <= c < width):
                    return False
                if overlap == 0:
                    return r != service_row
                return r != service_row or c < overlap

            while len(cells) < area:
                frontier: list[tuple[int, int]] = []
                for r, c in list(cells):
                    for nr, nc in _neighbors4(r, c):
                        if (nr, nc) in cells or not allowed(nr, nc):
                            continue
                        frontier.append((nr, nc))
                frontier = list(dict.fromkeys(frontier))
                if not frontier:
                    break
                cells.add(random.choice(frontier))

            if len(cells) != area:
                continue

            max_col = max(c for _r, c in cells)
            mask = np.zeros((panel_h, max_col + 1), dtype=bool)
            for r, c in cells:
                mask[r, c] = True
            if random.random() < 0.5:
                mask = np.fliplr(mask)
            if int(mask[service_row].sum()) == overlap:
                return mask

    return None


def _partitions(length: int, groups: int) -> list[list[tuple[int, int]]]:
    out: list[list[tuple[int, int]]] = []

    def rec(start: int, left: int, acc: list[tuple[int, int]]) -> None:
        if left == 1:
            acc.append((start, length))
            out.append(acc.copy())
            acc.pop()
            return
        for end in range(start + 1, length - left + 2):
            acc.append((start, end))
            rec(end, left - 1, acc)
            acc.pop()

    rec(0, groups, [])
    return out


def _solve_intervals(width: int, positive_ranges: list[tuple[int, int]], bottom_count: int) -> list[tuple[int, int]] | None:
    feasible: list[list[list[tuple[int, int]]]] = []

    for partition in _partitions(len(positive_ranges), bottom_count):
        candidate_lists: list[list[tuple[int, int]]] = []
        ok = True

        for idx, (start, end) in enumerate(partition):
            req_l = min(positive_ranges[j][0] for j in range(start, end))
            req_r = max(positive_ranges[j][1] for j in range(start, end))

            if idx == 0:
                hard_l = 0
            else:
                prev_start, prev_end = partition[idx - 1]
                hard_l = max(positive_ranges[j][1] for j in range(prev_start, prev_end)) + 2

            if idx == len(partition) - 1:
                hard_r = width - 1
            else:
                next_start, next_end = partition[idx + 1]
                hard_r = min(positive_ranges[j][0] for j in range(next_start, next_end)) - 2

            candidates = [
                (l, r)
                for l in range(hard_l, req_l + 1)
                for r in range(req_r, hard_r + 1)
                if 4 <= r - l + 1 <= 10
            ]
            if not candidates:
                ok = False
                break
            candidate_lists.append(candidates)

        if ok:
            feasible.append(candidate_lists)

    if not feasible:
        return None
    return [random.choice(candidates) for candidates in random.choice(feasible)]


def _validate_semantics(grid: np.ndarray) -> None:
    h, w = grid.shape
    assert h <= 30 and w <= 30

    sep = _separator_row(grid)
    top = grid[:sep, :]
    bottom = grid[sep + 1 :, :]
    assert top.shape == bottom.shape

    top_components = _components_by_color(top)
    bottom_components = _components_by_color(bottom)
    assert 2 <= len(top_components) <= 4
    assert 2 <= len(bottom_components) <= 3

    top_colors = [int(comp["color"]) for comp in top_components]
    bottom_colors = [int(comp["color"]) for comp in bottom_components]
    assert len(set(top_colors)) == len(top_colors)
    assert set(top_colors).issubset(set(TOP_COLORS))
    assert set(bottom_colors).issubset(set(BOTTOM_COLORS))

    top_areas = [int(comp["area"]) for comp in top_components]
    assert all(3 <= area <= 8 for area in top_areas)
    assert len(set(top_areas)) == len(top_areas)
    assert all(4 <= int(comp["area"]) <= 10 for comp in bottom_components)

    overlaps = []
    for comp in top_components:
        coords = comp["coords"]
        overlap = int(np.count_nonzero(bottom[coords[:, 0], coords[:, 1]] != BG))
        overlaps.append(overlap)
    assert sum(overlap > 0 for overlap in overlaps) >= 2
    assert all(0 <= overlap <= 7 for overlap in overlaps)
    assert any(
        top_areas[i] < top_areas[j] and overlaps[i] > overlaps[j]
        for i in range(len(top_areas))
        for j in range(len(top_areas))
    ), "area order must not be recoverable from overlap counts"

    stats = _top_stats(top, bottom)
    out = main(grid)
    assert out.shape == (3, len(stats) * 4 - 1)
    for idx, (_area, color, overlap) in enumerate(stats):
        block = out[:, idx * 4 : idx * 4 + 3]
        expected = np.zeros((3, 3), dtype=int)
        for r, c in SPIRAL[:overlap]:
            expected[r, c] = int(color)
        assert np.array_equal(block, expected)
    if len(stats) > 1:
        assert np.all(out[:, 3::4] == BG)


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(2, 4)
    panel_h = random.choice(PANEL_HEIGHT_CHOICES)
    panel_w = None

    areas = random.sample(range(3, 9), count)
    colors = random.sample(TOP_COLORS, count)

    zero_budget = 1 if count >= 3 and random.random() < 0.35 else 0
    overlaps = []
    for area in areas:
        low = 0 if zero_budget > 0 else 1
        overlap = random.randint(low, min(7, area))
        if overlap == 0:
            zero_budget -= 1
        overlaps.append(overlap)

    if sum(overlap > 0 for overlap in overlaps) < 2:
        return None
    if not any(
        areas[i] < areas[j] and overlaps[i] > overlaps[j]
        for i in range(count)
        for j in range(count)
    ):
        return None

    masks: list[np.ndarray] = []
    for area, overlap in zip(areas, overlaps):
        mask = _build_component_mask(panel_h, area, overlap)
        if mask is None:
            return None
        masks.append(mask)

    placement_order = list(range(count))
    random.shuffle(placement_order)
    if placement_order == sorted(range(count), key=lambda i: (areas[i], colors[i])):
        return None

    gaps = [random.randint(1, 3) for _ in range(count - 1)]
    left_margin = random.randint(0, 2)
    right_margin = random.randint(0, 2)

    starts = []
    col = left_margin
    for pos, idx in enumerate(placement_order):
        starts.append(col)
        col += masks[idx].shape[1]
        if pos < count - 1:
            col += gaps[pos]
    panel_w = col + right_margin
    if panel_w > MAX_PANEL_WIDTH:
        return None

    top = np.zeros((panel_h, panel_w), dtype=int)
    service_row = panel_h // 2
    positive_ranges: list[tuple[int, int]] = []

    for pos, idx in enumerate(placement_order):
        mask = masks[idx]
        c0 = starts[pos]
        top[:, c0 : c0 + mask.shape[1]][mask] = colors[idx]
        service_cols = np.flatnonzero(mask[service_row])
        if service_cols.size > 0:
            positive_ranges.append((c0 + int(service_cols[0]), c0 + int(service_cols[-1])))

    bottom_count = random.randint(2, min(3, len(positive_ranges)))
    intervals = _solve_intervals(panel_w, positive_ranges, bottom_count)
    if intervals is None:
        return None

    bottom = np.zeros((panel_h, panel_w), dtype=int)
    bottom_palette = random.choices(BOTTOM_COLORS, k=bottom_count)
    if bottom_count == 2 and random.random() < 0.7:
        bottom_palette = random.sample(BOTTOM_COLORS, 2)
    for color, (l, r) in zip(bottom_palette, intervals):
        bottom[service_row, l : r + 1] = int(color)

    grid = np.vstack([top, np.full((1, panel_w), SEPARATOR, dtype=int), bottom])

    try:
        _validate_semantics(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-overlap spiral puzzle input")
