from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _most_common(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected at least one value"
    return int(ranked[0][0])


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected at least one non-background marker"
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    return int(ranked[0][0])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start: int | None = None
    for idx in range(int(mask.size) + 1):
        active = bool(mask[idx]) if idx < int(mask.size) else False
        if active and start is None:
            start = idx
        elif not active and start is not None:
            groups.append((int(start), int(idx)))
            start = None
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group single-pixel markers by contiguous active rows and columns, then fill each
    non-empty row-group × column-group rectangle with the region's unique majority color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common(g)

    active_rows = np.any(g != bg, axis=1)
    active_cols = np.any(g != bg, axis=0)
    row_groups = _runs(active_rows)
    col_groups = _runs(active_cols)

    out = np.full(g.shape, bg, dtype=int)
    for r0, r1 in row_groups:
        for c0, c1 in col_groups:
            patch = g[r0:r1, c0:c1]
            markers = patch[patch != bg]
            if markers.size == 0:
                continue
            out[r0:r1, c0:c1] = _unique_mode(markers)
    return out


# --------------- generator ---------------

BG = 8
COLOR_POOL = tuple(color for color in range(10) if color != BG)
GROUP_COUNT_RANGE = (2, 4)
GROUP_SIZE_RANGE = (1, 4)
INNER_GAP_RANGE = (1, 3)
EDGE_GAP_RANGE = (0, 3)
GRID_SIZE_RANGE = (9, 24)
EXTRA_MARKER_CAP = 3
MAX_GENERATION_ATTEMPTS = 300


def _sample_axis() -> tuple[int, list[tuple[int, int]]]:
    min_len, max_len = GRID_SIZE_RANGE
    for _ in range(200):
        count = random.randint(*GROUP_COUNT_RANGE)
        sizes = [random.randint(*GROUP_SIZE_RANGE) for _ in range(count)]
        gaps = [random.randint(*INNER_GAP_RANGE) for _ in range(count - 1)]
        lead = random.randint(*EDGE_GAP_RANGE)
        trail = random.randint(*EDGE_GAP_RANGE)
        total = lead + sum(sizes) + sum(gaps) + trail
        if not (min_len <= total <= max_len):
            continue

        spans: list[tuple[int, int]] = []
        pos = lead
        for idx, size in enumerate(sizes):
            spans.append((int(pos), int(pos + size)))
            pos += size
            if idx < count - 1:
                pos += gaps[idx]
        return int(total), spans
    raise RuntimeError("failed to sample axis groups")


def _sample_occupancy(row_count: int, col_count: int) -> np.ndarray:
    for _ in range(200):
        occ = np.zeros((row_count, col_count), dtype=bool)

        for r in range(row_count):
            occ[r, random.randrange(col_count)] = True
        for c in range(col_count):
            occ[random.randrange(row_count), c] = True

        for r in range(row_count):
            for c in range(col_count):
                if not occ[r, c] and random.random() < 0.45:
                    occ[r, c] = True

        if occ.all():
            removable = [
                (r, c)
                for r in range(row_count)
                for c in range(col_count)
                if occ[r].sum() > 1 and occ[:, c].sum() > 1
            ]
            if removable:
                rr, cc = random.choice(removable)
                occ[rr, cc] = False

        if occ.any() and not occ.all():
            return occ
    raise RuntimeError("failed to sample a mixed occupancy matrix")


def _cover_positions(height: int, width: int) -> list[tuple[int, int]]:
    points: set[tuple[int, int]] = set()

    if height <= width:
        cols = list(range(width))
        random.shuffle(cols)
        for row in range(height):
            points.add((row, cols[row]))
        covered_cols = {col for _, col in points}
        for col in range(width):
            if col not in covered_cols:
                points.add((random.randrange(height), col))
    else:
        rows = list(range(height))
        random.shuffle(rows)
        for col in range(width):
            points.add((rows[col], col))
        covered_rows = {row for row, _ in points}
        for row in range(height):
            if row not in covered_rows:
                points.add((row, random.randrange(width)))

    remaining = [(r, c) for r in range(height) for c in range(width) if (r, c) not in points]
    max_extra = min(len(remaining), EXTRA_MARKER_CAP)
    extra = 0 if max_extra == 0 else random.randint(0, max_extra)
    if extra:
        points.update(random.sample(remaining, extra))

    pts = list(points)
    random.shuffle(pts)
    return pts


def _region_colors(count: int, dominant: int, palette: list[int]) -> list[int]:
    colors = [dominant] * count
    others = [color for color in palette if color != dominant]
    if count < 3 or not others or random.random() < 0.45:
        random.shuffle(colors)
        return colors

    dominant_count = random.randint(count // 2 + 1, count - 1)
    minority_total = count - dominant_count
    minority_color_count = random.randint(1, min(2, len(others), minority_total))
    chosen = random.sample(others, minority_color_count)
    splits = [0] * minority_color_count
    for idx in range(minority_total):
        splits[idx % minority_color_count] += 1
    random.shuffle(splits)

    colors = [dominant] * dominant_count
    for color, copies in zip(chosen, splits, strict=False):
        colors.extend([color] * copies)
    random.shuffle(colors)
    return colors


def _build_expected(
    shape: tuple[int, int],
    row_groups: list[tuple[int, int]],
    col_groups: list[tuple[int, int]],
    occupancy: np.ndarray,
    region_colors: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for row_idx, (r0, r1) in enumerate(row_groups):
        for col_idx, (c0, c1) in enumerate(col_groups):
            if not occupancy[row_idx, col_idx]:
                continue
            out[r0:r1, c0:c1] = int(region_colors[(row_idx, col_idx)])
    return out


def _gen_attempt() -> np.ndarray | None:
    height, row_groups = _sample_axis()
    width, col_groups = _sample_axis()
    occupancy = _sample_occupancy(len(row_groups), len(col_groups))

    palette_size = random.randint(2, min(6, len(COLOR_POOL)))
    palette = random.sample(list(COLOR_POOL), palette_size)

    grid = np.full((height, width), BG, dtype=int)
    dominant_by_region: dict[tuple[int, int], int] = {}

    for row_idx, (r0, r1) in enumerate(row_groups):
        for col_idx, (c0, c1) in enumerate(col_groups):
            if not occupancy[row_idx, col_idx]:
                continue

            dominant = random.choice(palette)
            dominant_by_region[(row_idx, col_idx)] = dominant

            points = _cover_positions(r1 - r0, c1 - c0)
            colors = _region_colors(len(points), dominant, palette)
            for (dr, dc), color in zip(points, colors, strict=False):
                grid[r0 + dr, c0 + dc] = int(color)

    expected = _build_expected(grid.shape, row_groups, col_groups, occupancy, dominant_by_region)

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
