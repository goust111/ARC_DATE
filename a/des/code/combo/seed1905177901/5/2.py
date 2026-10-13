from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 8
COLOR_POOL = [0, 1, 2, 3, 4, 5, 6, 7, 9]
GENERATION_RETRIES = 500

CLEAR_TEMPLATES = (
    ((0, 0), (0, 1), (1, 0), (2, 2)),
    ((0, 0), (0, 2), (1, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 3), (1, 0), (2, 2), (3, 3)),
)

DENSE_TEMPLATES = (
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (0, 2), (1, 0), (1, 2), (2, 1), (3, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
)


def _extract_seeds(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid != BG)]
    coords.sort()
    return [(r, c, int(grid[r, c])) for r, c in coords]


def _cross_cells(shape: tuple[int, int], row: int, col: int, hx: int, vy: int) -> set[tuple[int, int]]:
    height, width = shape
    cells = {(int(row), int(col))}
    for step in range(1, int(hx) + 1):
        if col - step >= 0:
            cells.add((int(row), int(col - step)))
        if col + step < width:
            cells.add((int(row), int(col + step)))
    for step in range(1, int(vy) + 1):
        if row - step >= 0:
            cells.add((int(row - step), int(col)))
        if row + step < height:
            cells.add((int(row + step), int(col)))
    return cells


def _seed_specs(
    seeds: list[tuple[int, int, int]], shape: tuple[int, int]
) -> list[dict[str, int | set[tuple[int, int]]]]:
    specs: list[dict[str, int | set[tuple[int, int]]]] = []
    for row, col, color in seeds:
        row_dists = [abs(int(col) - int(c2)) for r2, c2, _ in seeds if int(r2) == int(row) and int(c2) != int(col)]
        col_dists = [abs(int(row) - int(r2)) for r2, c2, _ in seeds if int(c2) == int(col) and int(r2) != int(row)]
        hx = 1 if not row_dists else max(0, (min(row_dists) - 1) // 2)
        vy = 1 if not col_dists else max(0, (min(col_dists) - 1) // 2)
        cells = _cross_cells(shape, int(row), int(col), int(hx), int(vy))
        specs.append(
            {
                "row": int(row),
                "col": int(col),
                "color": int(color),
                "hx": int(hx),
                "vy": int(vy),
                "area": int(len(cells)),
                "cells": cells,
            }
        )
    return specs


def _paint_output(seeds: list[tuple[int, int, int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for row, col, color in seeds:
        out[int(row), int(col)] = int(color)

    specs = _seed_specs(seeds, shape)
    specs.sort(key=lambda spec: (-int(spec["area"]), int(spec["row"]), int(spec["col"])))
    for spec in specs:
        color = int(spec["color"])
        for row, col in sorted(spec["cells"]):
            if int(out[row, col]) == BG:
                out[row, col] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored seed into a cross whose horizontal and vertical reach
    are determined by the nearest seed on the same row or column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    seeds = _extract_seeds(g)
    assert seeds, "expected at least one non-background seed"
    return _paint_output(seeds, g.shape)


def _sample_axis_positions(count: int, gap_low: int, gap_high: int) -> tuple[list[int], int] | None:
    for _ in range(200):
        start = random.randint(2, 5)
        positions = [start]
        for _idx in range(count - 1):
            positions.append(int(positions[-1] + random.randint(gap_low, gap_high)))
        size = int(positions[-1] + random.randint(2, 5) + 1)
        if 10 <= size <= 30:
            return positions, size
    return None


def _transform_layout(
    seeds: list[tuple[int, int, int]], shape: tuple[int, int]
) -> tuple[list[tuple[int, int, int]], tuple[int, int]]:
    height, width = shape
    cur = [(int(r), int(c), int(color)) for r, c, color in seeds]

    if random.random() < 0.35:
        cur = [(int(c), int(r), int(color)) for r, c, color in cur]
        height, width = width, height
    if random.random() < 0.5:
        cur = [(int(height - 1 - r), int(c), int(color)) for r, c, color in cur]
    if random.random() < 0.5:
        cur = [(int(r), int(width - 1 - c), int(color)) for r, c, color in cur]

    cur.sort()
    return cur, (int(height), int(width))


def _build_template_layout(kind: str) -> tuple[list[tuple[int, int, int]], tuple[int, int], str] | None:
    templates = CLEAR_TEMPLATES if kind == "clear" else DENSE_TEMPLATES
    template = random.choice(templates)
    row_count = 1 + max(int(r) for r, _ in template)
    col_count = 1 + max(int(c) for _, c in template)
    row_range = (3, 5) if kind == "clear" else (4, 8)
    col_range = (3, 5) if kind == "clear" else (4, 8)

    row_sample = _sample_axis_positions(row_count, *row_range)
    col_sample = _sample_axis_positions(col_count, *col_range)
    if row_sample is None or col_sample is None:
        return None

    rows, height = row_sample
    cols, width = col_sample
    colors = random.sample(COLOR_POOL, len(template))
    seeds = [(int(rows[r]), int(cols[c]), int(colors[idx])) for idx, (r, c) in enumerate(template)]
    seeds, shape = _transform_layout(seeds, (int(height), int(width)))
    return seeds, shape, kind


def _build_overlap_layout() -> tuple[list[tuple[int, int, int]], tuple[int, int], str] | None:
    for _ in range(200):
        top = random.randint(2, 4)
        left = random.randint(2, 4)

        row_mid_gap = random.randint(3, 5)
        min_row_far_gap = 2 * row_mid_gap + 1
        row_far_gap = random.randint(min_row_far_gap, min_row_far_gap + 4)

        col_mid_gap = random.randint(3, 5)
        min_col_far_gap = 2 * col_mid_gap + 1
        col_far_gap = random.randint(min_col_far_gap, min_col_far_gap + 4)

        rows = [int(top), int(top + row_mid_gap), int(top + row_far_gap)]
        cols = [int(left), int(left + col_mid_gap), int(left + col_mid_gap + col_far_gap)]
        height = int(rows[-1] + random.randint(2, 4) + 1)
        width = int(cols[-1] + random.randint(2, 4) + 1)
        if height > 30 or width > 30:
            continue

        colors = random.sample(COLOR_POOL, 4)
        seeds = [
            (rows[0], cols[0], colors[0]),
            (rows[2], cols[0], colors[1]),
            (rows[1], cols[1], colors[2]),
            (rows[1], cols[2], colors[3]),
        ]
        seeds, shape = _transform_layout(seeds, (height, width))
        return seeds, shape, "overlap"
    return None


def _spacing_ok(seeds: list[tuple[int, int, int]]) -> bool:
    for idx, (r1, c1, color1) in enumerate(seeds):
        if int(color1) == BG:
            return False
        for r2, c2, color2 in seeds[idx + 1 :]:
            if int(color1) == int(color2):
                return False
            if int(r1) == int(r2) and abs(int(c1) - int(c2)) < 3:
                return False
            if int(c1) == int(c2) and abs(int(r1) - int(r2)) < 3:
                return False
    return True


def _branch_mix_ok(seeds: list[tuple[int, int, int]]) -> bool:
    row_counts = Counter(int(r) for r, _c, _color in seeds)
    col_counts = Counter(int(c) for _r, c, _color in seeds)
    return (
        any(count >= 2 for count in row_counts.values())
        and any(count == 1 for count in row_counts.values())
        and any(count >= 2 for count in col_counts.values())
        and any(count == 1 for count in col_counts.values())
    )


def _conflict_stats(
    seeds: list[tuple[int, int, int]], shape: tuple[int, int]
) -> tuple[int, bool, int]:
    specs = _seed_specs(seeds, shape)
    seed_cells = {(int(spec["row"]), int(spec["col"])) for spec in specs}
    claims: dict[tuple[int, int], list[dict[str, int | set[tuple[int, int]]]]] = defaultdict(list)
    for spec in specs:
        for cell in spec["cells"]:
            if cell in seed_cells:
                continue
            claims[cell].append(spec)

    conflict_cells = 0
    max_depth = 0
    same_area_conflict = False
    for entries in claims.values():
        if len(entries) < 2:
            continue
        conflict_cells += 1
        max_depth = max(max_depth, len(entries))
        areas = [int(entry["area"]) for entry in entries]
        if len(set(areas)) != len(areas):
            same_area_conflict = True
    return conflict_cells, same_area_conflict, max_depth


def _scenario_ok(seeds: list[tuple[int, int, int]], shape: tuple[int, int], kind: str) -> bool:
    if not _spacing_ok(seeds):
        return False
    if not _branch_mix_ok(seeds):
        return False

    conflict_cells, same_area_conflict, max_depth = _conflict_stats(seeds, shape)
    if same_area_conflict or max_depth > 2:
        return False
    if kind == "clear" and conflict_cells != 0:
        return False
    if kind == "overlap" and not (1 <= conflict_cells <= 3):
        return False
    if kind == "dense" and conflict_cells > 4:
        return False

    inp = np.full(shape, BG, dtype=int)
    for row, col, color in seeds:
        inp[int(row), int(col)] = int(color)
    out = _paint_output(seeds, shape)
    if np.array_equal(inp, out):
        return False

    try:
        pred = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(pred, out))


def _gen_attempt() -> np.ndarray | None:
    kind = random.choices(["clear", "dense", "overlap"], weights=[4, 3, 3], k=1)[0]
    if kind == "overlap":
        layout = _build_overlap_layout()
    else:
        layout = _build_template_layout(kind)
    if layout is None:
        return None

    seeds, shape, actual_kind = layout
    if not _scenario_ok(seeds, shape, actual_kind):
        return None

    grid = np.full(shape, BG, dtype=int)
    for row, col, color in seeds:
        grid[int(row), int(col)] = int(color)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
