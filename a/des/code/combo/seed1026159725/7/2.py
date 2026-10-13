from __future__ import annotations

from collections import Counter
import random

import numpy as np


RANK_COLORS = (1, 3, 2, 4)  # blue, green, red, yellow

# --------------- shared helpers ---------------


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _normalize_mask(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    coords = np.argwhere(mask)
    assert coords.size, "shape must be non-empty"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    cropped = np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)
    return (int(cropped.shape[0]), int(cropped.shape[1])), cropped.tobytes()


def _edge_touch_count(mask: np.ndarray) -> int:
    if mask.shape[0] == 1 and mask.shape[1] == 1:
        return int(mask[0, 0])
    total = int(mask[0, :].sum() + mask[-1, :].sum() + mask[:, 0].sum() + mask[:, -1].sum())
    total -= int(mask[0, 0]) + int(mask[0, -1]) + int(mask[-1, 0]) + int(mask[-1, -1])
    return total


def _common_divisors(h: int, w: int) -> list[int]:
    return [d for d in range(2, min(h, w) + 1) if h % d == 0 and w % d == 0]


def _analyze_partition(
    grid: np.ndarray, cell_size: int, bg: int, fg: int
) -> dict | None:
    h, w = grid.shape
    if h % cell_size or w % cell_size:
        return None

    rows = h // cell_size
    cols = w // cell_size
    if rows < 2 or cols < 2:
        return None

    keys: list[tuple[tuple[int, int], bytes]] = []
    cells: list[tuple[slice, slice]] = []
    edge_touches = 0
    for br in range(rows):
        for bc in range(cols):
            rs = slice(br * cell_size, (br + 1) * cell_size)
            cs = slice(bc * cell_size, (bc + 1) * cell_size)
            cell = grid[rs, cs]
            colors = set(map(int, np.unique(cell)))
            if colors - {bg, fg}:
                return None
            mask = cell == fg
            if not mask.any():
                return None
            edge_touches += _edge_touch_count(mask)
            keys.append(_normalize_mask(mask))
            cells.append((rs, cs))

    freq = Counter(keys)
    repeated = sorted((count for count in freq.values() if count > 1), reverse=True)
    if not repeated:
        return None

    repeated_coverage = sum(count for count in freq.values() if count > 1)
    score = (
        repeated_coverage,
        len(repeated),
        -edge_touches,
        -cell_size,
    )
    return {
        "cell_size": int(cell_size),
        "rows": int(rows),
        "cols": int(cols),
        "keys": keys,
        "cells": cells,
        "freq": freq,
        "score": score,
    }


def _infer_partition(grid: np.ndarray) -> tuple[int, int, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    fg_colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg = fg_colors[0]

    candidates = []
    for cell_size in _common_divisors(*g.shape):
        info = _analyze_partition(g, cell_size, bg, fg)
        if info is not None:
            candidates.append(info)

    assert candidates, "failed to infer the cell partition"
    candidates.sort(key=lambda info: info["score"], reverse=True)
    assert len(candidates) == 1 or candidates[0]["score"] > candidates[1]["score"], (
        "partition should be uniquely determined"
    )
    return bg, fg, candidates[0]


# --------------- solver ---------------


def main(grid: np.ndarray) -> np.ndarray:
    """Group equal cell-local patterns and recolor repeated groups by descending frequency."""
    g = np.asarray(grid, dtype=int)
    bg, fg, info = _infer_partition(g)
    freq: Counter = info["freq"]

    ranked_groups = [
        (key, count)
        for key, count in freq.items()
        if count > 1
    ]
    ranked_groups.sort(key=lambda item: (-item[1], item[0]))
    color_by_key = {
        key: RANK_COLORS[i]
        for i, (key, _count) in enumerate(ranked_groups[: len(RANK_COLORS)])
    }

    out = g.copy()
    for key, (rs, cs) in zip(info["keys"], info["cells"]):
        if key not in color_by_key:
            continue
        cell = out[rs, cs]
        cell[cell == fg] = color_by_key[key]
    return out


# --------------- generator ---------------


FG = 5
BG_CHOICES = [0, 6, 7, 8, 9]
CELL_SIZE_CHOICES = [4, 5, 6, 7]
ROW_RANGE_BY_SIZE = {
    4: (3, 4),
    5: (3, 5),
    6: (3, 5),
    7: (3, 5),
}
COL_RANGE_BY_SIZE = {
    4: (3, 4),
    5: (3, 5),
    6: (3, 5),
    7: (3, 5),
}
REPEATED_GROUP_CHOICES = [1, 2, 3, 4]
MARGIN = 1
MAX_ATTEMPTS = 400


def _max_unique_shapes(cell_size: int) -> int:
    inner = cell_size - 2 * MARGIN
    return (1 << (inner * inner)) - 1


def _shape_from_key(key: tuple[tuple[int, int], bytes]) -> np.ndarray:
    shape, payload = key
    return np.frombuffer(payload, dtype=bool).reshape(shape).copy()


def _random_shape_key(cell_size: int) -> tuple[tuple[int, int], bytes]:
    inner = cell_size - 2 * MARGIN
    h = random.randint(1, inner)
    w = random.randint(1, inner)
    area = random.randint(1, h * w)
    flat = np.zeros(h * w, dtype=bool)
    for idx in random.sample(range(h * w), area):
        flat[idx] = True
    return _normalize_mask(flat.reshape(h, w))


def _sample_counts(total_cells: int, repeated_groups: int) -> list[int] | None:
    base = list(range(2, 2 + repeated_groups))
    if sum(base) > total_cells:
        return None

    extra = total_cells - sum(base)
    counts = base[:]
    while extra > 0:
        options = [i for i in range(repeated_groups) if i == 0 or counts[i] + 1 < counts[i - 1]]
        if not options:
            break
        idx = random.choice(options)
        counts[idx] += 1
        extra -= 1

    counts.sort(reverse=True)
    return counts


def _choose_layout(cell_size: int) -> tuple[int, int, list[int]] | None:
    row_lo, row_hi = ROW_RANGE_BY_SIZE[cell_size]
    col_lo, col_hi = COL_RANGE_BY_SIZE[cell_size]
    rows = random.randint(row_lo, row_hi)
    cols = random.randint(col_lo, col_hi)
    total_cells = rows * cols

    group_cap = min(4, total_cells // 2)
    repeated_groups = random.choice([g for g in REPEATED_GROUP_CHOICES if g <= group_cap])
    counts = _sample_counts(total_cells, repeated_groups)
    if counts is None:
        return None

    unique_needed = repeated_groups + (total_cells - sum(counts))
    if unique_needed > _max_unique_shapes(cell_size):
        return None
    return rows, cols, counts


def _sample_shape_keys(cell_size: int, needed: int) -> list[tuple[tuple[int, int], bytes]] | None:
    keys: list[tuple[tuple[int, int], bytes]] = []
    seen = set()
    for _ in range(500):
        if len(keys) == needed:
            return keys
        key = _random_shape_key(cell_size)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return None


def _paint_shape_in_cell(
    grid: np.ndarray,
    cell_r: int,
    cell_c: int,
    cell_size: int,
    key: tuple[tuple[int, int], bytes],
) -> None:
    mask = _shape_from_key(key)
    h, w = mask.shape
    max_r = cell_size - MARGIN - h
    max_c = cell_size - MARGIN - w
    r0 = cell_r * cell_size + random.randint(MARGIN, max_r)
    c0 = cell_c * cell_size + random.randint(MARGIN, max_c)
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = FG


def _expected_output(
    grid: np.ndarray, keys_by_cell: list[tuple[tuple[int, int], bytes]], cell_size: int
) -> np.ndarray:
    out = grid.copy()
    freq = Counter(keys_by_cell)
    ranked = [(key, count) for key, count in freq.items() if count > 1]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    color_by_key = {key: RANK_COLORS[i] for i, (key, _count) in enumerate(ranked[: len(RANK_COLORS)])}

    rows = grid.shape[0] // cell_size
    cols = grid.shape[1] // cell_size
    idx = 0
    for br in range(rows):
        for bc in range(cols):
            if keys_by_cell[idx] in color_by_key:
                rs = slice(br * cell_size, (br + 1) * cell_size)
                cs = slice(bc * cell_size, (bc + 1) * cell_size)
                cell = out[rs, cs]
                cell[cell == FG] = color_by_key[keys_by_cell[idx]]
            idx += 1
    return out


def _build_case() -> np.ndarray | None:
    cell_size = random.choice(CELL_SIZE_CHOICES)
    layout = _choose_layout(cell_size)
    if layout is None:
        return None
    rows, cols, repeated_counts = layout

    total_cells = rows * cols
    singleton_count = total_cells - sum(repeated_counts)
    all_shape_keys = _sample_shape_keys(cell_size, len(repeated_counts) + singleton_count)
    if all_shape_keys is None:
        return None

    repeated_keys = all_shape_keys[: len(repeated_counts)]
    singleton_keys = all_shape_keys[len(repeated_counts) :]

    assignments: list[tuple[tuple[int, int], bytes]] = []
    for key, count in zip(repeated_keys, repeated_counts):
        assignments.extend([key] * count)
    assignments.extend(singleton_keys)
    random.shuffle(assignments)

    bg = random.choice(BG_CHOICES)
    grid = np.full((rows * cell_size, cols * cell_size), bg, dtype=int)
    idx = 0
    for br in range(rows):
        for bc in range(cols):
            _paint_shape_in_cell(grid, br, bc, cell_size, assignments[idx])
            idx += 1

    try:
        bg_found, fg_found, info = _infer_partition(grid)
    except AssertionError:
        return None

    if bg_found != bg or fg_found != FG or info["cell_size"] != cell_size:
        return None

    expected = _expected_output(grid, assignments, cell_size)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
