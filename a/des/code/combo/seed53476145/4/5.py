from __future__ import annotations

import random

import numpy as np


def _separator_columns(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []

    solid_by_color: dict[int, list[int]] = {}
    for c in range(w):
        value = int(grid[0, c])
        if value == bg:
            continue
        if np.all(grid[:, c] == value):
            solid_by_color.setdefault(value, []).append(c)

    for color, cols in solid_by_color.items():
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1 = int(cols[i])
                c2 = int(cols[j])
                panel_w = c1
                if panel_w <= 0:
                    continue
                if c2 - c1 - 1 != panel_w:
                    continue
                if w - c2 - 1 != panel_w:
                    continue
                candidates.append((color, c1, c2))

    assert len(candidates) == 1, "expected one equal-width separator pair"
    return candidates[0][1], candidates[0][2]


def _panels(grid: np.ndarray, c1: int, c2: int) -> list[np.ndarray]:
    return [
        grid[:, :c1],
        grid[:, c1 + 1 : c2],
        grid[:, c2 + 1 :],
    ]


def _non_bg_colors(panel: np.ndarray, bg: int) -> set[int]:
    return {int(color) for color in np.unique(panel) if int(color) != bg}


def main(grid: np.ndarray) -> np.ndarray:
    """Keep only colors shared by all three panels, then overlay those objects left to right."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = 0
    assert np.any(g == bg), "expected black background"

    c1, c2 = _separator_columns(g, bg)
    panels = _panels(g, c1, c2)
    panel_h, panel_w = panels[0].shape
    assert all(panel.shape == (panel_h, panel_w) for panel in panels), "panels must match"

    common = set.intersection(*(_non_bg_colors(panel, bg) for panel in panels))
    out = np.full((panel_h, panel_w), bg, dtype=int)
    if not common:
        return out

    keep = np.array(sorted(common), dtype=int)
    for panel in panels:
        mask = np.isin(panel, keep)
        out[mask] = panel[mask]
    return out


# --------------- generator ---------------

BG = 0
SEP = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

PANEL_H_RANGE = (6, 10)
PANEL_W_RANGE = (6, 10)
MAX_PANEL_COLORS = 5

COMMON_COUNT_CHOICES = (0, 1, 1, 2, 2, 3)
PAIR_PATTERNS = ((1, 1, 0), (1, 0, 1), (0, 1, 1))
SINGLE_PATTERNS = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
NO_DISTRACTOR_PROB = 0.15
EMPTY_INTERSECTION_PROB = 0.15
FORCE_OVERLAP_PROB = 0.55

MAX_GENERATION_ATTEMPTS = 400
PATTERN_ATTEMPTS = 200

BASE_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
)


def _normalize_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rotate_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize_shape(tuple((c, -r) for r, c in cells))


def _flip_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize_shape(tuple((r, -c) for r, c in cells))


def _shape_dims(cells: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _translated_cells(
    cells: tuple[tuple[int, int], ...], top: int, left: int
) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in cells]


def _random_shape(min_area: int = 2, max_area: int = 5) -> tuple[tuple[int, int], ...]:
    choices = [shape for shape in BASE_SHAPES if min_area <= len(shape) <= max_area]
    cells = tuple(random.choice(choices))
    for _ in range(random.randrange(4)):
        cells = _rotate_shape(cells)
    if random.random() < 0.5:
        cells = _flip_shape(cells)
    return _normalize_shape(cells)


def _paint_object(
    panel: np.ndarray,
    forbidden: np.ndarray,
    cells: tuple[tuple[int, int], ...],
    top: int,
    left: int,
    color: int,
) -> None:
    h, w = panel.shape
    for rr, cc in _translated_cells(cells, top, left):
        panel[rr, cc] = color
        for nr in range(max(0, rr - 1), min(h, rr + 2)):
            for nc in range(max(0, cc - 1), min(w, cc + 2)):
                forbidden[nr, nc] = True


def _can_place(
    cells: tuple[tuple[int, int], ...],
    top: int,
    left: int,
    forbidden: np.ndarray,
    blocked: np.ndarray,
) -> bool:
    h, w = forbidden.shape
    for rr, cc in _translated_cells(cells, top, left):
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if forbidden[rr, cc] or blocked[rr, cc]:
            return False
    return True


def _random_position(
    cells: tuple[tuple[int, int], ...],
    forbidden: np.ndarray,
    blocked: np.ndarray,
) -> tuple[int, int] | None:
    h, w = forbidden.shape
    shape_h, shape_w = _shape_dims(cells)
    options = []
    for top in range(h - shape_h + 1):
        for left in range(w - shape_w + 1):
            if _can_place(cells, top, left, forbidden, blocked):
                options.append((top, left))
    if not options:
        return None
    return random.choice(options)


def _build_panel(
    panel_h: int,
    panel_w: int,
    specs: list[dict],
    blocked: np.ndarray,
) -> np.ndarray | None:
    panel = np.full((panel_h, panel_w), BG, dtype=int)
    forbidden = np.zeros((panel_h, panel_w), dtype=bool)

    ordered = sorted(
        specs,
        key=lambda spec: (0 if spec["pos"] is not None else 1, -len(spec["cells"])),
    )
    for spec in ordered:
        cells = spec["cells"]
        pos = spec["pos"]
        if pos is None:
            pos = _random_position(cells, forbidden, blocked)
            if pos is None:
                return None
        top, left = pos
        if not _can_place(cells, top, left, forbidden, blocked):
            return None
        _paint_object(panel, forbidden, cells, top, left, spec["color"])
    return panel


def _sample_presence_patterns() -> list[tuple[int, int, int]] | None:
    for _ in range(PATTERN_ATTEMPTS):
        common_count = random.choice(COMMON_COUNT_CHOICES)
        if common_count == 0 and random.random() > EMPTY_INTERSECTION_PROB:
            common_count = random.choice((1, 2, 2, 3))

        pair_count = random.randint(0, 3)
        single_count = random.randint(0, 3)
        if common_count > 0 and pair_count == 0 and single_count == 0 and random.random() > NO_DISTRACTOR_PROB:
            if random.random() < 0.5:
                pair_count = 1
            else:
                single_count = 1

        patterns: list[tuple[int, int, int]] = [(1, 1, 1)] * common_count
        patterns.extend(random.choice(PAIR_PATTERNS) for _ in range(pair_count))
        patterns.extend(random.choice(SINGLE_PATTERNS) for _ in range(single_count))
        if not patterns or len(patterns) > len(OBJECT_COLORS):
            continue

        counts = [sum(pattern[idx] for pattern in patterns) for idx in range(3)]
        if min(counts) < 2 or max(counts) > MAX_PANEL_COLORS:
            continue
        return patterns
    return None


def _expected_output(panels: list[np.ndarray]) -> np.ndarray:
    common = set.intersection(*(_non_bg_colors(panel, BG) for panel in panels))
    panel_h, panel_w = panels[0].shape
    out = np.full((panel_h, panel_w), BG, dtype=int)
    if not common:
        return out

    keep = np.array(sorted(common), dtype=int)
    for panel in panels:
        mask = np.isin(panel, keep)
        out[mask] = panel[mask]
    return out


def _compose_grid(panels: list[np.ndarray]) -> np.ndarray:
    panel_h, panel_w = panels[0].shape
    grid = np.full((panel_h, panel_w * 3 + 2), BG, dtype=int)
    grid[:, panel_w] = SEP
    grid[:, 2 * panel_w + 1] = SEP
    grid[:, :panel_w] = panels[0]
    grid[:, panel_w + 1 : 2 * panel_w + 1] = panels[1]
    grid[:, 2 * panel_w + 2 :] = panels[2]
    return grid


def _gen_attempt() -> np.ndarray | None:
    patterns = _sample_presence_patterns()
    if patterns is None:
        return None

    colors = random.sample(list(OBJECT_COLORS), len(patterns))
    panel_colors = [[], [], []]
    common_colors = []
    for color, pattern in zip(colors, patterns):
        if pattern == (1, 1, 1):
            common_colors.append(color)
        for idx, present in enumerate(pattern):
            if present:
                panel_colors[idx].append(color)

    panel_h = random.randint(*PANEL_H_RANGE)
    panel_w = random.randint(*PANEL_W_RANGE)
    blocked = [np.zeros((panel_h, panel_w), dtype=bool) for _ in range(3)]
    panel_specs: list[list[dict]] = [[], [], []]
    forced_keys: set[tuple[int, int]] = set()

    overlap_info = None
    if len(common_colors) >= 2 and random.random() < FORCE_OVERLAP_PROB:
        low, high = random.choice(((0, 1), (0, 2), (1, 2)))
        low_color, high_color = random.sample(common_colors, 2)
        overlap_cells = _random_shape(min_area=2, max_area=4)
        shape_h, shape_w = _shape_dims(overlap_cells)
        if shape_h > panel_h or shape_w > panel_w:
            return None
        top = random.randint(0, panel_h - shape_h)
        left = random.randint(0, panel_w - shape_w)
        overlap_cells_abs = _translated_cells(overlap_cells, top, left)

        panel_specs[low].append({"color": low_color, "cells": overlap_cells, "pos": (top, left)})
        panel_specs[high].append({"color": high_color, "cells": overlap_cells, "pos": (top, left)})
        forced_keys.add((low, low_color))
        forced_keys.add((high, high_color))

        for idx in range(3):
            if idx in (low, high):
                continue
            for rr, cc in overlap_cells_abs:
                blocked[idx][rr, cc] = True
        overlap_info = (low, high, low_color, high_color, overlap_cells_abs)

    for idx in range(3):
        random.shuffle(panel_colors[idx])
        for color in panel_colors[idx]:
            if (idx, color) in forced_keys:
                continue
            panel_specs[idx].append({"color": color, "cells": _random_shape(), "pos": None})

    panels = []
    for idx in range(3):
        panel = _build_panel(panel_h, panel_w, panel_specs[idx], blocked[idx])
        if panel is None:
            return None
        panels.append(panel)

    expected = _expected_output(panels)
    if overlap_info is not None:
        low, high, low_color, high_color, cells_abs = overlap_info
        for rr, cc in cells_abs:
            if int(panels[low][rr, cc]) != low_color:
                return None
            if int(panels[high][rr, cc]) != high_color:
                return None
            if int(expected[rr, cc]) != high_color:
                return None

    grid = _compose_grid(panels)
    try:
        out = main(grid)
    except Exception:
        return None
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
