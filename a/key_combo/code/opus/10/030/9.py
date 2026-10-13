from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

GRID_WIDTH = 9
GRID_HEIGHTS = (9, 10, 11)
SPLITS_BY_COUNT = {
    1: [(2,), (3,), (4,), (5,)],
    2: [(2, 5), (2, 6), (3, 5), (3, 6), (4, 6)],
}
COUNTER_SPLITS = {
    1: [(3,)],
    2: [(3, 6)],
}
LAYOUT_COUNTS = [(1, 1), (1, 2), (2, 1), (2, 2)]
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _passable_mask(grid: np.ndarray) -> np.ndarray:
    return (grid != GREY) & (grid != RED) & (grid != BLUE)


def _nearest_distance(pos: tuple[int, int], sources: np.ndarray) -> int:
    r, c = pos
    return int(np.min(np.abs(sources[:, 0] - r) + np.abs(sources[:, 1] - c)))


def _marker_fill_color(grid: np.ndarray, marker: tuple[int, int]) -> int:
    red_sources = np.argwhere(grid == RED)
    blue_sources = np.argwhere(grid == BLUE)
    assert red_sources.size and blue_sources.size
    d_red = _nearest_distance(marker, red_sources)
    d_blue = _nearest_distance(marker, blue_sources)
    return RED if d_red <= d_blue else BLUE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each chamber marker by nearest edge line, then flood-fill that chamber with the inherited color.

    Input structure:
    - The left border is a full red line and the right border is a full blue line.
    - Inside the grid, black open space is partitioned by thin grey orthogonal walls.
    - Each chamber contains one green singleton marker.

    Rule:
    - For each green marker, compute its Manhattan distance to the red edge line and to the blue edge line, ignoring walls.
    - Recolor that marker red when the red line is at least as near; otherwise recolor it blue.
    - Treat grey walls and the two colored edge lines as barriers, then 4-neighbor flood-fill the marker's chamber so every black cell in that chamber takes the marker's new color.
    - Keep the edge lines and walls unchanged.

    Key insight:
    - Geometry and topology use different constraints: nearest-edge color is decided through walls, but chamber filling is constrained by walls, so a right-side chamber can still become red.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    passable = _passable_mask(g)
    chamber_components = _components(passable)
    marker_colors = {
        (int(r), int(c)): _marker_fill_color(g, (int(r), int(c)))
        for r, c in np.argwhere(g == GREEN)
    }

    for comp in chamber_components:
        markers = [(r, c) for r, c in comp if g[r, c] == GREEN]
        if not markers:
            continue
        assert len(markers) == 1
        fill_color = marker_colors[markers[0]]
        for r, c in comp:
            if g[r, c] in (BLACK, GREEN):
                out[r, c] = fill_color
    return out


def _choose_layout() -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    h = random.choice(GRID_HEIGHTS)
    hr = random.randint(3, h - 4)
    top_count, bottom_count = random.choice(LAYOUT_COUNTS)
    counter_band = random.choice(["top", "bottom"])

    if counter_band == "top":
        top_splits = random.choice(COUNTER_SPLITS[top_count])
        bottom_splits = random.choice(SPLITS_BY_COUNT[bottom_count])
    else:
        top_splits = random.choice(SPLITS_BY_COUNT[top_count])
        bottom_splits = random.choice(COUNTER_SPLITS[bottom_count])
    return h, hr, top_splits, bottom_splits


def _build_grid(h: int, hr: int, top_splits: tuple[int, ...], bottom_splits: tuple[int, ...]) -> np.ndarray:
    grid = np.full((h, GRID_WIDTH), BLACK, dtype=int)
    grid[:, 0] = RED
    grid[:, -1] = BLUE
    grid[hr, 1:-1] = GREY
    for c in top_splits:
        grid[:hr, c] = GREY
    for c in bottom_splits:
        grid[hr + 1 :, c] = GREY
    return grid


def _component_stats(grid: np.ndarray) -> list[dict]:
    stats = []
    for comp in _components(_passable_mask(grid)):
        cols = [c for _r, c in comp]
        rows = [r for r, _c in comp]
        stats.append(
            {
                "cells": comp,
                "min_c": min(cols),
                "max_c": max(cols),
                "mean_c": float(sum(cols)) / len(cols),
                "min_r": min(rows),
                "max_r": max(rows),
            }
        )
    return stats


def _prefer_interior(cells: list[tuple[int, int]], h: int) -> list[tuple[int, int]]:
    interior = [(r, c) for r, c in cells if 0 < r < h - 1]
    return interior if interior else cells


def _pick_counter_marker(chamber: dict, h: int) -> tuple[int, int]:
    col4 = [(r, c) for r, c in chamber["cells"] if c == 4]
    assert col4
    return random.choice(_prefer_interior(col4, h))


def _pick_blue_marker(chamber: dict, h: int) -> tuple[int, int]:
    max_c = chamber["max_c"]
    candidates = [(r, c) for r, c in chamber["cells"] if c == max_c]
    return random.choice(_prefer_interior(candidates, h))


def _pick_generic_marker(chamber: dict, h: int) -> tuple[int, int]:
    cells = _prefer_interior(chamber["cells"], h)
    min_c = chamber["min_c"]
    max_c = chamber["max_c"]
    mean_c = chamber["mean_c"]

    if mean_c <= 4 and len([1 for _r, c in cells if c == min_c]) > 0 and random.random() < 0.6:
        edge_cells = [(r, c) for r, c in cells if c == min_c]
        return random.choice(edge_cells)
    if mean_c > 4 and len([1 for _r, c in cells if c == max_c]) > 0 and random.random() < 0.6:
        edge_cells = [(r, c) for r, c in cells if c == max_c]
        return random.choice(edge_cells)
    return random.choice(cells)


def _place_markers(grid: np.ndarray, chambers: list[dict]) -> None:
    h = grid.shape[0]
    counter_candidates = [ch for ch in chambers if ch["mean_c"] > 4 and any(c == 4 for _r, c in ch["cells"])]
    blue_candidates = [ch for ch in chambers if ch["max_c"] >= 5]
    assert counter_candidates and blue_candidates

    counter_chamber = random.choice(counter_candidates)
    other_blue = [ch for ch in blue_candidates if ch is not counter_chamber]
    assert other_blue
    blue_chamber = max(other_blue, key=lambda ch: (ch["mean_c"], ch["max_c"], len(ch["cells"])))

    chosen: dict[int, tuple[int, int]] = {}
    for idx, chamber in enumerate(chambers):
        if chamber is counter_chamber:
            chosen[idx] = _pick_counter_marker(chamber, h)
        elif chamber is blue_chamber:
            chosen[idx] = _pick_blue_marker(chamber, h)
        else:
            chosen[idx] = _pick_generic_marker(chamber, h)

    for r, c in chosen.values():
        assert grid[r, c] == BLACK
        grid[r, c] = GREEN


def _validate_structure(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    assert w == GRID_WIDTH and 9 <= h <= 11
    assert np.all(grid[:, 0] == RED)
    assert np.all(grid[:, -1] == BLUE)
    assert set(map(int, np.unique(grid))).issubset({BLACK, BLUE, RED, GREEN, GREY})

    chambers = _component_stats(grid)
    assert 4 <= len(chambers) <= 6

    marker_positions = {tuple(map(int, p)) for p in np.argwhere(grid == GREEN)}
    assert len(marker_positions) == len(chambers)
    for chamber in chambers:
        markers = [cell for cell in chamber["cells"] if cell in marker_positions]
        assert len(markers) == 1

    fill_colors = {}
    for pos in marker_positions:
        fill_colors[pos] = _marker_fill_color(grid, pos)

    assert RED in fill_colors.values()
    assert BLUE in fill_colors.values()

    right_red = False
    for chamber in chambers:
        marker = next(cell for cell in chamber["cells"] if cell in marker_positions)
        if fill_colors[marker] == RED and chamber["mean_c"] > 4:
            right_red = True
            break
    assert right_red
    return chambers


def _validate_semantics(grid: np.ndarray, chambers: list[dict]) -> None:
    out = main(grid)
    assert np.array_equal(out[:, 0], grid[:, 0])
    assert np.array_equal(out[:, -1], grid[:, -1])
    assert np.array_equal(out[grid == GREY], grid[grid == GREY])

    marker_positions = {tuple(map(int, p)) for p in np.argwhere(grid == GREEN)}
    changed = int(np.count_nonzero(out != grid))
    assert changed > len(marker_positions)

    for chamber in chambers:
        marker = next(cell for cell in chamber["cells"] if cell in marker_positions)
        expected_color = _marker_fill_color(grid, marker)
        values = {int(out[r, c]) for r, c in chamber["cells"]}
        assert values == {expected_color}


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        h, hr, top_splits, bottom_splits = _choose_layout()
        grid = _build_grid(h, hr, top_splits, bottom_splits)
        chambers = _component_stats(grid)
        try:
            _place_markers(grid, chambers)
            chambers = _validate_structure(grid)
            _validate_semantics(grid, chambers)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid distance-fill chamber puzzle")
