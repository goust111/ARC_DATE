import random
from typing import Iterable

import numpy as np


BLACK = 0
GRAY = 5
BAND_PALETTE = (1, 2, 3, 4)
OBSTACLE_PALETTE = (6, 8, 9)

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

_BASE_ARROW_SHAPES = {
    "right": (
        frozenset({(1, 0), (1, 1), (1, 2), (0, 3), (2, 3)}),
        frozenset({(1, 0), (1, 1), (1, 2), (0, 2), (2, 2)}),
    ),
    "left": (
        frozenset({(1, 1), (1, 2), (1, 3), (0, 0), (2, 0)}),
        frozenset({(1, 0), (1, 1), (1, 2), (0, 0), (2, 0)}),
    ),
    "up": (
        frozenset({(1, 1), (2, 1), (3, 1), (0, 0), (0, 2)}),
        frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
    ),
    "down": (
        frozenset({(0, 1), (1, 1), (2, 1), (3, 0), (3, 2)}),
        frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}),
    ),
}

ARROW_SHAPES: dict[frozenset[tuple[int, int]], str] = {}
for _direction, _variants in _BASE_ARROW_SHAPES.items():
    for _shape in _variants:
        ARROW_SHAPES[_shape] = _direction

GEN_RETRIES = 400
ARROW_PLACE_RETRIES = 180
OBSTACLE_PLACE_RETRIES = 80


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch each gray arrow straight ahead and leave a band-colored swept trail behind it.

    Input structure:
    - The background is black, with several full-width non-black band rows that partition
      the canvas into colored horizontal regions.
    - Gray arrows are 8-connected five-cell sprites: a length-3 shaft plus a two-pixel
      head that points up, down, left, or right.
    - Other non-black non-gray cells form solid rectangular blockers that never move.

    Rule:
    - Treat every full-width non-black row as a color band and every valid gray arrow as
      a movable object; process arrows in reading order so earlier trails become blockers.
    - Translate each whole arrow one cell at a time along its pointing direction until
      the next translated position would leave the grid or touch any non-black cell.
    - Color the swept cells vacated during that motion, excluding the arrow's final cells,
      with the color of the nearest band row at the same row index.
    - Leave each arrow gray at its resting place and clear any original gray cells that
      were not covered by the swept trail.

    Key insight:
    - The arrows move rigidly, but the trail does not inherit the arrow color; instead,
      each vacated pixel samples the horizontal band region it lies in, so long motions
      can change color as they pass between different band-row domains.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    band_rows = _find_band_rows(g)
    arrows = _find_arrows(g)

    state = g.copy()
    for arrow in arrows:
        final_cells, trail_cells, _steps = _simulate_arrow(state, arrow["cells"], arrow["direction"])
        for r, c in arrow["cells"]:
            state[r, c] = BLACK
        for r, c in trail_cells:
            state[r, c] = _nearest_band_color(band_rows, r)
        for r, c in final_cells:
            state[r, c] = GRAY
    return state


def _find_band_rows(grid: np.ndarray) -> list[tuple[int, int]]:
    bands = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if row[0] != BLACK and np.all(row == row[0]):
            bands.append((r, int(row[0])))
    assert bands, "expected at least one full-width band row"
    return bands


def _find_arrows(grid: np.ndarray) -> list[dict[str, object]]:
    mask = grid == GRAY
    components = _connected_components8(mask)
    arrows = []
    seen_cells = 0
    for component in components:
        if len(component) != 5:
            continue
        norm, bbox = _normalize(component)
        direction = ARROW_SHAPES.get(frozenset(norm))
        if direction is None:
            continue
        cells = frozenset(component)
        seen_cells += len(cells)
        arrows.append(
            {
                "cells": cells,
                "direction": direction,
                "bbox": bbox,
            }
        )
    assert arrows, "expected at least one gray arrow"
    assert seen_cells == int(np.count_nonzero(mask)), "all gray pixels must belong to arrows"
    arrows.sort(key=lambda arrow: (arrow["bbox"][0], arrow["bbox"][1]))
    return arrows


def _normalize(coords: Iterable[tuple[int, int]]) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    norm = sorted((r - r0, c - c0) for r, c in coords)
    return norm, (r0, c0, r1, c1)


def _connected_components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr in range(max(0, cr - 1), min(h, cr + 2)):
                    for nc in range(max(0, cc - 1), min(w, cc + 2)):
                        if (nr == cr and nc == cc) or seen[nr, nc] or not mask[nr, nc]:
                            continue
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _simulate_arrow(
    state: np.ndarray,
    start_cells: Iterable[tuple[int, int]],
    direction: str,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], int]:
    h, w = state.shape
    blocked = state.copy()
    current = set(start_cells)
    for r, c in current:
        blocked[r, c] = BLACK

    dr, dc = DIRS[direction]
    history: list[set[tuple[int, int]]] = []
    steps = 0
    while True:
        nxt = {(r + dr, c + dc) for r, c in current}
        if any(r < 0 or r >= h or c < 0 or c >= w or blocked[r, c] != BLACK for r, c in nxt):
            break
        history.append(set(current))
        current = nxt
        steps += 1

    swept = set()
    for coords in history:
        swept |= coords
    swept -= current
    return current, swept, steps


def _nearest_band_color(bands: list[tuple[int, int]], row: int) -> int:
    band_row, band_color = min(bands, key=lambda item: (abs(item[0] - row), item[0]))
    _ = band_row
    return band_color


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape == grid.shape and not np.array_equal(out, grid):
            return grid
    raise RuntimeError("failed to generate a valid arrow-band sample")


def _gen_attempt() -> np.ndarray | None:
    band_rows = _make_band_rows()
    h = band_rows[-1][0] + random.randint(2, 5)
    if h > 30:
        return None
    w = random.randint(14, 24)
    grid = np.zeros((h, w), dtype=int)

    for row, color in band_rows:
        grid[row, :] = color

    if not _place_obstacles(grid, band_rows):
        return None

    arrow_count = random.randint(2, 4)
    directions = _choose_directions(arrow_count)
    reserved = grid != BLACK
    arrow_cells: set[tuple[int, int]] = set()
    step_counts = []
    placed_vertical = 0
    placed_horizontal = 0

    for direction in directions:
        placed = False
        for _ in range(ARROW_PLACE_RETRIES):
            cells = _random_arrow_cells(grid.shape, direction)
            if cells is None:
                continue
            if any(reserved[r, c] for r, c in cells):
                continue
            if _touches_existing_arrow(cells, arrow_cells):
                continue

            probe = grid.copy()
            probe[(reserved) & (probe == BLACK)] = BAND_PALETTE[0]
            for r, c in cells:
                probe[r, c] = GRAY
            final_cells, trail_cells, steps = _simulate_arrow(probe, cells, direction)
            if steps < 1:
                continue
            footprint = set(cells) | final_cells | trail_cells
            if any(reserved[r, c] for r, c in footprint if (r, c) not in cells):
                continue
            if _touches_existing_arrow(footprint, arrow_cells):
                continue

            for r, c in cells:
                grid[r, c] = GRAY
                reserved[r, c] = True
                arrow_cells.add((r, c))
            for r, c in trail_cells | final_cells:
                reserved[r, c] = True

            step_counts.append(steps)
            if direction in ("up", "down"):
                placed_vertical += 1
            else:
                placed_horizontal += 1
            placed = True
            break
        if not placed:
            return None

    if placed_vertical == 0 or placed_horizontal == 0:
        return None
    if max(step_counts) < 2:
        return None
    return grid


def _make_band_rows() -> list[tuple[int, int]]:
    band_count = random.randint(2, 4)
    top_margin = random.randint(1, 4)
    gaps = [random.choice((3, 5)) for _ in range(band_count - 1)]
    rows = [top_margin]
    for gap in gaps:
        rows.append(rows[-1] + gap)
    colors = random.sample(BAND_PALETTE, band_count)
    return list(zip(rows, colors))


def _place_obstacles(grid: np.ndarray, band_rows: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    band_set = {row for row, _color in band_rows}
    occupied = grid != BLACK
    obstacle_count = random.randint(0, 3)
    for _ in range(obstacle_count):
        placed = False
        for _ in range(OBSTACLE_PLACE_RETRIES):
            rect_h = random.randint(1, 3)
            rect_w = random.randint(2, 6)
            if rect_h >= h or rect_w >= w - 1:
                continue
            r0 = random.randint(0, h - rect_h)
            c0 = random.randint(0, w - rect_w)
            rows = range(r0, r0 + rect_h)
            cols = range(c0, c0 + rect_w)
            if any(row in band_set for row in rows):
                continue
            if r0 == 0 or r0 + rect_h == h:
                continue
            if c0 == 0 or c0 + rect_w == w:
                continue
            r1 = r0 + rect_h
            c1 = c0 + rect_w
            pr0 = max(0, r0 - 1)
            pr1 = min(h, r1 + 1)
            pc0 = max(0, c0 - 1)
            pc1 = min(w, c1 + 1)
            if np.any(occupied[pr0:pr1, pc0:pc1]):
                continue
            color = random.choice(OBSTACLE_PALETTE)
            grid[r0:r1, c0:c1] = color
            occupied[pr0:pr1, pc0:pc1] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _choose_directions(count: int) -> list[str]:
    directions = [random.choice(("left", "right")), random.choice(("up", "down"))]
    while len(directions) < count:
        directions.append(random.choice(tuple(DIRS)))
    random.shuffle(directions)
    return directions


def _random_arrow_cells(shape: tuple[int, int], direction: str) -> set[tuple[int, int]] | None:
    h, w = shape
    template = random.choice(_BASE_ARROW_SHAPES[direction])
    rows = [r for r, _c in template]
    cols = [c for _r, c in template]
    box_h = max(rows) + 1
    box_w = max(cols) + 1
    if box_h > h or box_w > w:
        return None
    r0 = random.randint(0, h - box_h)
    c0 = random.randint(0, w - box_w)
    return {(r0 + r, c0 + c) for r, c in template}


def _touches_existing_arrow(
    cells: Iterable[tuple[int, int]],
    existing: set[tuple[int, int]],
) -> bool:
    if not existing:
        return False
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in existing:
                    return True
    return False
