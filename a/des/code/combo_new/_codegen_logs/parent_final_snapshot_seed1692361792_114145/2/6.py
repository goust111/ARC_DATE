from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 400


def _source_direction(row: int, col: int, height: int, width: int) -> tuple[int, int]:
    if row == 0 and 0 < col < width - 1:
        return (1, 0)
    if row == height - 1 and 0 < col < width - 1:
        return (-1, 0)
    if col == 0 and 0 < row < height - 1:
        return (0, 1)
    if col == width - 1 and 0 < row < height - 1:
        return (0, -1)
    raise AssertionError("sources must be non-corner edge pixels")


def _find_sources(grid: np.ndarray) -> list[tuple[int, int]]:
    height, width = map(int, grid.shape)
    sources: list[tuple[int, int]] = []
    for row in range(height):
        for col in range(width):
            if int(grid[row, col]) == BG:
                continue
            if row in {0, height - 1} or col in {0, width - 1}:
                sources.append((int(row), int(col)))
    sources.sort()
    return sources


def _find_mirrors(
    grid: np.ndarray, sources: list[tuple[int, int]]
) -> dict[tuple[int, int], tuple[str, int]]:
    height, width = map(int, grid.shape)
    source_set = set(sources)
    seen: set[tuple[int, int]] = set()
    lookup: dict[tuple[int, int], tuple[str, int]] = {}

    for row in range(1, height - 1):
        for col in range(1, width - 1):
            if int(grid[row, col]) == BG or (row, col) in source_set or (row, col) in seen:
                continue

            color = int(grid[row, col])
            q = deque([(row, col)])
            coords: list[tuple[int, int]] = []
            seen.add((row, col))
            while q:
                cur_row, cur_col = q.popleft()
                coords.append((int(cur_row), int(cur_col)))
                for d_row, d_col in NEIGHBORS4:
                    nxt = (int(cur_row + d_row), int(cur_col + d_col))
                    if nxt in seen:
                        continue
                    nr, nc = nxt
                    if not (1 <= nr < height - 1 and 1 <= nc < width - 1):
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen.add(nxt)
                    q.append(nxt)

            rows = sorted({r for r, _c in coords})
            cols = sorted({c for _r, c in coords})
            assert len(coords) >= 2, "mirror segments must have length at least 2"
            if len(rows) == 1:
                assert cols == list(range(cols[0], cols[-1] + 1)), "horizontal mirror must be contiguous"
                orientation = "h"
            else:
                assert len(cols) == 1, "mirror segments must be axis-aligned"
                assert rows == list(range(rows[0], rows[-1] + 1)), "vertical mirror must be contiguous"
                orientation = "v"

            for coord in coords:
                lookup[coord] = (orientation, color)

    return lookup


def _trace_ray(
    grid: np.ndarray,
    out: np.ndarray,
    start: tuple[int, int],
    mirror_lookup: dict[tuple[int, int], tuple[str, int]],
) -> None:
    height, width = map(int, grid.shape)
    row, col = start
    d_row, d_col = _source_direction(row, col, height, width)
    color = int(grid[row, col])

    for _ in range(height * width * 4):
        nxt_row = int(row + d_row)
        nxt_col = int(col + d_col)
        if not (0 <= nxt_row < height and 0 <= nxt_col < width):
            return

        mirror = mirror_lookup.get((nxt_row, nxt_col))
        if mirror is None:
            if int(grid[nxt_row, nxt_col]) == BG:
                out[nxt_row, nxt_col] = color
        else:
            orientation, mirror_color = mirror
            if (d_col != 0 and orientation == "v") or (d_row != 0 and orientation == "h"):
                d_row, d_col = -d_row, -d_col
                color = int(mirror_color)

        row, col = nxt_row, nxt_col

    raise AssertionError("ray simulation exceeded step limit")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace edge-emitted orthogonal light rays through colored horizontal and vertical mirror segments.

    Input structure:
    - The background is black.
    - One grid edge contains several single-pixel colored light sources.
    - Inside the grid, non-background pixels form short monochrome horizontal or vertical mirror segments.

    Rule:
    - Process sources in top-to-bottom, then left-to-right order.
    - Each source emits a ray straight inward from its edge location and paints every traversed background pixel with the ray's current color.
    - Mirror pixels themselves never change; a ray passes straight through a parallel mirror, but on hitting a perpendicular mirror it reverses direction and adopts that mirror segment's color.
    - Continue until the ray leaves the grid; later rays overwrite earlier trace colors on background cells they also traverse.

    Key insight:
    - The visible output is obtained by simulating each edge source independently: edge position determines the launch direction, mirror orientation determines whether the ray passes or bounces, and mirror color determines the post-reflection trace color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sources = _find_sources(g)
    mirrors = _find_mirrors(g, sources)
    out = g.copy()
    for source in sources:
        _trace_ray(g, out, source, mirrors)
    return out


def _sample_positions(axis_size: int, count: int) -> list[int]:
    candidates = list(range(1, axis_size - 1))
    random.shuffle(candidates)
    picked: list[int] = []
    for value in candidates:
        if all(abs(value - prev) >= 2 for prev in picked):
            picked.append(value)
            if len(picked) == count:
                return sorted(picked)
    return sorted(random.sample(list(range(1, axis_size - 1)), count))


def _to_cell(side: str, travel: int, orth: int, height: int, width: int) -> tuple[int, int]:
    if side == "left":
        return (int(orth), int(travel))
    if side == "right":
        return (int(orth), int(width - 1 - travel))
    if side == "top":
        return (int(travel), int(orth))
    return (int(height - 1 - travel), int(orth))


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(["left", "right", "top", "bottom"])
    height = random.randint(10, 28)
    width = random.randint(10, 28)
    grid = np.full((height, width), BG, dtype=int)

    if side in {"left", "right"}:
        source_axis_size = height
        travel_limit = width - 1
    else:
        source_axis_size = width
        travel_limit = height - 1

    max_sources = min(4, source_axis_size - 2)
    if max_sources < 2 or travel_limit < 6:
        return None

    source_count = random.randint(2, max_sources)
    source_positions = _sample_positions(source_axis_size, source_count)
    source_colors = random.sample(list(COLORS), source_count)
    source_set: set[tuple[int, int]] = set()
    for orth, color in zip(source_positions, source_colors):
        row, col = _to_cell(side, 0, orth, height, width)
        grid[row, col] = int(color)
        source_set.add((row, col))

    occupied: set[tuple[int, int]] = set()

    def can_place(cells: list[tuple[int, int]]) -> bool:
        if not cells:
            return False
        cell_set = set(cells)
        if len(cell_set) != len(cells):
            return False
        for row, col in cell_set:
            if not (1 <= row < height - 1 and 1 <= col < width - 1):
                return False
            if (row, col) in source_set or (row, col) in occupied:
                return False
            for d_row, d_col in NEIGHBORS4:
                neighbor = (int(row + d_row), int(col + d_col))
                if neighbor in occupied and neighbor not in cell_set:
                    return False
        return True

    def place_segment(cells: list[tuple[int, int]], color: int) -> None:
        for row, col in cells:
            grid[row, col] = int(color)
            occupied.add((row, col))

    def choose_color(*avoid: int) -> int:
        choices = [color for color in COLORS if color not in set(map(int, avoid))]
        if not choices:
            choices = list(COLORS)
        return int(random.choice(choices))

    def parallel_cells(orth: int, start: int, length: int) -> list[tuple[int, int]]:
        return [_to_cell(side, travel, orth, height, width) for travel in range(start, start + length)]

    def perpendicular_cells(travel: int, start_orth: int, length: int) -> list[tuple[int, int]]:
        return [_to_cell(side, travel, orth, height, width) for orth in range(start_orth, start_orth + length)]

    def place_parallel(orth: int, lo: int, hi: int, color: int) -> bool:
        candidates: list[list[tuple[int, int]]] = []
        for length in (2, 3, 4):
            for start in range(lo, hi - length + 2):
                cells = parallel_cells(orth, start, length)
                if can_place(cells):
                    candidates.append(cells)
        if not candidates:
            return False
        cells = random.choice(candidates)
        place_segment(cells, color)
        return True

    def place_perpendicular(orth: int, travel: int, color: int) -> bool:
        other_positions = [value for value in source_positions if value != orth]
        candidates: list[list[tuple[int, int]]] = []
        orth_limit = source_axis_size - 1
        for length in (2, 3, 4):
            lo = max(1, orth - length + 1)
            hi = min(orth, orth_limit - length)
            for start_orth in range(lo, hi + 1):
                span = range(start_orth, start_orth + length)
                if any(value in span for value in other_positions):
                    continue
                cells = perpendicular_cells(travel, start_orth, length)
                if can_place(cells):
                    candidates.append(cells)
        if not candidates:
            return False
        cells = random.choice(candidates)
        place_segment(cells, color)
        return True

    all_cases = ["reflect_only", "pass_only", "pass_then_reflect", "empty"]
    cases = [random.choice(["reflect_only", "pass_then_reflect"]), random.choice(["pass_only", "pass_then_reflect"])]
    while len(cases) < source_count:
        cases.append(random.choice(all_cases))
    random.shuffle(cases)

    source_specs = [
        {"orth": orth, "source_color": color, "case": case}
        for orth, color, case in zip(source_positions, source_colors, cases)
    ]
    source_specs.sort(key=lambda spec: {"pass_then_reflect": 0, "reflect_only": 1, "pass_only": 2, "empty": 3}[spec["case"]])

    for spec in source_specs:
        orth = int(spec["orth"])
        source_color = int(spec["source_color"])
        case = str(spec["case"])

        if case == "empty":
            continue

        if case == "reflect_only":
            travel = random.randint(3, travel_limit - 1)
            if not place_perpendicular(orth, travel, choose_color(source_color)):
                return None
            continue

        if case == "pass_only":
            seg_count = random.randint(1, 2)
            mirror_color = choose_color(source_color)
            for _ in range(seg_count):
                if not place_parallel(orth, 2, travel_limit - 1, mirror_color):
                    return None
            continue

        reflect_travel = random.randint(5, travel_limit - 1)
        mirror_color = choose_color(source_color)
        parallel_color = choose_color(source_color, mirror_color)
        parallel_count = random.randint(1, 2)
        for _ in range(parallel_count):
            if not place_parallel(orth, 2, reflect_travel - 2, parallel_color):
                return None
        if not place_perpendicular(orth, reflect_travel, mirror_color):
            return None

    def on_source_line(row: int, col: int) -> bool:
        return int(row) in source_positions if side in {"left", "right"} else int(col) in source_positions

    decoy_count = random.randint(1, 3)
    for _ in range(decoy_count):
        placed = False
        for _ in range(30):
            orientation = random.choice(["h", "v"])
            length = random.randint(2, 4)
            color = choose_color()
            if orientation == "h":
                row = random.randint(1, height - 2)
                start_col = random.randint(1, width - 1 - length)
                cells = [(int(row), int(col)) for col in range(start_col, start_col + length)]
            else:
                col = random.randint(1, width - 2)
                start_row = random.randint(1, height - 1 - length)
                cells = [(int(row), int(col)) for row in range(start_row, start_row + length)]

            if any(on_source_line(row, col) for row, col in cells):
                continue
            if not can_place(cells):
                continue
            place_segment(cells, color)
            placed = True
            break
        if not placed:
            continue

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
