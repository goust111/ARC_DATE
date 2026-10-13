from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PIVOT = 5
MARKER_COLORS = (BLUE, RED, GREEN)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
MIN_MARKER_DIST = 3
MAX_MARKER_DIST = 6
MIN_SIZE = 11
MAX_SIZE = 16
GENERATE_RETRIES = 400


def _cw(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _line_cells(
    pivot: tuple[int, int], direction: tuple[int, int], length: int
) -> list[tuple[int, int]]:
    pr, pc = pivot
    dr, dc = direction
    return [(pr + step * dr, pc + step * dc) for step in range(1, length + 1)]


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cell
    return 0 <= r < h and 0 <= c < w


def _make_spec(
    pivot: tuple[int, int],
    direction: tuple[int, int],
    dist: int,
    color: int | None = None,
) -> dict:
    original_path = _line_cells(pivot, direction, dist)
    rotated_path = _line_cells(pivot, _cw(direction), dist)
    spec = {
        "pivot": pivot,
        "direction": direction,
        "dist": dist,
        "source": original_path[-1],
        "end": original_path[0],
        "original_path": original_path,
        "rotated_path": rotated_path,
        "reserve": set([pivot, *original_path, *rotated_path]),
    }
    if color is not None:
        spec["color"] = int(color)
    return spec


def _candidate_for_direction(
    grid: np.ndarray, pivot: tuple[int, int], direction: tuple[int, int]
) -> dict | None:
    h, w = grid.shape
    pr, pc = pivot
    dr, dc = direction
    step = 1
    while True:
        cell = (pr + step * dr, pc + step * dc)
        if not _in_bounds((h, w), cell):
            return None
        value = int(grid[cell])
        if value == BG:
            step += 1
            continue
        if value in MARKER_COLORS and MIN_MARKER_DIST <= step <= MAX_MARKER_DIST:
            spec = _make_spec(pivot, direction, step, value)
            if all(int(grid[pos]) == BG for pos in spec["rotated_path"]):
                return spec
        return None


def _parse_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    present = set(map(int, np.unique(g)))
    assert present.issubset({BG, BLUE, RED, GREEN, YELLOW, PIVOT})
    assert YELLOW not in present, "input should not contain output-only yellow pixels"

    pivots = sorted(tuple(map(int, rc)) for rc in np.argwhere(g == PIVOT))
    assert 1 <= len(pivots) <= 2

    markers = {
        tuple(map(int, rc))
        for rc in np.argwhere(np.isin(g, np.array(MARKER_COLORS, dtype=int)))
    }
    assert len(markers) == len(pivots)

    objects: list[dict] = []
    used_sources: set[tuple[int, int]] = set()
    for pivot in pivots:
        candidates = [
            spec
            for direction in DIRS
            if (spec := _candidate_for_direction(g, pivot, direction)) is not None
        ]
        assert len(candidates) == 1, f"pivot {pivot} has {len(candidates)} marker candidates"
        spec = candidates[0]
        assert spec["source"] not in used_sources
        used_sources.add(spec["source"])
        objects.append(spec)

    assert used_sources == markers, "every marker must belong to exactly one pivot"
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored marker toward its grey pivot, then paint the clockwise-rotated image of its slide path in yellow.

    Input structure:
    - A black background contains 1 or 2 single-cell grey pivots.
    - Each pivot has exactly one colored singleton marker on the same row or column,
      with 2 to 5 blank cells between the marker and the pivot.
    - The straight sightline from marker to pivot is otherwise empty, and the
      clockwise-rotated copy of that path fits on the grid without conflicts.

    Rule:
    - For each pivot, find the unique aligned colored marker that can slide
      straight toward the pivot.
    - Move that marker to the cell adjacent to the pivot while keeping its color.
    - Record every cell on the original slide path from the marker's start to its
      stopping cell, inclusive.
    - Rotate that whole path 90 degrees clockwise around the pivot and draw the
      rotated cells in yellow.
    - Keep the grey pivots unchanged and process multiple pivots in reading order.

    Key insight:
    - The yellow output is not the marker itself rotated around the pivot; it is
      the entire slide trajectory rotated, so both the marker's direction and its
      travel distance are necessary to determine the result.
    """
    g = np.asarray(grid, dtype=int)
    objects = _parse_objects(g)

    out = g.copy()
    for obj in objects:
        out[obj["source"]] = BG
    for obj in objects:
        assert int(out[obj["end"]]) == BG
        out[obj["end"]] = int(obj["color"])
    for obj in objects:
        for cell in obj["rotated_path"]:
            assert int(out[cell]) == BG
            out[cell] = YELLOW
    return out


def _placement_valid(spec: dict, objects: list[dict], occupied: set[tuple[int, int]]) -> bool:
    if spec["reserve"] & occupied:
        return False

    pivot = spec["pivot"]
    source = spec["source"]
    for obj in objects:
        op = obj["pivot"]
        os = obj["source"]
        if pivot[0] == op[0] or pivot[1] == op[1]:
            return False
        if pivot[0] == os[0] or pivot[1] == os[1]:
            return False
        if source[0] == op[0] or source[1] == op[1]:
            return False
        if source[0] == os[0] or source[1] == os[1]:
            return False
        for a in (pivot, source):
            for b in (op, os):
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) < 3:
                    return False
    return True


def _validate_generated(grid: np.ndarray, expected_count: int) -> bool:
    try:
        objects = _parse_objects(grid)
        out = main(grid)
    except Exception:
        return False

    if len(objects) != expected_count:
        return False
    if int(np.count_nonzero(out == YELLOW)) != sum(obj["dist"] for obj in objects):
        return False

    for obj in objects:
        if int(out[obj["pivot"]]) != PIVOT:
            return False
        if int(out[obj["source"]]) != BG:
            return False
        if int(out[obj["end"]]) != int(obj["color"]):
            return False
        if not all(int(out[cell]) == YELLOW for cell in obj["rotated_path"]):
            return False
        if len(set(obj["original_path"])) != obj["dist"]:
            return False
        if len(set(obj["rotated_path"])) != obj["dist"]:
            return False
        if set(obj["original_path"]) & set(obj["rotated_path"]):
            return False

    return True


def _iter_specs_for_shape(shape: tuple[int, int], direction: tuple[int, int], dist: int):
    h, w = shape
    rows = list(range(h))
    cols = list(range(w))
    random.shuffle(rows)
    random.shuffle(cols)
    for pr in rows:
        for pc in cols:
            spec = _make_spec((pr, pc), direction, dist)
            if all(_in_bounds(shape, cell) for cell in spec["reserve"]):
                yield spec


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_SIZE, MAX_SIZE)
    w = random.randint(MIN_SIZE, MAX_SIZE)
    count = random.choice([1, 2])

    if count == 2 and random.random() < 0.5:
        colors = [random.choice(MARKER_COLORS)] * 2
    else:
        colors = [random.choice(MARKER_COLORS) for _ in range(count)]

    objects: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    for idx in range(count):
        pairs = [
            (direction, dist)
            for direction in DIRS
            for dist in range(MIN_MARKER_DIST, MAX_MARKER_DIST + 1)
        ]
        random.shuffle(pairs)

        placed = False
        for direction, dist in pairs:
            specs = list(_iter_specs_for_shape((h, w), direction, dist))
            random.shuffle(specs)
            for spec in specs:
                spec["color"] = int(colors[idx])
                if _placement_valid(spec, objects, occupied):
                    objects.append(spec)
                    occupied |= spec["reserve"]
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None

    grid = np.zeros((h, w), dtype=int)
    for obj in objects:
        grid[obj["pivot"]] = PIVOT
        grid[obj["source"]] = int(obj["color"])

    if not _validate_generated(grid, count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
