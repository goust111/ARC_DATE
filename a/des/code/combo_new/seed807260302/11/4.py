from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
OBJECT_COLOR = 5
LINE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 120

Coord = tuple[int, int]
Line = tuple[str, int, int]

_FALLBACK_SHAPES = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
)


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords.sort()
            comps.append(coords)
    return comps


def _find_mirror_lines(grid: np.ndarray, bg: int) -> tuple[list[Line], np.ndarray]:
    h, w = grid.shape
    lines: list[Line] = []
    mask = np.zeros_like(grid, dtype=bool)

    for r in range(h):
        row = grid[r, :]
        if np.all(row == row[0]) and int(row[0]) != bg:
            color = int(row[0])
            lines.append(("h", r, color))
            mask[r, :] = True

    for c in range(w):
        col = grid[:, c]
        if np.all(col == col[0]) and int(col[0]) != bg:
            color = int(col[0])
            lines.append(("v", c, color))
            mask[:, c] = True

    assert lines, "expected at least one full-width or full-height mirror line"
    return lines, mask


def _line_distance(coords: list[Coord], line: Line) -> int:
    axis, index, _color = line
    if axis == "h":
        return min(abs(r - index) for r, _c in coords)
    return min(abs(c - index) for _r, c in coords)


def _nearest_line(coords: list[Coord], lines: list[Line]) -> Line:
    distances = [(_line_distance(coords, line), i) for i, line in enumerate(lines)]
    best = min(distance for distance, _i in distances)
    matches = [lines[i] for distance, i in distances if distance == best]
    assert len(matches) == 1, "each object must have a unique nearest mirror line"
    return matches[0]


def _reflect_coords(coords: list[Coord], line: Line) -> list[Coord]:
    axis, index, _color = line
    if axis == "h":
        return [(2 * index - r, c) for r, c in coords]
    return [(r, 2 * index - c) for r, c in coords]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _background_color(g)

    lines, line_mask = _find_mirror_lines(g, bg)
    object_mask = (g != bg) & (~line_mask)
    objects = _components(object_mask)
    assert objects, "expected at least one off-line object"

    out = g.copy()
    for coords in objects:
        assert 3 <= len(coords) <= 6, "objects must be small 4-connected blobs"
        source_colors = {int(g[r, c]) for r, c in coords}
        assert len(source_colors) == 1, "each source object must be monochrome"
        line = _nearest_line(coords, lines)
        for rr, cc in _reflect_coords(coords, line):
            assert 0 <= rr < h and 0 <= cc < w, "reflected copy must stay inside the grid"
            if out[rr, cc] == bg:
                out[rr, cc] = line[2]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each small object across its nearest full-grid mirror line and color the copy with that line.

    Input structure:
    - The background is black.
    - Several single-pixel-thick colored lines span the full width or full height of the grid.
    - Away from those lines are several small 4-connected monochrome objects of 3 to 6 pixels.

    Rule:
    - Detect every full-width or full-height colored line as a mirror line.
    - For each off-line object, find the unique nearest mirror line by perpendicular distance.
    - Reflect the object's shape across that line and paint the reflected copy with the line's color, while leaving the original object and all lines unchanged.
    - If a reflected pixel lands on an already occupied foreground cell, skip that pixel instead of overwriting it.

    Key insight:
    - The line color serves both as the ownership cue for the nearest-line match and as the color used to draw the mirrored copy.
    """
    return _solve(grid)


def _normalize_shape(cells: set[Coord]) -> tuple[Coord, ...]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _transform_shape(shape: tuple[Coord, ...]) -> tuple[Coord, ...]:
    coords = list(shape)
    rotations = random.randrange(4)
    flip = random.random() < 0.5
    transformed: list[Coord] = []
    for r, c in coords:
        rr, cc = r, c
        for _ in range(rotations):
            rr, cc = cc, -rr
        if flip:
            cc = -cc
        transformed.append((rr, cc))
    return _normalize_shape(set(transformed))


def _shape_dims(shape: tuple[Coord, ...]) -> tuple[int, int]:
    return 1 + max(r for r, _c in shape), 1 + max(c for _r, c in shape)


def _random_irregular_shape() -> tuple[Coord, ...]:
    for _ in range(80):
        target = random.randint(3, 6)
        cells: set[Coord] = {(0, 0)}
        while len(cells) < target:
            frontier: list[Coord] = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells and -3 <= nr <= 3 and -3 <= nc <= 3:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        shape = _transform_shape(_normalize_shape(cells))
        sh, sw = _shape_dims(shape)
        if sh == 1 or sw == 1:
            continue
        if sh * sw == len(shape):
            continue
        return shape

    return _transform_shape(random.choice(_FALLBACK_SHAPES))


def _sample_spaced_positions(length: int, count: int, margin: int, gap: int) -> list[int] | None:
    choices = list(range(margin, length - margin))
    if len(choices) < count:
        return None
    for _ in range(120):
        positions = sorted(random.sample(choices, count))
        if all(positions[i + 1] - positions[i] >= gap for i in range(len(positions) - 1)):
            return positions
    return None


def _touches_object(coords: list[Coord], occupied: set[Coord]) -> bool:
    for r, c in coords:
        if (r, c) in occupied:
            return True
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return True
    return False


def _valid_anchor_ranges(
    shape: tuple[Coord, ...],
    line: Line,
    grid_shape: tuple[int, int],
    side: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = grid_shape
    sh, sw = _shape_dims(shape)
    axis, index, _color = line

    if axis == "h":
        if side < 0:
            r_min = max(0, 2 * index - (h - 1))
            r_max = index - sh
        else:
            r_min = index + 1
            r_max = min(h - sh, 2 * index - (sh - 1))
        c_min, c_max = 0, w - sw
    else:
        if side < 0:
            c_min = max(0, 2 * index - (w - 1))
            c_max = index - sw
        else:
            c_min = index + 1
            c_max = min(w - sw, 2 * index - (sw - 1))
        r_min, r_max = 0, h - sh

    if r_min > r_max or c_min > c_max:
        return None
    return (r_min, r_max), (c_min, c_max)


def _placed_coords(shape: tuple[Coord, ...], r0: int, c0: int) -> list[Coord]:
    return [(r0 + r, c0 + c) for r, c in shape]


def _try_place_object(
    grid: np.ndarray,
    lines: list[Line],
    occupied_objects: set[Coord],
    reserved_copies: set[Coord],
    records: list[tuple[int, int]],
) -> bool:
    h, w = grid.shape

    for _ in range(PLACE_ATTEMPTS):
        shape = _random_irregular_shape()
        target_idx = random.randrange(len(lines))
        side = random.choice((-1, 1))
        ranges = _valid_anchor_ranges(shape, lines[target_idx], (h, w), side)
        if ranges is None:
            continue
        (r_min, r_max), (c_min, c_max) = ranges
        r0 = random.randint(r_min, r_max)
        c0 = random.randint(c_min, c_max)
        coords = _placed_coords(shape, r0, c0)

        if any(grid[r, c] != BG_COLOR for r, c in coords):
            continue
        if any((r, c) in reserved_copies for r, c in coords):
            continue
        if _touches_object(coords, occupied_objects):
            continue

        try:
            line = _nearest_line(coords, lines)
        except AssertionError:
            continue
        if line != lines[target_idx]:
            continue

        reflected = _reflect_coords(coords, line)
        if any(not (0 <= r < h and 0 <= c < w) for r, c in reflected):
            continue
        if any(grid[r, c] != BG_COLOR for r, c in reflected):
            continue
        if any((r, c) in occupied_objects or (r, c) in reserved_copies for r, c in reflected):
            continue

        for r, c in coords:
            grid[r, c] = OBJECT_COLOR
            occupied_objects.add((r, c))
        reserved_copies.update(reflected)
        records.append((target_idx, side))
        return True

    return False


def _build_case() -> np.ndarray | None:
    orientation = random.choice(("h", "v"))
    h = random.randint(12, 24)
    w = random.randint(12, 24)
    axis_len = h if orientation == "h" else w
    max_lines = 3 if axis_len >= 17 else 2
    line_count = random.randint(2, max_lines)
    positions = _sample_spaced_positions(axis_len, line_count, margin=3, gap=5)
    if positions is None:
        return None

    line_colors = random.sample(LINE_COLORS, line_count)
    grid = np.full((h, w), BG_COLOR, dtype=int)
    lines: list[Line] = []
    for pos, color in zip(positions, line_colors):
        if orientation == "h":
            grid[pos, :] = color
            lines.append(("h", pos, color))
        else:
            grid[:, pos] = color
            lines.append(("v", pos, color))

    occupied_objects: set[Coord] = set()
    reserved_copies: set[Coord] = set()
    records: list[tuple[int, int]] = []

    object_count = random.randint(max(3, line_count), min(5, line_count + 2))
    for _ in range(object_count):
        if not _try_place_object(grid, lines, occupied_objects, reserved_copies, records):
            return None

    used_lines = {idx for idx, _side in records}
    used_sides = {side for _idx, side in records}
    if len(used_lines) < min(2, line_count):
        return None
    if used_sides != {-1, 1}:
        return None

    objects = _components(grid == OBJECT_COLOR)
    if len(objects) != object_count:
        return None
    try:
        if any(_nearest_line(coords, lines)[0] != orientation for coords in objects):
            return None
    except AssertionError:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-line ARC input")
