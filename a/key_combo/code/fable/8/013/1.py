from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
Orbit = tuple[Coord, ...]


def _reflection_orbits(shape: tuple[int, int]) -> list[Orbit]:
    height, width = map(int, shape)
    seen: set[frozenset[Coord]] = set()
    orbits: list[Orbit] = []

    for r in range(height):
        for c in range(width):
            orbit = frozenset(
                {
                    (r, c),
                    (height - 1 - r, c),
                    (r, width - 1 - c),
                    (height - 1 - r, width - 1 - c),
                }
            )
            if orbit in seen:
                continue
            seen.add(orbit)
            orbits.append(tuple(sorted(orbit)))

    return orbits


def _try_split_input(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    row_has_fg = np.any(grid != bg, axis=1)
    fg_rows = np.flatnonzero(row_has_fg)
    if fg_rows.size == 0:
        return None

    start = int(fg_rows[0])
    separator = None
    for r in range(start + 1, grid.shape[0]):
        if row_has_fg[r]:
            continue
        if np.any(row_has_fg[r + 1 :]):
            separator = r
            break

    if separator is None:
        return None

    top = grid[start:separator, :]
    coords = np.argwhere(top != bg)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    pattern = top[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    if min(pattern.shape) == 0:
        return None
    if pattern.shape[0] % 2 != 1 or pattern.shape[1] % 2 != 1:
        return None
    if not _visible_rows_and_cols(pattern != bg):
        return None
    debris = grid[separator + 1 :, :]
    return pattern, debris


def _split_input(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    counts = Counter(map(int, grid.ravel()))
    for bg, _count in counts.most_common():
        parts = _try_split_input(grid, int(bg))
        if parts is not None:
            pattern, debris = parts
            return int(bg), pattern, debris
    raise AssertionError("failed to infer background and split the input")


def _debris_size_map(debris: np.ndarray, bg: int) -> dict[int, int]:
    size_to_color: dict[int, int] = {}
    for color in sorted(set(map(int, debris.ravel())) - {bg}):
        count = int(np.count_nonzero(debris == color))
        assert count not in size_to_color, "debris color counts must be unique"
        size_to_color[count] = color
    return size_to_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a doubly symmetric pattern by copying visible orbit colors and using debris counts for fully missing orbits.

    Input structure:
    - The upper part of the grid contains an odd-sized multicolor pattern with horizontal and vertical mirror symmetry.
    - Some cells of that pattern are replaced by the background color; below it, after at least one empty row, lies a sparse debris field.
    - Each debris color appears as isolated pixels, and its pixel count matches one fully erased reflection orbit of size 1, 2, or 4.

    Rule:
    - Crop the upper pattern canvas and group its cells into four-way reflection orbits under horizontal and vertical mirroring.
    - If an orbit still contains visible colored cells, fill the whole orbit with that shared visible color.
    - If an orbit is entirely missing, fill it with the debris color whose number of pixels equals that orbit's size, then discard the debris area.

    Key insight:
    - Partial holes are solved locally by symmetry, but a completely erased orbit loses its color and must be identified from the only remaining invariant: the orbit size encoded by debris counts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, pattern, debris = _split_input(g)
    height, width = map(int, pattern.shape)

    size_to_color = _debris_size_map(debris, bg)
    out = pattern.copy()

    for orbit in _reflection_orbits(pattern.shape):
        visible = {int(pattern[r, c]) for r, c in orbit if int(pattern[r, c]) != bg}
        if visible:
            assert len(visible) == 1, "partially visible orbit must expose a single color"
            fill = visible.pop()
        else:
            orbit_size = len(orbit)
            assert orbit_size in size_to_color, "missing orbit must match a debris count"
            fill = size_to_color[orbit_size]
        for r, c in orbit:
            out[r, c] = fill

    return out


_BG = 0
_PATTERN_SIZE_CHOICES = (7, 9, 11, 13, 15)
_TOP_MARGIN_RANGE = (0, 2)
_SIDE_MARGIN_RANGE = (2, 5)
_GAP_RANGE = (1, 3)
_DEBRIS_HEIGHT_RANGE = (3, 6)
_PATTERN_COLOR_COUNT_RANGE = (3, 5)
_FULL_ORBIT_COUNT_CHOICES = (1, 2, 2, 3, 3)
_PARTIAL_ORBIT_COUNT_RANGE = (2, 6)
_GEN_RETRIES = 400


def _is_inner_orbit(orbit: Orbit, shape: tuple[int, int]) -> bool:
    height, width = map(int, shape)
    return all(0 < r < height - 1 and 0 < c < width - 1 for r, c in orbit)


def _visible_rows_and_cols(mask: np.ndarray) -> bool:
    return bool(np.all(np.any(mask, axis=1))) and bool(np.all(np.any(mask, axis=0)))


def _choose_full_orbits(orbits: list[Orbit], shape: tuple[int, int]) -> list[Orbit]:
    by_size: dict[int, list[Orbit]] = {1: [], 2: [], 4: []}
    for orbit in orbits:
        size = len(orbit)
        if size in by_size and _is_inner_orbit(orbit, shape):
            by_size[size].append(orbit)

    available_sizes = [size for size, cells in by_size.items() if cells]
    assert available_sizes, "expected at least one inner orbit"

    target_count = min(random.choice(_FULL_ORBIT_COUNT_CHOICES), len(available_sizes))
    picked_sizes = random.sample(available_sizes, target_count)
    return [random.choice(by_size[size]) for size in picked_sizes]


def _apply_holes(output: np.ndarray, full_orbits: list[Orbit], partial_orbits: list[Orbit]) -> np.ndarray:
    inp = output.copy()
    for orbit in full_orbits:
        for r, c in orbit:
            inp[r, c] = _BG

    for orbit in partial_orbits:
        cells = list(orbit)
        remove_count = 1 if len(cells) == 2 else random.randint(1, len(cells) - 1)
        for r, c in random.sample(cells, remove_count):
            inp[r, c] = _BG

    return inp


def _place_debris(canvas: np.ndarray, color: int, count: int, top_row: int) -> bool:
    height, width = map(int, canvas.shape)
    placed: list[Coord] = []
    candidates = [(r, c) for r in range(top_row, height) for c in range(width)]
    random.shuffle(candidates)

    for r, c in candidates:
        if int(canvas[r, c]) != _BG:
            continue
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in placed):
            continue
        placed.append((r, c))
        if len(placed) == count:
            for rr, cc in placed:
                canvas[rr, cc] = color
            return True

    return False


def _build_output_pattern(shape: tuple[int, int], colors: list[int]) -> np.ndarray:
    out = np.full(shape, _BG, dtype=int)
    for orbit in _reflection_orbits(shape):
        color = random.choice(colors)
        for r, c in orbit:
            out[r, c] = color
    return out


def _paint_orbit(grid: np.ndarray, orbit: Orbit, color: int) -> None:
    for r, c in orbit:
        grid[r, c] = color


def _gen_attempt() -> np.ndarray | None:
    pattern_h = random.choice(_PATTERN_SIZE_CHOICES)
    pattern_w = random.choice(_PATTERN_SIZE_CHOICES)
    top_margin = random.randint(*_TOP_MARGIN_RANGE)
    left_margin = random.randint(*_SIDE_MARGIN_RANGE)
    right_margin = random.randint(*_SIDE_MARGIN_RANGE)
    gap = random.randint(*_GAP_RANGE)
    debris_h = random.randint(*_DEBRIS_HEIGHT_RANGE)

    total_h = top_margin + pattern_h + gap + debris_h
    total_w = left_margin + pattern_w + right_margin
    if max(total_h, total_w) > 30:
        return None

    palette = list(range(1, 10))
    random.shuffle(palette)
    n_pattern_colors = random.randint(*_PATTERN_COLOR_COUNT_RANGE)
    pattern_colors = palette[:n_pattern_colors]

    output = _build_output_pattern((pattern_h, pattern_w), pattern_colors)
    orbits = _reflection_orbits(output.shape)
    full_orbits = _choose_full_orbits(orbits, output.shape)
    full_set = {orbit for orbit in full_orbits}

    used_colors = set(pattern_colors)
    remaining_colors = [color for color in range(1, 10) if color not in used_colors]
    if len(remaining_colors) < len(full_orbits):
        return None
    debris_colors = random.sample(remaining_colors, len(full_orbits))
    for orbit, color in zip(full_orbits, debris_colors):
        _paint_orbit(output, orbit, color)

    partial_candidates = [orbit for orbit in orbits if len(orbit) > 1 and orbit not in full_set]
    if len(partial_candidates) < _PARTIAL_ORBIT_COUNT_RANGE[0]:
        return None
    partial_count = random.randint(
        _PARTIAL_ORBIT_COUNT_RANGE[0],
        min(_PARTIAL_ORBIT_COUNT_RANGE[1], len(partial_candidates)),
    )
    partial_orbits = random.sample(partial_candidates, partial_count)

    pattern_in = _apply_holes(output, full_orbits, partial_orbits)
    visible_mask = pattern_in != _BG
    if not _visible_rows_and_cols(visible_mask):
        return None

    debris_plan = list(zip(debris_colors, [len(orbit) for orbit in full_orbits]))

    canvas = np.full((total_h, total_w), _BG, dtype=int)
    pattern_r0 = top_margin
    pattern_c0 = left_margin
    canvas[pattern_r0 : pattern_r0 + pattern_h, pattern_c0 : pattern_c0 + pattern_w] = pattern_in

    debris_top = top_margin + pattern_h + gap
    for color, count in debris_plan:
        if not _place_debris(canvas, color, count, debris_top):
            return None

    if not np.array_equal(main(canvas), output):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the symmetry debris task")
