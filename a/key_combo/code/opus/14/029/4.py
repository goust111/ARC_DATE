from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

GRID_HEIGHT_RANGE = (10, 14)
GRID_WIDTH_RANGE = (10, 14)
BLUE_SEGMENT_LENGTHS = (2, 3, 4)
RED_COMPONENT_COUNT_RANGE = (2, 5)
RED_COMPONENT_AREA_RANGE = (2, 5)
MAX_GENERATION_ATTEMPTS = 400


def _border_cycle(shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    coords: list[tuple[int, int]] = []
    for c in range(width):
        coords.append((0, c))
    for r in range(1, height):
        coords.append((r, width - 1))
    for c in range(width - 2, -1, -1):
        coords.append((height - 1, c))
    for r in range(height - 2, 0, -1):
        coords.append((r, 0))
    return coords


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _count_red_components(grid: np.ndarray) -> list[np.ndarray]:
    assert not np.any(grid[0, :] == RED)
    assert not np.any(grid[-1, :] == RED)
    assert not np.any(grid[:, 0] == RED)
    assert not np.any(grid[:, -1] == RED)
    return _component_coords(grid[1:-1, 1:-1] == RED)


def _blue_segment_start_and_length(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    blue_coords = [tuple(map(int, coord)) for coord in np.argwhere(grid == BLUE)]
    assert blue_coords, "expected a blue border segment"
    assert not np.any(grid[1:-1, 1:-1] == BLUE), "blue pixels must lie on the border"
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    assert not any(coord in corners for coord in blue_coords), "blue segment must not touch corners"

    side_cells = {
        "top": [c for c in range(1, w - 1) if grid[0, c] == BLUE],
        "right": [r for r in range(1, h - 1) if grid[r, w - 1] == BLUE],
        "bottom": [c for c in range(1, w - 1) if grid[h - 1, c] == BLUE],
        "left": [r for r in range(1, h - 1) if grid[r, 0] == BLUE],
    }
    populated = [name for name, values in side_cells.items() if values]
    assert len(populated) == 1, "blue pixels must occupy exactly one non-corner border side"

    values = side_cells[populated[0]]
    assert values == list(range(values[0], values[0] + len(values))), "blue border segment must be contiguous"

    border = _border_cycle(grid.shape)
    border_index = {coord: idx for idx, coord in enumerate(border)}
    indices = sorted(border_index[coord] for coord in blue_coords)
    assert indices == list(range(indices[0], indices[0] + len(indices))), "blue segment must be one path run"
    return int(indices[0]), int(len(indices))


def _tile_blue_border(grid: np.ndarray, start: int, seg_len: int, gap: int) -> np.ndarray:
    border = _border_cycle(grid.shape)
    perimeter = len(border)
    stride = int(seg_len + gap)
    assert perimeter % stride == 0, "segment length plus gap must tile the full border"
    out = grid.copy()
    for offset in range(0, perimeter, stride):
        for j in range(seg_len):
            r, c = border[(start + offset + j) % perimeter]
            out[r, c] = BLUE
    return out


def _expected_output(grid: np.ndarray) -> np.ndarray:
    start, seg_len = _blue_segment_start_and_length(grid)
    red_components = _count_red_components(grid)
    return _tile_blue_border(grid, start, seg_len, len(red_components))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat the blue border segment around the perimeter with a gap set by the number of interior red components.

    Input structure:
    - The background is black.
    - Exactly one contiguous blue segment lies on a single border side and does not touch corners.
    - The interior contains several red 4-connected components, while all non-red, non-blue cells stay black.

    Rule:
    - Count the red 4-connected components strictly inside the border and call that count K.
    - Unwrap the rectangular border clockwise into a cyclic path and locate the blue segment's start and length n.
    - Stamp the same blue segment every n + K border positions, so each copy is followed by a black gap of length K.
    - Keep every interior red component unchanged.

    Key insight:
    - The border periodicity is not visible from the border alone: the interior red component count is the gap parameter that determines the full tiling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert min(map(int, g.shape)) >= 4, "grid must have a non-empty interior"
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, RED}), "unexpected colors present"

    start, seg_len = _blue_segment_start_and_length(g)
    assert 2 <= seg_len <= 4, "blue segment length must be in [2, 4]"

    red_components = _count_red_components(g)
    assert RED_COMPONENT_COUNT_RANGE[0] <= len(red_components) <= RED_COMPONENT_COUNT_RANGE[1]
    for comp in red_components:
        assert RED_COMPONENT_AREA_RANGE[0] <= len(comp) <= RED_COMPONENT_AREA_RANGE[1]

    return _tile_blue_border(g, start, seg_len, len(red_components))


def _normalize_shape(cells: set[tuple[int, int]]) -> np.ndarray:
    arr = np.array(sorted(cells), dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _random_polyomino(area: int, max_h: int, max_w: int) -> np.ndarray | None:
    cells = {(0, 0)}
    while len(cells) < int(area):
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in cells:
                    continue
                trial = _normalize_shape(cells | {(nr, nc)})
                h = int(trial[:, 0].max()) + 1
                w = int(trial[:, 1].max()) + 1
                if h <= max_h and w <= max_w:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
        cells = set(map(tuple, _normalize_shape(cells).tolist()))
    return _normalize_shape(cells)


def _candidate_component_positions(
    occupied: set[tuple[int, int]],
    interior_h: int,
    interior_w: int,
    shape: np.ndarray,
    *,
    require_diagonal_touch: bool,
) -> list[set[tuple[int, int]]]:
    h = int(shape[:, 0].max()) + 1
    w = int(shape[:, 1].max()) + 1
    rel = [tuple(map(int, cell)) for cell in shape]
    candidates: list[set[tuple[int, int]]] = []
    for r0 in range(interior_h - h + 1):
        for c0 in range(interior_w - w + 1):
            cells = {(r0 + r, c0 + c) for r, c in rel}
            if cells & occupied:
                continue

            has_four_contact = False
            has_diagonal_touch = False
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in occupied and (nr, nc) not in cells:
                        has_four_contact = True
                        break
                if has_four_contact:
                    break
                for dr in (-1, 1):
                    for dc in (-1, 1):
                        if (r + dr, c + dc) in occupied:
                            has_diagonal_touch = True
            if has_four_contact:
                continue
            if require_diagonal_touch:
                if has_diagonal_touch:
                    candidates.append(cells)
            elif not has_diagonal_touch:
                candidates.append(cells)
    return candidates


def _place_red_components(
    grid: np.ndarray,
    component_count: int,
    *,
    force_diagonal_case: bool,
) -> bool:
    interior_h = grid.shape[0] - 2
    interior_w = grid.shape[1] - 2
    occupied: set[tuple[int, int]] = set()

    for idx in range(component_count):
        require_diagonal_touch = force_diagonal_case and idx == component_count - 1
        placed = False
        for _ in range(80):
            area = random.randint(*RED_COMPONENT_AREA_RANGE)
            shape = _random_polyomino(area, interior_h, interior_w)
            if shape is None:
                continue
            candidates = _candidate_component_positions(
                occupied,
                interior_h,
                interior_w,
                shape,
                require_diagonal_touch=require_diagonal_touch,
            )
            if not candidates:
                continue
            cells = random.choice(candidates)
            for r, c in cells:
                grid[r + 1, c + 1] = RED
            occupied |= cells
            placed = True
            break
        if not placed:
            return False

    comps = _count_red_components(grid)
    if len(comps) != component_count:
        return False
    if force_diagonal_case:
        interior_mask = grid[1:-1, 1:-1] == RED
        coords = [tuple(map(int, p)) for p in np.argwhere(interior_mask)]
        diagonal_pairs = 0
        coord_set = set(coords)
        for r, c in coords:
            for dr, dc in ((1, 1), (1, -1)):
                if (r + dr, c + dc) in coord_set:
                    diagonal_pairs += 1
        if diagonal_pairs == 0:
            return False
    return True


def _paint_source_segment(grid: np.ndarray, seg_len: int) -> None:
    h, w = grid.shape
    side_options = []
    if w - 2 >= seg_len:
        side_options.extend(["top", "bottom"])
    if h - 2 >= seg_len:
        side_options.extend(["left", "right"])
    side = random.choice(side_options)

    if side == "top":
        c0 = random.randint(1, w - seg_len - 1)
        grid[0, c0 : c0 + seg_len] = BLUE
    elif side == "bottom":
        c0 = random.randint(1, w - seg_len - 1)
        grid[h - 1, c0 : c0 + seg_len] = BLUE
    elif side == "left":
        r0 = random.randint(1, h - seg_len - 1)
        grid[r0 : r0 + seg_len, 0] = BLUE
    else:
        r0 = random.randint(1, h - seg_len - 1)
        grid[r0 : r0 + seg_len, w - 1] = BLUE


def _parameter_options() -> list[tuple[int, int, int, int]]:
    options = []
    for h in range(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1] + 1):
        for w in range(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1] + 1):
            perimeter = 2 * (h - 1) + 2 * (w - 1)
            for seg_len in BLUE_SEGMENT_LENGTHS:
                if max(h - 2, w - 2) < seg_len:
                    continue
                for red_count in range(RED_COMPONENT_COUNT_RANGE[0], RED_COMPONENT_COUNT_RANGE[1] + 1):
                    if perimeter % (seg_len + red_count) == 0:
                        options.append((h, w, seg_len, red_count))
    return options


PARAMETER_OPTIONS = _parameter_options()


def _validate_generated_instance(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert set(map(int, np.unique(grid))).issubset({BLACK, BLUE, RED})

    start, seg_len = _blue_segment_start_and_length(grid)
    comps = _count_red_components(grid)
    assert RED_COMPONENT_COUNT_RANGE[0] <= len(comps) <= RED_COMPONENT_COUNT_RANGE[1]
    for comp in comps:
        assert RED_COMPONENT_AREA_RANGE[0] <= len(comp) <= RED_COMPONENT_AREA_RANGE[1]

    expected = _expected_output(grid)
    actual = main(grid)
    assert np.array_equal(actual[1:-1, 1:-1], grid[1:-1, 1:-1]), "interior must remain unchanged"
    assert np.array_equal(actual, expected), "output must match the border-tiling rule"

    border = _border_cycle(grid.shape)
    perimeter = len(border)
    stride = seg_len + len(comps)
    assert perimeter % stride == 0
    blue_positions = {
        tuple(map(int, coord))
        for coord in np.argwhere(actual == BLUE)
    }
    expected_positions = set()
    for offset in range(0, perimeter, stride):
        for j in range(seg_len):
            expected_positions.add(border[(start + offset + j) % perimeter])
    assert blue_positions == expected_positions, "border copies must be complete and exact"


def _gen_attempt() -> np.ndarray | None:
    h, w, seg_len, red_count = random.choice(PARAMETER_OPTIONS)
    grid = np.zeros((h, w), dtype=int)
    _paint_source_segment(grid, seg_len)

    force_diagonal_case = red_count >= 2 and random.random() < 0.45
    if not _place_red_components(grid, red_count, force_diagonal_case=force_diagonal_case):
        return None

    try:
        _validate_generated_instance(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
