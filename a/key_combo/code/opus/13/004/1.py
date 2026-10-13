from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    comps.sort(key=lambda comp: (_bbox(comp)[1], _bbox(comp)[0], int(comp.shape[0])))
    return comps


def _is_hollow(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    obj = np.zeros((h, w), dtype=bool)
    obj[coords[:, 0] - r0, coords[:, 1] - c0] = True

    open_cells = np.pad(~obj, 1, constant_values=True)
    seen = np.zeros_like(open_cells, dtype=bool)
    q: deque[tuple[int, int]] = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        rr, cc = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = rr + dr, cc + dc
            if (
                0 <= nr < open_cells.shape[0]
                and 0 <= nc < open_cells.shape[1]
                and open_cells[nr, nc]
                and not seen[nr, nc]
            ):
                seen[nr, nc] = True
                q.append((nr, nc))

    trapped = open_cells[1:-1, 1:-1] & ~seen[1:-1, 1:-1]
    return bool(np.any(trapped))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split blue objects by topology so hollow ones pass through the divider and solid ones stop above it.

    Input structure:
    - Black is the background.
    - Exactly one full-width red horizontal divider lies in the lower half of the grid.
    - Several blue 4-connected objects sit strictly above the divider, with empty space below it.

    Rule:
    - Detect each blue object and test whether it encloses at least one black background region.
    - Clear every original blue pixel.
    - Hollow objects drop to the bottom row, punch out the red divider in their occupied columns, and recolor to green.
    - Non-hollow objects drop only until their bottom row is immediately above the divider and stay blue.

    Key insight:
    The divider behaves as a topological filter: only objects that truly trap background cells are allowed to pass through it.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    divider_rows = [r for r in range(h) if np.all(g[r] == RED)]
    assert len(divider_rows) == 1, "expected exactly one full-width red divider row"
    divider_r = int(divider_rows[0])

    blue_mask = g == BLUE
    blue_mask[divider_r:, :] = False
    objects = _find_components(blue_mask)
    assert objects, "expected blue objects above the divider"

    out = g.copy()
    out[g == BLUE] = BG

    for coords in objects:
        _r0, _c0, r1, _c1 = _bbox(coords)
        if _is_hollow(coords):
            shift = h - 1 - r1
            cols = np.unique(coords[:, 1])
            out[divider_r, cols] = BG
            color = GREEN
        else:
            shift = divider_r - 1 - r1
            color = BLUE

        moved = coords.copy()
        moved[:, 0] += shift
        out[moved[:, 0], moved[:, 1]] = color

    return out


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _transform_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    cells = {(int(r), int(c)) for r, c in shape}
    if random.random() < 0.5:
        cells = {(c, r) for r, c in cells}
    if random.random() < 0.5:
        max_r = max(r for r, _c in cells)
        cells = {(max_r - r, c) for r, c in cells}
    if random.random() < 0.5:
        max_c = max(c for _r, c in cells)
        cells = {(r, max_c - c) for r, c in cells}
    return _normalize(cells)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    return max(rows) + 1, max(cols) + 1


def _shape_array(shape: tuple[tuple[int, int], ...], top: int, left: int) -> np.ndarray:
    return np.array([(top + r, left + c) for r, c in shape], dtype=int)


def _composition(total: int, parts: int) -> list[int]:
    vals = [0] * parts
    for _ in range(total):
        vals[random.randrange(parts)] += 1
    return vals


def _place_intervals(widths: list[int], grid_width: int) -> list[int]:
    base = sum(widths) + max(0, len(widths) - 1)
    extra = grid_width - base
    assert extra >= 0
    gaps = _composition(extra, len(widths) + 1)
    starts: list[int] = []
    col = gaps[0]
    for idx, width in enumerate(widths):
        starts.append(col)
        col += width
        if idx < len(widths) - 1:
            col += 1 + gaps[idx + 1]
    return starts


HOLLOW_BASES = [
    (
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ),
    (
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
    ),
]

SOLID_BASES = [
    (
        (0, 0),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ),
    (
        (0, 0),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 2),
    ),
    (
        (0, 0),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
    ),
    (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ),
    (
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 1),
    ),
    (
        (0, 0),
        (1, 0),
        (2, 0),
        (2, 1),
        (2, 2),
    ),
]


def _random_hollow_shape() -> tuple[tuple[int, int], ...]:
    return _transform_shape(random.choice(HOLLOW_BASES))


def _random_solid_shape() -> tuple[tuple[int, int], ...]:
    return _transform_shape(random.choice(SOLID_BASES))


def _sample_shapes(count: int, hollow_count: int) -> list[tuple[str, tuple[tuple[int, int], ...]]]:
    labels = ["hollow"] * hollow_count + ["solid"] * (count - hollow_count)
    random.shuffle(labels)
    shapes: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    for label in labels:
        shape = _random_hollow_shape() if label == "hollow" else _random_solid_shape()
        shapes.append((label, shape))
    return shapes


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(15, 19)
    w = random.randint(18, 22)
    divider_r = random.randint(max(8, h // 2 + 1), h - 4)
    count = random.randint(4, 6)
    hollow_count = random.choice([1, 2])
    if hollow_count >= count:
        hollow_count = count - 1

    shapes = _sample_shapes(count, hollow_count)
    widths = [_shape_dims(shape)[1] for _kind, shape in shapes]
    min_width = sum(widths) + count - 1
    if min_width > w:
        return None

    starts = _place_intervals(widths, w)
    inp = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    inp[divider_r, :] = RED
    expected[divider_r, :] = RED

    seen_hollow = False
    seen_solid = False
    for (kind, shape), left in zip(shapes, starts):
        sh_h, sh_w = _shape_dims(shape)
        if kind == "hollow":
            seen_hollow = True
            landed_top = h - sh_h
            min_input_top = 0
            max_input_top = divider_r - sh_h - 1
            if max_input_top < min_input_top:
                return None
            input_top = random.randint(min_input_top, max_input_top)
            expected_coords = _shape_array(shape, landed_top, left)
            input_coords = _shape_array(shape, input_top, left)
            expected[divider_r, np.unique(expected_coords[:, 1])] = BG
            expected[expected_coords[:, 0], expected_coords[:, 1]] = GREEN
        else:
            seen_solid = True
            landed_top = divider_r - sh_h
            min_input_top = 0
            max_input_top = landed_top - 1
            if max_input_top < min_input_top:
                return None
            input_top = random.randint(min_input_top, max_input_top)
            expected_coords = _shape_array(shape, landed_top, left)
            input_coords = _shape_array(shape, input_top, left)
            expected[expected_coords[:, 0], expected_coords[:, 1]] = BLUE

        inp[input_coords[:, 0], input_coords[:, 1]] = BLUE

    if not (seen_hollow and seen_solid):
        return None
    if np.any(inp[divider_r + 1 :, :] != BG):
        return None
    if np.count_nonzero(expected == GREEN) == 0 or np.count_nonzero(expected == BLUE) == 0:
        return None
    if np.count_nonzero(expected[divider_r] == BG) == 0:
        return None
    if len(_find_components(inp == BLUE)) != count:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid topology-filter gravity input")
