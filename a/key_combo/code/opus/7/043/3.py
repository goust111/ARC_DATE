from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FG = 1
SPIRAL = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (2, 1),
    (2, 0),
    (1, 0),
    (1, 1),
)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120
SLOT_GAP = 2
SLOT_SIZE_RANGE = (5, 8)
AREA_VALUES = tuple(range(1, 9))


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _normalize(coords: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, c0 = min(rows), min(cols)
    arr = np.array(sorted((r - r0, c - c0) for r, c in coords), dtype=int)
    return arr


def _apply_dihedral(coords: np.ndarray) -> np.ndarray:
    pts = coords.copy()
    if random.random() < 0.5:
        pts[:, 1] *= -1
    k = random.randrange(4)
    for _ in range(k):
        pts = np.column_stack((-pts[:, 1], pts[:, 0]))
    return _normalize({(int(r), int(c)) for r, c in pts})


def _component_sizes(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    sizes: list[int] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            area = 0
            while q:
                cr, cc = q.popleft()
                area += 1
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            sizes.append(area)
    return sizes


def _gauge(area: int) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    for r, c in SPIRAL[: min(int(area), len(SPIRAL))]:
        patch[r, c] = FG
    return patch


def _render_output(areas: list[int]) -> np.ndarray:
    ordered = sorted(int(a) for a in areas)
    out = np.zeros((3, len(ordered) * 3 + max(0, len(ordered) - 1)), dtype=int)
    for i, area in enumerate(ordered):
        c0 = i * 4
        out[:, c0 : c0 + 3] = _gauge(area)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Encode foreground component areas as left-to-right spiral gauges sorted by size.

    Input structure:
    The input is a black background containing one foreground color. The
    foreground pixels form several disjoint 4-connected objects of different
    areas, with each object separated from the others by background space.

    Rule:
    Find every 4-connected foreground object and count its area. Sort those
    areas in ascending order. For each area, create a 3x3 blue gauge filled
    along the clockwise spiral path from the top-left for exactly that many
    cells, saturating at 9. Place the gauges in one horizontal row with a
    single black column between neighboring gauges.

    Key insight:
    The original object shapes and locations are discarded; only the sorted
    component areas survive, and each area is re-encoded by the fixed spiral
    fill order of a compact 3x3 gauge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fg_colors = sorted(int(v) for v in np.unique(g) if int(v) != BG)
    assert len(fg_colors) == 1, "expected exactly one non-background foreground color"

    areas = _component_sizes(g != BG)
    assert areas, "expected at least one foreground component"
    return _render_output(areas)


def _random_polyomino(area: int, max_h: int, max_w: int) -> np.ndarray | None:
    if area == 1:
        return np.array([[0, 0]], dtype=int)

    for _ in range(SHAPE_ATTEMPTS):
        if area <= max(max_h, max_w) and random.random() < 0.25:
            length = area
            if random.random() < 0.5 and length <= max_w:
                shape = np.array([(0, c) for c in range(length)], dtype=int)
            elif length <= max_h:
                shape = np.array([(r, 0) for r in range(length)], dtype=int)
            else:
                continue
            shape = _apply_dihedral(shape)
        else:
            cells: set[tuple[int, int]] = {(0, 0)}
            while len(cells) < area:
                frontier = {
                    nb
                    for r, c in cells
                    for nb in _neighbors4(r, c)
                    if nb not in cells
                }
                if not frontier:
                    break
                cells.add(random.choice(tuple(frontier)))
            if len(cells) != area:
                continue
            shape = _apply_dihedral(_normalize(cells))

        h = int(shape[:, 0].max()) + 1
        w = int(shape[:, 1].max()) + 1
        if h <= max_h and w <= max_w:
            return shape
    return None


def _layout_for_count(count: int) -> tuple[int, int]:
    if count <= 4:
        return 2, 2
    return random.choice(((2, 3), (3, 2)))


def _slot_origins(rows: int, cols: int, slot_h: int, slot_w: int) -> list[tuple[int, int]]:
    origins = []
    for r in range(rows):
        for c in range(cols):
            origins.append((1 + r * (slot_h + SLOT_GAP), 1 + c * (slot_w + SLOT_GAP)))
    return origins


def _place_shapes(
    shapes: list[np.ndarray],
    rows: int,
    cols: int,
    slot_h: int,
    slot_w: int,
) -> np.ndarray:
    height = 2 + rows * slot_h + (rows - 1) * SLOT_GAP
    width = 2 + cols * slot_w + (cols - 1) * SLOT_GAP
    grid = np.zeros((height, width), dtype=int)

    slots = _slot_origins(rows, cols, slot_h, slot_w)
    chosen_slots = random.sample(slots, len(shapes))
    random.shuffle(shapes)

    for shape, (r0, c0) in zip(shapes, chosen_slots):
        h = int(shape[:, 0].max()) + 1
        w = int(shape[:, 1].max()) + 1
        dr = random.randint(0, slot_h - h)
        dc = random.randint(0, slot_w - w)
        rr = shape[:, 0] + r0 + dr
        cc = shape[:, 1] + c0 + dc
        grid[rr, cc] = FG
    return grid


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(3, 6)
    areas = random.sample(AREA_VALUES, count)
    rows, cols = _layout_for_count(count)
    slot_h = random.randint(*SLOT_SIZE_RANGE)
    slot_w = random.randint(*SLOT_SIZE_RANGE)

    shapes: list[np.ndarray] = []
    for area in areas:
        shape = _random_polyomino(area, slot_h, slot_w)
        if shape is None:
            return None
        shapes.append(shape)

    grid = _place_shapes(shapes, rows, cols, slot_h, slot_w)
    if sorted(_component_sizes(grid == FG)) != sorted(areas):
        return None
    if not np.array_equal(main(grid), _render_output(areas)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
