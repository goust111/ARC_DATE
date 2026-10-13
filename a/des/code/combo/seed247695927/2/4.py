import math
import random
from collections import deque

import numpy as np


BG = 8  # azure
PALETTE = [0, 1, 2, 3, 4, 5, 6, 7, 9]
NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            row0, col0 = arr.min(axis=0)
            row1, col1 = arr.max(axis=0)
            top_row = arr[arr[:, 0] == row0]
            top_left = (int(row0), int(top_row[:, 1].min()))
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(len(coords)),
                    "bbox": (int(row0), int(col0), int(row1), int(col1)),
                    "top_left": top_left,
                }
            )

    return components


def _render_output(specs: list[tuple[tuple[int, int], int, int]]) -> np.ndarray:
    assert specs
    ordered = sorted(specs, key=lambda item: item[0])
    sides = [side for _top_left, _color, side in ordered]
    out_h = max(sides)
    out_w = sum(sides) + len(sides) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for idx, (_top_left, color, side) in enumerate(ordered):
        out[out_h - side : out_h, col : col + side] = color
        col += side
        if idx != len(ordered) - 1:
            col += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each foreground object into a filled square with equal area, preserving object order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _find_components(g)
    assert components, "expected at least one foreground object"

    specs = []
    areas = set()
    for comp in components:
        area = int(comp["area"])
        side = math.isqrt(area)
        assert side * side == area, "each object area must be a perfect square"
        assert area not in areas, "object areas must be distinct"
        areas.add(area)
        specs.append((comp["top_left"], int(comp["color"]), side))

    return _render_output(specs)


def _grow_polyomino(area: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    frontier = {(0, 1), (1, 0), (-1, 0), (0, -1)}

    while len(cells) < area:
        choices = list(frontier)
        random.shuffle(choices)
        scored = sorted(
            choices,
            key=lambda rc: sum((rc[0] + dr, rc[1] + dc) in cells for dr, dc in NEIGHBORS),
            reverse=True,
        )
        top_k = min(len(scored), random.randint(1, min(5, len(scored))))
        cell = random.choice(scored[:top_k])
        frontier.remove(cell)
        cells.add(cell)
        for dr, dc in NEIGHBORS:
            nxt = (cell[0] + dr, cell[1] + dc)
            if nxt not in cells:
                frontier.add(nxt)
    return cells


def _trimmed(coords: set[tuple[int, int]]) -> np.ndarray:
    arr = np.array(sorted(coords), dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    return arr


def _shape_variants(area: int) -> list[np.ndarray]:
    side = math.isqrt(area)
    variants = []

    variants.append(_trimmed({(r, c) for r in range(side) for c in range(side)}))

    for _ in range(4):
        grown = _trimmed(_grow_polyomino(area))
        variants.append(grown)

    unique = {}
    for arr in variants:
        key = tuple(map(tuple, arr.tolist()))
        unique[key] = arr
    return list(unique.values())


def _dilate_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    points = np.argwhere(mask)
    for r, c in points:
        r0 = max(0, int(r) - radius)
        r1 = min(h, int(r) + radius + 1)
        c0 = max(0, int(c) - radius)
        c1 = min(w, int(c) + radius + 1)
        out[r0:r1, c0:c1] = True
    return out


def _sample_shapes() -> list[tuple[int, np.ndarray]]:
    n_objects = random.randint(2, 5)
    candidate_sides = [1, 2, 3, 4, 5]
    if n_objects == 5:
        sides = candidate_sides
    else:
        sides = sorted(random.sample(candidate_sides, n_objects))

    shapes = []
    for side in sides:
        area = side * side
        shape = random.choice(_shape_variants(area))
        shapes.append((area, shape))
    return shapes


def _grid_size(shapes: list[tuple[int, np.ndarray]]) -> tuple[int, int]:
    max_h = max(int(shape[:, 0].max()) + 1 for _area, shape in shapes)
    max_w = max(int(shape[:, 1].max()) + 1 for _area, shape in shapes)
    rows = random.randint(max(10, max_h + 2), 22)
    cols = random.randint(max(10, max_w + 2), 24)
    return rows, cols


def _place_objects(
    rows: int, cols: int, shapes: list[tuple[int, np.ndarray]], colors: list[int]
) -> tuple[np.ndarray, list[tuple[tuple[int, int], int, int]]] | None:
    grid = np.full((rows, cols), BG, dtype=int)
    reserved = np.zeros((rows, cols), dtype=bool)
    placements = []

    order = list(range(len(shapes)))
    random.shuffle(order)
    for idx in order:
        area, shape = shapes[idx]
        color = colors[idx]
        h = int(shape[:, 0].max()) + 1
        w = int(shape[:, 1].max()) + 1

        candidates = []
        for r0 in range(rows - h + 1):
            for c0 in range(cols - w + 1):
                abs_coords = shape + np.array([r0, c0], dtype=int)
                if np.any(reserved[abs_coords[:, 0], abs_coords[:, 1]]):
                    continue
                candidates.append((r0, c0))

        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        abs_coords = shape + np.array([r0, c0], dtype=int)
        grid[abs_coords[:, 0], abs_coords[:, 1]] = color

        mask = np.zeros((rows, cols), dtype=bool)
        mask[abs_coords[:, 0], abs_coords[:, 1]] = True
        reserved |= _dilate_mask(mask, radius=1)

        top_row = abs_coords[:, 0].min()
        top_cols = abs_coords[abs_coords[:, 0] == top_row][:, 1]
        top_left = (int(top_row), int(top_cols.min()))
        placements.append((top_left, color, int(math.isqrt(area))))

    if not np.any(grid != BG):
        return None
    return grid, placements


def _expected_from_grid(grid: np.ndarray) -> np.ndarray:
    specs = []
    for comp in _find_components(grid):
        area = int(comp["area"])
        side = math.isqrt(area)
        if side * side != area:
            raise ValueError("invalid non-square area")
        specs.append((comp["top_left"], int(comp["color"]), side))
    return _render_output(specs)


def _gen_attempt() -> np.ndarray | None:
    shapes = _sample_shapes()
    rows, cols = _grid_size(shapes)
    colors = random.sample(PALETTE, len(shapes))
    placed = _place_objects(rows, cols, shapes, colors)
    if placed is None:
        return None

    grid, placements = placed
    expected = _render_output(placements)

    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    if not np.array_equal(expected, _expected_from_grid(grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
