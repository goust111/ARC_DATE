from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GEN_RETRIES = 400

SHAPE_LIBRARY = (
    np.array([[1]], dtype=int),
    np.array([[1, 1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
)


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4((h, w), rr, cc):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _build_windmill(grid: np.ndarray) -> np.ndarray:
    return np.block(
        [
            [grid, np.rot90(grid, -1)],
            [np.rot90(grid, 1), np.rot90(grid, 2)],
        ]
    )


def _quadrant_index(n: int, r: int, c: int) -> int:
    if r < n and c < n:
        return 0
    if r < n:
        return 1
    if c < n:
        return 2
    return 3


def _expected_output(grid: np.ndarray) -> np.ndarray:
    windmill = _build_windmill(grid)
    out = windmill.copy()
    n = grid.shape[0]
    for coords in _components_mask(windmill == BLUE):
        quadrants = {_quadrant_index(n, int(r), int(c)) for r, c in coords}
        if len(quadrants) >= 2:
            out[coords[:, 0], coords[:, 1]] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build the four-rotation windmill and recolor every blue component that spans multiple quadrants.

    Input structure:
    - The input is a square grid with black background and blue foreground pixels.
    - Blue pixels form one or more small 4-connected shapes.

    Rule:
    - Build a 2N x 2N windmill mosaic: original in the upper-left, 90-degree clockwise in the upper-right, 90-degree counterclockwise in the lower-left, and 180-degree rotation in the lower-right.
    - Find every 4-connected blue component in that mosaic.
    - Recolor a component red iff it contains pixels from at least two different quadrants of the mosaic.
    - Keep single-quadrant blue components blue and keep black background unchanged.

    Key insight:
    - The recoloring criterion is not an input-side property of a shape; it emerges only after the rotated copies are stitched together, because seam adjacencies can merge copies across quadrant borders.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] == g.shape[1], "windmill construction requires a square grid"
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE}), "expected only black and blue"
    return _expected_output(g)


def _rotate_shape(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _mask_coords(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    coords = []
    for r, c in np.argwhere(mask != 0):
        coords.append((int(r0 + r), int(c0 + c)))
    return coords


def _can_place_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    own = set(coords)
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] != BLACK:
            return False
        for nr, nc in _neighbors4((h, w), r, c):
            if (nr, nc) not in own and grid[nr, nc] == BLUE:
                return False
    return True


def _paint_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        grid[r, c] = BLUE


def _bridge_component(n: int) -> list[tuple[int, int]]:
    k = random.choice([n - 3, n - 2])
    if random.random() < 0.5:
        coords = {(k, c) for c in range(k, n)}
        coords |= {(r, k) for r in range(k, n)}
    else:
        coords = {(r, n - 1) for r in range(k, n)}
        coords |= {(n - 1, c) for c in range(k, n)}
    return sorted(coords)


def _random_shape(max_cells: int) -> np.ndarray:
    choices = [shape for shape in SHAPE_LIBRARY if int(shape.sum()) <= max_cells]
    return _rotate_shape(random.choice(choices))


def _try_place_interior(grid: np.ndarray, max_cells: int) -> bool:
    n = grid.shape[0]
    for _ in range(60):
        mask = _random_shape(max_cells)
        h, w = mask.shape
        if n - h - 1 < 1 or n - w - 1 < 1:
            continue
        r0 = random.randint(1, n - h - 1)
        c0 = random.randint(1, n - w - 1)
        coords = _mask_coords(mask, r0, c0)
        if _can_place_component(grid, coords):
            _paint_component(grid, coords)
            return True
    return False


def _try_place_edge_only(grid: np.ndarray, max_cells: int) -> bool:
    n = grid.shape[0]
    for _ in range(60):
        mask = _random_shape(max_cells)
        h, w = mask.shape
        edge = random.choice(["top", "left"])
        if edge == "top":
            if n - w - 2 < 0 or h > n - 1:
                continue
            r0 = 0
            c0 = random.randint(0, n - w - 2)
        else:
            if n - h - 2 < 0 or w > n - 1:
                continue
            r0 = random.randint(0, n - h - 2)
            c0 = 0
        coords = _mask_coords(mask, r0, c0)
        if any(r == n - 1 or c == n - 1 for r, c in coords):
            continue
        if _can_place_component(grid, coords):
            _paint_component(grid, coords)
            return True
    return False


def _semantic_ok(grid: np.ndarray) -> bool:
    if grid.shape[0] not in range(5, 10):
        return False
    if int(np.sum(grid == BLUE)) not in range(4, 19):
        return False
    if np.array_equal(grid, np.rot90(grid, -1)):
        return False
    if any(len(coords) == 0 for coords in _components_mask(grid == BLUE)):
        return False

    windmill = _build_windmill(grid)
    infos = []
    for coords in _components_mask(windmill == BLUE):
        quadrants = {_quadrant_index(grid.shape[0], int(r), int(c)) for r, c in coords}
        infos.append((coords, quadrants))
    if not any(len(quadrants) >= 2 for _, quadrants in infos):
        return False
    if not any(len(quadrants) == 1 for _, quadrants in infos):
        return False

    out = main(grid)
    expected = windmill.copy()
    for coords, quadrants in infos:
        color = RED if len(quadrants) >= 2 else BLUE
        if not np.all(out[coords[:, 0], coords[:, 1]] == color):
            return False
        expected[coords[:, 0], coords[:, 1]] = color
    return np.array_equal(out, expected)


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([5, 6, 7, 8, 9])
    grid = np.zeros((n, n), dtype=int)

    bridge = _bridge_component(n)
    _paint_component(grid, bridge)

    mode = random.choices(
        ["edge_only", "interior", "both"],
        weights=[5, 4, 3],
        k=1,
    )[0]
    plan = {
        "edge_only": ["edge"],
        "interior": ["interior"],
        "both": ["edge", "interior"],
    }[mode]

    for idx, kind in enumerate(plan):
        remaining = len(plan) - idx - 1
        budget = 18 - int(np.sum(grid == BLUE)) - remaining
        if budget <= 0:
            return None
        if kind == "edge":
            if not _try_place_edge_only(grid, min(4, budget)):
                return None
        else:
            if not _try_place_interior(grid, min(4, budget)):
                return None

    if random.random() < 0.35:
        budget = 18 - int(np.sum(grid == BLUE))
        if budget >= 1:
            placer = random.choice([_try_place_interior, _try_place_edge_only])
            placer(grid, min(2, budget))

    return grid if _semantic_ok(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid windmill-connectivity input")
