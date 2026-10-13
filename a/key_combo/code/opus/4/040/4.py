from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
ORANGE = 7

ORTH_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ALL_ADJ8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]

BASE_SHAPES = [
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 1], [0, 0, 1]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 1]], dtype=int),
]


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        queue = deque([(r, c)])
        seen[r, c] = True
        cells = []
        while queue:
            rr, cc = queue.popleft()
            cells.append((rr, cc))
            for nr, nc in neighbors4(rr, cc):
                if in_bounds(grid.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        parts.append(np.array(cells, dtype=int))
    return parts


def _find_cross_center(grid: np.ndarray) -> tuple[int, int]:
    centers = []
    for r, c in np.argwhere(grid == RED):
        r = int(r)
        c = int(c)
        if all(in_bounds(grid.shape, r + dr, c + dc) and grid[r + dr, c + dc] == BLUE for dr, dc in ORTH_DIRS):
            centers.append((r, c))
    assert len(centers) == 1, "expected exactly one two-color cross center"
    return centers[0]


def _extend_lines(grid: np.ndarray, center: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = np.array(grid, dtype=int, copy=True)
    orth_mask = out == BLUE
    diag_mask = out == RED
    cr, cc = center

    for dr, dc in ORTH_DIRS:
        r, c = cr + dr, cc + dc
        while in_bounds(out.shape, r, c):
            if out[r, c] == BLACK:
                out[r, c] = BLUE
                orth_mask[r, c] = True
            r += dr
            c += dc

    for dr, dc in DIAG_DIRS:
        r, c = cr + dr, cc + dc
        while in_bounds(out.shape, r, c):
            if out[r, c] == BLACK:
                out[r, c] = RED
                diag_mask[r, c] = True
            r += dr
            c += dc

    return out, orth_mask, diag_mask


def _touch_signature(cells: np.ndarray, orth_mask: np.ndarray, diag_mask: np.ndarray) -> tuple[bool, bool]:
    touch_orth = False
    touch_diag = False
    for r, c in cells:
        r = int(r)
        c = int(c)
        for nr, nc in neighbors4(r, c):
            if not in_bounds(orth_mask.shape, nr, nc):
                continue
            touch_orth |= bool(orth_mask[nr, nc])
            touch_diag |= bool(diag_mask[nr, nc])
    return touch_orth, touch_diag


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Extend the two-color cross into orthogonal and diagonal guide lines, then recolor each gray sprite by which guide system it touches.

    Input structure:
    - A black background contains one 3x3 two-color cross whose four orthogonal arm pixels are blue and whose center is red.
    - Several separate gray 4-connected sprites of size 3 to 6 are scattered away from the cross and away from one another.
    - The sprites do not lie on the cross's future extension rays; they can only touch those rays by 4-neighbor adjacency after the rays are extended.

    Rule:
    - Extend blue from the cross center along the four orthogonal directions to the grid boundary, painting only black cells and skipping over any existing non-black cells.
    - Extend red from the same center along the four diagonal directions to the grid boundary with the same skip-over behavior.
    - For each gray sprite, inspect the 4-neighborhood of all sprite pixels in the extended grid.
    - Recolor a sprite green if it touches only blue guide pixels, yellow if it touches only red guide pixels, orange if it touches both, and leave it gray if it touches neither.
    - Keep the original 3x3 cross unchanged.

    Key insight:
    - The blue orthogonal system and the red diagonal system are two different spatial guide networks, so the sprite color must encode which network or combination of networks reaches it.
    """
    grid = np.asarray(input_grid, dtype=int)
    center = _find_cross_center(grid)
    out, orth_mask, diag_mask = _extend_lines(grid, center)

    for coords in _components_of_color(grid, GRAY):
        touch_orth, touch_diag = _touch_signature(coords, orth_mask, diag_mask)
        if touch_orth and touch_diag:
            new_color = ORANGE
        elif touch_orth:
            new_color = GREEN
        elif touch_diag:
            new_color = YELLOW
        else:
            new_color = GRAY
        out[coords[:, 0], coords[:, 1]] = new_color

    return out


def _shape_variants() -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for base in BASE_SHAPES:
        for flip in (False, True):
            work = np.fliplr(base) if flip else base
            for k in range(4):
                rot = np.rot90(work, k)
                coords = np.argwhere(rot > 0)
                coords = coords - coords.min(axis=0)
                key = tuple(sorted((int(r), int(c)) for r, c in coords))
                if key not in seen:
                    seen.add(key)
                    variants.append(np.array(key, dtype=int))
    return variants


SHAPE_VARIANTS = _shape_variants()


def _make_cross_grid(shape: tuple[int, int], center: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    cr, cc = center
    grid[cr, cc] = RED
    for dr, dc in ORTH_DIRS:
        grid[cr + dr, cc + dc] = BLUE
    return grid


def _mark_blocked(blocked: np.ndarray, cells: np.ndarray) -> None:
    for r, c in cells:
        r = int(r)
        c = int(c)
        for dr, dc in ALL_ADJ8:
            nr = r + dr
            nc = c + dc
            if in_bounds(blocked.shape, nr, nc):
                blocked[nr, nc] = True


def _category_from_flags(touch_orth: bool, touch_diag: bool) -> str:
    if touch_orth and touch_diag:
        return "both"
    if touch_orth:
        return "orth"
    if touch_diag:
        return "diag"
    return "none"


def _placement_cells(shape_coords: np.ndarray, top: int, left: int) -> np.ndarray:
    return shape_coords + np.array([top, left], dtype=int)


def _is_valid_placement(
    grid: np.ndarray,
    blocked: np.ndarray,
    orth_mask: np.ndarray,
    diag_mask: np.ndarray,
    category: str,
    cells: np.ndarray,
) -> bool:
    for r, c in cells:
        r = int(r)
        c = int(c)
        if grid[r, c] != BLACK or blocked[r, c] or orth_mask[r, c] or diag_mask[r, c]:
            return False
    return _category_from_flags(*_touch_signature(cells, orth_mask, diag_mask)) == category


def _find_valid_placement(
    grid: np.ndarray,
    blocked: np.ndarray,
    orth_mask: np.ndarray,
    diag_mask: np.ndarray,
    category: str,
    rng: np.random.Generator,
) -> np.ndarray | None:
    h, w = grid.shape

    for _ in range(240):
        shape_coords = SHAPE_VARIANTS[int(rng.integers(0, len(SHAPE_VARIANTS)))]
        sh = int(shape_coords[:, 0].max()) + 1
        sw = int(shape_coords[:, 1].max()) + 1
        top = int(rng.integers(0, h - sh + 1))
        left = int(rng.integers(0, w - sw + 1))
        cells = _placement_cells(shape_coords, top, left)
        if _is_valid_placement(grid, blocked, orth_mask, diag_mask, category, cells):
            return cells

    chosen: np.ndarray | None = None
    seen = 0
    for shape_coords in SHAPE_VARIANTS:
        sh = int(shape_coords[:, 0].max()) + 1
        sw = int(shape_coords[:, 1].max()) + 1
        for top in range(h - sh + 1):
            for left in range(w - sw + 1):
                cells = _placement_cells(shape_coords, top, left)
                if not _is_valid_placement(grid, blocked, orth_mask, diag_mask, category, cells):
                    continue
                seen += 1
                if int(rng.integers(0, seen)) == 0:
                    chosen = cells
    return chosen


def _choose_center(rng: np.random.Generator, h: int, w: int) -> tuple[int, int]:
    row_lo = max(4, h // 4)
    row_hi = min(h - 5, h // 2)
    col_lo = max(4, w // 4)
    col_hi = min(w - 5, w // 2)
    return int(rng.integers(row_lo, row_hi + 1)), int(rng.integers(col_lo, col_hi + 1))


def _sample_categories(rng: np.random.Generator) -> list[str]:
    count = int(rng.integers(5, 8))
    categories = ["both", "orth", "diag"]
    extras: list[str] = []
    if rng.random() < 0.7:
        extras.append("none")
    pool = ["both", "orth", "diag", "none"]
    while len(categories) + len(extras) < count:
        extras.append(str(rng.choice(pool)))
    categories.extend(extras)
    rng.shuffle(categories)
    categories.sort(key=lambda name: {"both": 0, "orth": 1, "diag": 2, "none": 3}[name])
    return categories


def _build_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 27))
    w = int(rng.integers(18, 27))
    center = _choose_center(rng, h, w)
    grid = _make_cross_grid((h, w), center)
    _, orth_mask, diag_mask = _extend_lines(grid, center)

    blocked = np.zeros((h, w), dtype=bool)
    cross_cells = np.argwhere(grid != BLACK)
    _mark_blocked(blocked, cross_cells)

    for category in _sample_categories(rng):
        cells = _find_valid_placement(grid, blocked, orth_mask, diag_mask, category, rng)
        if cells is None:
            return None
        grid[cells[:, 0], cells[:, 1]] = GRAY
        _mark_blocked(blocked, cells)

    return grid


def _validate_generated_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert max(grid.shape) <= 30
    center = _find_cross_center(grid)
    empty_cross = _make_cross_grid(grid.shape, center)
    _, orth_mask, diag_mask = _extend_lines(empty_cross, center)

    gray_parts = _components_of_color(grid, GRAY)
    assert 5 <= len(gray_parts) <= 7

    counts = {"orth": 0, "diag": 0, "both": 0, "none": 0}
    for coords in gray_parts:
        assert 3 <= len(coords) <= 6
        for r, c in coords:
            assert not orth_mask[r, c]
            assert not diag_mask[r, c]
        counts[_category_from_flags(*_touch_signature(coords, orth_mask, diag_mask))] += 1

    assert counts["orth"] >= 1
    assert counts["diag"] >= 1
    assert counts["both"] >= 1

    out = main(grid)
    assert np.array_equal(out[grid == BLUE], _extend_lines(grid, center)[0][grid == BLUE])
    assert int(np.count_nonzero(out == BLUE)) > int(np.count_nonzero(grid == BLUE))
    assert int(np.count_nonzero(out == RED)) > int(np.count_nonzero(grid == RED))
    for coords in gray_parts:
        category = _category_from_flags(*_touch_signature(coords, orth_mask, diag_mask))
        expected_color = {"orth": GREEN, "diag": YELLOW, "both": ORANGE, "none": GRAY}[category]
        colors = set(int(out[r, c]) for r, c in coords)
        assert colors == {expected_color}


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(300):
        grid = _build_attempt(rng)
        if grid is None:
            continue
        _validate_generated_grid(grid)
        return grid
    raise RuntimeError("failed to generate a valid input")
