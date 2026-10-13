from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
TEAL = 8

Coord = tuple[int, int]
Shape = tuple[Coord, ...]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_corner(r: int, c: int, height: int, width: int) -> bool:
    return (r, c) in {
        (0, 0),
        (0, width - 1),
        (height - 1, 0),
        (height - 1, width - 1),
    }


def _find_unique_pixel(grid: np.ndarray, color: int) -> Coord:
    coords = np.argwhere(grid == color)
    assert coords.shape == (1, 2), f"expected exactly one pixel of color {color}"
    return int(coords[0, 0]), int(coords[0, 1])


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == color, dtype=bool)
    height, width = map(int, grid.shape)
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        queue: deque[Coord] = deque([(rr, cc)])
        seen[rr, cc] = True
        coords: list[Coord] = []

        while queue:
            cr, cc2 = queue.popleft()
            coords.append((cr, cc2))
            for nr, nc in _neighbors4(cr, cc2):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))

        components.append(np.array(coords, dtype=int))
    return components


def _quadrant_of_cell(
    r: int,
    c: int,
    upper_row: int,
    lower_row: int,
    left_col: int,
    right_col: int,
) -> str | None:
    if r < upper_row and c < left_col:
        return "tl"
    if r < upper_row and c > right_col:
        return "tr"
    if r > lower_row and c < left_col:
        return "bl"
    if r > lower_row and c > right_col:
        return "br"
    return None


def _quadrant_of_component(
    coords: np.ndarray,
    upper_row: int,
    lower_row: int,
    left_col: int,
    right_col: int,
) -> str | None:
    labels = {
        _quadrant_of_cell(int(r), int(c), upper_row, lower_row, left_col, right_col)
        for r, c in coords
    }
    return next(iter(labels)) if len(labels) == 1 else None


def _legend_colors(grid: np.ndarray) -> dict[str, int]:
    height, width = map(int, grid.shape)
    return {
        "tl": int(grid[0, 0]),
        "tr": int(grid[0, width - 1]),
        "bl": int(grid[height - 1, 0]),
        "br": int(grid[height - 1, width - 1]),
    }


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    out = g.copy()

    orange_r, orange_c = _find_unique_pixel(g, ORANGE)
    teal_r, teal_c = _find_unique_pixel(g, TEAL)
    upper_row, lower_row = sorted((orange_r, teal_r))
    left_col, right_col = sorted((orange_c, teal_c))
    legends = _legend_colors(g)

    for c in range(width):
        if not _is_corner(orange_r, c, height, width):
            out[orange_r, c] = ORANGE
    for r in range(height):
        if not _is_corner(r, orange_c, height, width):
            out[r, orange_c] = ORANGE

    for c in range(width):
        if not _is_corner(teal_r, c, height, width) and out[teal_r, c] != ORANGE:
            out[teal_r, c] = TEAL
    for r in range(height):
        if not _is_corner(r, teal_c, height, width) and out[r, teal_c] != ORANGE:
            out[r, teal_c] = TEAL

    for r, c in ((orange_r, teal_c), (teal_r, orange_c)):
        if not _is_corner(r, c, height, width):
            out[r, c] = MAGENTA

    for component in _components_of_color(g, GRAY):
        quadrant = _quadrant_of_component(component, upper_row, lower_row, left_col, right_col)
        if quadrant is None:
            continue
        fill = legends[quadrant]
        for r, c in component:
            rr = int(r)
            cc = int(c)
            if out[rr, cc] == GRAY:
                out[rr, cc] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the orange/teal cross rays, mark their off-center intersections, and recolor each quadrant's sprites from the corner legend.

    Input structure:
    - The grid has a black background with four single-pixel corner legend colors.
    - Exactly one orange pixel and one teal pixel lie away from the border and are not in the same row or column.
    - Several disconnected gray sprites lie fully inside the four quadrants defined by those two marker rows and columns.

    Rule:
    - Extend the orange marker to a full orange row and column, preserving the four corner legend pixels.
    - Extend the teal marker to a full teal row and column, again preserving the corners and any already-drawn orange line cells.
    - Recolor the two off-center orange/teal row-column crossings magenta.
    - Treat the two marker rows and two marker columns as quadrant boundaries, then recolor every gray sprite by the legend color of the quadrant containing all of its cells.

    Key insight:
    - The visible cross lines and the spatial recoloring use the same reference frame: the marker rows and columns both draw output structure and define which corner legend controls each sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    return _expected_output(g)


BASE_SHAPES: tuple[Shape, ...] = (
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)),
)


def _normalize_shape(shape: Shape) -> Shape:
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return tuple(sorted((r - min_r, c - min_c) for r, c in shape))


def _shape_variants(shape: Shape) -> set[Shape]:
    pts = list(shape)
    variants: set[Shape] = set()
    for flip in (1, -1):
        for rotate in range(4):
            transformed: list[Coord] = []
            for r, c in pts:
                rr, cc = r, c * flip
                for _ in range(rotate):
                    rr, cc = cc, -rr
                transformed.append((rr, cc))
            variants.add(_normalize_shape(tuple(transformed)))
    return variants


ALL_SHAPES: tuple[Shape, ...] = tuple(
    sorted({variant for shape in BASE_SHAPES for variant in _shape_variants(shape)})
)


def _shape_dims(shape: Shape) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _coord_set(coords: list[Coord] | np.ndarray) -> set[Coord]:
    return {(int(r), int(c)) for r, c in coords}


def _touches_existing(coords: list[Coord], occupied: set[Coord]) -> bool:
    cells = set(coords)
    for r, c in cells:
        if (r, c) in occupied:
            return True
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in occupied and (nr, nc) not in cells:
                    return True
    return False


def _quadrant_bounds(
    height: int,
    width: int,
    upper_row: int,
    lower_row: int,
    left_col: int,
    right_col: int,
) -> dict[str, tuple[int, int, int, int]]:
    return {
        "tl": (1, upper_row - 1, 1, left_col - 1),
        "tr": (1, upper_row - 1, right_col + 1, width - 2),
        "bl": (lower_row + 1, height - 2, 1, left_col - 1),
        "br": (lower_row + 1, height - 2, right_col + 1, width - 2),
    }


def _placement_candidates(
    bounds: tuple[int, int, int, int],
    shape: Shape,
    occupied: set[Coord],
) -> list[list[Coord]]:
    r0, r1, c0, c1 = bounds
    shape_h, shape_w = _shape_dims(shape)
    candidates: list[list[Coord]] = []
    for base_r in range(r0, r1 - shape_h + 2):
        for base_c in range(c0, c1 - shape_w + 2):
            coords = [(base_r + dr, base_c + dc) for dr, dc in shape]
            if not _touches_existing(coords, occupied):
                candidates.append(coords)
    return candidates


def _choose_markers(rng: np.random.Generator, height: int, width: int) -> tuple[Coord, Coord]:
    upper_row = int(rng.integers(5, height - 7))
    lower_row = int(rng.integers(upper_row + 2, height - 5))
    left_col = int(rng.integers(5, width - 7))
    right_col = int(rng.integers(left_col + 2, width - 5))

    orange_r, teal_r = (upper_row, lower_row) if bool(rng.integers(0, 2)) else (lower_row, upper_row)
    orange_c, teal_c = (left_col, right_col) if bool(rng.integers(0, 2)) else (right_col, left_col)
    return (orange_r, orange_c), (teal_r, teal_c)


def _place_sprites(
    rng: np.random.Generator,
    grid: np.ndarray,
    upper_row: int,
    lower_row: int,
    left_col: int,
    right_col: int,
) -> bool:
    height, width = map(int, grid.shape)
    bounds_by_quad = _quadrant_bounds(height, width, upper_row, lower_row, left_col, right_col)
    assignments = ["tl", "tr", "bl", "br"]
    total = int(rng.integers(4, 7))
    for _ in range(total - 4):
        assignments.append(str(rng.choice(("tl", "tr", "bl", "br"))))
    rng.shuffle(assignments)

    occupied = _coord_set(np.argwhere(grid != BLACK))
    for quadrant in assignments:
        shape_order = list(rng.permutation(len(ALL_SHAPES)))
        candidates: list[list[Coord]] = []
        for shape_idx in shape_order:
            shape = ALL_SHAPES[int(shape_idx)]
            candidates = _placement_candidates(bounds_by_quad[quadrant], shape, occupied)
            if candidates:
                break
        if not candidates:
            return False
        chosen = candidates[int(rng.integers(0, len(candidates)))]
        for r, c in chosen:
            grid[r, c] = GRAY
        occupied.update(chosen)
    return True


def _validate_input(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)
    assert 1 <= height <= 30 and 1 <= width <= 30
    assert int(g[0, 0]) == BLUE
    assert int(g[0, width - 1]) == RED
    assert int(g[height - 1, 0]) == GREEN
    assert int(g[height - 1, width - 1]) == YELLOW

    allowed = {BLACK, BLUE, RED, GREEN, YELLOW, GRAY, ORANGE, TEAL}
    assert set(map(int, np.unique(g))).issubset(allowed)

    orange_r, orange_c = _find_unique_pixel(g, ORANGE)
    teal_r, teal_c = _find_unique_pixel(g, TEAL)
    assert orange_r not in {0, height - 1} and teal_r not in {0, height - 1}
    assert orange_c not in {0, width - 1} and teal_c not in {0, width - 1}
    assert orange_r != teal_r and orange_c != teal_c

    upper_row, lower_row = sorted((orange_r, teal_r))
    left_col, right_col = sorted((orange_c, teal_c))

    components = _components_of_color(g, GRAY)
    assert 4 <= len(components) <= 6

    quadrant_counts = {"tl": 0, "tr": 0, "bl": 0, "br": 0}
    all_cells: list[set[Coord]] = []
    for component in components:
        coords = _coord_set(component)
        assert 3 <= len(coords) <= 6
        quadrant = _quadrant_of_component(component, upper_row, lower_row, left_col, right_col)
        assert quadrant is not None, "gray sprites must stay inside a single quadrant"
        quadrant_counts[quadrant] += 1
        for r, c in coords:
            assert 0 < r < height - 1 and 0 < c < width - 1
            assert r not in {orange_r, teal_r} and c not in {orange_c, teal_c}
        all_cells.append(coords)

    assert all(count >= 1 for count in quadrant_counts.values())
    for i, a in enumerate(all_cells):
        for b in all_cells[i + 1 :]:
            for r, c in a:
                for nr in range(r - 1, r + 2):
                    for nc in range(c - 1, c + 2):
                        assert (nr, nc) not in b, "sprites must stay visually separated"


def _validate_output_semantics(inp: np.ndarray, out: np.ndarray) -> None:
    expected = _expected_output(inp)
    assert np.array_equal(out, expected), "output does not follow the puzzle rules"

    orange_r, orange_c = _find_unique_pixel(inp, ORANGE)
    teal_r, teal_c = _find_unique_pixel(inp, TEAL)
    upper_row, lower_row = sorted((orange_r, teal_r))
    left_col, right_col = sorted((orange_c, teal_c))
    legends = _legend_colors(inp)

    for component in _components_of_color(inp, GRAY):
        quadrant = _quadrant_of_component(component, upper_row, lower_row, left_col, right_col)
        assert quadrant is not None
        expected_color = legends[quadrant]
        for r, c in component:
            rr = int(r)
            cc = int(c)
            if rr not in {orange_r, teal_r} and cc not in {orange_c, teal_c}:
                assert int(out[rr, cc]) == expected_color

    assert int(out[orange_r, teal_c]) == MAGENTA
    assert int(out[teal_r, orange_c]) == MAGENTA


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(13, 19))
    width = int(rng.integers(13, 19))
    grid = np.zeros((height, width), dtype=int)
    grid[0, 0] = BLUE
    grid[0, width - 1] = RED
    grid[height - 1, 0] = GREEN
    grid[height - 1, width - 1] = YELLOW

    (orange_r, orange_c), (teal_r, teal_c) = _choose_markers(rng, height, width)
    grid[orange_r, orange_c] = ORANGE
    grid[teal_r, teal_c] = TEAL

    upper_row, lower_row = sorted((orange_r, teal_r))
    left_col, right_col = sorted((orange_c, teal_c))
    if not _place_sprites(rng, grid, upper_row, lower_row, left_col, right_col):
        return None

    _validate_input(grid)
    out = main(grid)
    _validate_output_semantics(grid, out)
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
