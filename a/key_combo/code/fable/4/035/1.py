from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BACKGROUND = 0
FOREGROUND_COLORS = tuple(range(1, 10))
GENERATION_ATTEMPTS = 600


@dataclass(frozen=True)
class Square:
    """A monochromatic solid square, using inclusive (r, c) bbox bounds."""

    color: int
    size: int
    r0: int
    c0: int
    r1: int
    c1: int


def _extract_squares(grid: np.ndarray) -> list[Square]:
    """Recognize one solid square per non-black color."""
    squares: list[Square] = []
    colors = [int(v) for v in np.unique(grid) if int(v) != BACKGROUND]
    assert len(colors) >= 2, "expected several foreground squares"

    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.size > 0
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height == width, "every foreground object must be square"
        assert int(coords.shape[0]) == height * width, "squares must be solid"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), (
            "each color must form exactly one solid square"
        )
        assert r0 > 0 and c0 > 0 and r1 < grid.shape[0] - 1 and c1 < grid.shape[1] - 1, (
            "input squares must not touch the grid edge"
        )
        squares.append(Square(color, height, r0, c0, r1, c1))

    assert len({sq.color for sq in squares}) == len(squares), "square colors must be distinct"
    assert len({sq.size for sq in squares}) == len(squares), "square side lengths must be distinct"
    return squares


def main(grid: np.ndarray) -> np.ndarray:
    """Extend size-ordered square-colored ray bands, with earlier bands blocking later ones.

    Input structure:
    - A black background contains several separated, solid monochromatic squares.
    - Every square has a distinct visible color and a distinct side length, and no square touches the grid edge.

    Rule:
    - Process the squares from largest side length to smallest. From each of its four edges, advance a
      one-cell-deep strip as wide as that edge; if any cell of the next strip is non-black, stop that whole
      directional band, otherwise color the strip and continue to the grid edge.
    - Original squares and all bands painted by earlier squares are therefore obstacles and remain unchanged.

    Key insight:
    - Side-length rank is the hidden firing order, so previously painted ray bands become barriers for every
      smaller square and make the result depend on both sorting and collision timing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids cannot exceed 30 in either dimension"
    assert np.all(g >= 0), "colors must be non-negative integers"

    squares = sorted(_extract_squares(g), key=lambda sq: sq.size, reverse=True)
    out = g.copy()
    height, width = map(int, out.shape)

    for sq in squares:
        # Each candidate is the complete one-cell strip immediately ahead of
        # the band. A single occupied cell stops the complete band.
        for r in range(sq.r0 - 1, -1, -1):
            strip = out[r, sq.c0 : sq.c1 + 1]
            if np.any(strip != BACKGROUND):
                break
            strip[:] = sq.color

        for r in range(sq.r1 + 1, height):
            strip = out[r, sq.c0 : sq.c1 + 1]
            if np.any(strip != BACKGROUND):
                break
            strip[:] = sq.color

        for c in range(sq.c0 - 1, -1, -1):
            strip = out[sq.r0 : sq.r1 + 1, c]
            if np.any(strip != BACKGROUND):
                break
            strip[:] = sq.color

        for c in range(sq.c1 + 1, width):
            strip = out[sq.r0 : sq.r1 + 1, c]
            if np.any(strip != BACKGROUND):
                break
            strip[:] = sq.color

    return out


def _oracle(
    grid: np.ndarray,
    specs: list[Square],
    *,
    descending: bool = True,
    rays_are_obstacles: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    """Independent generator oracle plus collision trace.

    This deliberately uses coordinate lists rather than main()'s NumPy-slice
    implementation, so generation checks more than a call closing over itself.
    """
    original = grid.copy()
    out = grid.copy()
    height, width = map(int, grid.shape)
    ordered = sorted(specs, key=lambda sq: sq.size, reverse=descending)
    trace: dict[str, object] = {
        "square_blocks": 0,
        "partial_square_blocks": 0,
        "ray_blocks": 0,
        "edge_reaches": 0,
        "added_by_color": {},
    }

    for sq in ordered:
        before = int(np.count_nonzero(out))
        directions = (
            (-1, 0, sq.r0 - 1),
            (1, 0, sq.r1 + 1),
            (0, -1, sq.c0 - 1),
            (0, 1, sq.c1 + 1),
        )
        for dr, dc, position in directions:
            blocked = False
            while (0 <= position < height) if dr else (0 <= position < width):
                if dr:
                    cells = [(position, c) for c in range(sq.c0, sq.c1 + 1)]
                else:
                    cells = [(r, position) for r in range(sq.r0, sq.r1 + 1)]

                original_hit = any(original[r, c] != BACKGROUND for r, c in cells)
                ray_hit = any(
                    original[r, c] == BACKGROUND and out[r, c] != BACKGROUND
                    for r, c in cells
                )
                hit = original_hit or (rays_are_obstacles and ray_hit)
                if hit:
                    blocked = True
                    if original_hit:
                        trace["square_blocks"] = int(trace["square_blocks"]) + 1
                        if any(original[r, c] == BACKGROUND for r, c in cells):
                            trace["partial_square_blocks"] = int(trace["partial_square_blocks"]) + 1
                    if rays_are_obstacles and ray_hit:
                        trace["ray_blocks"] = int(trace["ray_blocks"]) + 1
                    break

                for r, c in cells:
                    out[r, c] = sq.color
                position += dr if dr else dc

            if not blocked:
                trace["edge_reaches"] = int(trace["edge_reaches"]) + 1

        added = int(np.count_nonzero(out)) - before
        added_by_color = trace["added_by_color"]
        assert isinstance(added_by_color, dict)
        added_by_color[sq.color] = added

    return out, trace


def _boxes_have_gap(a: Square, b: Square, gap: int = 2) -> bool:
    """Return whether two bboxes have at least `gap` black rows or columns between them."""
    return (
        a.r1 + gap < b.r0
        or b.r1 + gap < a.r0
        or a.c1 + gap < b.c0
        or b.c1 + gap < a.c0
    )


def _place_optional_square(
    height: int,
    width: int,
    size: int,
    color: int,
    existing: list[Square],
) -> Square | None:
    for _ in range(250):
        r0 = random.randint(1, height - size - 2)
        c0 = random.randint(1, width - size - 2)
        candidate = Square(color, size, r0, c0, r0 + size - 1, c0 + size - 1)
        if all(_boxes_have_gap(candidate, other) for other in existing):
            return candidate
    return None


def _orient(grid: np.ndarray, code: int) -> np.ndarray:
    """Apply one of eight shape-preserving/reflection/transposition variants."""
    if code < 4:
        return np.rot90(grid, code).copy()
    return np.rot90(np.fliplr(grid), code - 4).copy()


def _build_candidate() -> np.ndarray | None:
    height = random.randint(20, 30)
    width = random.randint(20, 30)
    count = random.choice((3, 4))
    sizes = sorted(random.sample((2, 3, 4, 5), count), reverse=True)
    colors = random.sample(FOREGROUND_COLORS, count)

    largest_size, aligned_size, diagonal_size = sizes[:3]
    r0 = random.randint(2, 5)
    c0 = random.randint(2, 5)
    largest = Square(
        colors[0], largest_size, r0, c0, r0 + largest_size - 1, c0 + largest_size - 1
    )

    # A smaller square sits partly across the largest square's downward band.
    # Because it is narrower, it tests that one occupied cell stops the whole band.
    aligned_min_r = largest.r1 + 4
    aligned_max_r = height - aligned_size - 2
    if aligned_min_r > aligned_max_r:
        return None
    ar0 = random.randint(aligned_min_r, aligned_max_r)
    ac0 = random.randint(largest.c0, largest.c1 - aligned_size + 1)
    aligned = Square(
        colors[1], aligned_size, ar0, ac0, ar0 + aligned_size - 1, ac0 + aligned_size - 1
    )

    # A diagonal square is outside both spans. Its inward ray must meet a band
    # painted by the largest square, rather than an original object.
    diagonal_min_r = largest.r1 + 4
    diagonal_max_r = height - diagonal_size - 2
    diagonal_min_c = largest.c1 + 4
    diagonal_max_c = width - diagonal_size - 2
    if diagonal_min_r > diagonal_max_r or diagonal_min_c > diagonal_max_c:
        return None
    dr0 = random.randint(diagonal_min_r, diagonal_max_r)
    dc0 = random.randint(diagonal_min_c, diagonal_max_c)
    diagonal = Square(
        colors[2], diagonal_size, dr0, dc0, dr0 + diagonal_size - 1, dc0 + diagonal_size - 1
    )

    specs = [largest, aligned, diagonal]
    if count == 4:
        optional = _place_optional_square(height, width, sizes[3], colors[3], specs)
        if optional is None:
            return None
        specs.append(optional)

    inp = np.zeros((height, width), dtype=int)
    for sq in specs:
        inp[sq.r0 : sq.r1 + 1, sq.c0 : sq.c1 + 1] = sq.color

    expected, trace = _oracle(inp, specs)
    reverse_order, _ = _oracle(inp, specs, descending=False)
    nonblocking, _ = _oracle(inp, specs, rays_are_obstacles=False)

    # Semantic checks: every named mechanism must visibly affect the answer.
    if int(trace["square_blocks"]) < 1 or int(trace["partial_square_blocks"]) < 1:
        return None
    if int(trace["ray_blocks"]) < 1 or int(trace["edge_reaches"]) < 2:
        return None
    added_by_color = trace["added_by_color"]
    if not isinstance(added_by_color, dict) or any(int(v) <= 0 for v in added_by_color.values()):
        return None
    if np.array_equal(inp, expected):
        return None
    if np.array_equal(reverse_order, expected):
        return None
    if np.array_equal(nonblocking, expected):
        return None
    density = float(np.count_nonzero(expected)) / float(expected.size)
    if not 0.12 <= density <= 0.68:
        return None
    foreground = int(np.count_nonzero(expected))
    if any(int(np.count_nonzero(expected == sq.color)) >= int(0.72 * foreground) for sq in specs):
        return None

    # This also independently validates object recognition and the descending
    # order used by main(), rather than merely trusting the construction.
    if not np.array_equal(main(inp), expected):
        return None

    orientation = random.randrange(8)
    oriented_inp = _orient(inp, orientation)
    oriented_expected = _orient(expected, orientation)
    if not np.array_equal(main(oriented_inp), oriented_expected):
        return None
    return oriented_inp


def generate_input() -> np.ndarray:
    """Generate varied, non-degenerate examples with both square and ray collisions."""
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid size-ordered ray-band input")
