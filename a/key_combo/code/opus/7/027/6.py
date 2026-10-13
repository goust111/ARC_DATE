from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


BACKGROUND = 0
SEPARATOR = 8
OBJECT_COLORS = (1, 2, 3, 4)
QUADRANT_SIZES = (7, 8, 9)
GENERATION_RETRIES = 400
QUADRANT_RETRIES = 200
PLACEMENT_RETRIES = 80


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _find_separator_cross(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = map(int, grid.shape)
    row_candidates = [
        (r, int(grid[r, 0]))
        for r in range(h)
        if np.all(grid[r, :] == grid[r, 0]) and int(grid[r, 0]) != bg
    ]
    col_candidates = [
        (c, int(grid[0, c]))
        for c in range(w)
        if np.all(grid[:, c] == grid[0, c]) and int(grid[0, c]) != bg
    ]
    crosses = [
        (r, c, color)
        for r, color in row_candidates
        for c, color2 in col_candidates
        if color == color2
    ]
    assert len(crosses) == 1, "expected one solid separator cross"
    return crosses[0]


def _split_quadrants(grid: np.ndarray, row: int, col: int) -> list[np.ndarray]:
    return [
        grid[:row, :col].copy(),
        grid[:row, col + 1 :].copy(),
        grid[row + 1 :, :col].copy(),
        grid[row + 1 :, col + 1 :].copy(),
    ]


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _count_objects(region: np.ndarray, bg: int) -> int:
    total = 0
    for color in sorted(set(map(int, region.flatten())) - {bg}):
        total += len(_component_coords(region == color))
    return total


def _rotate_clockwise(grid: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(grid, k=-(quarter_turns % 4)).copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each separator-defined quadrant by an angle encoded by its own object count.

    Input structure:
    - The background is black.
    - One solid non-background row and one solid non-background column form a
      cross that splits the odd square grid into four equal quadrants.
    - Each quadrant contains 1-4 separated monochrome objects on background,
      with the four quadrant object counts all different.

    Rule:
    - Delete the separator row and separator column.
    - Count the number N of 4-connected monochrome objects inside each quadrant.
    - Rotate that entire quadrant clockwise by (N-1) quarter-turns:
      1->0 degrees, 2->90 degrees, 3->180 degrees, 4->270 degrees.
    - Reassemble the rotated quadrants in their original upper-left,
      upper-right, lower-left, lower-right positions.

    Key insight:
    - The needed rotation is not fixed by quadrant position; each quadrant's
      own object count encodes how that same quadrant must be transformed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w and h % 2 == 1, "expected an odd square grid"

    bg = _get_bg(g)
    cross_r, cross_c, _sep = _find_separator_cross(g, bg)
    quadrants = _split_quadrants(g, cross_r, cross_c)
    assert len({quad.shape for quad in quadrants}) == 1, "quadrants must be equal-sized"

    rotated = []
    counts = []
    for quad in quadrants:
        count = _count_objects(quad, bg)
        assert 1 <= count <= 4, "each quadrant must contain 1-4 objects"
        counts.append(count)
        rotated.append(_rotate_clockwise(quad, count - 1))
    assert sorted(counts) == [1, 2, 3, 4], "quadrant object counts must be a permutation of 1..4"

    top = np.hstack(rotated[:2])
    bottom = np.hstack(rotated[2:])
    return np.vstack([top, bottom]).astype(int)


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_connected_mask(area: int, rng: np.random.Generator) -> np.ndarray:
    canvas = np.zeros((3, 3), dtype=bool)
    start = (int(rng.integers(0, 3)), int(rng.integers(0, 3)))
    cells = {start}
    canvas[start] = True
    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < 3 and 0 <= nc < 3 and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        choice = tuple(frontier)[int(rng.integers(0, len(frontier)))]
        cells.add(choice)
        canvas[choice] = True
    return _tight_crop(canvas)


def _sample_object_area(count: int, rng: np.random.Generator) -> int:
    if count == 1:
        choices = np.array([2, 3, 4, 5, 5])
    elif count == 2:
        choices = np.array([2, 2, 3, 4, 5])
    else:
        choices = np.array([2, 2, 3, 3, 4, 5])
    return int(choices[int(rng.integers(0, len(choices)))])


def _shape_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask)


def _candidate_positions(size: int, mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    return [(r, c) for r in range(1, size - h) for c in range(1, size - w)]


def _mark_blocked(blocked: np.ndarray, coords: Iterable[tuple[int, int]]) -> None:
    size = blocked.shape[0]
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    blocked[nr, nc] = True


def _generate_output_quadrant(size: int, count: int, rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(QUADRANT_RETRIES):
        quadrant = np.zeros((size, size), dtype=int)
        blocked = np.zeros((size, size), dtype=bool)
        blocked[0, :] = True
        blocked[-1, :] = True
        blocked[:, 0] = True
        blocked[:, -1] = True

        success = True
        for _obj_idx in range(count):
            placed = False
            for _ in range(PLACEMENT_RETRIES):
                area = _sample_object_area(count, rng)
                mask = _random_connected_mask(area, rng)
                coords = _shape_coords(mask)
                candidates = _candidate_positions(size, mask)
                if not candidates:
                    continue
                order = rng.permutation(len(candidates))
                color = int(OBJECT_COLORS[int(rng.integers(0, len(OBJECT_COLORS)))])
                for idx in order:
                    r0, c0 = candidates[int(idx)]
                    abs_coords = [(r0 + int(r), c0 + int(c)) for r, c in coords]
                    if any(blocked[r, c] for r, c in abs_coords):
                        continue
                    for r, c in abs_coords:
                        quadrant[r, c] = color
                    _mark_blocked(blocked, abs_coords)
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                success = False
                break

        if not success:
            continue

        if _count_objects(quadrant, BACKGROUND) != count:
            continue
        rotations = {
            _rotate_clockwise(quadrant, turns).tobytes()
            for turns in range(4)
        }
        if count > 1 and len(rotations) != 4:
            continue
        return quadrant
    return None


def _build_input(quadrants: list[np.ndarray]) -> np.ndarray:
    size = quadrants[0].shape[0]
    grid = np.zeros((2 * size + 1, 2 * size + 1), dtype=int)
    grid[size, :] = SEPARATOR
    grid[:, size] = SEPARATOR
    grid[:size, :size] = quadrants[0]
    grid[:size, size + 1 :] = quadrants[1]
    grid[size + 1 :, :size] = quadrants[2]
    grid[size + 1 :, size + 1 :] = quadrants[3]
    return grid


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        out = main(inp)
    except AssertionError:
        return False

    bg = _get_bg(inp)
    cross_r, cross_c, sep = _find_separator_cross(inp, bg)
    if sep != SEPARATOR or cross_r != cross_c:
        return False

    quadrants = _split_quadrants(inp, cross_r, cross_c)
    if len({quad.shape for quad in quadrants}) != 1:
        return False

    counts = [_count_objects(quad, bg) for quad in quadrants]
    if sorted(counts) != [1, 2, 3, 4]:
        return False

    rotated = [_rotate_clockwise(quad, count - 1) for quad, count in zip(quadrants, counts)]
    expected = np.vstack([np.hstack(rotated[:2]), np.hstack(rotated[2:])])
    if not np.array_equal(out, expected):
        return False

    no_rotation = np.vstack([np.hstack(quadrants[:2]), np.hstack(quadrants[2:])])
    if np.array_equal(out, no_rotation):
        return False

    areas_by_count = {}
    for quad, count in zip(quadrants, counts):
        if count > 1 and np.array_equal(quad, _rotate_clockwise(quad, count - 1)):
            return False
        areas_by_count[count] = int(np.count_nonzero(quad))
    area_sequence = [areas_by_count[k] for k in (1, 2, 3, 4)]
    if area_sequence == sorted(area_sequence):
        return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    size = int(QUADRANT_SIZES[int(rng.integers(0, len(QUADRANT_SIZES)))])
    counts = [1, 2, 3, 4]
    rng.shuffle(counts)

    output_quadrants: list[np.ndarray] = []
    for count in counts:
        quadrant = _generate_output_quadrant(size, count, rng)
        if quadrant is None:
            return None
        output_quadrants.append(quadrant)

    areas = [int(np.count_nonzero(q)) for q in output_quadrants]
    ordered_areas = [areas[counts.index(k)] for k in (1, 2, 3, 4)]
    if ordered_areas == sorted(ordered_areas):
        return None

    input_quadrants = [
        np.rot90(quadrant, k=(count - 1)).copy()
        for quadrant, count in zip(output_quadrants, counts)
    ]
    inp = _build_input(input_quadrants)
    if _semantic_checks(inp):
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
