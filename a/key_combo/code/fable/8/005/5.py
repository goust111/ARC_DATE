from __future__ import annotations

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(np.array(coords, dtype=int))

    return components


def _assert_isolated_single_pixels(grid: np.ndarray) -> None:
    comps = _components_mask(grid != BLACK)
    assert comps, "expected at least one colored pixel"
    assert all(len(comp) == 1 for comp in comps), "every foreground component must be a single pixel"


def _spiral_path(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    top, left = 0, 0
    bottom, right = h - 1, w - 1
    path: list[tuple[int, int]] = []

    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            path.append((top, c))
        top += 1

        for r in range(top, bottom + 1):
            path.append((r, right))
        right -= 1

        if top <= bottom:
            for c in range(right, left - 1, -1):
                path.append((bottom, c))
            bottom -= 1

        if left <= right:
            for r in range(bottom, top - 1, -1):
                path.append((r, left))
            left += 1

    assert len(path) == h * w
    return path


def _compact_sequence_to_tail(shape: tuple[int, int], colors: list[int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    path = _spiral_path(shape)
    tail = path[len(path) - len(colors) :]
    for (r, c), color in zip(tail, colors):
        out[r, c] = int(color)
    return out


def _spiral_sequence(grid: np.ndarray) -> list[int]:
    return [int(grid[r, c]) for r, c in _spiral_path(grid.shape) if int(grid[r, c]) != BLACK]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide all isolated pixels along the hidden clockwise spiral and compact them against the spiral's end.

    Input structure:
    - A black rectangular grid contains several colored single-pixel objects.
    - The hidden track is the clockwise spiral that starts at the top-left corner, first moves right, and visits every cell exactly once.
    - Pixel colors may repeat, and pixels may start anywhere on that spiral.

    Rule:
    - Read the colored pixels in their current order along the spiral track.
    - Slide every pixel forward toward the spiral's final center-side endpoint without letting any pair cross.
    - Continue until the colored pixels occupy the spiral's last consecutive block of cells, while preserving their spiral order and colors.

    Key insight:
    - The 2D scatter is really a 1D order-preserving compaction problem on an implicit spiral path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    _assert_isolated_single_pixels(g)

    colors = _spiral_sequence(g)
    return _compact_sequence_to_tail(g.shape, colors)


def _row_major_order(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(coords)


def _center_distance_order(shape: tuple[int, int], coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = (shape[0] - 1) / 2.0
    cc = (shape[1] - 1) / 2.0
    return sorted(coords, key=lambda rc: (abs(rc[0] - cr) + abs(rc[1] - cc), rc[0], rc[1]))


def _colors_in_order(grid: np.ndarray, coords: list[tuple[int, int]]) -> list[int]:
    return [int(grid[r, c]) for r, c in coords if int(grid[r, c]) != BLACK]


def _has_clearance(cell: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    r, c = cell
    return all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen)


def _sample_positions(
    shape: tuple[int, int], count: int, rng: np.random.Generator
) -> tuple[list[int], list[tuple[int, int]]] | None:
    path = _spiral_path(shape)
    all_ids = list(map(int, rng.permutation(len(path))))
    chosen_ids: list[int] = []
    chosen_cells: list[tuple[int, int]] = []

    for idx in all_ids:
        cell = path[idx]
        if _has_clearance(cell, chosen_cells):
            chosen_ids.append(int(idx))
            chosen_cells.append(cell)
            if len(chosen_ids) == count:
                break

    if len(chosen_ids) != count:
        return None
    order = np.argsort(chosen_ids)
    chosen_ids = [chosen_ids[int(i)] for i in order]
    chosen_cells = [chosen_cells[int(i)] for i in order]
    return chosen_ids, chosen_cells


def _sample_colors(count: int, rng: np.random.Generator) -> list[int] | None:
    want_repeats = bool(rng.integers(0, 2))
    for _ in range(100):
        if want_repeats:
            palette_size = int(rng.integers(2, min(5, count) + 1))
            palette = rng.choice(PALETTE, size=palette_size, replace=False)
            colors = list(map(int, rng.choice(palette, size=count, replace=True)))
            if len(set(colors)) == count:
                continue
        else:
            palette_size = min(count, int(rng.integers(2, min(6, count) + 1)))
            palette = rng.choice(PALETTE, size=palette_size, replace=False)
            colors = list(map(int, rng.choice(palette, size=count, replace=palette_size < count)))
        if len(set(colors)) < 2:
            continue
        if colors == colors[::-1]:
            continue
        return colors
    return None


def _build_input(
    shape: tuple[int, int], cells: list[tuple[int, int]], colors: list[int]
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for (r, c), color in zip(cells, colors):
        grid[r, c] = int(color)
    return grid


def _expected_output(shape: tuple[int, int], colors: list[int]) -> np.ndarray:
    return _compact_sequence_to_tail(shape, colors)


def _wrong_row_major(grid: np.ndarray) -> np.ndarray:
    coords = _row_major_order(list(map(tuple, np.argwhere(grid != BLACK))))
    return _compact_sequence_to_tail(grid.shape, _colors_in_order(grid, coords))


def _wrong_reverse_spiral(grid: np.ndarray) -> np.ndarray:
    colors = list(reversed(_spiral_sequence(grid)))
    return _compact_sequence_to_tail(grid.shape, colors)


def _wrong_center_sorted(grid: np.ndarray) -> np.ndarray:
    coords = list(map(tuple, np.argwhere(grid != BLACK)))
    ordered = _center_distance_order(grid.shape, coords)
    return _compact_sequence_to_tail(grid.shape, _colors_in_order(grid, ordered))


def _instance_is_valid(
    grid: np.ndarray, chosen_ids: list[int], chosen_cells: list[tuple[int, int]], colors: list[int]
) -> bool:
    path = _spiral_path(grid.shape)
    expected = _expected_output(grid.shape, colors)

    try:
        actual = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(grid, expected):
        return False

    if _spiral_sequence(grid) != colors:
        return False

    tail_ids = list(range(len(path) - len(colors), len(path)))
    if chosen_ids == tail_ids:
        return False
    if all(idx >= tail_ids[0] for idx in chosen_ids):
        return False

    if len({grid[r, c] for r, c in chosen_cells}) < 2:
        return False

    row_major_colors = _colors_in_order(grid, _row_major_order(chosen_cells))
    center_colors = _colors_in_order(grid, _center_distance_order(grid.shape, chosen_cells))
    reverse_colors = list(reversed(colors))

    if row_major_colors == colors:
        return False
    if center_colors == colors:
        return False
    if reverse_colors == colors:
        return False

    if np.array_equal(_wrong_row_major(grid), expected):
        return False
    if np.array_equal(_wrong_center_sorted(grid), expected):
        return False
    if np.array_equal(_wrong_reverse_spiral(grid), expected):
        return False

    tail = path[-len(colors) :]
    if {(int(r), int(c)) for r, c in np.argwhere(actual != BLACK)} != set(tail):
        return False

    return True


def _sample_shape(rng: np.random.Generator) -> tuple[int, int]:
    h = int(rng.integers(6, 13))
    w = int(rng.integers(6, 13))
    if bool(rng.integers(0, 2)):
        while h == w:
            w = int(rng.integers(6, 13))
    return h, w


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        rng = np.random.default_rng()
        shape = _sample_shape(rng)
        max_count = min(8, (shape[0] * shape[1]) // 6)
        count = int(rng.integers(4, max_count + 1))

        sampled = _sample_positions(shape, count, rng)
        if sampled is None:
            continue
        chosen_ids, chosen_cells = sampled

        colors = _sample_colors(count, rng)
        if colors is None:
            continue

        grid = _build_input(shape, chosen_cells, colors)
        if _instance_is_valid(grid, chosen_ids, chosen_cells, colors):
            return grid

    raise RuntimeError("failed to generate a non-degenerate spiral-compaction input")
