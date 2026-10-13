import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
MAROON = 9

RING_COLORS = [1, 2, 3, 6, 7, 8]
MAX_GENERATION_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _corner_origins(height: int, width: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (0, 0),
        "tr": (0, width - 3),
        "bl": (height - 3, 0),
        "br": (height - 3, width - 3),
    }


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    coords.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    coords.extend((r1, c) for c in range(c1 - 1, c0 - 1, -1))
    coords.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return coords


def _find_compass(grid: np.ndarray) -> tuple[str, int, int, int]:
    orthogonal = [(0, 1), (1, 0), (1, 2), (2, 1)]
    corner_cells = [(0, 0), (0, 2), (2, 0), (2, 2)]
    clockwise_arms = {(0, 1), (1, 0)}

    candidates: list[tuple[str, int, int, int]] = []
    for corner, (r0, c0) in _corner_origins(*grid.shape).items():
        patch = grid[r0 : r0 + 3, c0 : c0 + 3]
        if patch.shape != (3, 3):
            continue
        if int(patch[1, 1]) != GREY:
            continue
        if any(int(patch[r, c]) != BLACK for r, c in corner_cells):
            continue

        arm_values = [int(patch[r, c]) for r, c in orthogonal]
        if arm_values.count(MAROON) != 1 or arm_values.count(GREY) != 3:
            continue

        maroon_arm = orthogonal[arm_values.index(MAROON)]
        direction = 1 if maroon_arm in clockwise_arms else -1
        candidates.append((corner, r0, c0, direction))

    assert len(candidates) == 1, "expected exactly one corner compass"
    return candidates[0]


def _extract_rings(grid: np.ndarray) -> list[dict[str, object]]:
    remaining = grid != BLACK
    rings: list[dict[str, object]] = []

    while remaining.any():
        points = np.argwhere(remaining)
        r0, c0 = map(int, points.min(axis=0))
        r1, c1 = map(int, points.max(axis=0))
        coords = _perimeter_coords(r0, c0, r1, c1)
        values = np.array([int(grid[r, c]) for r, c in coords], dtype=int)

        assert np.all(values != BLACK), "ring perimeter must be fully occupied"
        marker_count = int(np.count_nonzero(values == YELLOW))
        assert 1 <= marker_count <= 3, "each ring must have 1-3 yellow markers"

        ring_colors = {int(v) for v in values if int(v) != YELLOW}
        assert len(ring_colors) == 1, "each ring must have exactly one non-yellow color"

        for r, c in coords:
            remaining[r, c] = False

        rings.append(
            {
                "coords": coords,
                "values": values,
                "marker_count": marker_count,
            }
        )

    assert rings, "expected at least one ring"
    return rings


def _rotate_ring(values: np.ndarray, direction: int, steps: int) -> np.ndarray:
    return np.roll(values, shift=direction * steps)


def main(grid: np.ndarray) -> np.ndarray:
    """Read a corner compass and rotate each concentric rectangular ring by its own marker count.

    Input structure:
    - The background is black.
    - Away from one corner compass, the grid contains several concentric one-cell-thick rectangular rings separated by one black cell.
    - Each ring uses one base color and has 1-3 yellow marker pixels replacing ring cells.
    - Exactly one corner contains a 3x3 compass whose center is grey and whose four orthogonal arms are grey except for one maroon arm.

    Rule:
    - Find the compass in a grid corner and read the maroon arm: above or left of the center means rotate clockwise, below or right means rotate counterclockwise.
    - For each colored ring, count its yellow markers; that count is the number of perimeter steps for that ring.
    - Traverse each rectangle perimeter from the top-left corner, across the top edge, down the right edge, across the bottom edge, and up the left edge, then apply the cyclic shift in the chosen global direction.
    - Erase the compass back to black and keep only the rotated rings in the output.

    Key insight:
    - The compass contributes one global direction, while each ring contributes its own local step count via the number of yellow markers; solving the task requires combining those two signals on the same perimeter-rotation operation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert min(g.shape) >= 3

    work = g.copy()
    _, compass_r0, compass_c0, direction = _find_compass(work)
    work[compass_r0 : compass_r0 + 3, compass_c0 : compass_c0 + 3] = BLACK

    rings = _extract_rings(work)
    out = np.zeros_like(work)
    for ring in rings:
        coords = ring["coords"]
        values = ring["values"]
        marker_count = int(ring["marker_count"])
        rotated = _rotate_ring(values, direction=direction, steps=marker_count)
        for (r, c), value in zip(coords, rotated):
            out[r, c] = int(value)

    return out


def _choose_inner_size(ring_count: int) -> tuple[int, int] | None:
    max_inner = min(9, 25 - 4 * (ring_count - 1))
    if max_inner < 3:
        return None
    inner_h = int(_RNG.integers(3, max_inner + 1))
    inner_w = int(_RNG.integers(3, max_inner + 1))
    return inner_h, inner_w


def _sample_layout(outer_h: int, outer_w: int, corner: str) -> tuple[int, int, int, int] | None:
    if corner.startswith("t"):
        top = int(_RNG.integers(4, 7))
        bottom = int(_RNG.integers(1, 6))
    else:
        top = int(_RNG.integers(1, 6))
        bottom = int(_RNG.integers(4, 7))

    if corner.endswith("l"):
        left = int(_RNG.integers(4, 7))
        right = int(_RNG.integers(1, 6))
    else:
        left = int(_RNG.integers(1, 6))
        right = int(_RNG.integers(4, 7))

    height = top + outer_h + bottom
    width = left + outer_w + right
    if height > 30 or width > 30:
        return None
    return height, width, top, left


def _sample_marker_indices(length: int, count: int, shift: int) -> list[int] | None:
    for _ in range(200):
        picks = sorted(int(x) for x in _RNG.choice(length, size=count, replace=False))
        moved = sorted((idx + shift) % length for idx in picks)
        if picks != moved:
            return picks
    return None


def _paint_compass(grid: np.ndarray, corner: str, direction: int) -> None:
    r0, c0 = _corner_origins(*grid.shape)[corner]
    grid[r0 : r0 + 3, c0 : c0 + 3] = BLACK
    grid[r0 + 1, c0 + 1] = GREY

    for dr, dc in [(0, 1), (1, 0), (1, 2), (2, 1)]:
        grid[r0 + dr, c0 + dc] = GREY

    if direction == 1:
        dr, dc = [(0, 1), (1, 0)][int(_RNG.integers(0, 2))]
    else:
        dr, dc = [(1, 2), (2, 1)][int(_RNG.integers(0, 2))]
    grid[r0 + dr, c0 + dc] = MAROON


def _gen_attempt() -> np.ndarray | None:
    ring_count = int(_RNG.integers(1, 6))
    inner_size = _choose_inner_size(ring_count)
    if inner_size is None:
        return None
    inner_h, inner_w = inner_size

    outer_h = inner_h + 4 * (ring_count - 1)
    outer_w = inner_w + 4 * (ring_count - 1)
    corner = ["tl", "tr", "bl", "br"][int(_RNG.integers(0, 4))]
    layout = _sample_layout(outer_h, outer_w, corner)
    if layout is None:
        return None
    height, width, top, left = layout

    direction = 1 if bool(_RNG.integers(0, 2)) else -1
    colors = list(_RNG.choice(RING_COLORS, size=ring_count, replace=False))

    grid = np.full((height, width), BLACK, dtype=int)
    shift_sign = direction
    changed_rings = 0

    for depth, color in enumerate(colors):
        r0 = top + 2 * depth
        c0 = left + 2 * depth
        r1 = top + outer_h - 1 - 2 * depth
        c1 = left + outer_w - 1 - 2 * depth
        coords = _perimeter_coords(r0, c0, r1, c1)
        length = len(coords)
        marker_count = int(_RNG.integers(1, 4))
        marker_indices = _sample_marker_indices(length, marker_count, shift_sign * marker_count)
        if marker_indices is None:
            return None

        for r, c in coords:
            grid[r, c] = int(color)
        for idx in marker_indices:
            r, c = coords[idx]
            grid[r, c] = YELLOW

        moved_indices = sorted((idx + shift_sign * marker_count) % length for idx in marker_indices)
        if moved_indices != marker_indices:
            changed_rings += 1

    _paint_compass(grid, corner, direction)

    if changed_rings == 0:
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
