import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PATTERN_COLORS = (BLUE, RED, GREEN)

GEN_ATTEMPTS = 400
GEN_SIDES = (13, 15, 17, 19)
GEN_DISTANCES = (1, 2, 3, 4)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _dihedral_variants(patch: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rot = np.rot90(patch, k)
        variants.append(rot)
        variants.append(np.fliplr(rot))
    return variants


def _is_good_pattern(patch: np.ndarray) -> bool:
    if patch.shape != (3, 3):
        return False
    colors = set(map(int, np.unique(patch)))
    if colors != set(PATTERN_COLORS):
        return False
    counts = {color: int(np.count_nonzero(patch == color)) for color in PATTERN_COLORS}
    if any(count < 2 or count > 4 for count in counts.values()):
        return False
    if len({tuple(map(int, row)) for row in patch}) != 3:
        return False
    if len({tuple(map(int, patch[:, c])) for c in range(3)}) != 3:
        return False
    if np.array_equal(patch[0], patch[2]):
        return False
    if np.array_equal(patch[:, 0], patch[:, 2]):
        return False
    return sum(np.array_equal(patch, variant) for variant in _dihedral_variants(patch)) == 1


def _random_pattern() -> np.ndarray:
    for _ in range(200):
        cells = [random.choice(PATTERN_COLORS) for _ in range(9)]
        patch = np.array(cells, dtype=int).reshape(3, 3)
        if _is_good_pattern(patch):
            return patch
    raise RuntimeError("failed to sample a valid 3x3 pattern")


def _find_pattern_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = (grid != BLACK) & (grid != YELLOW)
    b = _bbox(mask)
    assert b is not None, "expected one colored pattern"
    r0, c0, r1, c1 = b
    assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "pattern must occupy a 3x3 box"
    patch_mask = mask[r0 : r1 + 1, c0 : c1 + 1]
    assert bool(np.all(patch_mask)), "pattern box must be fully filled"
    assert int(mask.sum()) == 9, "pattern must contain exactly 9 cells"
    return r0, c0, r1, c1


def _scan_distance(grid: np.ndarray, start: tuple[int, int], step: tuple[int, int]) -> int:
    r, c = map(int, start)
    dr, dc = map(int, step)
    distance = 0
    while True:
        r += dr
        c += dc
        assert 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1], "expected a yellow marker on each axis"
        value = int(grid[r, c])
        if value == YELLOW:
            return distance + 1
        assert value == BLACK, "axis between pattern and marker must stay empty"
        distance += 1


def _expand_patch(patch: np.ndarray, d_up: int, d_down: int, d_left: int, d_right: int) -> np.ndarray:
    out = np.asarray(patch, dtype=int).copy()
    if d_up:
        out = np.vstack([np.repeat(out[[0]], d_up, axis=0), out])
    if d_down:
        out = np.vstack([out, np.repeat(out[[-1]], d_down, axis=0)])
    if d_left:
        out = np.hstack([np.repeat(out[:, [0]], d_left, axis=1), out])
    if d_right:
        out = np.hstack([out, np.repeat(out[:, [-1]], d_right, axis=1)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode four axial marker distances, then expand the central 3x3 pattern by asymmetric edge-copy padding.

    Input structure:
    - The grid has black background, exactly one central 3x3 non-black/non-yellow color patch, and four yellow marker pixels.
    - The 3x3 patch uses only blue, red, and green, while the yellow markers lie on the patch's center row/column extensions.
    - There is exactly one yellow marker in each cardinal direction, separated from the patch by black cells.

    Rule:
    - Find the unique 3x3 color patch and read the four distances from its top, bottom, left, and right edge centers to the first yellow marker along that axis.
    - Expand the patch upward, downward, leftward, and rightward by those four distances.
    - Each expansion layer copies the current outermost row or column; apply vertical expansion first, then horizontal expansion so left/right writes determine corner cells.

    Key insight:
    - The yellow markers are not part of the output object: their axial offsets encode four independent padding thicknesses for the edge-replication transform.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    r0, c0, r1, c1 = _find_pattern_bbox(g)
    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2

    d_up = _scan_distance(g, (r0, mid_c), (-1, 0))
    d_down = _scan_distance(g, (r1, mid_c), (1, 0))
    d_left = _scan_distance(g, (mid_r, c0), (0, -1))
    d_right = _scan_distance(g, (mid_r, c1), (0, 1))

    yellow_coords = np.argwhere(g == YELLOW)
    assert len(yellow_coords) == 4, "expected exactly four yellow markers"
    assert len({d_up, d_down, d_left, d_right}) == 4, "marker distances must be distinct"

    return _expand_patch(patch, d_up, d_down, d_left, d_right)


def _build_input(
    height: int,
    width: int,
    patch: np.ndarray,
    distances: dict[str, int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    mid_r = height // 2
    mid_c = width // 2
    r0, c0 = mid_r - 1, mid_c - 1
    grid[r0 : r0 + 3, c0 : c0 + 3] = patch

    grid[r0 - distances["up"], mid_c] = YELLOW
    grid[r0 + 2 + distances["down"], mid_c] = YELLOW
    grid[mid_r, c0 - distances["left"]] = YELLOW
    grid[mid_r, c0 + 2 + distances["right"]] = YELLOW
    return grid


def _valid_generated(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if out.shape[0] < 5 or out.shape[1] < 5:
        return False
    if np.array_equal(out[:3, :3], out[-3:, -3:]):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        height = random.choice(GEN_SIDES)
        width = random.choice(GEN_SIDES)
        patch = _random_pattern()
        dist_values = list(GEN_DISTANCES)
        random.shuffle(dist_values)
        distances = {
            "up": dist_values[0],
            "down": dist_values[1],
            "left": dist_values[2],
            "right": dist_values[3],
        }
        grid = _build_input(height, width, patch, distances)
        if _valid_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
