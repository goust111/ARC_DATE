import random
from collections import deque

import numpy as np


# Lightweight helper subset following /solvers/arc_utils.py conventions:
# grids use shape (H, W), coordinates are always (r, c), and 4-neighborhood
# means orthogonal adjacency only.
BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

MAX_MASK_PIXELS = 15
RETRY_LIMIT = 600
SEEN_OUTPUT_SIGNATURES: set[bytes] = set()


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _full_height_separator_column(grid: np.ndarray) -> int:
    cols = np.where(np.all(grid == GREY, axis=0))[0]
    assert len(cols) == 1, "expected exactly one full-height grey separator"
    sep = int(cols[0])
    assert sep > 0 and sep < grid.shape[1] - 1, "separator must split two non-empty panels"
    assert sep == grid.shape[1] - sep - 1, "left and right panels must have equal width"
    return sep


def _extract_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    sep = _full_height_separator_column(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape
    assert np.all(np.isin(left, [BLACK, BLUE]))
    assert np.all(np.isin(right, [BLACK, RED]))
    return left, right


def _wall_from_masks(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    wall = np.zeros(left.shape, dtype=int)
    wall[(left == BLUE) & (right == RED)] = GREEN
    return wall


def _classify_with_wall(wall_grid: np.ndarray) -> np.ndarray:
    out = np.asarray(wall_grid, dtype=int).copy()
    h, w = out.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def push(r: int, c: int) -> None:
        if out[r, c] == BLACK and not exterior[r, c]:
            exterior[r, c] = True
            q.append((r, c))

    for r in range(h):
        push(r, 0)
        push(r, w - 1)
    for c in range(w):
        push(0, c)
        push(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds((h, w), nr, nc) and out[nr, nc] == BLACK and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    out[exterior] = YELLOW
    out[out == BLACK] = ORANGE
    return out


def _mask_component_count(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r, c in coords:
        if seen[r, c]:
            continue
        count += 1
        q: deque[tuple[int, int]] = deque([(int(r), int(c))])
        seen[r, c] = True
        while q:
            rr, cc = q.popleft()
            for nr, nc in _neighbors4(rr, cc):
                if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
    return count


def _rect_ring_coords(top: int, left: int, height: int, width: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(left, left + width):
        coords.append((top, c))
        coords.append((top + height - 1, c))
    for r in range(top + 1, top + height - 1):
        coords.append((r, left))
        coords.append((r, left + width - 1))
    return coords


def _interior_coords(top: int, left: int, height: int, width: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(top + 1, top + height - 1)
        for c in range(left + 1, left + width - 1)
    ]


def _paint_mask(shape: tuple[int, int], coords: list[tuple[int, int]], color: int) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, c in coords:
        grid[r, c] = color
    return grid


def _choose_disjoint_extra_sets(interior: list[tuple[int, int]], max_extra: int):
    assert len(interior) >= 3
    assert max_extra >= 1
    shuffled = interior[:]
    random.shuffle(shuffled)

    choices: list[tuple[int, int]] = []
    for left_count in range(1, max_extra + 1):
        for right_count in range(1, max_extra + 1):
            if left_count + right_count < len(interior):
                choices.append((left_count, right_count))
    assert choices

    left_count, right_count = random.choice(choices)
    left_extra = shuffled[:left_count]
    right_extra = shuffled[left_count : left_count + right_count]
    return left_extra, right_extra


def _panel_wall_variants(input_grid: np.ndarray):
    left, right = _extract_panels(input_grid)
    and_wall = _wall_from_masks(left, right)

    left_wall = np.zeros(left.shape, dtype=int)
    left_wall[left == BLUE] = GREEN

    right_wall = np.zeros(right.shape, dtype=int)
    right_wall[right == RED] = GREEN

    union_wall = np.zeros(left.shape, dtype=int)
    union_wall[(left == BLUE) | (right == RED)] = GREEN
    return and_wall, left_wall, right_wall, union_wall


def _validate_generated_grid(input_grid: np.ndarray) -> bool:
    try:
        left, right = _extract_panels(input_grid)
        blue_mask = left == BLUE
        red_mask = right == RED
        if not (3 <= int(blue_mask.sum()) <= MAX_MASK_PIXELS):
            return False
        if not (3 <= int(red_mask.sum()) <= MAX_MASK_PIXELS):
            return False
        if _mask_component_count(blue_mask) != 1 or _mask_component_count(red_mask) != 1:
            return False
        if np.array_equal(blue_mask, red_mask):
            return False

        and_wall, left_wall, right_wall, union_wall = _panel_wall_variants(input_grid)
        if int(np.sum(and_wall == GREEN)) < 8:
            return False

        out = main(input_grid)
        if not np.any(out == GREEN):
            return False
        if not np.any(out == YELLOW):
            return False
        orange_count = int(np.sum(out == ORANGE))
        if orange_count < 2:
            return False

        and_out = _classify_with_wall(and_wall)
        left_out = _classify_with_wall(left_wall)
        right_out = _classify_with_wall(right_wall)
        union_out = _classify_with_wall(union_wall)
        if not np.array_equal(out, and_out):
            return False
        if np.array_equal(out, left_out):
            return False
        if np.array_equal(out, right_out):
            return False
        if np.array_equal(out, union_out):
            return False

        # The interior wall pixels injected into the individual masks must matter.
        if np.array_equal(and_wall, left_wall) or np.array_equal(and_wall, right_wall):
            return False

        return True
    except AssertionError:
        return False


def main(grid: np.ndarray) -> np.ndarray:
    """Use the overlap of two separated masks as a wall, then color exterior and enclosed background differently.

    Input structure:
    - The grid has black background and one full-height grey separator column splitting it into equal left and right panels.
    - The left panel contains a blue/black mask and the right panel contains a red/black mask in the same panel coordinate system.
    - Each panel's foreground forms one 4-connected mask shape, while the separator is layout-only.

    Rule:
    - Remove the separator conceptually and align the two panels cell by cell.
    - Paint green exactly where the left panel is blue and the right panel is red; these overlap cells form the wall in the output panel.
    - Flood-fill from the output panel border through black cells: border-reachable background becomes yellow, and any remaining black cells enclosed by the green wall become orange.

    Key insight:
    - The logical AND is only the first step: it creates a dynamic wall whose topology must then be analyzed with a border flood-fill to distinguish outside from inside.
    """
    left, right = _extract_panels(grid)
    wall = _wall_from_masks(left, right)
    return _classify_with_wall(wall)


def _attempt_generate() -> np.ndarray | None:
    panel_h = random.randint(9, 13)
    panel_w = random.randint(9, 14)

    ring_sizes = [
        (3, 5),
        (4, 4),
        (5, 3),
        (3, 6),
        (4, 5),
        (5, 4),
        (6, 3),
    ]
    valid_sizes = []
    for ring_h, ring_w in ring_sizes:
        if ring_h + 2 > panel_h or ring_w + 2 > panel_w:
            continue
        wall_size = 2 * ring_h + 2 * ring_w - 4
        hole_size = (ring_h - 2) * (ring_w - 2)
        extra_budget = MAX_MASK_PIXELS - wall_size
        if hole_size >= 3 and extra_budget >= 1:
            valid_sizes.append((ring_h, ring_w, wall_size, hole_size, extra_budget))
    if not valid_sizes:
        return None

    ring_h, ring_w, wall_size, _hole_size, extra_budget = random.choice(valid_sizes)
    top = random.randint(1, panel_h - ring_h - 1)
    left = random.randint(1, panel_w - ring_w - 1)

    ring_coords = _rect_ring_coords(top, left, ring_h, ring_w)
    interior = _interior_coords(top, left, ring_h, ring_w)
    max_extra = min(2, extra_budget)
    left_extra, right_extra = _choose_disjoint_extra_sets(interior, max_extra)

    left_panel = _paint_mask((panel_h, panel_w), ring_coords + left_extra, BLUE)
    right_panel = _paint_mask((panel_h, panel_w), ring_coords + right_extra, RED)
    separator = np.full((panel_h, 1), GREY, dtype=int)
    input_grid = np.concatenate([left_panel, separator, right_panel], axis=1)

    if not _validate_generated_grid(input_grid):
        return None

    output_signature = main(input_grid).tobytes() + str(main(input_grid).shape).encode()
    if output_signature in SEEN_OUTPUT_SIGNATURES:
        return None
    SEEN_OUTPUT_SIGNATURES.add(output_signature)
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate input for the overlap-wall task")
