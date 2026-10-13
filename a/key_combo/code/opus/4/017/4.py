import random
from collections import deque

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREY = 5


GENERATION_ATTEMPTS = 500
HEIGHT_RANGE = (5, 9)
PANEL_WIDTH_RANGE = (4, 7)
TARGET_COMPONENT_RANGE = (2, 5)

# Small normalized 4-connected shapes for the overlap mask.
SHAPES_BY_AREA = {
    1: [((0, 0),)],
    2: [
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    ],
    3: [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (1, 0), (1, 1)),
    ],
    4: [
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
    ],
}


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _find_separator_col(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    full_grey_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == Color.GREY)]
    assert len(full_grey_cols) == 1, "expected exactly one full-height grey separator column"
    sep = int(full_grey_cols[0])
    assert np.count_nonzero(g == Color.GREY) == g.shape[0], "grey should appear only in the separator column"
    return sep


def _component_areas(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    areas: list[int] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        area = 0
        while q:
            cr, cc = q.popleft()
            area += 1
            for nr, nc in neighbors4(cr, cc):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        areas.append(area)
    return areas


def _split_masks(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    sep = _find_separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape[1] == right.shape[1] and left.shape[1] > 0, "left/right masks must have equal positive width"
    return left == Color.BLUE, right == Color.BLUE


def _overlap_mask(grid: np.ndarray) -> np.ndarray:
    left_mask, right_mask = _split_masks(grid)
    return left_mask & right_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Intersect the two blue masks split by the grey divider and encode the overlap component count as a red diagonal matrix.

    Input structure:
    - The grid contains a single full-height grey separator column that splits the canvas into two equal-width panels.
    - Each side panel is a black background with blue foreground pixels forming one or more 4-connected regions.
    - The two side panels are aligned pixel-for-pixel and may differ away from their common overlap.

    Rule:
    - Locate the grey separator and read the left and right blue masks on either side.
    - Keep only positions where both masks are blue, then count the 4-connected components in that overlap.
    - The overlap components themselves are blue, but the output ignores their geometry except for the count M.
    - Return an M by M black grid whose main diagonal is red.

    Key insight:
    - Neither side mask is the answer by itself; only the fragmentation of their logical AND matters, and that count alone determines the output size.
    """
    overlap = _overlap_mask(grid)
    areas = _component_areas(overlap)
    assert areas, "expected at least one overlap component"

    # The largest overlap component is well-defined in generated tasks, but the
    # final output uses a fixed red diagonal regardless of that component's color.
    _largest_area = max(areas)
    assert _largest_area >= 1

    m = len(areas)
    out = np.full((m, m), Color.BLACK, dtype=int)
    np.fill_diagonal(out, Color.RED)
    return out


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return 1 + max(r for r, _ in shape), 1 + max(c for _, c in shape)


def _translated(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + r, c0 + c) for r, c in shape]


def _can_place(coords: list[tuple[int, int]], occupied: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    for r, c in coords:
        if not _in_bounds(shape, r, c):
            return False
        if (r, c) in occupied:
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _build_shared_mask(height: int, width: int, target_components: int) -> np.ndarray | None:
    for _ in range(120):
        mask = np.zeros((height, width), dtype=bool)
        occupied: set[tuple[int, int]] = set()

        largest_area = random.choice((3, 4))
        area_plan = [largest_area] + [random.choice((1, 1, 2)) for _ in range(target_components - 1)]

        ok = True
        for area in area_plan:
            shape = random.choice(SHAPES_BY_AREA[area])
            sh, sw = _shape_dims(shape)
            candidates = [
                (r0, c0)
                for r0 in range(height - sh + 1)
                for c0 in range(width - sw + 1)
            ]
            random.shuffle(candidates)
            placed = False
            for r0, c0 in candidates:
                coords = _translated(shape, r0, c0)
                if _can_place(coords, occupied, (height, width)):
                    for r, c in coords:
                        mask[r, c] = True
                        occupied.add((r, c))
                    placed = True
                    break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        areas = sorted(_component_areas(mask), reverse=True)
        if len(areas) != target_components:
            continue
        if len(areas) >= 2 and areas[0] == areas[1]:
            continue
        return mask
    return None


def _adjacent_candidates(
    available: set[tuple[int, int]],
    anchor_mask: np.ndarray,
    *,
    strict: bool,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r, c in available:
        touches = any(
            _in_bounds(anchor_mask.shape, nr, nc) and anchor_mask[nr, nc]
            for nr, nc in neighbors4(r, c)
        )
        if touches == strict:
            candidates.append((r, c))
    return candidates


def _grow_extra_mask(
    shared: np.ndarray,
    available: set[tuple[int, int]],
    target_cells: int,
) -> np.ndarray:
    extra = np.zeros_like(shared, dtype=bool)
    if target_cells <= 0 or not available:
        return extra

    cluster_count = random.randint(1, min(3, target_cells))
    remaining = target_cells
    for cluster_index in range(cluster_count):
        if not available or remaining <= 0:
            break
        clusters_left = cluster_count - cluster_index
        max_here = remaining - (clusters_left - 1)
        cluster_size = random.randint(1, max_here)

        if random.random() < 0.8:
            seeds = _adjacent_candidates(available, shared | extra, strict=True)
        else:
            seeds = list(available)
        if not seeds:
            seeds = list(available)
        seed = random.choice(seeds)

        cluster = [seed]
        available.remove(seed)
        extra[seed] = True

        while len(cluster) < cluster_size:
            growth = [
                cell
                for cell in available
                if any(cell == nbr for base in cluster for nbr in neighbors4(*base))
            ]
            if not growth:
                break
            cell = random.choice(growth)
            cluster.append(cell)
            available.remove(cell)
            extra[cell] = True

        remaining -= len(cluster)
    return extra


def _build_masks(shared: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    height, width = shared.shape
    available = {(r, c) for r in range(height) for c in range(width) if not shared[r, c]}
    if len(available) < 2:
        return None

    left_target = random.randint(1, min(6, max(1, len(available) // 3)))
    left_extra = _grow_extra_mask(shared, available, left_target)

    if not available:
        return None

    right_target = random.randint(1, min(6, max(1, len(available) // 2)))
    right_extra = _grow_extra_mask(shared, available, right_target)
    if not right_extra.any():
        return None

    left = shared | left_extra
    right = shared | right_extra
    if np.array_equal(left, right):
        return None
    return left, right


def _assemble_grid(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    height, width = left.shape
    grid = np.full((height, width * 2 + 1), Color.BLACK, dtype=int)
    grid[:, :width][left] = Color.BLUE
    grid[:, width + 1 :][right] = Color.BLUE
    grid[:, width] = Color.GREY
    return grid


def _is_valid_input(grid: np.ndarray, target_components: int) -> bool:
    try:
        left_mask, right_mask = _split_masks(grid)
    except AssertionError:
        return False

    if np.array_equal(left_mask, right_mask):
        return False

    overlap = left_mask & right_mask
    areas = _component_areas(overlap)
    if len(areas) != target_components:
        return False
    if not (TARGET_COMPONENT_RANGE[0] <= len(areas) <= TARGET_COMPONENT_RANGE[1]):
        return False
    sorted_areas = sorted(areas, reverse=True)
    if len(sorted_areas) >= 2 and sorted_areas[0] == sorted_areas[1]:
        return False

    expected = np.full((target_components, target_components), Color.BLACK, dtype=int)
    np.fill_diagonal(expected, Color.RED)
    return np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    target_components = random.randint(*TARGET_COMPONENT_RANGE)
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*PANEL_WIDTH_RANGE)

    shared = _build_shared_mask(height, width, target_components)
    if shared is None:
        return None

    masks = _build_masks(shared)
    if masks is None:
        return None
    left, right = masks

    grid = _assemble_grid(left, right)
    if not _is_valid_input(grid, target_components):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
