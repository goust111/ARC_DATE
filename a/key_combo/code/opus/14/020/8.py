from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

SIZE = 11
CENTER = SIZE // 2
RADIUS = SIZE // 2
MARKER_POOL = (BLUE, GREEN, YELLOW, PINK, ORANGE, TEAL, MAROON)
MAX_GENERATION_ATTEMPTS = 500


def _chebyshev_distance(r: int, c: int, center: int) -> int:
    return max(abs(int(r) - center), abs(int(c) - center))


def _distance_grid(size: int) -> np.ndarray:
    center = size // 2
    rr = np.arange(size)[:, None]
    cc = np.arange(size)[None, :]
    return np.maximum(np.abs(rr - center), np.abs(cc - center))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _reachable_non_walls(wall_mask: np.ndarray) -> np.ndarray:
    h, w = wall_mask.shape
    reachable = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not wall_mask[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not wall_mask[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not wall_mask[nr, nc] and not reachable[nr, nc]:
                reachable[nr, nc] = True
                queue.append((nr, nc))
    return reachable


def _extract_ring_colors(grid: np.ndarray) -> dict[int, int]:
    h, w = grid.shape
    center = h // 2
    distances = _distance_grid(h)
    colors: dict[int, int] = {}
    marker_mask = (grid != BLACK) & (grid != RED)
    for dist in range(center + 1):
        coords = np.argwhere(marker_mask & (distances == dist))
        assert len(coords) == 1, f"expected exactly one marker on ring {dist}"
        r, c = map(int, coords[0])
        colors[dist] = int(grid[r, c])
    assert len(set(colors.values())) == center + 1
    return colors


def _ring_fill_from_colors(size: int, ring_colors: dict[int, int]) -> np.ndarray:
    distances = _distance_grid(size)
    out = np.empty((size, size), dtype=int)
    for dist, color in ring_colors.items():
        out[distances == dist] = int(color)
    return out


def _semantic_transform(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ring_colors = _extract_ring_colors(grid)
    out = _ring_fill_from_colors(grid.shape[0], ring_colors)
    wall_mask = grid == RED
    out[wall_mask] = RED
    reachable = _reachable_non_walls(wall_mask)
    enclosed = (~wall_mask) & (~reachable)
    out[enclosed] = GREY
    return out, reachable, enclosed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the whole grid with concentric square-ring colors, then gray out the non-red pockets sealed by red walls.

    Input structure:
    - The grid is an odd square on a black background with a single center marker and one non-red marker on each Chebyshev-distance ring.
    - Additional red pixels form one-cell-thick wall segments that may create closed chambers.
    - Marker positions are clues only: each marker's distance from the center identifies a ring color.

    Rule:
    - For every cell, compute its Chebyshev distance from the center and repaint it with the color of the unique marker on that distance ring.
    - Keep every red input cell red; these red cells override the ring coloring and act as impassable walls.
    - Flood-fill from all border cells through 4-connected non-red cells, then recolor every non-red cell that is unreachable from the border to gray.

    Key insight:
    - The puzzle combines two independent overlays on the same canvas: distance-defined square rings create the base image, and red-wall enclosure determines which parts of that base image are later overwritten by gray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w and h % 2 == 1, "expected an odd square grid"
    assert h >= 3, "grid is too small"
    center = h // 2
    assert int(g[center, center]) not in (BLACK, RED), "center marker must be present"

    ring_colors = _extract_ring_colors(g)
    assert set(ring_colors) == set(range(center + 1))

    out = _ring_fill_from_colors(h, ring_colors)
    wall_mask = g == RED
    out[wall_mask] = RED

    reachable = _reachable_non_walls(wall_mask)
    enclosed = (~wall_mask) & (~reachable)
    assert np.any(enclosed), "expected at least one enclosed non-red region"
    out[enclosed] = GREY
    return out


def _rect_perimeter(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _rect_interior(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _all_frame_candidates() -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    dist_grid = _distance_grid(SIZE)
    for r0 in range(1, SIZE - 3):
        for c0 in range(1, SIZE - 3):
            for height in range(4, 7):
                for width in range(4, 7):
                    r1 = r0 + height - 1
                    c1 = c0 + width - 1
                    if r1 >= SIZE - 1 or c1 >= SIZE - 1:
                        continue
                    perimeter = _rect_perimeter(r0, c0, r1, c1)
                    if (CENTER, CENTER) in perimeter:
                        continue
                    interior = _rect_interior(r0, c0, r1, c1)
                    if not interior:
                        continue
                    interior_dists = {int(dist_grid[r, c]) for r, c in interior}
                    if len(interior_dists) < 2:
                        continue
                    candidates.append(
                        {
                            "bbox": (r0, c0, r1, c1),
                            "perimeter": perimeter,
                            "interior": interior,
                            "interior_dists": interior_dists,
                        }
                    )
    return candidates


FRAME_CANDIDATES = _all_frame_candidates()


def _place_open_distractor(
    rng: np.random.Generator,
    occupied: set[tuple[int, int]],
    reserved_boxes: list[tuple[int, int, int, int]],
) -> tuple[set[tuple[int, int]], tuple[int, int, int, int]] | None:
    options = FRAME_CANDIDATES[:]
    rng.shuffle(options)
    for cand in options:
        bbox = cand["bbox"]
        assert isinstance(bbox, tuple)
        if any(not _boxes_separated(bbox, other, gap=1) for other in reserved_boxes):
            continue
        perimeter = set(cand["perimeter"])
        if perimeter & occupied:
            continue
        r0, c0, r1, c1 = bbox
        removable = []
        removable.extend((r0, c) for c in range(c0 + 1, c1))
        removable.extend((r1, c) for c in range(c0 + 1, c1))
        removable.extend((r, c0) for r in range(r0 + 1, r1))
        removable.extend((r, c1) for r in range(r0 + 1, r1))
        gap_cell = removable[int(rng.integers(0, len(removable)))]
        perimeter.remove(gap_cell)
        return perimeter, bbox
    return None


def _choose_frames(rng: np.random.Generator) -> list[dict[str, object]] | None:
    target_count = 1 if float(rng.random()) < 0.55 else 2
    options = FRAME_CANDIDATES[:]
    rng.shuffle(options)
    chosen: list[dict[str, object]] = []
    for cand in options:
        bbox = cand["bbox"]
        assert isinstance(bbox, tuple)
        if any(not _boxes_separated(bbox, other["bbox"], gap=1) for other in chosen):
            continue
        chosen.append(cand)
        if len(chosen) == target_count:
            return chosen
    return None


def _marker_positions(
    rng: np.random.Generator, wall_mask: np.ndarray
) -> dict[int, tuple[int, int]] | None:
    distances = _distance_grid(SIZE)
    layout_id = int(rng.integers(0, 8))
    positions: dict[int, tuple[int, int]] = {}

    def score(rc: tuple[int, int]) -> tuple[int, int]:
        r, c = rc
        variants = (
            (r, c),
            (r, -c),
            (-r, c),
            (-r, -c),
            (c, r),
            (c, -r),
            (-c, r),
            (-c, -r),
        )
        return variants[layout_id]

    for dist in range(RADIUS + 1):
        coords = [tuple(map(int, rc)) for rc in np.argwhere((distances == dist) & (~wall_mask))]
        if not coords:
            return None
        if float(rng.random()) < 0.5:
            coords.sort(key=score)
            positions[dist] = coords[0]
        else:
            positions[dist] = coords[int(rng.integers(0, len(coords)))]
    return positions


def _validate_input(
    grid: np.ndarray,
    distractor_bbox: tuple[int, int, int, int] | None,
) -> bool:
    if grid.shape != (SIZE, SIZE):
        return False
    if int(grid[CENTER, CENTER]) in (BLACK, RED):
        return False
    if np.count_nonzero(grid == RED) < 12:
        return False

    try:
        ring_colors = _extract_ring_colors(grid)
    except AssertionError:
        return False
    if set(ring_colors) != set(range(RADIUS + 1)):
        return False

    wall_mask = grid == RED
    distances = _distance_grid(SIZE)
    for dist in range(RADIUS + 1):
        if not np.any((distances == dist) & (~wall_mask)):
            return False

    out, reachable, enclosed = _semantic_transform(grid)
    if not np.any(enclosed):
        return False
    if not np.any(out == GREY):
        return False

    enclosed_components = _components(enclosed)
    if not enclosed_components:
        return False
    if max(len({int(distances[r, c]) for r, c in comp}) for comp in enclosed_components) < 2:
        return False

    enclosed_dists = {int(distances[r, c]) for r, c in np.argwhere(enclosed)}
    for dist in enclosed_dists:
        same_ring_reachable = np.any((distances == dist) & reachable)
        if not same_ring_reachable:
            return False

    ring_only = _ring_fill_from_colors(SIZE, ring_colors)
    if len(set(int(v) for v in ring_only[enclosed])) < 2:
        return False
    if np.array_equal(out, ring_only):
        return False

    if distractor_bbox is not None:
        r0, c0, r1, c1 = distractor_bbox
        interior = np.zeros_like(grid, dtype=bool)
        interior[r0 + 1 : r1, c0 + 1 : c1] = True
        if np.any(interior & enclosed):
            return False
        if not np.any(interior & reachable):
            return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, out)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    frames = _choose_frames(rng)
    if frames is None:
        return None

    grid = np.zeros((SIZE, SIZE), dtype=int)
    occupied: set[tuple[int, int]] = set()
    reserved_boxes: list[tuple[int, int, int, int]] = []
    for frame in frames:
        perimeter = frame["perimeter"]
        bbox = frame["bbox"]
        assert isinstance(perimeter, set)
        assert isinstance(bbox, tuple)
        occupied |= perimeter
        reserved_boxes.append(bbox)

    distractor_bbox: tuple[int, int, int, int] | None = None
    if float(rng.random()) < 0.65:
        distractor = _place_open_distractor(rng, occupied, reserved_boxes)
        if distractor is not None:
            distractor_cells, distractor_bbox = distractor
            occupied |= distractor_cells
            reserved_boxes.append(distractor_bbox)

    for r, c in occupied:
        grid[r, c] = RED

    marker_positions = _marker_positions(rng, grid == RED)
    if marker_positions is None:
        return None

    palette = list(MARKER_POOL)
    rng.shuffle(palette)
    ring_colors = palette[: RADIUS + 1]
    for dist, color in enumerate(ring_colors):
        r, c = marker_positions[dist]
        grid[r, c] = int(color)

    if not _validate_input(grid, distractor_bbox):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
