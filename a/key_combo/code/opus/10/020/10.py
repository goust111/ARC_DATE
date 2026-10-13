import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

RING_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

GEN_RETRIES = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _extract_segments(grid: np.ndarray, color: int, slope: int) -> list[list[tuple[int, int]]]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(grid == color)]
    grouped: dict[int, list[tuple[int, int]]] = {}
    for r, c in coords:
        key = r - c if slope == 1 else r + c
        grouped.setdefault(key, []).append((r, c))

    segments: list[list[tuple[int, int]]] = []
    dc = 1 if slope == 1 else -1
    for cells in grouped.values():
        cells.sort()
        run = [cells[0]]
        for r, c in cells[1:]:
            pr, pc = run[-1]
            if r == pr + 1 and c == pc + dc:
                run.append((r, c))
            else:
                segments.append(run)
                run = [(r, c)]
        segments.append(run)
    return segments


def _replicate_segment(segment: Iterable[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    cells = list(segment)
    covered: set[tuple[int, int]] = set()
    for k in range(-w, w + 1):
        shift = 3 * k
        for r, c in cells:
            nc = c + shift
            if 0 <= nc < w and 0 <= r < h:
                covered.add((r, nc))
    return covered


def _find_template_ring_cells(grid: np.ndarray) -> set[tuple[int, int]]:
    h, w = grid.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r, c] != BLACK:
                continue
            ring_cells = {(r + dr, c + dc) for dr, dc in RING_OFFSETS}
            if all(grid[rr, cc] == GREEN for rr, cc in ring_cells):
                return ring_cells
    return set()


def _coverage_sets(grid: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    red_cover: set[tuple[int, int]] = set()
    for segment in _extract_segments(grid, RED, slope=1):
        red_cover |= _replicate_segment(segment, grid.shape)

    blue_cover: set[tuple[int, int]] = set()
    for segment in _extract_segments(grid, BLUE, slope=-1):
        blue_cover |= _replicate_segment(segment, grid.shape)

    return red_cover, blue_cover


def _apply_transform(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    ring_cells = _find_template_ring_cells(out)
    red_cover, blue_cover = _coverage_sets(out)

    for r, c in red_cover:
        if out[r, c] == BLACK:
            out[r, c] = RED

    for r, c in blue_cover:
        if out[r, c] == BLACK:
            out[r, c] = BLUE

    intersections = red_cover & blue_cover
    for r, c in intersections:
        out[r, c] = YELLOW

    for r, c in intersections:
        for dr, dc in RING_OFFSETS:
            rr, cc = r + dr, c + dc
            if _in_bounds(out.shape, rr, cc) and out[rr, cc] == BLACK:
                out[rr, cc] = GREEN

    for r, c in ring_cells:
        out[r, c] = BLACK

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate both diagonal-segment families every three columns, mark their crossings yellow, stamp green rings there, and erase the source ring template.

    Input structure:
    - The grid is mostly black and contains 2-3 red descending diagonal segments, 2-3 blue ascending diagonal segments, and one isolated green 3x3 hollow ring template.
    - Each red segment is a straight left-top to right-bottom run; each blue segment is a straight right-top to left-bottom run.
    - The green ring is only a template instruction and does not overlap the segment families.

    Rule:
    - For every red segment, copy that finite segment onto every parallel diagonal obtained by shifting it horizontally by multiples of three columns, painting only black cells red.
    - Do the same for every blue segment, again only onto black cells.
    - Any cell covered by both replicated families becomes yellow.
    - Around every yellow crossing, stamp a 3x3 green hollow ring but only on black neighbor cells; keep the yellow center and any existing red/blue cells unchanged.
    - Finally erase the original green template ring back to black.

    Key insight:
    - The output crossings are not present in the input; they emerge only after independently repeating the two diagonal segment families on their mod-3 parallel offsets, while the input green ring is a one-shot stencil that disappears after use.
    """
    return _apply_transform(np.asarray(grid, dtype=int))


def _segment_candidates(shape: tuple[int, int], slope: int) -> list[list[tuple[int, int]]]:
    h, w = shape
    candidates: list[list[tuple[int, int]]] = []
    for length in range(3, 6):
        if slope == 1:
            for r0 in range(h - length + 1):
                for c0 in range(w - length + 1):
                    candidates.append([(r0 + i, c0 + i) for i in range(length)])
        else:
            for r0 in range(h - length + 1):
                for c0 in range(length - 1, w):
                    candidates.append([(r0 + i, c0 - i) for i in range(length)])
    return candidates


def _touches_reserved(cells: Iterable[tuple[int, int]], reserved: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in reserved:
                    return True
    return False


def _place_seed_segments(
    shape: tuple[int, int],
    count: int,
    slope: int,
    reserved: set[tuple[int, int]],
) -> list[list[tuple[int, int]]] | None:
    candidates = _segment_candidates(shape, slope)
    random.shuffle(candidates)
    placed: list[list[tuple[int, int]]] = []
    occupied = set(reserved)
    for cells in candidates:
        if _touches_reserved(cells, occupied):
            continue
        placed.append(cells)
        occupied.update(cells)
        if len(placed) == count:
            return placed
    return None


def _pairwise_far_enough(points: list[tuple[int, int]], gap: int) -> bool:
    for i, (r1, c1) in enumerate(points):
        for r2, c2 in points[i + 1 :]:
            if max(abs(r1 - r2), abs(c1 - c2)) < gap:
                return False
    return True


def _has_unique_family_contribution(families: list[set[tuple[int, int]]]) -> bool:
    for i, family in enumerate(families):
        others: set[tuple[int, int]] = set()
        for j, other in enumerate(families):
            if i != j:
                others |= other
        if not (family - others):
            return False
    return True


def _template_centers(
    shape: tuple[int, int],
    seed_cells: set[tuple[int, int]],
    red_cover: set[tuple[int, int]],
    blue_cover: set[tuple[int, int]],
    intersections: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    centers: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            area = {(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)}
            if area & seed_cells:
                continue
            if area & red_cover:
                continue
            if area & blue_cover:
                continue
            if any(max(abs(r - ir), abs(c - ic)) <= 1 for ir, ic in intersections):
                continue
            centers.append((r, c))
    return centers


def _build_input_grid(
    shape: tuple[int, int],
    red_segments: list[list[tuple[int, int]]],
    blue_segments: list[list[tuple[int, int]]],
    template_center: tuple[int, int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for segment in red_segments:
        for r, c in segment:
            grid[r, c] = RED
    for segment in blue_segments:
        for r, c in segment:
            grid[r, c] = BLUE
    r, c = template_center
    for dr, dc in RING_OFFSETS:
        grid[r + dr, c + dc] = GREEN
    return grid


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(11, 15)
    w = random.randint(11, 15)
    shape = (h, w)

    red_count = random.randint(2, 3)
    blue_count = random.randint(2, 3)

    red_segments = _place_seed_segments(shape, red_count, slope=1, reserved=set())
    if red_segments is None:
        return None

    red_seed_cells = {cell for segment in red_segments for cell in segment}
    blue_segments = _place_seed_segments(shape, blue_count, slope=-1, reserved=red_seed_cells)
    if blue_segments is None:
        return None

    red_families = [_replicate_segment(segment, shape) for segment in red_segments]
    blue_families = [_replicate_segment(segment, shape) for segment in blue_segments]

    if not _has_unique_family_contribution(red_families):
        return None
    if not _has_unique_family_contribution(blue_families):
        return None

    red_cover = set().union(*red_families)
    blue_cover = set().union(*blue_families)
    intersections = red_cover & blue_cover

    if not (2 <= len(intersections) <= 10):
        return None
    if not _pairwise_far_enough(sorted(intersections), gap=3):
        return None
    if not any(1 <= r < h - 1 and 1 <= c < w - 1 for r, c in intersections):
        return None

    seed_cells = red_seed_cells | {cell for segment in blue_segments for cell in segment}
    centers = _template_centers(shape, seed_cells, red_cover, blue_cover, intersections)
    if not centers:
        return None

    template_center = random.choice(centers)
    grid = _build_input_grid(shape, red_segments, blue_segments, template_center)

    red_found = _extract_segments(grid, RED, slope=1)
    blue_found = _extract_segments(grid, BLUE, slope=-1)
    if not (2 <= len(red_found) <= 3 and 2 <= len(blue_found) <= 3):
        return None
    if not _find_template_ring_cells(grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
