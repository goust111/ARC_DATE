from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

MAX_ATTEMPTS = 500
FIVE_MARKER_PATTERNS = (
    (0, 1, 2, 3, 4),
    (0, 1, 2, 3, 5),
    (0, 1, 2, 4, 5),
)
SIX_MARKER_PATTERN = (0, 1, 2, 3, 4, 5)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _trace_rays(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {BLACK, RED, GREEN}

    starts = np.argwhere(g == RED)
    assert len(starts) == 1
    start_r, start_c = map(int, starts[0])
    assert start_c == 0

    marker_mask = g == GREEN
    out = g.copy()
    opened_markers: set[tuple[int, int]] = set()
    seen_rays: set[tuple[int, int, int, int]] = set()
    stack: list[tuple[int, int, int, int]] = [(start_r, start_c, 0, 1)]
    h, w = g.shape

    while stack:
        r, c, dr, dc = stack.pop()
        state = (r, c, dr, dc)
        if state in seen_rays:
            continue
        seen_rays.add(state)

        rr, cc = r, c
        while True:
            nr, nc = rr + dr, cc + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break

            if marker_mask[nr, nc]:
                marker = (int(nr), int(nc))
                if marker not in opened_markers:
                    opened_markers.add(marker)
                    if dr == 0:
                        stack.append((rr, cc, -1, 0))
                        stack.append((rr, cc, 1, 0))
                    else:
                        stack.append((rr, cc, 0, -1))
                        stack.append((rr, cc, 0, 1))
                break

            if g[nr, nc] == BLACK:
                out[nr, nc] = RED
            rr, cc = nr, nc

    return out


def _corner_exterior(wall_mask: np.ndarray) -> np.ndarray:
    h, w = wall_mask.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r, c in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        if not wall_mask[r, c] and not exterior[r, c]:
            exterior[r, c] = True
            q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not wall_mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    return exterior


def _label_enclosed_regions(wall_mask: np.ndarray, exterior: np.ndarray) -> np.ndarray:
    h, w = wall_mask.shape
    region_ids = np.zeros((h, w), dtype=int)
    q: deque[tuple[int, int]] = deque()
    next_id = 0

    for r in range(h):
        for c in range(w):
            if wall_mask[r, c] or exterior[r, c] or region_ids[r, c] != 0:
                continue

            next_id += 1
            region_ids[r, c] = next_id
            q.append((r, c))
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not wall_mask[nr, nc]
                        and not exterior[nr, nc]
                        and region_ids[nr, nc] == 0
                    ):
                        region_ids[nr, nc] = next_id
                        q.append((nr, nc))

    return region_ids


def _marker_region_counts(grid: np.ndarray, region_ids: np.ndarray) -> dict[int, int]:
    h, w = grid.shape
    counts = {rid: 0 for rid in range(1, int(region_ids.max()) + 1)}

    for mr, mc in map(tuple, np.argwhere(grid == GREEN)):
        adjacent_regions = set()
        for nr, nc in _neighbors4(int(mr), int(mc)):
            if 0 <= nr < h and 0 <= nc < w and region_ids[nr, nc] > 0:
                adjacent_regions.add(int(region_ids[nr, nc]))
        for rid in adjacent_regions:
            counts[rid] += 1

    return counts


def _analyze(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, int], np.ndarray]:
    ray_grid = _trace_rays(grid)
    wall_mask = (ray_grid == RED) | (grid == GREEN)
    exterior = _corner_exterior(wall_mask)
    region_ids = _label_enclosed_regions(wall_mask, exterior)
    counts = _marker_region_counts(np.asarray(grid, dtype=int), region_ids)

    out = ray_grid.copy()
    for rid, count in counts.items():
        out[region_ids == rid] = BLUE if count % 2 == 1 else YELLOW
    return out, region_ids, counts, ray_grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a recursive red ray network from the left-edge start and fill each enclosed region by the parity of its green markers.

    Input structure:
    - The grid has black background, exactly one red start pixel on the left edge, and several non-edge green marker pixels.
    - Green markers are isolated control/count objects and are preserved in place.
    - No blue or yellow appears in the input; those colors are reserved for output fills.

    Rule:
    - Emit a red ray from the start toward the right, painting black cells until the ray reaches the first green marker in that direction or the grid boundary.
    - On the first hit to a marker, branch from the red cell just before that marker into the two perpendicular directions; process these branches depth-first with down preferred over up and right preferred over left.
    - After all rays finish, treat the red ray network together with all green markers as walls; flood from the four corners through remaining black cells to identify the exterior.
    - Each non-exterior black component is an enclosed region. Count the green markers that are 4-neighbor-adjacent to that region; fill the region blue for odd counts and yellow for even counts.

    Key insight:
    - The green markers have two inseparable roles: they determine where the recursive ray network turns, and the resulting enclosed regions are colored by the parity of the markers touching each region.
    """
    out, _region_ids, _counts, _ray_grid = _analyze(np.asarray(grid, dtype=int))
    return out


def _has_odd_and_even(counts: dict[int, int]) -> bool:
    if not counts:
        return False
    values = list(counts.values())
    return any(v % 2 == 1 for v in values) and any(v % 2 == 0 for v in values)


def _marker_spacing_ok(markers: list[tuple[int, int]], candidate: tuple[int, int]) -> bool:
    r, c = candidate
    for mr, mc in markers:
        if max(abs(r - mr), abs(c - mc)) <= 1:
            return False
    return True


def _sample_base_points() -> tuple[int, int, int, list[tuple[int, int]]] | None:
    h = random.randint(16, 22)
    w = random.randint(16, 24)
    start_r = random.randint(h // 3, (2 * h) // 3)
    first_c = random.randint(3, 5)

    top_max = start_r - 2
    bottom_max = h - start_r - 3
    if top_max < 3 or bottom_max < 4:
        return None

    top_span = random.randint(3, top_max)
    bottom_span = random.randint(3, bottom_max - 1)
    right_c = random.randint(first_c + 5, w - 3)

    points = [
        (start_r, first_c),
        (start_r - top_span, first_c - 1),
        (start_r + bottom_span, first_c - 1),
        (start_r - top_span + 1, right_c),
        (start_r + bottom_span - 1, right_c),
        (start_r + bottom_span + 1, right_c - 1),
    ]
    if any(r <= 0 or c <= 0 or r >= h - 1 or c >= w - 1 for r, c in points):
        return None

    markers: list[tuple[int, int]] = []
    for point in points:
        if not _marker_spacing_ok(markers, point):
            return None
        markers.append(point)

    return h, w, start_r, points


def _add_passive_markers(
    grid: np.ndarray,
    region_ref: np.ndarray,
    counts_ref: dict[int, int],
    ray_ref: np.ndarray,
    need: int,
) -> np.ndarray | None:
    g = grid.copy()
    h, w = g.shape
    markers = list(map(tuple, np.argwhere(g == GREEN)))
    red_ref = ray_ref == RED

    for _ in range(need):
        candidates = [
            (r, c)
            for r in range(1, h - 1)
            for c in range(1, w - 1)
            if g[r, c] == BLACK and _marker_spacing_ok(markers, (r, c))
        ]
        random.shuffle(candidates)

        placed = False
        for candidate in candidates:
            trial = g.copy()
            trial[candidate] = GREEN
            _out2, region2, counts2, ray2 = _analyze(trial)
            if (
                np.array_equal(region2, region_ref)
                and counts2 == counts_ref
                and np.array_equal(ray2 == RED, red_ref)
            ):
                g = trial
                markers.append(candidate)
                placed = True
                break

        if not placed:
            return None

    return g


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_base_points()
    if sampled is None:
        return None
    h, w, start_r, points = sampled

    target_marker_count = random.randint(5, 8)
    base_pattern = random.choice(FIVE_MARKER_PATTERNS) if target_marker_count == 5 else SIX_MARKER_PATTERN

    grid = np.zeros((h, w), dtype=int)
    grid[start_r, 0] = RED
    for idx in base_pattern:
        r, c = points[idx]
        grid[r, c] = GREEN

    try:
        out, region_ids, counts, ray_grid = _analyze(grid)
    except AssertionError:
        return None

    if int(region_ids.max()) < 2 or not _has_odd_and_even(counts):
        return None

    extras_needed = target_marker_count - len(base_pattern)
    if extras_needed > 0:
        grid = _add_passive_markers(grid, region_ids, counts, ray_grid, extras_needed)
        if grid is None:
            return None
        out, region_ids, counts, _ray_grid = _analyze(grid)
        if int(region_ids.max()) < 2 or not _has_odd_and_even(counts):
            return None

    if not (5 <= int(np.count_nonzero(grid == GREEN)) <= 8):
        return None
    if not np.any(out == BLUE) or not np.any(out == YELLOW):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
