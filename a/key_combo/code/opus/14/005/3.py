import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

_MAX_GEN_ATTEMPTS = 500


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _strictly_inside_bbox(bbox: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 < r < r1 and c0 < c < c1


def _classify_objects(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]]:
    obstacle_comps: list[list[tuple[int, int]]] = []
    obstacle_cells: set[tuple[int, int]] = set()
    mixed_mask = (grid == BLUE) | (grid == RED)

    for comp in _components_from_mask(mixed_mask):
        if not all(_strictly_inside_bbox(frame_bbox, r, c) for r, c in comp):
            continue
        if any(int(grid[r, c]) == BLUE for r, c in comp):
            obstacle_comps.append(comp)
            obstacle_cells.update(comp)

    markers: list[tuple[int, int]] = []
    for r, c in np.argwhere(grid == RED):
        rr, cc = int(r), int(c)
        if _strictly_inside_bbox(frame_bbox, rr, cc) and (rr, cc) not in obstacle_cells:
            markers.append((rr, cc))
    return obstacle_comps, markers


def _forbidden_mask(
    shape: tuple[int, int],
    grid: np.ndarray,
    obstacle_comps: list[list[tuple[int, int]]],
) -> np.ndarray:
    h, w = shape
    forbidden = np.zeros((h, w), dtype=bool)
    for comp in obstacle_comps:
        radius = sum(1 for r, c in comp if int(grid[r, c]) == BLUE)
        for r, c in comp:
            for dr in range(-radius, radius + 1):
                rr = r + dr
                if not (0 <= rr < h):
                    continue
                span = radius - abs(dr)
                c0 = max(0, c - span)
                c1 = min(w - 1, c + span)
                forbidden[rr, c0 : c1 + 1] = True
    return forbidden


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp blue hollow rings around red markers inside a green frame, but suppress ring pixels that fall in obstacle-defined exclusion zones.

    Input structure:
    - The grid has a black background and one green hollow rectangular frame.
    - Inside the frame are isolated red marker pixels plus one or more 4-connected mixed red/blue obstacle blocks.
    - Red pixels outside the frame are distractors and must be ignored.

    Rule:
    - Use the green rectangle's bounding box to select the active area, then identify obstacle components as the interior 4-connected red/blue components that contain at least one blue pixel.
    - For each such obstacle, count its blue pixels B and mark as forbidden every cell within Manhattan distance <= B of any obstacle cell.
    - For each interior red marker that is not part of an obstacle, try to paint its 8-neighborhood blue; skip any target cell that is forbidden, is another marker center, or already belongs to a preserved non-background input object.

    Key insight:
    - The red pixels inside the frame split into two roles: isolated red cells are stamp centers, while red cells attached to blue obstacle components only encode exclusion zones that carve away parts of each ring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_bbox = _bbox_of_mask(g == GREEN)
    assert frame_bbox is not None, "expected a green frame"

    obstacle_comps, markers = _classify_objects(g, frame_bbox)
    forbidden = _forbidden_mask(g.shape, g, obstacle_comps)
    marker_set = set(markers)

    out = g.copy()
    for r, c in markers:
        for rr, cc in neighbors8(r, c):
            if not in_bounds(g.shape, rr, cc):
                continue
            if (rr, cc) in marker_set:
                continue
            if forbidden[rr, cc]:
                continue
            if int(g[rr, cc]) != BLACK:
                continue
            out[rr, cc] = BLUE
    return out


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _bbox_manhattan_distance(bbox: tuple[int, int, int, int], r: int, c: int) -> int:
    r0, c0, r1, c1 = bbox
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return dr + dc


def _random_connected_shape(size: int) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(tuple(frontier)))
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    return norm


def _shape_bbox(shape: list[tuple[int, int]]) -> tuple[int, int]:
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    return max_r + 1, max_c + 1


def _translate_shape(shape: list[tuple[int, int]], anchor_r: int, anchor_c: int) -> list[tuple[int, int]]:
    return [(anchor_r + r, anchor_c + c) for r, c in shape]


def _cells_have_gap(
    cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    min_chebyshev: int,
) -> bool:
    for cell in cells:
        for other in occupied:
            if _chebyshev(cell, other) < min_chebyshev:
                return False
    return True


def _place_obstacles(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    obstacle_count: int,
) -> list[list[tuple[int, int]]] | None:
    r0, c0, r1, c1 = frame_bbox
    safe_r0, safe_c0 = r0 + 2, c0 + 2
    safe_r1, safe_c1 = r1 - 2, c1 - 2
    if safe_r0 > safe_r1 or safe_c0 > safe_c1:
        return None

    obstacle_comps: list[list[tuple[int, int]]] = []
    occupied: set[tuple[int, int]] = set()

    for _ in range(obstacle_count):
        placed = False
        for _ in range(120):
            size = random.randint(3, 6)
            blue_count = random.randint(1, min(3, size - 1))
            shape = _random_connected_shape(size)
            sh, sw = _shape_bbox(shape)
            max_r = safe_r1 - sh + 1
            max_c = safe_c1 - sw + 1
            if max_r < safe_r0 or max_c < safe_c0:
                continue

            anchor_r = random.randint(safe_r0, max_r)
            anchor_c = random.randint(safe_c0, max_c)
            cells = _translate_shape(shape, anchor_r, anchor_c)
            if not _cells_have_gap(cells, occupied, min_chebyshev=2):
                continue

            blue_cells = set(random.sample(cells, blue_count))
            for r, c in cells:
                grid[r, c] = BLUE if (r, c) in blue_cells else RED
            obstacle_comps.append(cells)
            occupied.update(cells)
            placed = True
            break
        if not placed:
            return None
    return obstacle_comps


def _marker_stamp_targets(
    grid: np.ndarray,
    markers: list[tuple[int, int]],
    forbidden: np.ndarray,
    marker: tuple[int, int],
) -> set[tuple[int, int]]:
    marker_set = set(markers)
    targets: set[tuple[int, int]] = set()
    r, c = marker
    for rr, cc in neighbors8(r, c):
        if not in_bounds(grid.shape, rr, cc):
            continue
        if (rr, cc) in marker_set:
            continue
        if forbidden[rr, cc]:
            continue
        if int(grid[rr, cc]) != BLACK:
            continue
        targets.add((rr, cc))
    return targets


def _marker_candidates(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    forbidden: np.ndarray,
    obstacle_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    candidates: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(grid[r, c]) != BLACK or forbidden[r, c]:
                continue
            if any(_chebyshev((r, c), obstacle) < 2 for obstacle in obstacle_cells):
                continue

            blocked = 0
            free_black = 0
            for rr, cc in neighbors8(r, c):
                if not in_bounds(grid.shape, rr, cc):
                    continue
                if forbidden[rr, cc]:
                    blocked += 1
                elif int(grid[rr, cc]) == BLACK:
                    free_black += 1
            if blocked >= 1 and free_black >= 2:
                candidates.append((r, c))
    return candidates


def _choose_markers(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    forbidden: np.ndarray,
    obstacle_cells: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    candidates = _marker_candidates(grid, frame_bbox, forbidden, obstacle_cells)
    if len(candidates) < 2:
        return None

    marker_count = random.choice([2, 2, 3, 3, 4])
    if len(candidates) < marker_count:
        marker_count = len(candidates)
    if marker_count < 2:
        return None

    want_overlap = marker_count >= 2 and random.random() < 0.5
    close_pairs = [
        (a, b)
        for i, a in enumerate(candidates)
        for b in candidates[i + 1 :]
        if 1 <= _chebyshev(a, b) <= 2
    ]

    for _ in range(200):
        chosen: list[tuple[int, int]] = []
        if want_overlap and close_pairs:
            a, b = random.choice(close_pairs)
            chosen = [a, b]

        pool = candidates[:]
        random.shuffle(pool)
        for cand in pool:
            if cand in chosen:
                continue
            if want_overlap and len(chosen) >= 2:
                if any(_chebyshev(cand, other) < 3 for other in chosen):
                    continue
            elif any(_chebyshev(cand, other) < 3 for other in chosen):
                continue
            chosen.append(cand)
            if len(chosen) == marker_count:
                break
        if len(chosen) != marker_count:
            continue

        stamp_sets = [_marker_stamp_targets(grid, chosen, forbidden, marker) for marker in chosen]
        if any(len(stamps) < 2 for stamps in stamp_sets):
            continue
        if want_overlap and len(chosen) >= 2:
            overlap_ok = False
            for i in range(len(chosen)):
                for j in range(i + 1, len(chosen)):
                    if stamp_sets[i] & stamp_sets[j]:
                        overlap_ok = True
                        break
                if overlap_ok:
                    break
            if not overlap_ok:
                continue
        return chosen
    return None


def _place_distractors(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    count: int,
) -> bool:
    h, w = grid.shape
    near: list[tuple[int, int]] = []
    far: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BLACK:
                continue
            d = _bbox_manhattan_distance(frame_bbox, r, c)
            if 1 <= d <= 4:
                near.append((r, c))
            else:
                far.append((r, c))

    random.shuffle(near)
    random.shuffle(far)
    chosen: list[tuple[int, int]] = []
    for cand in near + far:
        if any(_chebyshev(cand, other) < 2 for other in chosen):
            continue
        chosen.append(cand)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        return False

    for r, c in chosen:
        grid[r, c] = RED
    return True


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = GREEN
    grid[r1, c0 : c1 + 1] = GREEN
    grid[r0 : r1 + 1, c0] = GREEN
    grid[r0 : r1 + 1, c1] = GREEN


def _build_attempt() -> np.ndarray | None:
    h = random.randint(17, 24)
    w = random.randint(17, 24)
    grid = np.full((h, w), BLACK, dtype=int)

    frame_h = random.randint(11, min(16, h - 2))
    frame_w = random.randint(11, min(16, w - 2))
    top = random.randint(1, h - frame_h - 1)
    left = random.randint(1, w - frame_w - 1)
    frame_bbox = (top, left, top + frame_h - 1, left + frame_w - 1)
    _draw_frame(grid, frame_bbox)

    obstacle_count = random.choice([1, 2, 2, 3])
    obstacle_comps = _place_obstacles(grid, frame_bbox, obstacle_count)
    if obstacle_comps is None:
        return None

    obstacle_cells = {cell for comp in obstacle_comps for cell in comp}
    forbidden = _forbidden_mask(grid.shape, grid, obstacle_comps)

    markers = _choose_markers(grid, frame_bbox, forbidden, obstacle_cells)
    if markers is None:
        return None
    for r, c in markers:
        grid[r, c] = RED

    distractor_count = random.choice([1, 2])
    if not _place_distractors(grid, frame_bbox, distractor_count):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    # Re-check the intended structure after markers are inserted.
    obstacle_comps2, markers2 = _classify_objects(grid, frame_bbox)
    if len(obstacle_comps2) != len(obstacle_comps) or len(markers2) != len(markers):
        return None

    forbidden2 = _forbidden_mask(grid.shape, grid, obstacle_comps2)
    for marker in markers2:
        blocked_neighbors = sum(
            1
            for rr, cc in neighbors8(*marker)
            if in_bounds(grid.shape, rr, cc) and forbidden2[rr, cc]
        )
        if blocked_neighbors == 0:
            return None
        if not _marker_stamp_targets(grid, markers2, forbidden2, marker):
            return None

    if int(np.sum(out == BLUE)) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
