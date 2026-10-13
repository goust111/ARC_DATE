from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
ENDPOINT = 2
MASK = 5
WALL = 8

RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 400


def _mirror_col(width: int, col: int) -> int:
    return int(width - 1 - col)


def _gray_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == MASK)
    assert len(coords) > 0, "expected a gray occluding rectangle"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == MASK), "gray cells must form a solid rectangle"
    return int(r0), int(c0), int(r1), int(c1)


def _restore_from_mirror(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    r0, c0, r1, c1 = _gray_bbox(g)
    width = g.shape[1]
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            mc = _mirror_col(width, c)
            mirrored = int(g[r, mc])
            assert mirrored != MASK, "gray rectangle may not cross the symmetry axis"
            out[r, c] = mirrored
    return out


def _endpoint_coords(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    pts = np.argwhere(np.asarray(grid, dtype=int) == ENDPOINT)
    assert len(pts) == 2, "expected exactly two red endpoints"
    pts = [tuple(map(int, pt)) for pt in pts]
    pts.sort()
    return pts[0], pts[1]


def _path_analysis(
    grid: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]] | None]:
    g = np.asarray(grid, dtype=int)
    start, goal = _endpoint_coords(g)
    h, w = g.shape
    passable = g != WALL

    dist = np.full((h, w), -1, dtype=int)
    ways = np.zeros((h, w), dtype=np.int8)
    parent = np.full((h, w, 2), -1, dtype=int)
    q: deque[tuple[int, int]] = deque([start])
    dist[start] = 0
    ways[start] = 1

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not passable[nr, nc]:
                continue
            nd = int(dist[r, c] + 1)
            if dist[nr, nc] == -1:
                dist[nr, nc] = nd
                ways[nr, nc] = ways[r, c]
                parent[nr, nc] = (r, c)
                q.append((nr, nc))
            elif dist[nr, nc] == nd:
                ways[nr, nc] = min(2, int(ways[nr, nc]) + int(ways[r, c]))

    goal_dist = int(dist[goal])
    goal_ways = int(ways[goal])
    if goal_dist < 0 or goal_ways != 1:
        return goal_dist, goal_ways, None

    path_rev = [goal]
    cur = goal
    while cur != start:
        pr, pc = map(int, parent[cur])
        assert pr >= 0
        cur = (pr, pc)
        path_rev.append(cur)
    path = list(reversed(path_rev))
    return goal_dist, goal_ways, path


def _paint_solution(grid: np.ndarray) -> np.ndarray:
    dist, ways, path = _path_analysis(grid)
    assert dist >= 0 and ways == 1 and path is not None, "expected a unique shortest path"
    out = np.asarray(grid, dtype=int).copy()
    for r, c in path:
        out[r, c] = ENDPOINT
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the mirrored hidden wall segment, then draw the unique shortest route between the red endpoints.

    Input structure:
    - The grid has black traversable cells, cyan wall cells, one solid gray occluding rectangle, and exactly two red endpoint pixels.
    - If the gray rectangle were removed, the cyan wall layout would be left-right mirror-symmetric across the grid's vertical centerline.
    - The endpoints lie outside the gray rectangle.

    Rule:
    - Replace every gray cell with the content of its horizontally mirrored partner cell.
    - In the restored grid, treat cyan as blocking and move only in four directions through non-cyan cells.
    - Find the unique shortest path between the two red endpoints and color every cell on that path red.

    Key insight:
    - The gray rectangle hides the missing half of a symmetric wall pattern, and the concealed wall cells invalidate the tempting shortcut that appears before restoration.
    """
    restored = _restore_from_mirror(np.asarray(grid, dtype=int))
    return _paint_solution(restored)


def _build_clean_grid() -> tuple[np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int, int, int]]:
    h = int(RNG.integers(13, 26))
    w = int(RNG.choice(np.arange(13, 26, 2)))
    mid = w // 2
    grid = np.full((h, w), WALL, dtype=int)

    branch_row_candidates = list(range(2, h - 2, 2))
    branch_count = int(RNG.integers(4, min(7, len(branch_row_candidates)) + 1))
    branch_rows = sorted(map(int, RNG.choice(branch_row_candidates, size=branch_count, replace=False)))
    pair_index = int(RNG.integers(0, branch_count - 1))
    r1, r2 = branch_rows[pair_index], branch_rows[pair_index + 1]

    length_lo = 3
    length_hi = max(length_lo + 1, mid - 1)
    lengths = [int(RNG.integers(length_lo, length_hi)) for _ in branch_rows]
    special_length = int(RNG.integers(length_lo, length_hi))
    lengths[pair_index] = special_length
    lengths[pair_index + 1] = special_length

    grid[1 : h - 1, mid] = BG
    for row, length in zip(branch_rows, lengths):
        grid[row, mid - length : mid + length + 1] = BG

    # Add a few mirrored isolated background pockets as distractors, without touching the routed tree.
    for _ in range(int(RNG.integers(0, 3))):
        for _attempt in range(20):
            rr = int(RNG.integers(1, h - 1))
            cc = int(RNG.integers(1, mid - 1))
            mc = _mirror_col(w, cc)
            cells = [(rr, cc), (rr, mc)]
            if any(grid[r, c] != WALL for r, c in cells):
                continue
            neighbors = []
            for r, c in cells:
                neighbors.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])
            if any(0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BG for nr, nc in neighbors):
                continue
            for r, c in cells:
                grid[r, c] = BG
            break

    c_left = int(mid - special_length)
    start = (int(r1), c_left)
    goal = (int(r2), c_left)
    grid[start] = ENDPOINT
    grid[goal] = ENDPOINT

    link_offset = int(RNG.integers(1, special_length))
    link_col = int(c_left + link_offset)
    max_width = int(mid - link_col)
    mask_width = int(RNG.integers(1, min(4, max_width + 1)))
    mask_bbox = (int(r1), int(link_col), int(r2), int(link_col + mask_width - 1))
    return grid, start, goal, mask_bbox


def _apply_mask(clean: np.ndarray, mask_bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = clean.copy()
    r0, c0, r1, c1 = mask_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = MASK
    return out


def _flip_horiz(grid: np.ndarray) -> np.ndarray:
    return np.asarray(grid[:, ::-1], dtype=int)


def _validate_clean_grid(clean: np.ndarray) -> None:
    temp = clean.copy()
    temp[temp == ENDPOINT] = BG
    assert np.array_equal(temp, temp[:, ::-1]), "wall/background pattern must be mirror-symmetric"
    dist, ways, path = _path_analysis(clean)
    assert dist >= 8, "path should be long enough to be visually meaningful"
    assert ways == 1 and path is not None, "restored grid must have a unique shortest path"


def _validate_candidate(clean: np.ndarray, masked: np.ndarray, mask_bbox: tuple[int, int, int, int]) -> None:
    restored = _restore_from_mirror(masked)
    assert np.array_equal(restored, clean), "mirror restoration must recover the clean grid exactly"

    r0, c0, r1, c1 = mask_bbox
    hidden = clean[r0 : r1 + 1, c0 : c1 + 1]
    assert np.any(hidden == WALL), "mask must hide at least one true wall cell"
    assert np.any(hidden == BG), "mask should also hide some background to expose mixed recovery"

    dist_clean, ways_clean, path_clean = _path_analysis(clean)
    assert dist_clean >= 0 and ways_clean == 1 and path_clean is not None

    naive = masked.copy()
    naive[naive == MASK] = BG
    dist_naive, ways_naive, _ = _path_analysis(naive)
    assert dist_naive >= 0
    assert dist_naive < dist_clean or ways_naive != 1, "restoration must change routing semantics"

    solved = main(masked)
    assert not np.any(solved == MASK), "output may not contain gray cells"
    assert np.count_nonzero(solved == ENDPOINT) > 2, "solution should draw a visible red path"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        clean, _start, _goal, mask_bbox = _build_clean_grid()
        _validate_clean_grid(clean)
        masked = _apply_mask(clean, mask_bbox)
        if RNG.random() < 0.5:
            clean = _flip_horiz(clean)
            masked = _flip_horiz(masked)
            r0, c0, r1, c1 = mask_bbox
            w = clean.shape[1]
            mask_bbox = (r0, _mirror_col(w, c1), r1, _mirror_col(w, c0))
        try:
            _validate_candidate(clean, masked, mask_bbox)
        except AssertionError:
            continue
        return masked
    raise RuntimeError("failed to generate a valid mirror-restore routing task")
