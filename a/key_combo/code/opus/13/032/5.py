from __future__ import annotations

import math
import random

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
REGION_COLORS = (1, 2, 3, 4, 6, 8)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _ordered_path(mask: np.ndarray) -> list[tuple[int, int]]:
    comps = _components(mask)
    assert len(comps) == 1, "expected exactly one grey path"
    pts = {tuple(map(int, cell)) for cell in comps[0]}
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    endpoints = []
    for cell in pts:
        neigh = [n for n in _neighbors4(*cell) if n in pts]
        adjacency[cell] = neigh
        if len(neigh) == 1:
            endpoints.append(cell)
        else:
            assert len(neigh) == 2, "path must be non-branching and one-pixel wide"
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    path = [min(endpoints)]
    prev = None
    cur = path[0]
    while True:
        nxt = [cell for cell in adjacency[cur] if cell != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path walk must be unique"
        prev, cur = cur, nxt[0]
        path.append(cur)
    assert len(path) == len(pts), "path traversal must cover every path pixel"
    return path


def _divider_coords(grid: np.ndarray) -> np.ndarray:
    comps = _components(grid == ORANGE)
    assert len(comps) == 1, "expected exactly one orange divider"
    return comps[0]


def _region_components(grid: np.ndarray) -> list[dict]:
    regions = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) not in (BG, GREY, ORANGE)):
        comps = _components(grid == color)
        assert len(comps) == 1, "each region color must form exactly one component"
        regions.append({"color": int(color), "coords": comps[0]})
    assert len(regions) >= 2, "expected at least two colored regions"
    return regions


def _region_angle(region_coords: np.ndarray, center_r: float, center_c: float) -> float:
    centroid_r, centroid_c = region_coords.mean(axis=0)
    return math.atan2(float(centroid_r - center_r), float(centroid_c - center_c)) % (2 * math.pi)


def _ordered_regions(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    divider = _divider_coords(grid)
    center_r, center_c = divider.mean(axis=0)
    ordered = []
    for region in _region_components(grid):
        angle = _region_angle(region["coords"], center_r, center_c)
        ordered.append((angle, region))
    ordered.sort(key=lambda item: item[0])
    angles = [angle for angle, _region in ordered]
    for left, right in zip(angles, angles[1:]):
        assert not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9), "region order must be unique"
    return [region for _angle, region in ordered], divider


def _apply_rotation(grid: np.ndarray, regions: list[dict], k: int, *, erase_path: bool) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    n = len(regions)
    for i, region in enumerate(regions):
        source = regions[(i - k) % n]
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = int(source["color"])
    if erase_path:
        out[grid == GREY] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the surrounding region colors by a path-length-dependent cyclic offset, then erase the path.

    Input structure:
    - Black background contains one orange central divider, several separated monochrome colored regions, and one grey one-pixel-wide path.
    - The grey path is 4-connected, has exactly two endpoints, and does not touch the colored regions or the divider.
    - Each non-black, non-grey, non-orange color forms one 4-connected region around the divider.

    Rule:
    - Count the number of grey path pixels to get L.
    - Count the colored regions to get N, then sort those regions by centroid angle around the orange divider centroid in screen-coordinate clockwise order starting from the right.
    - Recolor region i with the original color of region (i - (L mod N)) mod N, keeping every region's shape and position fixed, then erase the grey path to black.

    Key insight:
    - The grey path is only a counter: its pixel count determines how far the surrounding region colors rotate around the central divider.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    path = _ordered_path(g == GREY)
    regions, _divider = _ordered_regions(g)
    return _apply_rotation(g, regions, len(path) % len(regions), erase_path=True)


def _normalize_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _transform_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(cells, dtype=int)
    h = int(arr[:, 0].max()) + 1
    w = int(arr[:, 1].max()) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in cells:
        patch[r, c] = 1
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    patch = np.rot90(patch, random.randrange(4))
    return [tuple(map(int, cell)) for cell in np.argwhere(patch > 0)]


def _random_polyomino(area: int, max_h: int = 4, max_w: int = 4) -> list[tuple[int, int]]:
    for _ in range(120):
        cells = {(0, 0)}
        frontier = set(_neighbors4(0, 0))
        while frontier and len(cells) < area:
            candidate = random.choice(tuple(frontier))
            frontier.remove(candidate)
            new_cells = set(cells)
            new_cells.add(candidate)
            rows = [r for r, _c in new_cells]
            cols = [c for _r, c in new_cells]
            if max(rows) - min(rows) + 1 > max_h or max(cols) - min(cols) + 1 > max_w:
                continue
            cells = new_cells
            for nr, nc in _neighbors4(*candidate):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if len(cells) == area:
            return _transform_cells(_normalize_cells(cells))
    raise RuntimeError("failed to sample a region polyomino")


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _cells_have_gap(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if (r, c) in occupied:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _slot_offsets(rad_r: int, rad_c: int) -> list[tuple[int, int]]:
    diag_r = max(3, rad_r - 1)
    diag_c = max(3, rad_c - 1)
    return [
        (0, rad_c),
        (diag_r, diag_c),
        (rad_r, 0),
        (diag_r, -diag_c),
        (0, -rad_c),
        (-diag_r, -diag_c),
        (-rad_r, 0),
        (-diag_r, diag_c),
    ]


def _sample_slot_indices(n: int) -> list[int]:
    for _ in range(100):
        indices = sorted(random.sample(range(8), n))
        gaps = []
        for i, left in enumerate(indices):
            right = indices[(i + 1) % n]
            gap = (right - left) % 8
            gaps.append(gap)
        if max(gaps) <= 3 and any(i % 2 == 0 for i in indices) and any(i % 2 == 1 for i in indices):
            return indices
    raise RuntimeError("failed to sample surrounding region slots")


def _place_divider(grid: np.ndarray, center: tuple[int, int]) -> set[tuple[int, int]]:
    center_r, center_c = center
    v_arm, h_arm = random.choice(((1, 1), (2, 1), (1, 2)))
    cells = {(center_r, center_c)}
    for dr in range(-v_arm, v_arm + 1):
        cells.add((center_r + dr, center_c))
    for dc in range(-h_arm, h_arm + 1):
        cells.add((center_r, center_c + dc))
    for r, c in cells:
        grid[r, c] = ORANGE
    return cells


def _place_regions(
    grid: np.ndarray,
    center: tuple[int, int],
    divider_cells: set[tuple[int, int]],
    n_regions: int,
) -> list[dict] | None:
    h, w = grid.shape
    center_r, center_c = center
    rad_r = min(center_r - 3, h - center_r - 4)
    rad_c = min(center_c - 3, w - center_c - 4)
    if rad_r < 4 or rad_c < 4:
        return None

    offsets = _slot_offsets(rad_r, rad_c)
    colors = random.sample(list(REGION_COLORS), n_regions)
    occupied = set(divider_cells)
    placed_regions = []

    for slot_index, color in zip(_sample_slot_indices(n_regions), colors):
        base_r = center_r + offsets[slot_index][0]
        base_c = center_c + offsets[slot_index][1]
        placed = None
        for _ in range(80):
            cells = _random_polyomino(random.randint(3, 6))
            jitter_r = random.randint(-1, 1)
            jitter_c = random.randint(-1, 1)
            centroid = np.array(cells, dtype=float).mean(axis=0)
            shift_r = int(round(base_r + jitter_r - centroid[0]))
            shift_c = int(round(base_c + jitter_c - centroid[1]))
            translated = {(r + shift_r, c + shift_c) for r, c in cells}
            if not all(_in_bounds((h, w), r, c) for r, c in translated):
                continue
            if not _cells_have_gap(translated, occupied):
                continue
            if any(r in (0, h - 1) or c in (0, w - 1) for r, c in translated):
                continue
            placed = translated
            break
        if placed is None:
            return None
        coords = np.array(sorted(placed), dtype=int)
        grid[coords[:, 0], coords[:, 1]] = int(color)
        occupied |= placed
        placed_regions.append({"color": int(color), "coords": coords})
    return placed_regions


def _build_path(allowed: np.ndarray, length: int) -> list[tuple[int, int]] | None:
    starts = [tuple(map(int, cell)) for cell in np.argwhere(allowed)]
    random.shuffle(starts)

    def candidates(cur: tuple[int, int], used: set[tuple[int, int]]) -> list[tuple[int, int]]:
        out = []
        for nr, nc in _neighbors4(*cur):
            if not _in_bounds(allowed.shape, nr, nc) or not allowed[nr, nc] or (nr, nc) in used:
                continue
            bad = False
            for ar, ac in _neighbors4(nr, nc):
                if (ar, ac) in used and (ar, ac) != cur:
                    bad = True
                    break
            if not bad:
                degree = 0
                for ar, ac in _neighbors4(nr, nc):
                    if _in_bounds(allowed.shape, ar, ac) and allowed[ar, ac] and (ar, ac) not in used:
                        degree += 1
                out.append((degree, (nr, nc)))
        random.shuffle(out)
        out.sort(key=lambda item: item[0], reverse=True)
        return [cell for _degree, cell in out]

    def dfs(path: list[tuple[int, int]], used: set[tuple[int, int]]):
        if len(path) == length:
            return list(path)
        for nxt in candidates(path[-1], used):
            used.add(nxt)
            path.append(nxt)
            result = dfs(path, used)
            if result is not None:
                return result
            path.pop()
            used.remove(nxt)
        return None

    for start in starts:
        if not any(
            _in_bounds(allowed.shape, nr, nc) and allowed[nr, nc]
            for nr, nc in _neighbors4(*start)
        ):
            continue
        result = dfs([start], {start})
        if result is not None:
            return result
    return None


def _manual_expected(grid: np.ndarray) -> tuple[np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    path = _ordered_path(g == GREY)
    regions, _divider = _ordered_regions(g)
    k = len(path) % len(regions)
    return _apply_rotation(g, regions, k, erase_path=True), len(path), len(regions)


def _validate_grid(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert 15 <= g.shape[0] <= 19 and 15 <= g.shape[1] <= 19
    divider = _divider_coords(g)
    assert len(divider) in (5, 7), "divider should contain 5 or 7 pixels"

    path = _ordered_path(g == GREY)
    assert 3 <= len(path) <= 15

    regions, _divider = _ordered_regions(g)
    n_regions = len(regions)
    assert 4 <= n_regions <= 6
    assert len({region["color"] for region in regions}) == n_regions

    k = len(path) % n_regions
    assert k not in (0, 1), "rotation must be non-trivial and not equal to one step"

    occupied = g != BG
    path_mask = g == GREY
    protected = g == ORANGE
    moving = (g != BG) & (g != GREY) & (g != ORANGE)
    assert not np.any(_dilate4(protected) & path_mask), "path must not touch divider"
    assert not np.any(_dilate4(moving) & path_mask), "path must not touch colored regions"

    center_r, center_c = divider.mean(axis=0)
    angles = [_region_angle(region["coords"], center_r, center_c) for region in regions]
    wrapped = angles + [angles[0] + 2 * math.pi]
    min_gap = min(b - a for a, b in zip(wrapped, wrapped[1:]))
    assert min_gap > 0.30, "region angles must be clearly separated"

    expected, path_len, region_count = _manual_expected(g)
    actual = main(g)
    assert path_len == len(path) and region_count == n_regions
    assert np.array_equal(actual, expected), "solver output disagrees with independent oracle"

    erase_only = _apply_rotation(g, regions, 0, erase_path=True)
    fixed_one = _apply_rotation(g, regions, 1, erase_path=True)
    keep_path = _apply_rotation(g, regions, k, erase_path=False)
    assert not np.array_equal(expected, erase_only), "rotation step must matter"
    assert not np.array_equal(expected, fixed_one), "path length must not collapse to a fixed one-step rotation"
    assert not np.array_equal(expected, keep_path), "path erasure must matter"
    assert not np.array_equal(expected, g), "output should differ from input"
    assert occupied.sum() > 0


def _generate_once() -> np.ndarray | None:
    h = random.randint(15, 19)
    w = random.randint(15, 19)
    grid = np.zeros((h, w), dtype=int)

    center = (h // 2, w // 2)
    divider_cells = _place_divider(grid, center)
    n_regions = random.choice((4, 5, 6))
    placed_regions = _place_regions(grid, center, divider_cells, n_regions)
    if placed_regions is None:
        return None

    target_k = random.randint(2, n_regions - 1)
    valid_lengths = [length for length in range(3, 16) if length % n_regions == target_k]
    path_length = random.choice(valid_lengths)

    blocked = _dilate4(grid != BG)
    allowed = ~blocked
    allowed[0, :] = False
    allowed[-1, :] = False
    allowed[:, 0] = False
    allowed[:, -1] = False
    path = _build_path(allowed, path_length)
    if path is None:
        return None
    for r, c in path:
        grid[r, c] = GREY

    try:
        _validate_grid(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-rotation ARC grid")
