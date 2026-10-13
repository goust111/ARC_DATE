from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GENERATION_RETRIES = 400
POLYLINE_RETRIES = 250
PLACEMENT_RETRIES = 250


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(g.shape, rr, cc):
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            components.append({"color": color, "coords": coords, "area": len(coords)})
    return components


def _count_bends(coords: list[tuple[int, int]]) -> int:
    coord_set = set(coords)
    endpoints = 0
    bends = 0

    for r, c in coords:
        neighbors = [(rr, cc) for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if (rr, cc) in coord_set]
        deg = len(neighbors)
        assert deg in (1, 2), "expected a simple 1-pixel-wide open polyline"
        if deg == 1:
            endpoints += 1
            continue
        (r1, c1), (r2, c2) = neighbors
        v1 = (r1 - r, c1 - c)
        v2 = (r2 - r, c2 - c)
        if v1 != (-v2[0], -v2[1]):
            bends += 1

    assert endpoints == 2, "a valid polyline must have exactly two endpoints"
    assert bends >= 1, "puzzle instances require at least one bend per polyline"
    return bends


def _ring_output(bend_to_color: dict[int, int]) -> np.ndarray:
    max_bend = max(bend_to_color)
    side = 2 * max_bend - 1
    center = max_bend - 1
    rr = np.arange(side)[:, None]
    cc = np.arange(side)[None, :]
    layers = np.maximum(np.abs(rr - center), np.abs(cc - center)) + 1

    out = np.zeros((side, side), dtype=int)
    for bend, color in bend_to_color.items():
        out[layers == int(bend)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place each polyline's color onto the concentric square ring indexed by its bend count.

    Input structure:
    - The grid has a black background and several 4-connected monochrome foreground components.
    - Each foreground component is a width-1 open polyline with a unique color and a unique number of bends.
    - The bend counts include at least one missing value between 1 and the maximum bend count.

    Rule:
    - Detect every foreground polyline and count how many corner turns it contains.
    - Let B be the largest bend count that appears; create a square output of side 2*B-1.
    - Paint the cells whose Chebyshev-distance layer index is b with the color of the polyline that has b bends.
      Thus b=1 is the center cell, b=2 is the next square ring, and any missing layer stays black.

    Key insight:
    - The output order is determined by the polylines' turn topology, not their areas, positions, or bounding boxes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = BLACK if np.any(g == BLACK) else _get_bg(g)
    components = _find_components(g, bg)
    assert components, "expected at least one foreground polyline"

    bend_to_color: dict[int, int] = {}
    for comp in components:
        bend_count = _count_bends(comp["coords"])
        assert bend_count not in bend_to_color, "bend counts must be unique"
        bend_to_color[bend_count] = int(comp["color"])

    return _ring_output(bend_to_color)


def _normalize_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return [(r - r0, c - c0) for r, c in coords]


def _coords_to_patch(coords: list[tuple[int, int]], color: int) -> np.ndarray:
    norm = _normalize_coords(coords)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in norm:
        patch[r, c] = int(color)
    return patch


def _path_has_only_local_touches(coords: list[tuple[int, int]]) -> bool:
    for i, (r1, c1) in enumerate(coords):
        for j in range(i + 1, len(coords)):
            if j - i <= 2:
                continue
            r2, c2 = coords[j]
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _rotate_flip_patch(patch: np.ndarray) -> np.ndarray:
    out = np.rot90(patch, random.randrange(4))
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.asarray(out, dtype=int)


def _make_polyline_patch(bend_count: int, color: int) -> tuple[np.ndarray, int]:
    for _ in range(POLYLINE_RETRIES):
        direction = random.randrange(4)
        directions = [direction]
        for _turn in range(bend_count):
            direction = (direction + random.choice((-1, 1))) % 4
            directions.append(direction)

        segment_lengths = [random.randint(1, 5) for _ in range(bend_count + 1)]
        coords = [(0, 0)]
        for direction, length in zip(directions, segment_lengths):
            dr, dc = DIRS4[direction]
            for _step in range(length):
                nr = coords[-1][0] + dr
                nc = coords[-1][1] + dc
                coords.append((nr, nc))

        if len(set(coords)) != len(coords):
            continue
        if not _path_has_only_local_touches(coords):
            continue

        patch = _coords_to_patch(coords, color)
        if max(patch.shape) > 11:
            continue
        patch = _rotate_flip_patch(patch)
        if _count_bends(_find_components(patch, BLACK)[0]["coords"]) != bend_count:
            continue
        return patch, len(coords)
    raise RuntimeError(f"failed to generate a {bend_count}-bend polyline")


def _can_place(occupied: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    coords = np.argwhere(patch != BLACK)
    for pr, pc in coords:
        rr = top + int(pr)
        cc = left + int(pc)
        if occupied[rr, cc]:
            return False
        r0 = max(0, rr - 1)
        r1 = min(occupied.shape[0], rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(occupied.shape[1], cc + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> bool:
    h, w = patch.shape
    max_r = grid.shape[0] - h
    max_c = grid.shape[1] - w
    positions = [(r, c) for r in range(max_r + 1) for c in range(max_c + 1)]
    random.shuffle(positions)
    for top, left in positions[:PLACEMENT_RETRIES]:
        if not _can_place(occupied, patch, top, left):
            continue
        mask = patch != BLACK
        grid[top : top + h, left : left + w][mask] = patch[mask]
        for pr, pc in np.argwhere(mask):
            occupied[top + int(pr), left + int(pc)] = True
        return True
    return False


def _strictly_increasing(values: list[int]) -> bool:
    return all(a < b for a, b in zip(values, values[1:]))


def _strictly_decreasing(values: list[int]) -> bool:
    return all(a > b for a, b in zip(values, values[1:]))


def _sample_bend_counts() -> list[int]:
    max_bend = random.randint(4, 7)
    n_objects = random.randint(3, min(5, max_bend - 1))
    counts = sorted(random.sample(list(range(1, max_bend)), n_objects - 1) + [max_bend])
    assert len(counts) == len(set(counts))
    assert any(b not in counts for b in range(1, max_bend + 1))
    return counts


def _validate_case(inp: np.ndarray) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    comps = _find_components(inp, BLACK)
    if len(comps) < 3:
        return False

    colors = [int(comp["color"]) for comp in comps]
    if len(colors) != len(set(colors)):
        return False

    bend_area_pairs = []
    bend_to_color: dict[int, int] = {}
    for comp in comps:
        try:
            bend = _count_bends(comp["coords"])
        except AssertionError:
            return False
        if bend in bend_to_color:
            return False
        bend_to_color[bend] = int(comp["color"])
        bend_area_pairs.append((bend, int(comp["area"])))

    max_bend = max(bend_to_color)
    if max_bend <= len(comps):
        return False
    if not any(b not in bend_to_color for b in range(1, max_bend + 1)):
        return False

    areas_by_bend = [area for _bend, area in sorted(bend_area_pairs)]
    if len(set(areas_by_bend)) != len(areas_by_bend):
        return False
    if _strictly_increasing(areas_by_bend) or _strictly_decreasing(areas_by_bend):
        return False

    out = main(inp)
    if out.shape != (2 * max_bend - 1, 2 * max_bend - 1):
        return False
    if not np.array_equal(out, _ring_output(bend_to_color)):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    counts = _sample_bend_counts()
    colors = random.sample(list(range(1, 10)), len(counts))
    pairs = list(zip(counts, colors))
    random.shuffle(pairs)

    patches: list[tuple[int, int, np.ndarray]] = []
    for bend, color in pairs:
        patch, area = _make_polyline_patch(bend, color)
        patches.append((bend, area, patch))

    ordered_areas = [area for _bend, area, _patch in sorted(patches)]
    if len(set(ordered_areas)) != len(ordered_areas):
        return None
    if _strictly_increasing(ordered_areas) or _strictly_decreasing(ordered_areas):
        return None

    grid_h = random.randint(18, 28)
    grid_w = random.randint(18, 28)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)

    for _bend, _area, patch in sorted(patches, key=lambda item: item[1], reverse=True):
        if not _place_patch(grid, occupied, patch):
            return None

    if not _validate_case(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
