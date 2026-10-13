from __future__ import annotations

import random
from collections import Counter
from typing import Iterable

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 600
MIN_OBJECT_GAP_ANY = 2
MIN_OBJECT_GAP_SAME = 4
SHAPE_TILE = 3
SHAPE_PITCH = 4

_BASE_SHAPES = [
    np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 1], [0, 0, 0]], dtype=int),
    np.array([[1, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=int),
]


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _color_components(grid: np.ndarray) -> dict[int, list[np.ndarray]]:
    colors = [int(c) for c in np.unique(grid) if int(c) != BACKGROUND]
    return {color: _components_of_color(grid, color) for color in colors}


def _color_counts(grid: np.ndarray) -> dict[int, int]:
    return {color: len(comps) for color, comps in _color_components(grid).items()}


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each color family by its component count and keep overlap cells for rarer-color families.

    Input structure:
    - The grid has black background cells and 3-4 foreground colors.
    - Each foreground color forms several small 4-connected irregular components, and each color's component count is unique.
    - Components may sit near the border, so expanded regions can clip against the grid boundary.

    Rule:
    - Count the 4-connected components of every foreground color and call that count K for the color.
    - For each component of that color, if K > 1, expand its bounding box outward by K-1 cells on all four sides and fill that rectangle with the same color, clipped to the grid.
    - If expanded rectangles of different colors overlap on background cells, keep the color with smaller K.
    - Preserve every original foreground pixel exactly as it was; expansion never overwrites original objects.

    Key insight:
    - The same per-color component count drives both the expansion radius and the overlap priority, and K=1 is a special no-expansion case even when the component's own bbox has empty cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps_by_color = _color_components(g)
    counts = {color: len(comps) for color, comps in comps_by_color.items()}
    assert counts, "expected foreground colors"
    assert len(set(counts.values())) == len(counts), "foreground colors must have distinct component counts"

    h, w = g.shape
    best_k = np.full((h, w), 10**9, dtype=int)
    best_color = np.zeros((h, w), dtype=int)

    for color, comps in comps_by_color.items():
        k = counts[color]
        if k <= 1:
            continue
        radius = k - 1
        for coords in comps:
            r0, c0, r1, c1 = _bbox(coords)
            er0 = max(0, r0 - radius)
            ec0 = max(0, c0 - radius)
            er1 = min(h - 1, r1 + radius)
            ec1 = min(w - 1, c1 + radius)
            k_view = best_k[er0 : er1 + 1, ec0 : ec1 + 1]
            c_view = best_color[er0 : er1 + 1, ec0 : ec1 + 1]
            better = k < k_view
            k_view[better] = k
            c_view[better] = color

    out = g.copy()
    fill_mask = (g == BACKGROUND) & (best_color != BACKGROUND)
    out[fill_mask] = best_color[fill_mask]
    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_shape(rng: random.Random) -> np.ndarray:
    mask = np.array(rng.choice(_BASE_SHAPES), copy=True)
    mask = np.rot90(mask, rng.randrange(4))
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    if rng.random() < 0.5:
        mask = np.flipud(mask)
    return _trim_mask(mask.astype(bool))


def _tile_origins(size: int, preferred_start: int) -> list[int]:
    origins: list[int] = []
    start = preferred_start
    while start + SHAPE_TILE - 1 < size:
        origins.append(start)
        start += SHAPE_PITCH
    return origins


def _choose_grid_shape(rng: random.Random, total_components: int) -> tuple[int, int, int, int]:
    for _ in range(200):
        h = rng.randint(15, 20)
        w = rng.randint(15, 20)
        row_start = rng.randint(0, 1)
        col_start = rng.randint(0, 1)
        rows = _tile_origins(h, row_start)
        cols = _tile_origins(w, col_start)
        if len(rows) * len(cols) >= total_components:
            return h, w, row_start, col_start
    raise RuntimeError("failed to choose a roomy grid")


def _sample_counts(rng: random.Random, n_colors: int) -> list[int]:
    choices = [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]
    if n_colors == 4:
        return [1, 2, 3, 4]
    weights = [4, 4, 2, 1]
    return list(rng.choices(choices, weights=weights, k=1)[0])


def _tile_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _assign_slots(
    colors: list[int],
    counts: dict[int, int],
    slots: list[tuple[int, int]],
    center: tuple[float, float],
    rng: random.Random,
    spread_mode: bool,
) -> dict[int, list[tuple[int, int]]] | None:
    order = sorted(colors, key=lambda color: counts[color], reverse=True)
    assignments: dict[int, list[tuple[int, int]]] = {color: [] for color in colors}
    used: set[tuple[int, int]] = set()

    def slot_score(slot: tuple[int, int], color: int) -> float:
        interact = sum(1 for other in used if _tile_distance(slot, other) <= 1)
        edge = min(slot[0], slot[1])
        central = abs(slot[0] - center[0]) + abs(slot[1] - center[1])
        if spread_mode:
            return -abs(interact - 1) * 4.0 - central * 0.25 - edge * 0.1 + rng.random() * 0.25
        return interact * 6.0 - central * 0.8 - edge * 0.2 + rng.random() * 0.2 - counts[color] * 0.05

    def rec(color_idx: int) -> bool:
        if color_idx == len(order):
            return True
        color = order[color_idx]
        need = counts[color]

        def place_one(k: int) -> bool:
            if k == need:
                return rec(color_idx + 1)
            candidates = [slot for slot in slots if slot not in used]
            candidates = [
                slot
                for slot in candidates
                if all(_tile_distance(slot, prev) >= 2 for prev in assignments[color])
            ]
            candidates.sort(key=lambda slot: slot_score(slot, color), reverse=True)
            for slot in candidates:
                used.add(slot)
                assignments[color].append(slot)
                if place_one(k + 1):
                    return True
                assignments[color].pop()
                used.remove(slot)
            return False

        return place_one(0)

    return assignments if rec(0) else None


def _shape_cells_in_tile(origin: tuple[int, int], rng: random.Random) -> np.ndarray:
    mask = _random_shape(rng)
    h, w = mask.shape
    r_off = rng.randint(0, SHAPE_TILE - h)
    c_off = rng.randint(0, SHAPE_TILE - w)
    rr, cc = np.where(mask)
    coords = np.column_stack((rr + origin[0] + r_off, cc + origin[1] + c_off))
    return coords.astype(int)


def _min_chebyshev_gap(a: np.ndarray, b: np.ndarray) -> int:
    best = 10**9
    for r1, c1 in a:
        for r2, c2 in b:
            best = min(best, max(abs(int(r1) - int(r2)), abs(int(c1) - int(c2))))
    return int(best)


def _place_components(
    h: int,
    w: int,
    colors: list[int],
    counts: dict[int, int],
    assignments: dict[int, list[tuple[int, int]]],
    row_origins: list[int],
    col_origins: list[int],
    rng: random.Random,
) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    placed: list[tuple[int, np.ndarray]] = []
    origin_lookup = {(ri, ci): (row_origins[ri], col_origins[ci]) for ri in range(len(row_origins)) for ci in range(len(col_origins))}

    for color in colors:
        for slot in assignments[color]:
            success = False
            tile_origin = origin_lookup[slot]
            for _ in range(40):
                coords = _shape_cells_in_tile(tile_origin, rng)
                if np.any(grid[coords[:, 0], coords[:, 1]] != BACKGROUND):
                    continue
                okay = True
                for other_color, other_coords in placed:
                    gap = _min_chebyshev_gap(coords, other_coords)
                    limit = MIN_OBJECT_GAP_SAME if other_color == color else MIN_OBJECT_GAP_ANY
                    if gap < limit:
                        okay = False
                        break
                if not okay:
                    continue
                grid[coords[:, 0], coords[:, 1]] = color
                placed.append((color, coords))
                success = True
                break
            if not success:
                return None
    return grid


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    comps_by_color = _color_components(g)
    counts = {color: len(comps) for color, comps in comps_by_color.items()}
    claims: dict[tuple[int, int], list[tuple[int, int]]] = {}
    h, w = g.shape
    for color, comps in comps_by_color.items():
        k = counts[color]
        if k <= 1:
            continue
        radius = k - 1
        for coords in comps:
            r0, c0, r1, c1 = _bbox(coords)
            for r in range(max(0, r0 - radius), min(h - 1, r1 + radius) + 1):
                for c in range(max(0, c0 - radius), min(w - 1, c1 + radius) + 1):
                    claims.setdefault((r, c), []).append((k, color))
    out = g.copy()
    for (r, c), items in claims.items():
        if int(g[r, c]) != BACKGROUND:
            continue
        k, color = min(items, key=lambda item: item[0])
        out[r, c] = color
    return out


def _expanded_boxes(grid: np.ndarray) -> dict[int, list[tuple[int, int, int, int]]]:
    comps_by_color = _color_components(grid)
    counts = {color: len(comps) for color, comps in comps_by_color.items()}
    h, w = grid.shape
    boxes: dict[int, list[tuple[int, int, int, int]]] = {}
    for color, comps in comps_by_color.items():
        k = counts[color]
        radius = k - 1
        boxes[color] = []
        if radius <= 0:
            continue
        for coords in comps:
            r0, c0, r1, c1 = _bbox(coords)
            boxes[color].append(
                (
                    max(0, r0 - radius),
                    max(0, c0 - radius),
                    min(h - 1, r1 + radius),
                    min(w - 1, c1 + radius),
                )
            )
    return boxes


def _validate_grid(grid: np.ndarray, want_clipping: bool) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False

    comps_by_color = _color_components(grid)
    colors = sorted(comps_by_color)
    if not (3 <= len(colors) <= 4):
        return False

    counts = {color: len(comps) for color, comps in comps_by_color.items()}
    if len(set(counts.values())) != len(counts):
        return False
    if max(counts.values()) > 4:
        return False
    if 1 not in counts.values():
        return False

    all_comps = [(color, comp) for color, comps in comps_by_color.items() for comp in comps]
    for color, comp in all_comps:
        area = int(len(comp))
        if not (3 <= area <= 6):
            return False
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1) * (c1 - c0 + 1) == area:
            return False

    for i, (color_a, comp_a) in enumerate(all_comps):
        for color_b, comp_b in all_comps[i + 1 :]:
            gap = _min_chebyshev_gap(comp_a, comp_b)
            limit = MIN_OBJECT_GAP_SAME if color_a == color_b else MIN_OBJECT_GAP_ANY
            if gap < limit:
                return False

    out = main(grid)
    oracle = _oracle_output(grid)
    if not np.array_equal(out, oracle):
        return False
    if np.array_equal(out, grid):
        return False

    boxes = _expanded_boxes(grid)
    h, w = grid.shape
    conflict_cells = 0
    preserved_overlap = 0
    clipped_boxes = 0
    for r in range(h):
        for c in range(w):
            claims: list[tuple[int, int]] = []
            for color, rects in boxes.items():
                for r0, c0, r1, c1 in rects:
                    if r0 <= r <= r1 and c0 <= c <= c1:
                        claims.append((counts[color], color))
                        break
            if len(claims) >= 2 and int(grid[r, c]) == BACKGROUND:
                conflict_cells += 1
                winner = min(claims, key=lambda item: item[0])[1]
                if int(out[r, c]) != winner:
                    return False
            if int(grid[r, c]) != BACKGROUND:
                pixel_color = int(grid[r, c])
                if any(color != pixel_color for _, color in claims):
                    preserved_overlap += 1

    for rects in boxes.values():
        for r0, c0, r1, c1 in rects:
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                clipped_boxes += 1

    if conflict_cells == 0:
        return False
    if preserved_overlap == 0:
        return False
    if want_clipping and clipped_boxes == 0:
        return False

    coverage = float(np.count_nonzero(out)) / float(out.size)
    coverage_cap = 0.68 if len(colors) == 4 else 0.62
    if coverage > coverage_cap:
        return False
    return True


def _generate_candidate(rng: random.Random) -> np.ndarray | None:
    n_colors = 4 if rng.random() < 0.3 else 3
    counts_list = _sample_counts(rng, n_colors)
    total_components = sum(counts_list)
    if n_colors == 4:
        h, w, row_start, col_start = 20, 20, 0, 0
    else:
        h, w, row_start, col_start = _choose_grid_shape(rng, total_components)
    row_origins = _tile_origins(h, row_start)
    col_origins = _tile_origins(w, col_start)

    colors = rng.sample(list(PALETTE), n_colors)
    counts = {color: count for color, count in zip(colors, counts_list)}
    slots = [(ri, ci) for ri in range(len(row_origins)) for ci in range(len(col_origins))]
    center = ((len(row_origins) - 1) / 2.0, (len(col_origins) - 1) / 2.0)

    assignments = _assign_slots(
        colors,
        counts,
        slots,
        center,
        rng,
        spread_mode=(n_colors == 3 and total_components >= 8),
    )
    if assignments is None:
        return None

    grid = _place_components(h, w, colors, counts, assignments, row_origins, col_origins, rng)
    if grid is None:
        return None

    want_clipping = rng.random() < 0.4
    if not _validate_grid(grid, want_clipping):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
