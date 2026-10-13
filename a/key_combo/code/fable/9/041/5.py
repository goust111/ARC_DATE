from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


Coord = tuple[int, int]

_UNCLAIMED = -1
_WALL = -2
_SEAM = -3

_PALETTE = list(range(10))
_SYMMETRIES = ("id", "flipud", "fliplr", "rot180", "transpose")


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _detect_wall_color(grid: np.ndarray, bg: int) -> int:
    boundary = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    colors = {int(v) for v in boundary if int(v) != bg}
    assert len(colors) == 1, "expected exactly one wall color on the border"
    return colors.pop()


def _seed_sources(grid: np.ndarray, bg: int, wall: int) -> dict[int, list[Coord]]:
    sources: dict[int, list[Coord]] = defaultdict(list)
    for r, c in np.argwhere((grid != bg) & (grid != wall)):
        sources[int(grid[r, c])].append((int(r), int(c)))
    assert sources, "expected at least one seed blob"
    return {color: coords for color, coords in sources.items()}


def _simulate_flood(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    wall = _detect_wall_color(g, bg)
    sources = _seed_sources(g, bg, wall)

    owner = np.full(g.shape, _UNCLAIMED, dtype=int)
    owner[g == wall] = _WALL
    out = g.copy()

    frontier: dict[int, set[Coord]] = {}
    for color, coords in sources.items():
        cell_set = set(coords)
        frontier[color] = cell_set
        for r, c in cell_set:
            owner[r, c] = color

    while frontier:
        proposals: dict[Coord, set[int]] = defaultdict(set)
        for color, cells in frontier.items():
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if not _in_bounds(g.shape, nr, nc):
                        continue
                    if owner[nr, nc] != _UNCLAIMED or int(g[nr, nc]) != bg:
                        continue
                    proposals[(nr, nc)].add(color)

        next_frontier: dict[int, set[Coord]] = defaultdict(set)
        for (r, c), colors in proposals.items():
            if len(colors) == 1:
                color = next(iter(colors))
                owner[r, c] = color
                out[r, c] = color
                next_frontier[color].add((r, c))
            else:
                owner[r, c] = _SEAM
                out[r, c] = bg

        frontier = {color: cells for color, cells in next_frontier.items() if cells}

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood all seed blobs through the open space around walls, keeping equal-time clashes between different colors as background seams.

    Input structure:
    - A dominant background surrounds one wall color whose pixels form winding barrier segments with gaps and touch the outer border.
    - Several interior seed blobs use the remaining colors; some puzzles may contain two separate blobs of the same color.
    - Open cells may look straight-line close to one seed while actually being faster to reach from another only by routing through wall gaps.

    Rule:
    - Treat every non-wall, non-background seed pixel as an initial source and expand all sources simultaneously in 4-neighborhood steps through background cells only.
    - A background cell takes the color of the source color that reaches it first.
    - If different colors arrive on the same step, that cell stays background forever and does not propagate further; if the simultaneous arrivals all share one color, the cell fills normally with that color.
    - Keep every wall pixel unchanged.

    Key insight:
    - The partition is defined by wall-respecting propagation time, not by straight-line closeness; same-color fronts merge seamlessly, but different-color equal-time fronts carve non-propagating seams that can block side chambers.
    """
    return _simulate_flood(grid)


def _apply_symmetry(grid: np.ndarray, variant: str) -> np.ndarray:
    if variant == "id":
        return grid.copy()
    if variant == "flipud":
        return np.flipud(grid).copy()
    if variant == "fliplr":
        return np.fliplr(grid).copy()
    if variant == "rot180":
        return np.flipud(np.fliplr(grid)).copy()
    if variant == "transpose":
        return grid.T.copy()
    raise ValueError(f"unknown symmetry variant: {variant}")


def _map_coord(coord: Coord, shape: tuple[int, int], variant: str) -> Coord:
    r, c = coord
    h, w = shape
    if variant == "id":
        return r, c
    if variant == "flipud":
        return h - 1 - r, c
    if variant == "fliplr":
        return r, w - 1 - c
    if variant == "rot180":
        return h - 1 - r, w - 1 - c
    if variant == "transpose":
        return c, r
    raise ValueError(f"unknown symmetry variant: {variant}")


def _map_coords(coords: list[Coord], shape: tuple[int, int], variant: str) -> list[Coord]:
    return [_map_coord(coord, shape, variant) for coord in coords]


def _distance_map(shape: tuple[int, int], walls: np.ndarray, sources: list[Coord]) -> np.ndarray:
    dist = np.full(shape, 10**9, dtype=int)
    q: deque[Coord] = deque()
    for r, c in sources:
        dist[r, c] = 0
        q.append((r, c))
    while q:
        r, c = q.popleft()
        nd = dist[r, c] + 1
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(shape, nr, nc) or walls[nr, nc]:
                continue
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                q.append((nr, nc))
    return dist


def _seed_components(grid: np.ndarray, bg: int, wall: int) -> dict[int, list[list[Coord]]]:
    seen = np.zeros(grid.shape, dtype=bool)
    components: dict[int, list[list[Coord]]] = defaultdict(list)
    for r, c in np.argwhere((grid != bg) & (grid != wall)):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        color = int(grid[r, c])
        q: deque[Coord] = deque([(r, c)])
        seen[r, c] = True
        comp: list[Coord] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if not _in_bounds(grid.shape, nr, nc) or seen[nr, nc]:
                    continue
                if int(grid[nr, nc]) != color:
                    continue
                seen[nr, nc] = True
                q.append((nr, nc))
        components[color].append(comp)
    return {color: comps for color, comps in components.items()}


def _manhattan_partition(grid: np.ndarray, bg: int, wall: int) -> np.ndarray:
    seeds = _seed_sources(grid, bg, wall)
    out = grid.copy()
    for r, c in np.argwhere(grid == bg):
        best = None
        winners: set[int] = set()
        for color, coords in seeds.items():
            dist = min(abs(r - sr) + abs(c - sc) for sr, sc in coords)
            if best is None or dist < best:
                best = dist
                winners = {color}
            elif dist == best:
                winners.add(color)
        out[r, c] = bg if len(winners) != 1 else next(iter(winners))
    return out


def _sample_distinct(exclude: set[int], k: int) -> list[int]:
    return random.sample([c for c in _PALETTE if c not in exclude], k)


def _build_canonical_case() -> tuple[np.ndarray, dict]:
    same_color_pair = random.random() < 0.65

    bg = random.choice(_PALETTE)
    wall, color_a, color_b, color_c = _sample_distinct({bg}, 4)

    gate_top = random.randint(2, 4)
    half_span = random.randint(4, 7)
    seam_row = gate_top + half_span
    gate_bottom = seam_row + half_span
    height = gate_bottom + random.randint(3, 6)

    chamber_width = random.randint(2, 3)
    seed_offset = chamber_width + random.randint(2, 3)
    left_wall_col = random.randint(seed_offset + 1, seed_offset + 4)
    corridor_col = left_wall_col + 1
    right_wall_col = left_wall_col + 2
    chamber_wall_col = right_wall_col + chamber_width + 1
    width = chamber_wall_col + seed_offset + random.randint(2, 5)
    assert width <= 30 and height <= 30

    chamber_radius = 2 if half_span == 4 else random.randint(2, 3)
    chamber_top = seam_row - chamber_radius
    chamber_bottom = seam_row + chamber_radius

    grid = np.full((height, width), bg, dtype=int)
    grid[:, left_wall_col] = wall
    grid[:, right_wall_col] = wall
    grid[gate_top, left_wall_col] = bg
    grid[gate_bottom, right_wall_col] = bg
    grid[seam_row, right_wall_col] = bg

    grid[chamber_top : chamber_bottom + 1, chamber_wall_col] = wall
    grid[chamber_top, right_wall_col : chamber_wall_col + 1] = wall
    grid[chamber_bottom, right_wall_col : chamber_wall_col + 1] = wall

    left_seed_col = left_wall_col - seed_offset
    right_seed_col = right_wall_col + seed_offset

    a1 = (gate_top, left_seed_col)
    a2 = (gate_bottom, left_seed_col)
    b1 = (gate_bottom, right_seed_col)

    grid[a1] = color_a
    grid[a2] = color_a if same_color_pair else color_c
    grid[b1] = color_b

    chamber_cells = [
        (r, c)
        for r in range(chamber_top + 1, chamber_bottom)
        for c in range(right_wall_col + 1, chamber_wall_col)
    ]

    meta = {
        "bg": bg,
        "wall": wall,
        "same_color_pair": same_color_pair,
        "pair_color": color_a,
        "pair_sources": [a1, a2],
        "other_sources": [b1],
        "seam_cell": (seam_row, corridor_col),
        "chamber_cells": chamber_cells,
    }
    return grid, meta


def _validate_semantics(grid: np.ndarray, meta: dict) -> bool:
    bg = meta["bg"]
    wall = meta["wall"]
    out = main(grid)

    if not np.array_equal(out[grid == wall], grid[grid == wall]):
        return False

    original_bg = grid == bg
    seam_mask = original_bg & (out == bg)
    if not seam_mask.any():
        return False

    seam_r, seam_c = meta["seam_cell"]
    if int(out[seam_r, seam_c]) != bg:
        return False

    for cell in meta["chamber_cells"]:
        if int(out[cell]) != bg:
            return False

    simple = _manhattan_partition(grid, bg, wall)
    routing_mismatch = original_bg & (out != bg) & (simple != out)
    if not routing_mismatch.any():
        return False

    if meta["same_color_pair"]:
        walls = grid == wall
        comp_dists = [_distance_map(grid.shape, walls, [src]) for src in meta["pair_sources"]]
        other_dists = [
            _distance_map(grid.shape, walls, [src]) for src in meta["other_sources"]
        ]
        pair_color = meta["pair_color"]
        same_color_meet = False
        for r, c in np.argwhere(original_bg):
            d0 = int(comp_dists[0][r, c])
            d1 = int(comp_dists[1][r, c])
            if d0 >= 10**9 or d0 != d1:
                continue
            if any(int(other[r, c]) <= d0 for other in other_dists):
                continue
            if int(out[r, c]) == pair_color:
                same_color_meet = True
                break
        if not same_color_meet:
            return False

    return True


def _attempt_case() -> np.ndarray | None:
    grid, meta = _build_canonical_case()
    variant = random.choice(_SYMMETRIES)
    transformed = _apply_symmetry(grid, variant)
    transformed_meta = {
        "bg": meta["bg"],
        "wall": meta["wall"],
        "same_color_pair": meta["same_color_pair"],
        "pair_color": meta["pair_color"],
        "pair_sources": _map_coords(meta["pair_sources"], grid.shape, variant),
        "other_sources": _map_coords(meta["other_sources"], grid.shape, variant),
        "seam_cell": _map_coord(meta["seam_cell"], grid.shape, variant),
        "chamber_cells": _map_coords(meta["chamber_cells"], grid.shape, variant),
    }

    try:
        if _validate_semantics(transformed, transformed_meta):
            return transformed
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid flood-around-walls input")
