from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 7  # orange
ALL_COLORS = tuple(range(10))
FG_COLORS = tuple(c for c in ALL_COLORS if c != BG)
EDGES = ("top", "bottom", "left", "right")
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400
_RNG = random.Random()


def _components_by_color(grid: np.ndarray, ignore_mask: np.ndarray | None = None) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            if ignore_mask is not None and bool(ignore_mask[r, c]):
                seen[r, c] = True
                continue
            color = int(grid[r, c])
            if color == BG:
                seen[r, c] = True
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                        if ignore_mask is not None and bool(ignore_mask[nr, nc]):
                            seen[nr, nc] = True
                            continue
                        if int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            out.append((color, pts))
    return out


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _nearest_edge(points: list[tuple[int, int]], shape: tuple[int, int]) -> str:
    h, w = shape
    r0, c0, r1, c1 = _bbox(points)
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    distances = {
        "top": cr,
        "bottom": (h - 1) - cr,
        "left": cc,
        "right": (w - 1) - cc,
    }
    best_edge = min(distances, key=distances.get)
    best_value = distances[best_edge]
    ties = [edge for edge, value in distances.items() if abs(value - best_value) < 1e-9]
    assert len(ties) == 1, "nearest edge must be unique"
    return best_edge


def _flip_points(points: list[tuple[int, int]], edge: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(points)
    if edge == "top":
        return [(2 * r0 - 1 - r, c) for r, c in points]
    if edge == "bottom":
        return [(2 * r1 + 1 - r, c) for r, c in points]
    if edge == "left":
        return [(r, 2 * c0 - 1 - c) for r, c in points]
    assert edge == "right"
    return [(r, 2 * c1 + 1 - c) for r, c in points]


def _legend_mapping(grid: np.ndarray) -> dict[int, int]:
    row = grid[-1]
    mapping: dict[int, int] = {}
    c = 0
    while c < grid.shape[1]:
        if int(row[c]) == BG:
            c += 1
            continue
        assert c + 1 < grid.shape[1], "legend entries must be adjacent pairs"
        src = int(row[c])
        dst = int(row[c + 1])
        assert src != BG and dst != BG and src != dst, "legend pair must be two distinct foreground colors"
        assert src not in mapping, "legend source colors must be unique"
        mapping[src] = dst
        c += 2
    assert mapping, "legend must contain at least one color pair"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 6 and w >= 8

    mapping = _legend_mapping(g)
    scene_ignore = np.zeros_like(g, dtype=bool)
    scene_ignore[-1, :] = True
    comps = _components_by_color(g, ignore_mask=scene_ignore)

    out = g.copy()
    out[-1, :] = BG

    for color, points in comps:
        assert color in mapping, "every scene object color must appear in the legend"
        edge = _nearest_edge(points, g.shape)
        flipped = _flip_points(points, edge)
        for rr, cc in flipped:
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = mapping[color]
    return out


def _normalize(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _random_shape(rng: random.Random) -> list[tuple[int, int]]:
    target_area = rng.randint(3, 6)
    max_h = rng.randint(2, 4)
    max_w = rng.randint(2, 4)
    cells = {(0, 0)}
    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < max_h and 0 <= nc < max_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return _random_shape(rng)
        cells.add(rng.choice(frontier))
    if len({r for r, _ in cells}) == 1 or len({c for _, c in cells}) == 1:
        if rng.random() < 0.5:
            return _random_shape(rng)
    return _normalize(cells)


def _shape_dims(shape: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _place_shape(shape: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + r, c0 + c) for r, c in shape]


def _halo(points: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in points:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _valid_edge_for_points(points: list[tuple[int, int]], shape: tuple[int, int], edge: str) -> bool:
    try:
        return _nearest_edge(points, shape) == edge
    except AssertionError:
        return False


def _candidate_positions(
    edge: str,
    shp_h: int,
    shp_w: int,
    grid_h: int,
    grid_w: int,
) -> list[tuple[int, int]]:
    scene_h = grid_h - 1
    positions: list[tuple[int, int]] = []
    if edge == "top":
        for r0 in range(shp_h, min(shp_h + 3, scene_h - shp_h + 1)):
            for c0 in range(1, grid_w - shp_w - 1):
                positions.append((r0, c0))
    elif edge == "bottom":
        low = max(shp_h - 1, scene_h - 2 * shp_h)
        high = scene_h - shp_h
        for r0 in range(low, high + 1):
            for c0 in range(1, grid_w - shp_w - 1):
                positions.append((r0, c0))
    elif edge == "left":
        for c0 in range(shp_w, min(shp_w + 3, grid_w - shp_w)):
            for r0 in range(1, scene_h - shp_h):
                positions.append((r0, c0))
    else:
        low = max(shp_w - 1, grid_w - 2 * shp_w)
        high = grid_w - shp_w - 1
        for c0 in range(low, high + 1):
            for r0 in range(1, scene_h - shp_h):
                positions.append((r0, c0))
    return positions


def _generate_case(rng: random.Random) -> np.ndarray:
    h = rng.randint(12, 18)
    w = rng.randint(12, 20)
    grid = np.full((h, w), BG, dtype=int)

    object_count = rng.randint(2, 4)
    chosen_edges = rng.sample(list(EDGES), object_count)
    rng.shuffle(chosen_edges)

    palette = list(rng.sample(FG_COLORS, 2 * object_count))
    source_colors = palette[:object_count]
    complement_colors = palette[object_count:]
    order = list(range(object_count))
    rng.shuffle(order)
    legend_pairs = [(source_colors[i], complement_colors[i]) for i in order]

    used_visual: set[tuple[int, int]] = set()
    used_output: set[tuple[int, int]] = set()
    placed: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []

    for edge, color in zip(chosen_edges, source_colors, strict=False):
        shape = _random_shape(rng)
        shp_h, shp_w = _shape_dims(shape)
        candidates = _candidate_positions(edge, shp_h, shp_w, h, w)
        rng.shuffle(candidates)

        found = None
        for r0, c0 in candidates:
            points = _place_shape(shape, r0, c0)
            if not _valid_edge_for_points(points, grid.shape, edge):
                continue
            flipped = _flip_points(points, edge)
            if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in flipped):
                continue
            scene_points = set(points)
            output_points = set(flipped)
            if scene_points & used_visual:
                continue
            if output_points & used_output:
                continue
            if _halo(points, h, w) & used_visual:
                continue
            if _halo(flipped, h, w) & used_output:
                continue
            if scene_points & output_points:
                continue
            found = (points, flipped)
            break

        if found is None:
            raise ValueError("failed to place object")

        points, flipped = found
        for r, c in points:
            grid[r, c] = color
        used_visual |= set(points)
        used_output |= set(flipped)
        placed.append((color, points, flipped))

    start = rng.randint(0, w - 2 * object_count)
    for idx, (src, dst) in enumerate(legend_pairs):
        c = start + 2 * idx
        grid[h - 1, c] = src
        grid[h - 1, c + 1] = dst

    ignore_bottom = np.zeros_like(grid, dtype=bool)
    ignore_bottom[-1, :] = True
    mapping = _legend_mapping(grid)
    assert set(mapping) == set(source_colors)
    comps = _components_by_color(grid, ignore_mask=ignore_bottom)
    assert len(comps) == object_count
    for color, points, flipped in placed:
        assert _nearest_edge(points, grid.shape) in EDGES
        assert mapping[color] != color
        assert set(flipped).isdisjoint(set(points))
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _generate_case(_RNG)
            out = main(grid)
            assert isinstance(out, np.ndarray)
            assert np.all(out[-1, :] == BG)
            return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
