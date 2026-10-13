from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

MARKER_COLORS = {BLUE, RED}
QUADRANT_DIRS = {
    "ul": (-1, -1),
    "ur": (-1, 1),
    "dl": (1, -1),
    "dr": (1, 1),
}

GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as (N,2) coord arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(np.any((rows == 0) | (rows == h - 1) | (cols == 0) | (cols == w - 1)))


def _ray_cells(start: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    h, w = shape
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    return cells


def _direction_from_marker(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[str, tuple[int, int]]:
    mr, mc = marker
    r0, c0, r1, c1 = bbox
    top_half = 2 * mr <= r0 + r1
    left_half = 2 * mc <= c0 + c1
    if top_half and left_half:
        key = "ul"
    elif top_half and not left_half:
        key = "ur"
    elif not top_half and left_half:
        key = "dl"
    else:
        key = "dr"
    return key, QUADRANT_DIRS[key]


def _analyze_regions(grid: np.ndarray) -> list[dict]:
    passable = grid != GREEN
    regions: list[dict] = []
    for coords in _components(passable):
        vals = grid[coords[:, 0], coords[:, 1]]
        marker_mask = np.isin(vals, list(MARKER_COLORS))
        marker_coords = coords[marker_mask]
        region = {
            "coords": coords,
            "bbox": _bbox(coords),
            "touches_edge": _touches_edge(coords, grid.shape),
            "markers": marker_coords,
        }
        if marker_coords.size:
            marker = tuple(map(int, marker_coords[0]))
            color = int(grid[marker])
            region["marker"] = marker
            region["marker_color"] = color
            region["direction_key"], region["direction"] = _direction_from_marker(marker, region["bbox"])
        regions.append(region)
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor edge-reachable background grey and let each enclosed marker emit a diagonal ray set by its room quadrant.

    Input structure:
    - The grid has black background, green 1-cell-thick barrier walls, and singleton blue/red markers.
    - Green walls partition the passable black space into 4-connected regions; enclosed regions each contain exactly one marker.
    - Edge-reachable regions contain no markers, while enclosed regions have bbox size at least 2x2 and the marker is not on the bbox mid-axes.

    Rule:
    - Treat every non-green cell as passable and split those cells into 4-connected regions.
    - Recolor black cells in every edge-touching region to grey, but keep enclosed regions black.
    - For each enclosed region, use the marker's quadrant inside that region's bbox to choose one diagonal direction.
    - Starting from the marker, paint its color along that diagonal until the next step would leave the grid, overwriting any existing color.

    Key insight:
    - Topology decides which regions participate at all, and only then does the marker's relative position inside its enclosed region encode the ray direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(np.unique(g)).issubset({BLACK, BLUE, RED, GREEN})

    regions = _analyze_regions(g)
    assert regions

    out = g.copy()
    for region in regions:
        coords = region["coords"]
        markers = region["markers"]
        if region["touches_edge"]:
            assert len(markers) == 0
            black_mask = g[coords[:, 0], coords[:, 1]] == BLACK
            black_coords = coords[black_mask]
            out[black_coords[:, 0], black_coords[:, 1]] = GREY
            continue

        assert len(markers) == 1
        marker = region["marker"]
        color = region["marker_color"]
        direction = region["direction"]
        for rr, cc in _ray_cells(marker, direction, g.shape):
            out[rr, cc] = color

    return out


def _draw_rect_ring(grid: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    grid[r0, c0 : c1 + 1] = GREEN
    grid[r1, c0 : c1 + 1] = GREEN
    grid[r0 : r1 + 1, c0] = GREEN
    grid[r0 : r1 + 1, c1] = GREEN


def _draw_segment(grid: np.ndarray, p0: tuple[int, int], p1: tuple[int, int]) -> None:
    r0, c0 = p0
    r1, c1 = p1
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        grid[r0, lo : hi + 1] = GREEN
    else:
        assert c0 == c1
        lo, hi = sorted((r0, r1))
        grid[lo : hi + 1, c0] = GREEN


def _quadrant_cells(rect: tuple[int, int, int, int], key: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if 2 * r == (r0 + 1) + (r1 - 1) or 2 * c == (c0 + 1) + (c1 - 1):
                continue
            top = 2 * r <= (r0 + 1) + (r1 - 1)
            left = 2 * c <= (c0 + 1) + (c1 - 1)
            if key == "ul" and top and left:
                cells.append((r, c))
            elif key == "ur" and top and not left:
                cells.append((r, c))
            elif key == "dl" and not top and left:
                cells.append((r, c))
            elif key == "dr" and not top and not left:
                cells.append((r, c))
    return cells


def _pick_room_rect(
    rng: np.random.Generator,
    zone: tuple[int, int, int, int],
    zone_name: str,
) -> tuple[int, int, int, int] | None:
    zr0, zr1, zc0, zc1 = zone
    zh = zr1 - zr0 + 1
    zw = zc1 - zc0 + 1
    max_h = min(8, zh)
    max_w = min(8, zw)
    if max_h < 5 or max_w < 5:
        return None
    oh = int(rng.integers(5, max_h + 1))
    ow = int(rng.integers(5, max_w + 1))
    r_slack = zr1 - oh - zr0 + 1
    c_slack = zc1 - ow - zc0 + 1
    r_jitter = int(rng.integers(0, min(3, max(1, r_slack + 1))))
    c_jitter = int(rng.integers(0, min(3, max(1, c_slack + 1))))
    if zone_name in {"TL", "TR"}:
        r0 = zr0 + r_jitter
    else:
        r0 = zr1 - oh + 1 - r_jitter
    if zone_name in {"TL", "BL"}:
        c0 = zc0 + c_jitter
    else:
        c0 = zc1 - ow + 1 - c_jitter
    return (r0, c0, r0 + oh - 1, c0 + ow - 1)


def _room_connector(
    rng: np.random.Generator,
    rect: tuple[int, int, int, int],
    zone_name: str,
    cross_r: int,
    cross_c: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = rect
    inward_sides = {
        "TL": ("bottom", "right"),
        "TR": ("bottom", "left"),
        "BL": ("top", "right"),
        "BR": ("top", "left"),
    }[zone_name]
    side = inward_sides[int(rng.integers(0, 2))]
    if side == "top":
        c = int(rng.integers(c0 + 1, c1))
        return (r0, c), (cross_r, c)
    if side == "bottom":
        c = int(rng.integers(c0 + 1, c1))
        return (r1, c), (cross_r, c)
    if side == "left":
        r = int(rng.integers(r0 + 1, r1))
        return (r, c0), (r, cross_c)
    r = int(rng.integers(r0 + 1, r1))
    return (r, c1), (r, cross_c)


def _choose_room_count(rng: np.random.Generator, max_rooms: int) -> int:
    options = [n for n in (4, 3) if n <= max_rooms]
    if not options:
        options = [2]
    weights = {4: 5, 3: 3, 2: 1}
    total = sum(weights[n] for n in options)
    pick = int(rng.integers(total))
    acc = 0
    for n in options:
        acc += weights[n]
        if pick < acc:
            return n
    return options[-1]


def _validate_input(grid: np.ndarray) -> None:
    assert grid.ndim == 2 and min(grid.shape) >= 18 and max(grid.shape) <= 30
    assert set(np.unique(grid)).issubset({BLACK, BLUE, RED, GREEN})

    green_comps = _components(grid == GREEN)
    assert len(green_comps) == 1

    regions = _analyze_regions(grid)
    closed = [r for r in regions if not r["touches_edge"]]
    open_regions = [r for r in regions if r["touches_edge"]]
    assert len(open_regions) >= 1
    assert len(closed) >= 2

    used_rays: set[tuple[int, int]] = set()
    all_closed_cells: list[set[tuple[int, int]]] = []
    marker_positions = set()
    directions = set()
    for region in closed:
        coords = {tuple(map(int, cell)) for cell in region["coords"]}
        all_closed_cells.append(coords)
        assert len(region["markers"]) == 1
        marker = region["marker"]
        marker_positions.add(marker)
        directions.add(region["direction_key"])
        r0, c0, r1, c1 = region["bbox"]
        assert (r1 - r0 + 1) >= 2 and (c1 - c0 + 1) >= 2
        assert len(coords) >= 4
        assert 2 * marker[0] != r0 + r1
        assert 2 * marker[1] != c0 + c1

    assert len(directions) == len(closed)

    for region in open_regions:
        assert len(region["markers"]) == 0

    overwritten_open = 0
    green_hits = 0
    other_closed_cells = [set().union(*(all_closed_cells[:i] + all_closed_cells[i + 1 :])) for i in range(len(closed))]
    for i, region in enumerate(closed):
        ray = _ray_cells(region["marker"], region["direction"], grid.shape)
        assert len(ray) >= 4
        own_cells = all_closed_cells[i]
        others = other_closed_cells[i]
        for cell in ray:
            assert cell not in used_rays
            used_rays.add(cell)
            if cell in others:
                raise AssertionError("ray may not enter another enclosed region")
            val = int(grid[cell])
            if cell not in own_cells and val == BLACK:
                overwritten_open += 1
            if val == GREEN:
                green_hits += 1
        assert all(cell not in marker_positions or cell == region["marker"] for cell in ray)

    assert overwritten_open > 0
    assert green_hits >= len(closed)

    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out == GREY)
    assert np.count_nonzero(out == GREEN) < np.count_nonzero(grid == GREEN)
    for region in closed:
        marker = region["marker"]
        assert out[marker] == grid[marker]


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(22, 29))
    w = int(rng.integers(22, 29))
    cross_r = int(rng.integers(10, h - 10))
    cross_c = int(rng.integers(10, w - 10))

    grid = np.full((h, w), BLACK, dtype=int)
    grid[cross_r, :] = GREEN
    grid[:, cross_c] = GREEN

    zones = {
        "TL": (2, cross_r - 4, 2, cross_c - 4),
        "TR": (2, cross_r - 4, cross_c + 4, w - 3),
        "BL": (cross_r + 4, h - 3, 2, cross_c - 4),
        "BR": (cross_r + 4, h - 3, cross_c + 4, w - 3),
    }
    viable = {name: zone for name, zone in zones.items() if zone[0] <= zone[1] and zone[2] <= zone[3]}
    viable = {name: zone for name, zone in viable.items() if (zone[1] - zone[0] + 1) >= 5 and (zone[3] - zone[2] + 1) >= 5}
    if len(viable) < 2:
        return None

    room_count = _choose_room_count(rng, len(viable))
    zone_names = list(viable)
    rng.shuffle(zone_names)
    zone_names = zone_names[:room_count]

    direction_keys = list(QUADRANT_DIRS)
    rng.shuffle(direction_keys)
    direction_keys = direction_keys[:room_count]

    colors = [BLUE, RED]
    while len(colors) < room_count:
        colors.append(BLUE if rng.random() < 0.5 else RED)
    colors = colors[:room_count]
    rng.shuffle(colors)

    for zone_name, direction_key, color in zip(zone_names, direction_keys, colors):
        rect = _pick_room_rect(rng, viable[zone_name], zone_name)
        if rect is None:
            return None
        _draw_rect_ring(grid, rect)
        p0, p1 = _room_connector(rng, rect, zone_name, cross_r, cross_c)
        _draw_segment(grid, p0, p1)
        cells = _quadrant_cells(rect, direction_key)
        if not cells:
            return None
        marker = cells[int(rng.integers(0, len(cells)))]
        if grid[marker] != BLACK:
            return None
        grid[marker] = color

    try:
        _validate_input(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid enclosed-ray puzzle input")


if __name__ == "__main__":
    sample = generate_input()
    print(sample)
    print(main(sample))
