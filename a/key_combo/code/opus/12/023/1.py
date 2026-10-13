from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

BODY_RECOLORS = [RED, GREEN, YELLOW, ORANGE]
MARKER_POOL = [RED, GREEN, YELLOW, PINK, ORANGE, TEAL, MAROON]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _left_side_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        if grid[r, 0] != GREY:
            seen[r, 0] = True
            q.append((r, 0))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != GREY and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _round_half_up(x: float) -> int:
    return int(np.floor(float(x) + 0.5))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _wall_distances(shape: tuple[int, int], wall_cells: list[tuple[int, int]]) -> np.ndarray:
    h, w = shape
    dist = np.full((h, w), 10**9, dtype=int)
    for r in range(h):
        for c in range(w):
            dist[r, c] = min(abs(r - wr) + abs(c - wc) for wr, wc in wall_cells)
    return dist


def _analyze_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    left_mask = _left_side_mask(g)
    wall_cells = [tuple(map(int, p)) for p in np.argwhere(g == GREY)]
    assert wall_cells, "expected a grey wall"

    comps = _component_coords(~np.isin(g, [BLACK, GREY]))
    info = []
    for coords in comps:
        values = g[coords[:, 0], coords[:, 1]]
        marker_mask = values != BLUE
        assert int(marker_mask.sum()) == 1, "each component must contain exactly one marker"
        marker_idx = int(np.flatnonzero(marker_mask)[0])
        marker = tuple(map(int, coords[marker_idx]))
        centroid = coords.mean(axis=0)
        centroid_cell = (_round_half_up(centroid[0]), _round_half_up(centroid[1]))
        assert 0 <= centroid_cell[0] < g.shape[0] and 0 <= centroid_cell[1] < g.shape[1]
        assert g[centroid_cell] != GREY, "component centroid rounded onto wall"
        side = "left" if left_mask[centroid_cell] else "right"
        body = coords[values == BLUE]
        info.append(
            {
                "coords": coords,
                "body": body,
                "marker": marker,
                "marker_color": int(g[marker]),
                "distance": int(min(abs(marker[0] - wr) + abs(marker[1] - wc) for wr, wc in wall_cells)),
                "side": side,
                "bbox": _bbox(coords),
            }
        )
    return info


def _apply_recolor(grid: np.ndarray, comps: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    grouped = {"left": [], "right": []}
    for comp in comps:
        grouped[comp["side"]].append(comp)

    for side_comps in grouped.values():
        ordered = sorted(
            side_comps,
            key=lambda comp: (comp["distance"], comp["marker"][0], comp["marker"][1]),
        )
        assert len(ordered) <= len(BODY_RECOLORS), "puzzle supports at most four components per side"
        distances = [comp["distance"] for comp in ordered]
        assert len(set(distances)) == len(distances), "marker distances must be distinct within each side"
        for idx, comp in enumerate(ordered):
            color = BODY_RECOLORS[idx]
            body = comp["body"]
            out[body[:, 0], body[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each side's marked blue components by chaining them in order of marker-to-wall distance.

    Input structure:
    - The grid has a black background and a single 4-connected grey wall that runs from the top edge to the bottom edge without touching the left or right edges.
    - On both sides of the wall sit several disjoint 4-connected components made of blue body cells plus exactly one non-blue marker cell inside each component.
    - The wall separates the canvas into a left side and a right side, and every component lies entirely on one side of that partition.

    Rule:
    - Flood from the left grid edge through all non-wall cells to determine the left side; every other non-wall cell belongs to the right side.
    - Group components by the side containing the rounded centroid of the component.
    - Within each side, measure the marker cell's Manhattan distance to the nearest grey wall cell and sort components from nearest to farthest.
    - Recolor only the blue body cells of the first, second, third, and fourth components in that side-order to red, green, yellow, and orange respectively; keep marker cells, wall cells, and background unchanged.

    Key insight:
    - The same grey wall serves both as the partition that decides which components belong together and as the distance reference that orders each side's recoloring chain.
    """
    g = np.asarray(grid, dtype=int)
    return _apply_recolor(g, _analyze_components(g))


def _make_wall_path(rng: random.Random, h: int, w: int) -> list[tuple[int, int]]:
    col = rng.randint(2, w - 3)
    path = [(0, col)]
    for r in range(h - 1):
        if 0 < r < h - 2:
            shift = rng.choice([-2, -1, 0, 1, 2])
            target = min(w - 3, max(2, col + shift))
            while col != target:
                col += 1 if target > col else -1
                path.append((r, col))
        path.append((r + 1, col))
    return path


def _random_shape(rng: random.Random, size: int) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(rng.choice(frontier))
        r0 = min(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        norm = {(r - r0, c - c0) for r, c in cells}
        rows = [r for r, _ in norm]
        cols = [c for _, c in norm]
        bh = max(rows) - min(rows) + 1
        bw = max(cols) - min(cols) + 1
        if bh == 1 or bw == 1:
            continue
        if bh * bw == len(norm):
            continue
        marker = rng.choice(list(norm))
        return norm, marker
    raise RuntimeError("failed to sample irregular component shape")


def _mark_forbidden(mask: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                mask[nr, nc] = True
        mask[r, c] = True


def _place_side_components(
    rng: random.Random,
    grid: np.ndarray,
    side_mask: np.ndarray,
    count: int,
    colors: list[int],
    forbidden: np.ndarray,
    dist_map: np.ndarray,
) -> bool:
    h, w = grid.shape
    by_distance: dict[int, list[tuple[int, int]]] = {}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if side_mask[r, c] and not forbidden[r, c]:
                d = int(dist_map[r, c])
                by_distance.setdefault(d, []).append((r, c))
    usable_distances = [d for d, cells in by_distance.items() if cells]
    if len(usable_distances) < count:
        return False

    for _ in range(100):
        chosen_distances = sorted(rng.sample(usable_distances, count))
        placed = []
        local_forbidden = forbidden.copy()
        ok = True
        for marker_color, dist in zip(colors, chosen_distances):
            anchors = by_distance[dist][:]
            rng.shuffle(anchors)
            placed_here = None
            for anchor in anchors:
                if local_forbidden[anchor]:
                    continue
                for _ in range(60):
                    size = rng.randint(4, 8)
                    shape, marker_local = _random_shape(rng, size)
                    dr = anchor[0] - marker_local[0]
                    dc = anchor[1] - marker_local[1]
                    cells = {(dr + r, dc + c) for r, c in shape}
                    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                        continue
                    if any(not side_mask[r, c] for r, c in cells):
                        continue
                    if any(local_forbidden[r, c] for r, c in cells):
                        continue
                    placed_here = (cells, anchor, marker_color)
                    break
                if placed_here is not None:
                    break
            if placed_here is None:
                ok = False
                break
            cells, marker, color = placed_here
            _mark_forbidden(local_forbidden, cells)
            placed.append((cells, marker, color))
        if not ok:
            continue
        for cells, marker, color in placed:
            for r, c in cells:
                grid[r, c] = BLUE
            grid[marker] = color
        forbidden[:] = local_forbidden
        return True
    return False


def _wall_is_simple_path(grid: np.ndarray) -> bool:
    wall = grid == GREY
    comps = _component_coords(wall)
    if len(comps) != 1:
        return False
    h, w = grid.shape
    wall_cells = {tuple(map(int, p)) for p in comps[0]}
    if any(c in (0, w - 1) for _r, c in wall_cells):
        return False
    if not any(r == 0 for r, _c in wall_cells) or not any(r == h - 1 for r, _c in wall_cells):
        return False
    degrees = []
    for r, c in wall_cells:
        deg = sum((nr, nc) in wall_cells for nr, nc in _neighbors4(r, c))
        degrees.append(deg)
        if deg > 2:
            return False
    end_count = sum(deg == 1 for deg in degrees)
    return end_count == 2


def _validate_generated_grid(grid: np.ndarray, side_counts: tuple[int, int]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not _wall_is_simple_path(grid):
        return False

    try:
        comps = _analyze_components(grid)
    except AssertionError:
        return False
    grouped = {"left": [], "right": []}
    for comp in comps:
        coords = comp["coords"]
        area = int(coords.shape[0])
        if not (4 <= area <= 8):
            return False
        r0, c0, r1, c1 = comp["bbox"]
        if (r1 - r0 + 1) == 1 or (c1 - c0 + 1) == 1:
            return False
        if (r1 - r0 + 1) * (c1 - c0 + 1) == area:
            return False
        grouped[comp["side"]].append(comp)

    if len(grouped["left"]) != side_counts[0] or len(grouped["right"]) != side_counts[1]:
        return False
    for side in ("left", "right"):
        distances = [comp["distance"] for comp in grouped[side]]
        if len(set(distances)) != len(distances):
            return False
    expected = _apply_recolor(grid, comps)
    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return np.array_equal(out, expected)


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 18)
    w = rng.randint(14, 18)
    grid = np.zeros((h, w), dtype=int)

    wall_path = _make_wall_path(rng, h, w)
    for r, c in wall_path:
        grid[r, c] = GREY

    if not _wall_is_simple_path(grid):
        return None

    left_mask = _left_side_mask(grid)
    side_masks = {
        "left": left_mask & (grid != GREY),
        "right": (~left_mask) & (grid != GREY),
    }

    counts = rng.choice([(3, 3), (3, 4), (4, 3)])
    marker_colors = rng.sample(MARKER_POOL, counts[0] + counts[1])
    dist_map = _wall_distances(grid.shape, wall_path)
    forbidden = np.zeros_like(grid, dtype=bool)
    forbidden[grid == GREY] = True

    if not _place_side_components(
        rng,
        grid,
        side_masks["left"],
        counts[0],
        marker_colors[: counts[0]],
        forbidden,
        dist_map,
    ):
        return None
    if not _place_side_components(
        rng,
        grid,
        side_masks["right"],
        counts[1],
        marker_colors[counts[0] :],
        forbidden,
        dist_map,
    ):
        return None

    if not _validate_generated_grid(grid, counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        seed = random.randrange(1 << 30)
        grid = _generate_attempt(random.Random(seed))
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wall-partition recolor puzzle")
