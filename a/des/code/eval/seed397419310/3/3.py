from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
TOWER_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")

# Generator configuration.
GRID_MIN = 16
GRID_MAX = 30
OBJECT_COUNT_OPTIONS = (1, 2, 3, 4)
OBJECT_COUNT_WEIGHTS = (3, 4, 3, 2)
RECT_H_RANGE = (5, 10)
RECT_W_RANGE = (5, 10)
GAP_COUNT_RANGE = (1, 3)
PLACEMENT_RETRIES = 120
SCENE_RETRIES = 400
PADDING = 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _gap_side(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = gap
    r0, c0, r1, c1 = bbox
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    raise AssertionError(f"gap {gap} is not on bbox perimeter {bbox}")


def _ray_direction(side: str) -> tuple[int, int]:
    if side == "top":
        return -1, 0
    if side == "bottom":
        return 1, 0
    if side == "left":
        return 0, -1
    if side == "right":
        return 0, 1
    raise AssertionError(f"unknown side {side}")


def _is_allowed_path_cell(
    r: int,
    c: int,
    bbox: tuple[int, int, int, int],
    tower: tuple[int, int],
    gap: tuple[int, int],
) -> bool:
    if (r, c) == tower or (r, c) == gap:
        return True
    r0, c0, r1, c1 = bbox
    return r0 < r < r1 and c0 < c < c1


def _neighbor_order(r: int, c: int, goal: tuple[int, int]) -> list[tuple[int, int]]:
    gr, gc = goal
    ordered: list[tuple[int, int]] = []
    if c != gc:
        ordered.append((0, 1 if gc > c else -1))
    if r != gr:
        ordered.append((1 if gr > r else -1, 0))
    for step in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        if step not in ordered:
            ordered.append(step)
    return ordered


def _connection_path(
    bbox: tuple[int, int, int, int],
    tower: tuple[int, int],
    gap: tuple[int, int],
) -> list[tuple[int, int]]:
    q = deque([tower])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {tower: None}
    while q:
        cur = q.popleft()
        if cur == gap:
            break
        r, c = cur
        for dr, dc in _neighbor_order(r, c, gap):
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if nxt in parent:
                continue
            if not _is_allowed_path_cell(nr, nc, bbox, tower, gap):
                continue
            parent[nxt] = cur
            q.append(nxt)

    assert gap in parent, f"no interior path from tower {tower} to gap {gap}"

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = gap
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _analyze_wall(
    g: np.ndarray, cells: list[tuple[int, int]]
) -> dict[str, object]:
    bbox = _bbox(cells)
    r0, c0, r1, c1 = bbox
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "wall bbox must enclose an interior"

    perim = _perimeter_cells(bbox)
    perim_set = set(perim)
    cell_set = set(cells)
    assert cell_set <= perim_set, "blue wall must stay on the rectangle perimeter"

    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    assert corners <= cell_set, "corners must remain blue"

    gaps: list[tuple[int, int]] = []
    for rc in perim:
        if rc in cell_set:
            continue
        r, c = rc
        assert rc not in corners, "gap may not be a corner"
        assert int(g[r, c]) == BLACK, "input gaps must be black"
        gaps.append(rc)
    assert GAP_COUNT_RANGE[0] <= len(gaps) <= GAP_COUNT_RANGE[1], "each wall must have 1-3 gaps"

    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    tower_mask = (inner != BLACK) & (inner != BLUE)
    tower_coords = np.argwhere(tower_mask)
    assert tower_coords.shape[0] == 1, "each region must contain exactly one signal tower pixel"
    tr, tc = map(int, tower_coords[0])
    tower = (r0 + 1 + tr, c0 + 1 + tc)
    color = int(g[tower])

    return {"bbox": bbox, "gaps": gaps, "tower": tower, "color": color}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each wall gap with its inner tower color, connect the tower to the gap,
    and emit an outward ray from that repaired gap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    walls = _components(g == BLUE)
    assert walls, "expected at least one blue wall"

    objects = [_analyze_wall(g, cells) for cells in walls]
    out = g.copy()

    # First realize all repairs and tower-to-gap connections.
    for obj in objects:
        bbox = obj["bbox"]
        tower = obj["tower"]
        color = int(obj["color"])
        gaps = obj["gaps"]

        for gap in gaps:
            out[gap] = color
        for gap in gaps:
            for r, c in _connection_path(bbox, tower, gap):
                if int(out[r, c]) != BLUE or (r, c) == gap:
                    out[r, c] = color

    # Then emit all rays on top of the repaired scene.
    h, w = out.shape
    for obj in objects:
        color = int(obj["color"])
        bbox = obj["bbox"]
        gaps = obj["gaps"]
        for gap in gaps:
            dr, dc = _ray_direction(_gap_side(gap, bbox))
            r, c = gap[0] + dr, gap[1] + dc
            while 0 <= r < h and 0 <= c < w and int(out[r, c]) == BLACK:
                out[r, c] = color
                r += dr
                c += dc

    return out


def _expanded_cells(
    bbox: tuple[int, int, int, int], h: int, w: int, pad: int = PADDING
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {
        (r, c)
        for r in range(max(0, r0 - pad), min(h, r1 + pad + 1))
        for c in range(max(0, c0 - pad), min(w, c1 + pad + 1))
    }


def _sample_gaps(bbox: tuple[int, int, int, int], sides: list[str]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    gaps: list[tuple[int, int]] = []
    for side in sides:
        if side == "top":
            gaps.append((r0, random.randint(c0 + 1, c1 - 1)))
        elif side == "bottom":
            gaps.append((r1, random.randint(c0 + 1, c1 - 1)))
        elif side == "left":
            gaps.append((random.randint(r0 + 1, r1 - 1), c0))
        else:
            assert side == "right"
            gaps.append((random.randint(r0 + 1, r1 - 1), c1))
    return gaps


def _ray_cells(
    gap: tuple[int, int], side: str, h: int, w: int
) -> list[tuple[int, int]]:
    dr, dc = _ray_direction(side)
    r, c = gap[0] + dr, gap[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _draw_object(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    gaps: list[tuple[int, int]],
    tower: tuple[int, int],
    color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    for r, c in _perimeter_cells(bbox):
        grid[r, c] = BLUE
    for r, c in gaps:
        grid[r, c] = BLACK
    grid[tower] = color


def _sample_object_count() -> int:
    return random.choices(OBJECT_COUNT_OPTIONS, weights=OBJECT_COUNT_WEIGHTS, k=1)[0]


def _grid_shape_for(n_objects: int) -> tuple[int, int]:
    low = max(GRID_MIN, 12 + 2 * n_objects)
    return random.randint(low, GRID_MAX), random.randint(low, GRID_MAX)


def _place_scene() -> np.ndarray | None:
    n_objects = _sample_object_count()
    h, w = _grid_shape_for(n_objects)
    grid = np.full((h, w), BLACK, dtype=int)

    reserved_solids: set[tuple[int, int]] = set()
    reserved_rays: set[tuple[int, int]] = set()
    colors_used: list[int] = []

    for _obj_idx in range(n_objects):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            rect_h = random.randint(*RECT_H_RANGE)
            rect_w = random.randint(*RECT_W_RANGE)
            if rect_h >= h - 2 or rect_w >= w - 2:
                continue

            r0 = random.randint(1, h - rect_h - 1)
            c0 = random.randint(1, w - rect_w - 1)
            bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)

            expanded = _expanded_cells(bbox, h, w)
            if expanded & reserved_solids:
                continue
            if expanded & reserved_rays:
                continue

            gap_count = random.randint(*GAP_COUNT_RANGE)
            sides = random.sample(SIDES, gap_count)
            gaps = _sample_gaps(bbox, sides)

            ray_cells: set[tuple[int, int]] = set()
            valid = True
            for gap in gaps:
                side = _gap_side(gap, bbox)
                rcells = set(_ray_cells(gap, side, h, w))
                if rcells & reserved_solids:
                    valid = False
                    break
                if rcells & reserved_rays:
                    valid = False
                    break
                ray_cells |= rcells
            if not valid:
                continue

            tr = random.randint(r0 + 1, bbox[2] - 1)
            tc = random.randint(c0 + 1, bbox[3] - 1)
            tower = (tr, tc)

            if random.random() < 0.35 and colors_used:
                color = random.choice(colors_used)
            else:
                color = random.choice(TOWER_COLORS)
            colors_used.append(color)

            _draw_object(grid, bbox, gaps, tower, color)
            reserved_solids |= expanded
            reserved_rays |= ray_cells
            placed = True
            break
        if not placed:
            return None

    return grid


def _valid_input(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    if np.array_equal(out, grid):
        return False
    if not np.any((grid != BLACK) & (grid != BLUE)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(SCENE_RETRIES):
        grid = _place_scene()
        if grid is not None and _valid_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
