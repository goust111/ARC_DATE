from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            comps.append(
                {
                    "color": color,
                    "coords": tuple(sorted(cells)),
                    "area": len(cells),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _drop_coords(
    coords: tuple[tuple[int, int], ...], occupancy: np.ndarray, h: int, w: int
) -> tuple[tuple[int, int], ...]:
    shifted = list(coords)
    while True:
        next_cells = []
        blocked = False
        for r, c in shifted:
            nr = r + 1
            if nr >= h or occupancy[nr, c]:
                blocked = True
                break
            next_cells.append((nr, c))
        if blocked:
            return tuple(shifted)
        shifted = next_cells


def _extract_objects_and_markers(grid: np.ndarray) -> tuple[int, list[dict], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    bg = _most_common_color(g)
    comps = _find_components(g, bg)
    assert comps, "expected markers and floating objects"

    markers: dict[int, int] = {}
    objects: list[dict] = []
    for comp in comps:
        coords = comp["coords"]
        if comp["area"] == 1 and coords[0][0] == h - 1:
            color = int(comp["color"])
            assert color not in markers, "marker colors must be unique"
            markers[color] = int(coords[0][1])
        else:
            objects.append(comp)

    assert objects, "expected floating objects"
    assert len(objects) == len(markers), "objects and same-color ground markers must match one-to-one"
    object_colors = {int(obj["color"]) for obj in objects}
    assert object_colors == set(markers), "every object must have exactly one same-color ground marker"
    return bg, objects, markers


def _simulate_from_parts(
    shape: tuple[int, int], bg: int, objects: list[dict], markers: dict[int, int], descending: bool
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    occupancy = np.zeros((h, w), dtype=bool)
    ordered = sorted(
        objects,
        key=lambda obj: (
            -int(obj["area"]) if descending else int(obj["area"]),
            int(obj["bbox"][0]),
            int(obj["bbox"][1]),
        ),
    )

    for obj in ordered:
        color = int(obj["color"])
        r0, c0, r1, c1 = obj["bbox"]
        width = c1 - c0 + 1
        assert width % 2 == 1, "object bbox width must be odd"
        center_col = (c0 + c1) // 2
        target_col = int(markers[color])
        dc = target_col - center_col
        shifted = tuple((int(r), int(c + dc)) for r, c in obj["coords"])
        assert all(0 <= r < h and 0 <= c < w for r, c in shifted), "aligned object must stay in bounds"
        landed = _drop_coords(shifted, occupancy, h, w)
        for r, c in landed:
            occupancy[r, c] = True
            out[r, c] = color

    for color, col in markers.items():
        if not occupancy[h - 1, col]:
            out[h - 1, col] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align each floating object to its same-color ground marker column, then drop larger objects first to build the stack.

    Input structure:
    - The background is black, with several disconnected monochromatic floating objects above the floor.
    - Each floating object has a unique color and an odd-width bounding box.
    - The bottom row contains one isolated marker pixel per object, using the same color as that object.

    Rule:
    - For each object, horizontally translate it so the center column of its bounding box matches the column of its same-color marker.
    - Then process the objects in strictly decreasing area order and let each translated shape fall straight down.
    - A falling object stops when any of its cells would move below the grid or onto a previously landed object; uncovered ground markers remain visible.

    Key insight:
    - The marker only chooses the destination column, while the final pile depends on dropping the aligned shapes in descending area order rather than by position.
    """
    g = np.asarray(grid, dtype=int)
    bg, objects, markers = _extract_objects_and_markers(g)
    return _simulate_from_parts(g.shape, bg, objects, markers, descending=True)


def _random_connected_shape(target_area: int, width: int, max_height: int) -> tuple[tuple[int, int], ...] | None:
    for _ in range(80):
        height = random.randint(2, max_height)
        start = (random.randrange(height), random.randrange(width))
        cells = {start}
        frontier = set()

        def add_frontier(cell: tuple[int, int]) -> None:
            r, c = cell
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        add_frontier(start)
        while len(cells) < target_area and frontier:
            nxt = random.choice(tuple(frontier))
            frontier.remove(nxt)
            cells.add(nxt)
            add_frontier(nxt)

        if len(cells) != target_area:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if 0 not in cols or width - 1 not in cols:
            continue
        r0, c0 = min(rows), min(cols)
        norm = tuple(sorted((r - r0, c - c0) for r, c in cells))
        return norm
    return None


def _make_unique_shapes(n_objects: int) -> list[dict] | None:
    widths = [3, 5, 7]
    area_choices = list(range(4, 15))
    random.shuffle(area_choices)
    used_areas: set[int] = set()
    shapes: list[dict] = []

    for _ in range(n_objects):
        random.shuffle(area_choices)
        made = False
        for area in area_choices:
            if area in used_areas:
                continue
            possible_widths = [w for w in widths if w <= area and area <= w * 4]
            random.shuffle(possible_widths)
            for width in possible_widths:
                coords = _random_connected_shape(area, width, max_height=4)
                if coords is None:
                    continue
                rows = [r for r, _ in coords]
                cols = [c for _, c in coords]
                shapes.append(
                    {
                        "coords": coords,
                        "area": area,
                        "height": max(rows) + 1,
                        "width": max(cols) + 1,
                    }
                )
                used_areas.add(area)
                made = True
                break
            if made:
                break
        if not made:
            return None
    return shapes


def _can_place(
    occupied: np.ndarray, coords: tuple[tuple[int, int], ...], top: int, left: int, pad: int
) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        rr = top + r
        cc = left + c
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if occupied[rr, cc]:
            return False
        for nr in range(max(0, rr - pad), min(h, rr + pad + 1)):
            for nc in range(max(0, cc - pad), min(w, cc + pad + 1)):
                if occupied[nr, nc]:
                    return False
    return True


def _stamp(grid: np.ndarray, occupied: np.ndarray, coords: tuple[tuple[int, int], ...], top: int, left: int, color: int) -> tuple[tuple[int, int], ...]:
    placed = []
    for r, c in coords:
        rr = top + r
        cc = left + c
        grid[rr, cc] = int(color)
        occupied[rr, cc] = True
        placed.append((rr, cc))
    return tuple(placed)


def _place_initial_objects(grid: np.ndarray, shapes: list[dict], colors: list[int]) -> list[dict] | None:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    order = list(range(len(shapes)))
    random.shuffle(order)
    for idx in order:
        shape = shapes[idx]
        coords = shape["coords"]
        obj_h = shape["height"]
        obj_w = shape["width"]
        candidates = []
        row_max = h - obj_h - 5
        if row_max < 1:
            return None
        for top in range(1, row_max + 1):
            for left in range(0, w - obj_w + 1):
                if _can_place(occupied, coords, top, left, pad=1):
                    candidates.append((top, left))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        color = colors[idx]
        placed = _stamp(grid, occupied, coords, top, left, color)
        rows = [r for r, _ in placed]
        cols = [c for _, c in placed]
        objects.append(
            {
                "color": color,
                "coords": placed,
                "area": len(placed),
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
            }
        )
    return objects


def _marker_columns(widths: list[int], grid_w: int) -> list[int] | None:
    half_max = max(w // 2 for w in widths)
    center_min = half_max + 2
    center_max = grid_w - half_max - 3
    if center_min > center_max:
        return None

    offset_pool = [-3, -2, -1, 0, 1, 2, 3]
    for _ in range(80):
        center = random.randint(center_min, center_max)
        random.shuffle(offset_pool)
        chosen = sorted(offset_pool[: len(widths)])
        cols = [center + off for off in chosen]
        if len(set(cols)) != len(cols):
            continue
        if all((w // 2) <= c <= grid_w - 1 - (w // 2) for c, w in zip(cols, widths)):
            if max(cols) - min(cols) <= 6:
                return cols
    return None


def _build_candidate() -> np.ndarray | None:
    h = random.randint(15, 22)
    w = random.randint(16, 26)
    n_objects = random.randint(3, 5)
    shapes = _make_unique_shapes(n_objects)
    if shapes is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    colors = random.sample(PALETTE, n_objects)
    objects = _place_initial_objects(grid, shapes, colors)
    if objects is None:
        return None

    objects_by_color = {int(obj["color"]): obj for obj in objects}
    widths = [objects_by_color[color]["bbox"][3] - objects_by_color[color]["bbox"][1] + 1 for color in colors]
    cols = _marker_columns(widths, w)
    if cols is None:
        return None

    markers: dict[int, int] = {}
    for color, col in zip(colors, cols):
        grid[h - 1, col] = int(color)
        markers[int(color)] = int(col)

    moved = False
    for obj in objects:
        center = (obj["bbox"][1] + obj["bbox"][3]) // 2
        if markers[int(obj["color"])] != center:
            moved = True
            break
    if not moved:
        return None

    out_desc = _simulate_from_parts(grid.shape, BG, objects, markers, descending=True)
    out_asc = _simulate_from_parts(grid.shape, BG, objects, markers, descending=False)
    if np.array_equal(out_desc, out_asc):
        return None
    if not np.array_equal(main(grid), out_desc):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid non-degenerate input")
