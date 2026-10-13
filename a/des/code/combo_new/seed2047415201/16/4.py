from __future__ import annotations

from collections import deque
import random

import numpy as np


AZURE = 8
GREEN = 3
YELLOW = 4
MAROON = 9

CARDINAL_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = map(int, mask.shape)
    seen = np.zeros((height, width), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr in range(height):
        for sc in range(width):
            if seen[sr, sc] or not bool(mask[sr, sc]):
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            comp: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                comp.append((int(r), int(c)))
                for rr, cc in _neighbors4(r, c):
                    if not (0 <= rr < height and 0 <= cc < width):
                        continue
                    if seen[rr, cc] or not bool(mask[rr, cc]):
                        continue
                    seen[rr, cc] = True
                    q.append((rr, cc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _arrow_direction(cells: list[tuple[int, int]], height: int, width: int) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(cells)
    assert len(cells) == 3, "each arrow legend must be a three-pixel L"
    assert r1 == r0 + 1 and c1 == c0 + 1, "arrow legend must fit a 2x2 box"

    full = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    missing = list(full - set(cells))
    assert len(missing) == 1, "arrow L must have exactly one open corner"
    mr, mc = missing[0]
    open_dr = -1 if mr == r0 else 1
    open_dc = -1 if mc == c0 else 1

    touches_horizontal_edge = r0 == 0 or r1 == height - 1
    touches_vertical_edge = c0 == 0 or c1 == width - 1
    if touches_horizontal_edge and not touches_vertical_edge:
        return open_dr, 0
    if touches_vertical_edge and not touches_horizontal_edge:
        return 0, open_dc
    return open_dr, open_dc


def _find_arrows(g: np.ndarray) -> list[dict]:
    height, width = map(int, g.shape)
    arrows = []
    for comp in _components(g == GREEN):
        if not any(r in (0, height - 1) or c in (0, width - 1) for r, c in comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if len(comp) != 3 or r1 != r0 + 1 or c1 != c0 + 1:
            continue
        arrows.append({"cells": comp, "direction": _arrow_direction(comp, height, width)})
    return arrows


def _find_regions(g: np.ndarray) -> list[dict]:
    height, width = map(int, g.shape)
    passable = (g == AZURE) | (g == YELLOW)
    regions = []
    for comp in _components(passable):
        if any(r in (0, height - 1) or c in (0, width - 1) for r, c in comp):
            continue
        cells = set(comp)
        walls: set[tuple[int, int]] = set()
        for r, c in cells:
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells and int(g[rr, cc]) == MAROON:
                    walls.add((int(rr), int(cc)))
        if walls:
            regions.append({"cells": cells, "walls": walls, "bbox": _bbox(cells)})
    regions.sort(key=lambda reg: reg["bbox"])
    return regions


def _manhattan_between(a: list[tuple[int, int]] | set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    return min(abs(ar - br) + abs(ac - bc) for ar, ac in a for br, bc in b)


def _assign_directions(arrows: list[dict], regions: list[dict]) -> dict[int, tuple[int, int]]:
    assigned: dict[int, tuple[int, tuple[int, int]]] = {}
    for arrow in arrows:
        ranked = [
            (_manhattan_between(arrow["cells"], reg["walls"]), idx)
            for idx, reg in enumerate(regions)
        ]
        ranked.sort()
        dist, idx = ranked[0]
        previous = assigned.get(idx)
        if previous is None or dist < previous[0]:
            assigned[idx] = (dist, arrow["direction"])
    return {idx: direction for idx, (_dist, direction) in assigned.items()}


def _slide_points(
    region: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    ordered = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    stopped: set[tuple[int, int]] = set()
    result: list[tuple[int, int]] = []
    for r0, c0 in ordered:
        r, c = int(r0), int(c0)
        while True:
            rr, cc = r + dr, c + dc
            if (rr, cc) not in region or (rr, cc) in stopped:
                break
            if dr != 0 and dc != 0 and ((r + dr, c) not in region or (r, c + dc) not in region):
                break
            r, c = rr, cc
        stopped.add((r, c))
        result.append((r, c))
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside maroon rooms according to nearby green edge-arrow legends.

    Input structure:
    - The grid background is Azure; Maroon one-cell walls enclose several rectangular regions.
    - Each enclosed region contains one or more Yellow floating pixels.
    - Green three-pixel L legends sit on the grid edge; each legend is spatially closest to the wall of the region it controls.

    Rule:
    - Read each L legend's open side as a movement direction, and associate that legend to the enclosed region with the nearest Maroon wall segment.
    - In every controlled region, clear the original Yellow pixels and slide them along the legend direction until the next step would hit a Maroon wall or an already stopped Yellow pixel.
    - Process pixels from the destination side inward so Yellow pixels stack against the collision wall while preserving row/column order.
    - Erase the Green edge legends to Azure.

    Key insight:
    The legends are not matched by color; their nearest wall segment identifies the target region, while the L opening encodes that region's local gravity direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height <= 30 and width <= 30, "ARC grids must fit within 30x30"

    arrows = _find_arrows(g)
    regions = _find_regions(g)
    assert arrows and regions, "expected edge arrows and enclosed regions"
    directions = _assign_directions(arrows, regions)

    out = g.copy()
    for arrow in arrows:
        for r, c in arrow["cells"]:
            out[r, c] = AZURE

    for idx, region in enumerate(regions):
        if idx not in directions:
            continue
        dr, dc = directions[idx]
        movers = [(r, c) for r, c in region["cells"] if int(g[r, c]) == YELLOW]
        if not movers:
            continue
        for r, c in movers:
            out[r, c] = AZURE
        for r, c in _slide_points(region["cells"], movers, dr, dc):
            out[r, c] = YELLOW

    return out


def _wall_cells(room: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = room
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _interior_cells(room: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = room
    return {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 2) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_rooms(height: int, width: int, count: int) -> list[tuple[int, int, int, int]] | None:
    rooms: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        for _attempt in range(160):
            room_h = random.randint(5, min(8, height - 4))
            room_w = random.randint(5, min(9, width - 4))
            if room_h >= height - 2 or room_w >= width - 2:
                continue
            r0 = random.randint(2, height - room_h - 2)
            c0 = random.randint(2, width - room_w - 2)
            room = (r0, c0, r0 + room_h - 1, c0 + room_w - 1)
            if all(not _boxes_too_close(room, prev) for prev in rooms):
                rooms.append(room)
                break
        else:
            return None
    return rooms


def _arrow_cells_for_direction(
    height: int,
    width: int,
    direction: tuple[int, int],
    anchor: int,
    edge: str,
    variant: int,
) -> list[tuple[int, int]] | None:
    dr, dc = direction
    if edge == "top":
        r0, c0 = 0, anchor
    elif edge == "bottom":
        r0, c0 = height - 2, anchor
    elif edge == "left":
        r0, c0 = anchor, 0
    elif edge == "right":
        r0, c0 = anchor, width - 2
    else:
        return None

    if not (0 <= r0 < height - 1 and 0 <= c0 < width - 1):
        return None

    if dr < 0:
        missing = (r0, c0 + variant)
    elif dr > 0:
        missing = (r0 + 1, c0 + variant)
    elif dc < 0:
        missing = (r0 + variant, c0)
    else:
        missing = (r0 + variant, c0 + 1)

    full = [(r, c) for r in (r0, r0 + 1) for c in (c0, c0 + 1)]
    return [(r, c) for r, c in full if (r, c) != missing]


def _reserved_ok(cells: list[tuple[int, int]], reserved: set[tuple[int, int]], height: int, width: int) -> bool:
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) in reserved:
                    return False
    return True


def _candidate_arrows(
    room: tuple[int, int, int, int],
    direction: tuple[int, int],
    height: int,
    width: int,
) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = room
    row_center = (r0 + r1) // 2
    col_center = (c0 + c1) // 2
    candidates: list[list[tuple[int, int]]] = []

    if direction[0] != 0:
        edges = ["top", "bottom"]
        base = col_center - 1
        anchors = [base + delta for delta in range(-4, 5)]
        random.shuffle(edges)
        random.shuffle(anchors)
        for edge in edges:
            for anchor in anchors:
                if not (1 <= anchor <= width - 3):
                    continue
                for variant in random.sample([0, 1], 2):
                    cells = _arrow_cells_for_direction(height, width, direction, anchor, edge, variant)
                    if cells is not None:
                        candidates.append(cells)
    else:
        edges = ["left", "right"]
        base = row_center - 1
        anchors = [base + delta for delta in range(-4, 5)]
        random.shuffle(edges)
        random.shuffle(anchors)
        for edge in edges:
            for anchor in anchors:
                if not (1 <= anchor <= height - 3):
                    continue
                for variant in random.sample([0, 1], 2):
                    cells = _arrow_cells_for_direction(height, width, direction, anchor, edge, variant)
                    if cells is not None:
                        candidates.append(cells)
    return candidates


def _closest_room_index(cells: list[tuple[int, int]], rooms: list[tuple[int, int, int, int]]) -> int:
    distances = [(_manhattan_between(cells, _wall_cells(room)), idx) for idx, room in enumerate(rooms)]
    distances.sort()
    return distances[0][1]


def _place_arrows(
    rooms: list[tuple[int, int, int, int]],
    directions: list[tuple[int, int]],
    height: int,
    width: int,
) -> list[list[tuple[int, int]]] | None:
    reserved: set[tuple[int, int]] = set()
    arrows: list[list[tuple[int, int]]] = []
    order = list(range(len(rooms)))
    random.shuffle(order)
    chosen: dict[int, list[tuple[int, int]]] = {}

    for idx in order:
        candidates = _candidate_arrows(rooms[idx], directions[idx], height, width)
        random.shuffle(candidates)
        for cells in candidates:
            if not _reserved_ok(cells, reserved, height, width):
                continue
            ranked = [(_manhattan_between(cells, _wall_cells(room)), room_idx) for room_idx, room in enumerate(rooms)]
            ranked.sort()
            if ranked[0][1] != idx:
                continue
            if len(ranked) > 1 and ranked[0][0] >= ranked[1][0]:
                continue
            chosen[idx] = cells
            for r, c in cells:
                reserved.add((r, c))
            break
        else:
            return None

    for idx in range(len(rooms)):
        arrows.append(chosen[idx])
    return arrows


def _paint_room(grid: np.ndarray, room: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = room
    grid[r0 : r1 + 1, c0] = MAROON
    grid[r0 : r1 + 1, c1] = MAROON
    grid[r0, c0 : c1 + 1] = MAROON
    grid[r1, c0 : c1 + 1] = MAROON


def _yellow_candidates(room: tuple[int, int, int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room
    cells = sorted(_interior_cells(room))
    dr, dc = direction
    if dr < 0:
        cells = [(r, c) for r, c in cells if r > r0 + 1]
    elif dr > 0:
        cells = [(r, c) for r, c in cells if r < r1 - 1]
    elif dc < 0:
        cells = [(r, c) for r, c in cells if c > c0 + 1]
    else:
        cells = [(r, c) for r, c in cells if c < c1 - 1]
    random.shuffle(cells)
    return cells


def _expected_output(
    grid: np.ndarray,
    rooms: list[tuple[int, int, int, int]],
    directions: list[tuple[int, int]],
    arrows: list[list[tuple[int, int]]],
) -> np.ndarray:
    out = grid.copy()
    for cells in arrows:
        for r, c in cells:
            out[r, c] = AZURE
    for room, (dr, dc) in zip(rooms, directions):
        region = _interior_cells(room)
        movers = [(r, c) for r, c in region if int(grid[r, c]) == YELLOW]
        for r, c in movers:
            out[r, c] = AZURE
        for r, c in _slide_points(region, movers, dr, dc):
            out[r, c] = YELLOW
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 30)
    width = random.randint(18, 30)
    room_count = random.randint(2, 4)
    rooms = _place_rooms(height, width, room_count)
    if rooms is None:
        return None
    rooms.sort()

    directions = random.sample(list(CARDINAL_DIRS), room_count)
    arrows = _place_arrows(rooms, directions, height, width)
    if arrows is None:
        return None

    grid = np.full((height, width), AZURE, dtype=int)
    for room in rooms:
        _paint_room(grid, room)
    for cells in arrows:
        for r, c in cells:
            grid[r, c] = GREEN

    for room, direction in zip(rooms, directions):
        candidates = _yellow_candidates(room, direction)
        if not candidates:
            return None
        count = random.randint(1, min(4, len(candidates)))
        for r, c in candidates[:count]:
            grid[r, c] = YELLOW

    expected = _expected_output(grid, rooms, directions, arrows)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
