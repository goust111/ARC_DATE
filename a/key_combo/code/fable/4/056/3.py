from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
TEAL = 8
PALETTE = (YELLOW, GREEN, RED, TEAL)

GRID_MIN = 20
GRID_MAX = 30
CAVITY_COUNT_RANGE = (2, 4)
PLACEMENT_ATTEMPTS = 80
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _enclosed_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, grid.shape)
    passable = grid != GREY
    exterior = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def add(r: int, c: int) -> None:
        if passable[r, c] and not exterior[r, c]:
            exterior[r, c] = True
            queue.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(0, c)
        add(h - 1, c)

    while queue:
        r, c = queue.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w:
                add(rr, cc)

    seen = exterior.copy()
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not passable[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            seen[r, c] = True
            queue.append((r, c))
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _distance_map(cells: list[tuple[int, int]], source: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    comp_mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        comp_mask[r, c] = True

    dist = np.full(shape, -1, dtype=int)
    sr, sc = source
    dist[sr, sc] = 0
    queue: deque[tuple[int, int]] = deque([source])
    while queue:
        r, c = queue.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < shape[0] and 0 <= cc < shape[1] and comp_mask[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                queue.append((rr, cc))

    return dist


def _palette_step(marker_color: int, distance: int) -> int:
    return int(PALETTE[(PALETTE.index(int(marker_color)) + int(distance)) % len(PALETTE)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint concentric cyclic rings inside each marked cavity using geodesic distance from the marker.

    Input structure:
    - Black background cells surround several grey-walled enclosed cavities.
    - Each cavity interior is a 4-connected non-grey region; some cavities contain
      exactly one non-grey marker pixel and others are entirely black.
    - Marker colors come from the visible cycle yellow, green, red, teal.

    Rule:
    - Treat grey cells as walls and find every non-wall region that is not connected
      to the grid boundary.
    - For each enclosed region with exactly one marker, compute 4-neighbor shortest
      path distance from that marker through the cavity interior.
    - Recolor the whole marked cavity by advancing around the fixed palette
      yellow -> green -> red -> teal one step per distance layer; leave walls,
      the exterior, and unmarked cavities unchanged.

    Key insight:
    - The rings follow geodesic distance inside the cavity, so internal wall fins
      bend the color bands and make them differ from simple Manhattan circles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for component in _enclosed_components(g):
        markers = [(r, c) for r, c in component if int(g[r, c]) not in (BLACK, GREY)]
        assert len(markers) <= 1, "each cavity must have at most one marker"
        if not markers:
            continue

        marker = markers[0]
        marker_color = int(g[marker])
        assert marker_color in PALETTE, "marker must use the visible cyclic palette"

        dist = _distance_map(component, marker, g.shape)
        for r, c in component:
            assert dist[r, c] >= 0
            out[r, c] = _palette_step(marker_color, dist[r, c])

    return out


def _connected_walkable(tile: np.ndarray) -> bool:
    walkable = tile != GREY
    coords = np.argwhere(walkable)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        r, c = queue.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < tile.shape[0] and 0 <= cc < tile.shape[1] and walkable[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                queue.append((rr, cc))
    return len(seen) == int(np.count_nonzero(walkable))


def _make_cavity(style: str) -> np.ndarray:
    if style == "rect":
        inner_h = random.randint(4, 8)
        inner_w = random.randint(4, 9)
    elif style == "h_snake":
        inner_h = random.randint(5, 8)
        inner_w = random.randint(7, 11)
    else:
        inner_h = random.randint(7, 11)
        inner_w = random.randint(5, 8)

    tile = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
    tile[0, :] = GREY
    tile[-1, :] = GREY
    tile[:, 0] = GREY
    tile[:, -1] = GREY

    if style == "h_snake":
        step = random.choice((2, 3))
        cols = list(range(2, inner_w - 1, step))
        if not cols:
            cols = [inner_w // 2]
        for i, c in enumerate(cols):
            if i % 2 == 0:
                tile[1:inner_h, 1 + c] = GREY
            else:
                tile[2 : inner_h + 1, 1 + c] = GREY
    elif style == "v_snake":
        step = random.choice((2, 3))
        rows = list(range(2, inner_h - 1, step))
        if not rows:
            rows = [inner_h // 2]
        for i, r in enumerate(rows):
            if i % 2 == 0:
                tile[1 + r, 1:inner_w] = GREY
            else:
                tile[1 + r, 2 : inner_w + 1] = GREY

    assert _connected_walkable(tile), "cavity interior must stay connected"
    return tile


def _pick_marker_cell(tile: np.ndarray) -> tuple[int, int]:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(tile == BLACK)]
    anchor = random.choice(("tl", "tr", "bl", "br"))
    if anchor == "tl":
        score = lambda cell: (cell[0] + cell[1], random.random())
    elif anchor == "tr":
        score = lambda cell: (cell[0] - cell[1], random.random())
    elif anchor == "bl":
        score = lambda cell: (-cell[0] + cell[1], random.random())
    else:
        score = lambda cell: (-cell[0] - cell[1], random.random())
    cells.sort(key=score)
    return random.choice(cells[: min(6, len(cells))])


def _can_place(occupied: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(occupied.shape[0], top + h + 1)
    c1 = min(occupied.shape[1], left + w + 1)
    return not np.any(occupied[r0:r1, c0:c1])


def _place_tile(grid: np.ndarray, occupied: np.ndarray, tile: np.ndarray) -> bool:
    h, w = map(int, tile.shape)
    if grid.shape[0] - h - 1 < 1 or grid.shape[1] - w - 1 < 1:
        return False
    for _ in range(PLACEMENT_ATTEMPTS):
        top = random.randint(1, grid.shape[0] - h - 1)
        left = random.randint(1, grid.shape[1] - w - 1)
        if not _can_place(occupied, top, left, tile.shape):
            continue
        grid[top : top + h, left : left + w] = tile
        occupied[top : top + h, left : left + w] = True
        return True
    return False


def _has_detour(grid: np.ndarray) -> bool:
    for component in _enclosed_components(grid):
        markers = [(r, c) for r, c in component if int(grid[r, c]) in PALETTE]
        if len(markers) != 1:
            continue
        marker = markers[0]
        dist = _distance_map(component, marker, grid.shape)
        mr, mc = marker
        for r, c in component:
            if dist[r, c] > abs(r - mr) + abs(c - mc):
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    cavity_count = random.randint(*CAVITY_COUNT_RANGE)
    marked_count = random.randint(1, cavity_count)
    if marked_count == cavity_count and cavity_count > 1 and random.random() < 0.55:
        marked_count -= 1

    marked_indices = set(random.sample(range(cavity_count), marked_count))
    force_detour = bool(marked_indices) and random.random() < 0.75
    forced_index = random.choice(tuple(marked_indices)) if force_detour else None

    for idx in range(cavity_count):
        if idx == forced_index:
            style = random.choice(("h_snake", "v_snake"))
        else:
            style = random.choices(
                ["rect", "h_snake", "v_snake"],
                weights=[3, 2, 2],
                k=1,
            )[0]

        tile = _make_cavity(style)
        if idx in marked_indices:
            marker_r, marker_c = _pick_marker_cell(tile)
            tile[marker_r, marker_c] = random.choice(PALETTE)

        if not _place_tile(grid, occupied, tile):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(out != grid):
        return None
    if force_detour and not _has_detour(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid geodesic-ripple input")
