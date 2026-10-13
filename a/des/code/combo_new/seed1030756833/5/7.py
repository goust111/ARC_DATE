from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
KEY_COLORS = (2, 3, 4, 6, 7, 8, 9)

GENERATION_RETRIES = 500
ROOM_PLACE_RETRIES = 350


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr in range(h):
        for sc in range(w):
            if not bool(mask[sr, sc]) or bool(seen[sr, sc]):
                continue
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            comp: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells = [(r0, c) for c in range(c0, c1 + 1)]
    cells.extend((r1, c) for c in range(c0, c1 + 1))
    cells.extend((r, c0) for r in range(r0 + 1, r1))
    cells.extend((r, c1) for r in range(r0 + 1, r1))
    return cells


def _find_rooms(g: np.ndarray) -> list[dict]:
    rooms: list[dict] = []
    blue_components = _components4(g == BLUE)
    assert blue_components, "expected at least one blue rectangular frame"

    for comp in blue_components:
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frames must enclose an interior"

        border = set(_border_cells(r0, c0, r1, c1))
        assert set(comp) == border, "each blue component must be one closed rectangular border"

        keys: list[tuple[int, int, int]] = []
        locks: list[tuple[int, int]] = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                value = int(g[r, c])
                if value == BLACK:
                    continue
                if value == GREY:
                    locks.append((r, c))
                    continue
                assert value != BLUE, "room interiors may not contain blue wall pixels"
                keys.append((r, c, value))

        assert len(keys) == 1, "each room must contain exactly one key pixel"
        assert 1 <= len(locks) <= 6, "each room must contain one to six grey locks"
        kr, kc, key_color = keys[0]
        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "key": (kr, kc),
                "color": key_color,
                "locks": locks,
            }
        )

    key_colors = [int(room["color"]) for room in rooms]
    assert len(key_colors) == len(set(key_colors)), "key colors must be unique"
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unlock the nearest grey locks in each blue-framed room according to the global key-color order.

    Input structure:
    - Black is the background.
    - Blue pixels form several separated one-pixel-thick closed rectangular room frames.
    - Each room interior contains exactly one key pixel whose color is not Black, Blue, or Grey.
    - Each room interior also contains one to six Grey lock pixels.

    Rule:
    - Read all key colors by the first position where their key appears, scanning top-to-bottom and then left-to-right.
    - The 1-based rank R of a room's key color is the number of Grey locks to remove in that room.
    - Within a room, remove locks from nearest to farthest by Manhattan distance to the key, breaking ties by row and then column.
    - If the room has fewer than R locks, remove every lock; keys and blue frames are preserved.

    Key insight:
    - A room cannot be solved in isolation: the key's global reading-order rank decides how many locks are opened, and local distance order decides which locks open first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rooms = _find_rooms(g)
    ordered = sorted(rooms, key=lambda room: room["key"])
    color_rank = {int(room["color"]): rank for rank, room in enumerate(ordered, start=1)}

    out = g.copy()
    for room in rooms:
        kr, kc = room["key"]
        rank = int(color_rank[int(room["color"])])
        locks = sorted(
            room["locks"],
            key=lambda pos: (abs(pos[0] - kr) + abs(pos[1] - kc), pos[0], pos[1]),
        )
        for r, c in locks[:rank]:
            out[r, c] = BLACK

    return out


def _paint_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    for r, c in _border_cells(r0, c0, r1, c1):
        grid[r, c] = BLUE


def _try_place_rooms(height: int, width: int, count: int) -> list[tuple[int, int, int, int]] | None:
    reserved = np.zeros((height, width), dtype=bool)
    rooms: list[tuple[int, int, int, int]] = []

    for _ in range(count):
        placed = False
        for _attempt in range(ROOM_PLACE_RETRIES):
            rect_h = random.randint(5, 8)
            rect_w = random.randint(5, 8)
            if rect_h + 2 > height or rect_w + 2 > width:
                continue

            r0 = random.randint(1, height - rect_h - 1)
            c0 = random.randint(1, width - rect_w - 1)
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1

            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(height - 1, r1 + 1)
            cc1 = min(width - 1, c1 + 1)
            if reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue

            reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            rooms.append((r0, c0, r1, c1))
            placed = True
            break

        if not placed:
            return None

    return rooms


def _sample_lock_count(rank: int, partial_rank: int, shortage_rank: int) -> int:
    if rank == partial_rank:
        return random.randint(rank + 1, 6)
    if rank == shortage_rank:
        return random.randint(1, rank - 1)
    return random.randint(1, 6)


def _gen_attempt() -> np.ndarray | None:
    room_count = random.randint(3, 6)
    min_dim = 18 if room_count <= 4 else 24 if room_count == 5 else 27
    height = random.randint(min_dim, 30)
    width = random.randint(min_dim, 30)

    rooms = _try_place_rooms(height, width, room_count)
    if rooms is None:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    colors = random.sample(KEY_COLORS, room_count)
    specs: list[dict] = []

    for idx, (r0, c0, r1, c1) in enumerate(rooms):
        _paint_frame(grid, r0, c0, r1, c1)
        interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        key = random.choice(interior)
        grid[key] = int(colors[idx])
        specs.append({"bbox": (r0, c0, r1, c1), "key": key, "color": int(colors[idx])})

    ranked_specs = sorted(specs, key=lambda spec: spec["key"])
    shortage_rank = random.randint(2, room_count)
    partial_options = [rank for rank in range(1, min(room_count, 5) + 1) if rank != shortage_rank]
    if not partial_options:
        return None
    partial_rank = random.choice(partial_options)

    for rank, spec in enumerate(ranked_specs, start=1):
        r0, c0, r1, c1 = spec["bbox"]
        cells = [
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if (r, c) != spec["key"]
        ]
        lock_count = _sample_lock_count(rank, partial_rank, shortage_rank)
        if lock_count > len(cells):
            return None
        for r, c in random.sample(cells, lock_count):
            grid[r, c] = GREY

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid seed1030756833/5/7 input")
